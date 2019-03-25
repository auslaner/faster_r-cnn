import argparse
import logging
import os
import random
import sqlite3
from sqlite3 import Error

import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PIL import Image
from utils.tfannotation import TFAnnotation

from config import pollinator_config as config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to save record files")
args = vars(ap.parse_args())


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None


def get_filename(frame_number, count, video, frame=False):
    if frame:
        file_name = "-".join([video[:-4], "frame", str(frame_number), str(count)]) + ".png"
    else:
        file_name = "-".join([video[:-4], str(frame_number), str(count)]) + ".png"
    return file_name


def select_all_examples(conn):
    """
    Query all pollinator log entries in the logentry table.
    :param conn: the Connection object
    :return: List of database rows
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM tfodinfo;")

    rows = cur.fetchall()

    return rows


def path_exists(p):
    image = cv2.imread(p)
    return image is not None


def main(_):
    # Open the classes output file
    f = open(os.path.sep.join([args["output"], "classes.pbtxt"]), "w")

    # Loop over the classes
    for k, v in config.CLASSES.items():
        # Construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)

    # Close the output classes file
    f.close()

    # Grab a connection to the database file
    conn = create_connection("log.db")

    # Initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    rows = select_all_examples(conn)

    # Keep a count for verification images
    vimg_cnt = 0

    # Loop over the individual rows, skipping the header
    for num, row in enumerate(rows):
        print("*" * 35)
        print("[*] Looping over database row {}...\n".format(num))
        # Break the row into components
        _, _, frame_fname, label, start_x, start_y, end_x, end_y, multi_poll = row

        logging.debug("Start X: {}".format(start_x))
        logging.debug("Start Y: {}".format(start_y))
        logging.debug("End X: {}".format(end_x))
        logging.debug("End Y: {}".format(end_y))

        # Minor classes are pooled into 'Other' class
        label = label if label in config.CLASSES.keys() else "Other"

        # Adapt file path for server
        frame_fname = frame_fname.split("/")[-1]

        logging.debug("Frame Filename: {}".format(frame_fname))

        # Build the path to the input image, then grab any other
        # bounding boxes + label associated with the image
        # path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, "Frames", "Pollinator", frame_fname])
        if path_exists(p):
            b = D.get(p, [])
    
            logging.debug("Frame Path: {}".format(p))
    
            # Build a tuple consisting of the label and bounding box,
            # then update the list and store it in the dictionary
            b.append((label, (start_x, start_y, end_x, end_y)))
            D[p] = b
        else:
            print("[!] Image not found at path {}".format(p))

    # Create training and testing splits from our data dictionary
    train_keys, test_keys = train_test_split(list(D.keys()),
                                             test_size=config.TEST_SIZE, random_state=42)
    # Split the test keys further to create a validation set
    test_keys, validation_keys = train_test_split(test_keys, random_state=42)

    # Initialize the data split files
    datasets = [
        ("train", train_keys, os.path.sep.join([args["output"], "training.record"])),
        ("test", test_keys, os.path.sep.join([args["output"], "testing.record"])),
        ("validation", validation_keys, os.path.sep.join([args["output"], "validation.record"]))
    ]

    # Loop over the datasets
    for d_type, keys, output_path in datasets:
        # Initialize the TensorFlow writer and intialize the total
        # number of examples written to file
        print("[INFO] Processing '{}'...".format(d_type))
        writer = tf.python_io.TFRecordWriter(output_path)
        total = 0

        # Loop over all the keys in the current set
        for k in keys:
            # Load the input image from disk as a TensorFlow object
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)

            # Load the image from disk again, this time as a PIL
            # object
            pil_image = Image.open(k)
            w, h = pil_image.size[:2]

            # Parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            # Initialize the annotation object used to store
            # information regarding the bounding box + labels
            tf_annot = TFAnnotation()
            tf_annot.image = encoded
            tf_annot.encoding = encoding
            tf_annot.filename = filename
            tf_annot.width = w
            tf_annot.height = h

            # Loop over the bounding boxes + labels associated with
            # the image
            for label, (start_x, start_y, end_x, end_y) in D[k]:
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                x_min = start_x / w
                x_max = end_x / w
                y_min = start_y / h
                y_max = end_y / h

                """
                Occasionally save an image for manual inspection
                """
                if config.VISUALLY_CHECK and random.randint(0, 9) == 0:
                    # Load the input image from disk and denormalize the
                    # bounding box coordinates
                    image = cv2.imread(k)
                    start_x = int(x_min * w)
                    start_y = int(y_min * h)
                    end_x = int(x_max * w)
                    end_y = int(y_max * h)

                    # Draw the bounding box on the image
                    cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                                  (0, 255, 0), 2)

                    # Add the label
                    cv2.putText(image, label, (start_x - 10, start_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0),
                                2, cv2.LINE_AA)

                    # Save the labeled image
                    print("[*] Saving verification image {}...".format(vimg_cnt))
                    cv2.imwrite("Verify-" + str(total) + ".png", image)
                    vimg_cnt += 1

                # Update the bounding boxes + label lists
                tf_annot.x_mins.append(x_min)
                tf_annot.x_maxs.append(x_max)
                tf_annot.y_mins.append(y_min)
                tf_annot.y_maxs.append(y_max)
                tf_annot.text_labels.append(label.encode("utf8"))
                tf_annot.classes.append(config.CLASSES[label])
                tf_annot.difficult.append(0)

                # Increment the total number of examples
                total += 1

            # Encode the data point attributes using the Tensorflow
            # helper functions
            features = tf.train.Features(feature=tf_annot.build())
            example = tf.train.Example(features=features)

            # Add the example to the writer
            writer.write(example.SerializeToString())

        # Close the writer and print diagnostic information to the
        # user
        writer.close()
        print("[INFO] {} examples saved for '{}'".format(total,
                                                         d_type))


# Check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()
