from config import lisa_config as config
from utils.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os


def main(_):
    # Open the classes output file
    f = open(config.CLASSES_FILE, "w")

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

    # Initialize a data dictionary used to map each image filename
    # to all bounding boxes associated with the image, then load
    # the contents of the annotations file
    D = {}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")

    # Loop over the individual rows, skipping the header
    for row in rows[1:]:
        # Break the row into components
        row = row.split(",")[0].split(";")
        image_path, label, start_x, start_y, end_x, end_y, _ = row
        start_x, start_y = float(start_x), float(start_y)
        end_x, end_y = float(end_x), float(end_y)

        # if we are not interested in the label, ignore it
        if label not in config.CLASSES:
            continue

        # Build the path to the input image, then grab any other
        # bounding boxes + label associated with the image
        # path, labels, and bounding box lists, respectively
        p = os.path.sep.join([config.BASE_PATH, image_path])
        b = D.get(p, [])

        # Build a tuple consisting of the label and bounding box,
        # then update the list and store it in the dictionary
        b.append((label, start_x, start_y, end_x, end_y))
        D[p] = b

    # Create training and testing splits from our data dictionary
    train_keys, test_keys = train_test_split(list(D.keys()),
                                             test_size=config.TEST_SIZE, random_state=42)

    # Initialize the data split files
    datasets = [
        ("train", train_keys, config.TRAIN_RECORD),
        ("test", test_keys, config.TEST_RECORD)
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
