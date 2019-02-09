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
