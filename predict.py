import argparse
import cv2
import imutils
import numpy as np
import os
import tensorflow as tf

from imutils.video import FileVideoStream
from object_detection.utils import label_map_util

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
                help="labels file")
ap.add_argument("-v", "--video", required=True,
                help="path to video file")
ap.add_argument("-n", "--num-classes", type=int, required=True,
                help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# Initialize a set of colors for our class labels
COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

# Initialize the model
model = tf.Graph()

# Create a context manager that makes this model the default one for
# execution
with model.as_default():
    # Initialize the graph definition
    graph_def = tf.GraphDef()

    # Load the graph from disk
    with tf.gfile.GFile(args["model"], "rb") as f:
        serialized_graph = f.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name="")

# Load the class labels from disk
label_map = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=args["num_classes"],
    use_display_name=True
)
category_idx = label_map_util.create_category_index(categories)

vs = FileVideoStream(args["video"]).start()
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
model_num = args["model"].split(os.path.sep)[-3]
annotated_video = cv2.VideoWriter("annotated_video" + "-" + model_num + ".avi", fourcc, 5, (640, 480))

# Create a session to perform inference
with model.as_default():
    with tf.Session(graph=model) as sess:
        # Grab a reference to the input image tensor and the boxes
        # tensor
        image_tensor = model.get_tensor_by_name("image_tensor:0")
        boxes_tensor = model.get_tensor_by_name("detection_boxes:0")

        # For each bounding box we would like to know the score
        # (i.e. probability) and class label
        scores_tensor = model.get_tensor_by_name("detection_scores:0")
        classes_tensor = model.get_tensor_by_name("detection_classes:0")
        num_detections = model.get_tensor_by_name("num_detections:0")

        # Load the image from disk
        while vs.more():
            image = vs.read()
            if image is None:
                continue

            (h, w) = image.shape[:2]

            # Check to see if we should resize along the width
            if w > h and w > 1000:
                image = imutils.resize(image, width=1000)

            # Otherwise, check to see if we should resize along
            # the height
            elif h > w and h > 1000:
                image = imutils.resize(image, height=1000)

            # Prepare the image for detection
            h, w = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            # Perform inference and compute the bounding boxes,
            # probabilities, and class labels
            boxes, scores, labels, n = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections],
                feed_dict={image_tensor: image}
            )

            # Squeeze the lists into a single dimension
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            # Loop over the bounding box predictions
            for box, score, label in zip(boxes, scores, labels):
                # If the predicted probability is less than the minimum
                # confidence, ignore it
                if score < args["min_confidence"]:
                    continue

                # Scale the bounding box from the range [0, 1] to [w, h]
                start_y, start_x, end_y, end_x = box
                start_x = int(start_x * w)
                start_y = int(start_y * h)
                end_x = int(end_x * w)
                end_y = int(end_y * h)

                # Draw the prediction on the output image
                label = category_idx[label]
                idx = int(label["id"]) - 1
                label = "{}: {:.2f}".format(label["name"], score)
                cv2.rectangle(output, (start_x, start_y), (end_x, end_y),
                              COLORS[idx], 2)
                y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                cv2.putText(output, label, (start_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 1)

            # Write the labeled frame to the output video file
            annotated_video.write(output)

annotated_video.release()
