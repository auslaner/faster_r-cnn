import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from imutils.video import FileVideoStream

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
                help="labels file")
ap.add_argument("-n", "--num-classes", type=int, required=True,
                help="# of class labels")
args = vars(ap.parse_args())

FEW_POLLINATOR_VIDEO = "/scratch/general/lustre/u6000791/pollinators/data/videos/02-2018.05.24_07.11.34-23.mpg"
MULTIPLE_POLLINATOR_VIDEO = "/scratch/general/lustre/u6000791/pollinators/data/videos/02-2018.05.30_16.54.34-01.mpg"

FRAMES_WITH_POLLINATORS_FEW = [8044, 8045, 8046, 8047, 8049, 8050, 8066, 8068, 8069, 8069, 8070, 8071, 8072, 8073,
                               8074, 8075, 8079, 8080, 8081]

FRAMES_WITH_POLLINATORS_MANY = {26: 1, 27: 1, 37: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 1, 53: 1,
                                54: 1, 57: 1, 58: 1, 59: 1, 60: 2, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1,
                                68: 1, 69: 1, 70: 1, 71: 1, 72: 1, 73: 1, 74: 1, 76: 1, 78: 1, 79: 1, 80: 1, 81: 1,
                                82: 1, 83: 1, 84: 1, 85: 1, 86: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1,
                                98: 1, 99: 1, 100: 1, 101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1,
                                109: 1, 110: 1, 111: 2, 112: 2, 113: 2, 114: 2, 115: 2, 116: 2, 117: 2, 118: 2, 119: 2,
                                120: 2, 121: 1}
errors = {"Type I": [], "Type II": 0}

# Total frames evaluated
TOTAL_FRAMES = 8202

# Total pollinators
TOTAL_POLLINATORS = len(FRAMES_WITH_POLLINATORS_FEW) + sum(FRAMES_WITH_POLLINATORS_MANY.values())


def plot_eval(num_evals, correct):
    # Get a list of correct scores without the None values
    pure_correct = [score for score in correct if score is not None]
    bin_width = 0.1

    # Plot distribution of Type I Error
    if len(errors["Type I"]) > 0:
        plt.hist(errors["Type I"], bins=np.arange(min(errors["Type I"]), max(errors["Type I"]) + bin_width, bin_width))
        plt.title("Model Confidence Distribution of Type I Errors")
        plt.ylabel("Frequency")
        plt.xlabel("Confidence Score")
        plt.show()
    else:
        print("[!] Zero Type I Errors!")

    # Plot distribution of correct evals
    if len(pure_correct) > 0:
        plt.hist(pure_correct, bins=np.arange(min(pure_correct), max(pure_correct) + bin_width, bin_width))
        plt.title("Model Confidence Distribution of Correct Evaluations")
        plt.ylabel("Frequency")
        plt.xlabel("Confidence Score")
        plt.show()
    else:
        print("[!] Zero correct classifications on frames with pollinators!")

    # Plot comparison of errors and correct evals
    cor_pct = len(correct)/num_evals
    values = [cor_pct, len(errors["Type I"])/num_evals, errors["Type II"]/num_evals]
    labels = ["Correct", "Type I Error", "Type II Error"]
    explode = (0.1, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(values, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")
    plt.show()

    # Print some additional stats
    print("Average confidence of Type I Error: {}".format(np.mean(errors["Type I"])))
    print("Average confidence of correct classifications: {}".format(np.mean(pure_correct)))
    print("Total accuracy: {}".format(cor_pct))
    print("Frame classification accuracy: {}".format(len(correct) / TOTAL_FRAMES))
    print("Accuracy on frames with pollinators: {:.2%}"
          .format(len(pure_correct) / TOTAL_POLLINATORS))


def main():
    # Initialize the model
    model = tf.Graph()

    # Initialize variables to hold info about progress
    num_evals = 0
    correct = []

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

    # Type I Error check; most frames in the early part of this video do not contain pollinators
    few_vs = FileVideoStream(FEW_POLLINATOR_VIDEO).start()
    print("[*] Evaluating video with few pollinators...\n")
    num_evals = evaluate_video(model, few_vs, "few", num_evals, correct)
    few_vs.stop()

    print("Total Evals: {}\nType I Errors: {}\nType II Errors: {}\nCorrect Evals: {}".format(num_evals,
                                                                                             len(errors["Type I"]),
                                                                                             errors["Type II"],
                                                                                             len(correct)))

    # Evaluate on video with multiple blurry pollinators
    many_vs = FileVideoStream(MULTIPLE_POLLINATOR_VIDEO).start()
    print("[*] Evaluating video with many pollinators...\n")
    num_evals = evaluate_video(model, many_vs, "many", num_evals, correct)
    many_vs.stop()

    print("Total Evals: {}\nType I Errors: {}\nType II Errors: {}\nCorrect Evals: {}".format(num_evals,
                                                                                             len(errors["Type I"]),
                                                                                             errors["Type II"],
                                                                                             len(correct)))
    plot_eval(num_evals, correct)


def evaluate_video(model, video_stream, video_type, total_evals, correct):
    global errors
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

            frame = 0

            # Load the image from disk
            while video_stream.more():
                image = video_stream.read()
                frame += 1
                if image is None:
                    continue

                total_evals += 1

                print("Frame: {}".format(frame))

                if video_type == "few":
                    if frame > 8081:
                        return total_evals
                else:
                    truth = FRAMES_WITH_POLLINATORS_MANY.get(frame, None)
                    if frame > 121:
                        return total_evals

                (h, w) = image.shape[:2]

                # Check to see if we should resize along the width
                if w > h and w > 1000:
                    image = imutils.resize(image, width=1000)

                # Otherwise, check to see if we should resize along
                # the height
                elif h > w and h > 1000:
                    image = imutils.resize(image, height=1000)

                # Prepare the image for detection
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
                detects = 0
                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.75:
                        continue

                    detects += 1
                    if detects > 1:
                        total_evals += 1

                    print("\tDetection: {}".format(detects))
                    print("\tScore: {}".format(score))
                    print("\tEval Number: {}".format(total_evals))

                    if video_type == "few":
                        if frame in FRAMES_WITH_POLLINATORS_FEW:
                            if detects == 1:
                                correct.append(score)
                                print("\tCorrect: {}".format(len(correct)))
                            else:
                                errors["Type I"].append(score)
                                print("\tType I Error: {}".format(len(errors["Type I"])))
                        else:
                            errors["Type I"].append(score)
                            print("\tType I Error: {}".format(len(errors["Type I"])))
                    else:
                        if truth:
                            if detects <= truth:
                                correct.append(score)
                                print("\tCorrect: {}".format(len(correct)))
                            else:
                                errors["Type I"].append(score)
                                print("\tType I Error: {}".format(len(errors["Type I"])))
                        else:
                            errors["Type I"].append(score)
                            print("\tType I Error: {}".format(len(errors["Type I"])))

                if detects == 0:
                    if video_type == "few" and frame in FRAMES_WITH_POLLINATORS_FEW:
                        errors["Type II"] += 1
                        print("\t Type II Error: {}".format(errors["Type II"]))
                    elif video_type == "many" and truth:
                        errors["Type II"] += 1
                        print("\t Type II Error: {}".format(errors["Type II"]))
                    else:
                        correct.append(None)
                        print("\tCorrect: {}".format(len(correct)))

    return total_evals


if __name__ == "__main__":
    main()
