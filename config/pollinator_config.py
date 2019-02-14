import os

# Initialize the base path for the pollinator dataset
BASE_PATH = "/scratch/general/lustre/u6000791/pollinators"

# Build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "records/classes.pbtxt"])

# Initialize the test split size
TEST_SIZE = 0.25

# Initialize the class labels dictionary
CLASSES = {"Pollinator": 1}

# Show images to see if things are working as we expect
VISUALLY_CHECK = False
