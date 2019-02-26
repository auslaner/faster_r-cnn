import os

# Initialize the base path for the pollinator dataset
BASE_PATH = "/scratch/general/lustre/u6000791/pollinators"

# Build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH, "data/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH, "data/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH, "data/classes.pbtxt"])

# Initialize the test split size
TEST_SIZE = 0.25

# Initialize the class labels dictionary
CLASSES = {
    "Osmia": 1,
    "Masarinae": 2,
    "Other": 3}

# Show images to see if things are working as we expect
VISUALLY_CHECK = True
