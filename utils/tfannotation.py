from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature


class TFAnnotation:
    def __init__(self):
        # Initialize the bounding box and label lists
        self.x_mins = []
        self.x_maxs = []
        self.y_mins = []
        self.y_maxs = []
        self.text_labels = []
        self.classes = []
        self.difficult = []

        # Initialize additional variables, including the image itself,
        # spatial dimensions, encoding, and filename
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        # Encode the attributes using their respective Tensorflow
        # encoding function
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.image)
