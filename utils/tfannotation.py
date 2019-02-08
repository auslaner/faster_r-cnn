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
        image = bytes_feature(self.image)
        x_mins = float_list_feature(self.x_mins)
        x_maxs = float_list_feature(self.x_maxs)
        y_mins = float_list_feature(self.y_mins)
        y_maxs = float_list_feature(self.y_maxs)
        text_labels = bytes_list_feature(self.text_labels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)

        data = {
            "image/height": h,
            "image/width": w,
            "image/filename": filename,
            "image/source_id": filename,
            "image/encoded": image,
            "image/format": encoding,
            "image/object/bbox/xmin": x_mins,
            "image/object/bbox/xmax": x_maxs,
            "image/object/bbox/ymin": y_mins,
            "image/object/bbox/ymax": y_maxs,
            "image/object/class/text": text_labels,
            "image/object/class/label": classes,
            "image/object/difficult": difficult,
        }

        # Return the data dictionary
        return data
