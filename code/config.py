from easydict import EasyDict
from pathlib import Path

config = EasyDict()


config.data = Path("data")
config.label_dir = config.data / "labeled"
config.valid_dir = config.data / "valid"
config.unlabeled_dir = config.data / "unlabeled"
config.noise_dir = config.data / "noise"
config.labeled_safe_dir = config.data / "labeled_safe"
config.preprocessed_dir = config.data / "preprocessed"
config.yolo_labeled_dir = config.data / "yolo_labeled"
config.weights_dir = Path("weights")

# yolo
config.yolo = EasyDict()

## architecture
config.yolo.input_size = (608, 608)  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True
config.yolo.classes = str(config.label_dir / "classes.txt")
config.yolo.safe_classes = str(config.labeled_safe_dir / "classes.txt")

# misc
config.yolo.weights_type = "yolo"
config.yolo.label_weights = str(config.weights_dir / "label.weights")
