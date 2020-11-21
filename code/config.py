import abc
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
config.preprocessed_valid_dir = config.data / "preprocessed_valid"
config.yolo_labeled_dir = config.data / "yolo_labeled"
config.weights_dir = Path("weights")

# yolo
config.yolo = EasyDict()

## architecture
config.yolo.input_size = 832  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True

# classes
config.yolo.classes = str(config.label_dir / "classes.txt")
config.yolo.safe_classes = str(config.labeled_safe_dir / "classes.txt")
config.yolo.stripped_classes = str(config.preprocessed_valid_dir / "classes.txt")

# misc
config.yolo.weights_type = "yolo"
config.yolo.label_weights = str(config.weights_dir / "label.weights")
config.yolo.safe_label_weights = str(config.weights_dir / "safe_label.weights")
config.yolo.stripped_weights = str(config.weights_dir / "stripped_best.weights")

# removes classes from dataset
config.labels_to_remove = [
    "edge_tl",
    "edge_tr",
    "edge_br",
    "edge_bl",
    "t_left",
    "t_top",
    "t_right",
    "t_bot",
    "cross"
]

# removes classes and the file where the class is present from dataset
config.labels_and_files_to_remove = [
    "bat_left",
    "bat_top",
    "bat_right",
    "bat_bot",
    "res_us_hor",
    "res_us_ver",
    "lamp_de_hor",
    "lamp_de_ver",
    "lamp_us_hor",
    "lamp_us_ver",
    "ind_us_hor",
    "ind_us_ver",
]
