import abc
from easydict import EasyDict
from pathlib import Path

config = EasyDict()


# base
config.data = Path("data")

# training / augmented training
config.train_dir = config.data / "train"
config.train_preprocessed_dir = config.data / "train_preprocessed"

# validation / augmented validation
config.valid_dir = config.data / "valid"
config.valid_preprocessed_dir = config.data / "valid_preprocessed"

# preprocessing stuff
config.yolo_labeled_dir = config.data / "yolo_labeled"
config.unlabeled_dir = config.data / "unlabeled"

# bg fg
config.foregrounds_dir = config.data / "foregrounds"
config.backgrounds_dir = config.data / "backgrounds"
config.merged_dir = config.data / "merged"

# other
config.noise_dir = config.data / "noise"
config.labeled_safe_dir = config.data / "labeled_safe"

# weights
config.weights_dir = Path("weights")

########
# yolo #
########
config.yolo = EasyDict()

## architecture
config.yolo.input_size = 832  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True
config.yolo.weights_type = "yolo"

config.yolo.architecture_type = "stripped"

# classes and corresponding trained weights
architecture_type = {
    # uses only german symbols without edges and T's
    "stripped": (
        str(config.train_preprocessed_dir / "classes.txt"),
        str(config.weights_dir / "stripped_best.weights"),
    ),
    # contains all labels without edges
    "safe": (
        str(config.labeled_safe_dir / "classes.txt"),
        str(config.weights_dir / "safe_label.weights"),
    ),
    # contains edges, T's, crosses and old stuff like US shit
    "edges": (
        str(config.train_dir / "classes.txt"),
        str(config.weights_dir / "label.weights"),
    ),
}

classes, weights = architecture_type[config.yolo.architecture_type]
config.yolo.classes = classes
config.yolo.weights = weights
config.yolo.full_classes = architecture_type["edges"][0]


#################
# augmentations #
#################

config.augment = EasyDict()

config.augment.resize = 1000

# whether to perform flip and rotation on the dataset
config.augment.perform_train = False
config.augment.perform_valid = False
config.augment.perform_merged = False

# will exclude merged entirelly
config.augment.exclude_merged = True


# removes classes from dataset
# fmt: off
config.labels_to_remove = [
    "edge_tl",
    "edge_tr",
    "edge_br",
    "edge_bl",
    "t_left",
    "t_top",
    "t_right",
    "t_bot",
    "cross",

    ### ALL ###
    # "diode_left",
    # "diode_top",
    # "diode_right",
    # "diode_bot",
    # "bat_left",
    # "bat_top",
    # "bat_right",
    # "bat_bot",
    # "res_de_hor",
    # "res_de_ver",
    # "res_us_hor",
    # "res_us_ver",
    # "cap_hor",
    # "cap_ver",
    # "gr_left",
    # "gr_top",
    # "gr_right",
    # "gr_bot",
    # "lamp_de_hor",
    # "lamp_de_ver",
    # "lamp_us_hor",
    # "lamp_us_ver",
    # "ind_de_hor",
    # "ind_de_ver",
    # "ind_us_hor",
    # "ind_us_ver",
    # "source_hor",
    # "source_ver",
    # "current_hor",
    # "current_ver",
    # "edge_tl",
    # "edge_tr",
    # "edge_br",
    # "edge_bl",
    # "t_left",
    # "t_top",
    # "t_right",
    # "t_bot",
    # "cross",
]
# fmt: on

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
