from easydict import EasyDict
from pathlib import Path
import utils

config = EasyDict()


# base
config.data = Path("data")

# training / augmented training
config.train_dir = config.data / "train"
config.train_out_dir = config.data / "train_out"

# validation / augmented validation
config.valid_dir = config.data / "valid"
config.valid_out_dir = config.data / "valid_out"

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
config.yolo.input_size = 608  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True
config.yolo.weights_type = "yolo"
config.yolo.pretrained_weights = config.weights_dir / "yolov4-tiny-small.weights"

## training
config.yolo.run_eagerly = (
    # True
    # if utils.isme()
    # else False
    False
)
config.yolo.checkpoint_dir = Path("checkpoints")
config.yolo.preload_dataset = True
config.yolo.batch_size = 2 if utils.isme() else 8
config.yolo.subdivisions = 32 if utils.isme() else 8
config.yolo.validation_steps = 1 if utils.isme() else 2
config.yolo.validation_frequency = 2 if utils.isme() else 10
config.yolo.loss = "ciou"
config.yolo.lr = 1e-4
config.yolo.epochs = 4000
config.yolo.workers = 12 if utils.isme() else 16

# classes and corresponding trained weights
architecture_type = {
    # uses only german symbols without edges and T's
    "stripped": (
        str(config.train_out_dir / "classes.txt"),
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

# config.yolo.architecture_type = "edges"
config.yolo.architecture_type = "stripped"
classes, weights = architecture_type[config.yolo.architecture_type]
config.yolo.classes = classes
config.yolo.weights = weights
config.yolo.full_classes = architecture_type["edges"][0]



###############
# mobile_unet #
###############

config.unet = EasyDict()

# net
config.unet.n_classes = 2
config.unet.input_size = 448  # 448 #224 #608 #416  #288
config.unet.channels = 3
config.unet.weights = Path("mobile_unet/weights/best.pth")

# training
config.unet.lr = 3e-4
config.unet.batch_size = 32 if not utils.isme() else 8
config.unet.subdivision = 4 if not utils.isme() else 2
# minibatch_size = batch_size / subdivision
config.unet.n_epochs = 1000

# lr scheduler
config.unet.lr_decay = "fixed"  # "cos" # "linear", "schedule", step
config.unet.lr_decay_fixed = [300, 500]
config.unet.lr_burn_in = 10

# loss functions
config.unet.focal_alpha = 0.1
config.unet.focal_gamma = 2
config.unet.focal_reduction = "mean"

# optimizers
config.unet.amsgrad = True
config.unet.decay = 0.00005
config.unet.betas = (0.90, 0.999)

# priority[pretrained] > priority[checkpoint]
config.unet.pretrained_path = None
config.unet.pretrained_path = Path("weights/mobilenet_v2_rgb.pth")
config.unet.checkpoint_path = None
# config.unet.checkpoint_path = Path("weights/checkpoint.pth")
config.unet.output_dir = Path("outputs")

# utility
config.unet.random_state = 42
config.unet.n_workers = 16 if not utils.isme() else 1

###################
## augmentations ##
###################

config.augment = EasyDict()

config.augment.yolo = EasyDict()
config.augment.yolo.img_params = EasyDict()
config.augment.yolo.img_params.channels = 1
config.augment.yolo.img_params.keep_ar = True
config.augment.yolo.img_params.resize = 1000

config.augment.unet = EasyDict()
config.augment.unet.img_params = EasyDict()
config.augment.unet.img_params.channels = 1  # config.unet.channels
config.augment.unet.img_params.keep_ar = True
config.augment.unet.img_params.resize = 640

# whether to perform flip and rotation on the dataset
config.augment.perform_train = True
config.augment.perform_valid = False
config.augment.perform_merged = True

# will exclude merged entirelly
config.augment.exclude_merged = True

# transition occurs always with the clock (90°)
config.augment.label_transition_rotation = {
    "diode_left": "diode_top",
    "diode_top": "diode_right",
    "diode_right": "diode_bot",
    "diode_bot": "diode_left",
    "bat_left": "bat_top",
    "bat_top": "bat_right",
    "bat_right": "bat_bot",
    "bat_bot": "bat_left",
    "res_de_hor": "res_de_ver",
    "res_de_ver": "res_de_hor",
    "res_us_hor": "res_us_ver",
    "res_us_ver": "res_us_hor",
    "cap_hor": "cap_ver",
    "cap_ver": "cap_hor",
    "gr_left": "gr_top",
    "gr_top": "gr_right",
    "gr_right": "gr_bot",
    "gr_bot": "gr_left",
    "lamp_de_hor": "lamp_de_ver",
    "lamp_de_ver": "lamp_de_hor",
    "lamp_us_hor": "lamp_us_ver",
    "lamp_us_ver": "lamp_us_hor",
    "ind_de_hor": "ind_de_ver",
    "ind_de_ver": "ind_de_hor",
    "ind_us_hor": "ind_us_ver",
    "ind_us_ver": "ind_us_hor",
    "source_hor": "source_ver",
    "source_ver": "source_hor",
    "current_hor": "current_ver",
    "current_ver": "current_hor",
    "edge_tl": "edge_tr",
    "edge_tr": "edge_br",
    "edge_br": "edge_bl",
    "edge_bl": "edge_tl",
    "t_left": "t_top",
    "t_top": "t_right",
    "t_right": "t_bot",
    "t_bot": "t_left",
    "cross": "cross",
}

# flip over y axis
config.augment.label_transition_flip = {
    "diode_left": "diode_right",
    "diode_top": "diode_top",
    "diode_right": "diode_left",
    "diode_bot": "diode_bot",
    "bat_left": "bat_right",
    "bat_top": "bat_top",
    "bat_right": "bat_left",
    "bat_bot": "bat_bot",
    "res_de_hor": "res_de_hor",
    "res_de_ver": "res_de_ver",
    "res_us_hor": "res_us_hor",
    "res_us_ver": "res_us_ver",
    "cap_hor": "cap_hor",
    "cap_ver": "cap_ver",
    "gr_left": "gr_right",
    "gr_top": "gr_top",
    "gr_right": "gr_left",
    "gr_bot": "gr_bot",
    "lamp_de_hor": "lamp_de_hor",
    "lamp_de_ver": "lamp_de_ver",
    "lamp_us_hor": "lamp_us_hor",
    "lamp_us_ver": "lamp_us_ver",
    "ind_de_hor": "ind_de_hor",
    "ind_de_ver": "ind_de_ver",
    "ind_us_hor": "ind_us_hor",
    "ind_us_ver": "ind_us_ver",
    "source_hor": "source_hor",
    "source_ver": "source_ver",
    "current_hor": "current_hor",
    "current_ver": "current_ver",
    "edge_tl": "edge_tr",
    "edge_tr": "edge_tl",
    "edge_br": "edge_bl",
    "edge_bl": "edge_br",
    "t_left": "t_right",
    "t_top": "t_top",
    "t_right": "t_left",
    "t_bot": "t_bot",
    "cross": "cross",
}

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
