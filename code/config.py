from easydict import EasyDict
from pathlib import Path

import easydict
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

# test / augmented test
config.test_dir = config.data / "test"
config.test_out_dir = config.data / "test_out"

# evaluation
config.eval_dir = config.data / "eval"

# preprocessing stuff
config.yolo_labeled_dir = config.data / "yolo_labeled"
config.unlabeled_dir = config.data / "unlabeled"

# bg fg
config.foregrounds_dir = config.data / "foregrounds"
config.backgrounds_dir = config.data / "backgrounds"
config.merged_dir = config.data / "merged"

# texts
config.texts_dir = config.data / "texts"

# unused_data
config.unused_data_dir = config.data / "unused_data"

# mnist
config.mnist_dir = config.data / "mnist"
config.mnist_train_dir = config.mnist_dir / "train"
config.mnist_test_dir = config.mnist_dir / "test"

# arrows
config.arrows_dir = config.data / "arrows"

# other
config.noise_dir = config.data / "noise"
config.labeled_safe_dir = config.data / "labeled_safe"

# weights
config.weights_dir = Path("weights")

config.train = EasyDict()
config.train.mean = 0.631572
config.train.std = 0.126536

########
# yolo #
########
config.yolo = EasyDict()


## architecture
config.yolo.input_size = 608  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True
config.yolo.weights_type = "yolo" # yolo, tf
config.yolo.activation = "leaky"  # leaky, hswish
config.yolo.backbone = "yolo"  # yolo, mobilenetv3-large, mobilenetv3-small
config.yolo.pretrained_weights = config.weights_dir / "yolov4-tiny-small.weights"

## training
config.yolo.batch_size = 2 if utils.isme() else 16
config.yolo.accumulation_steps = 8 if utils.isme() else 4
config.yolo.real_batch_size = config.yolo.batch_size * config.yolo.accumulation_steps
config.yolo.loss = "ciou"  # "ciou", "eiou", "diou"
config.yolo.loss_gamma = 0.0  # 0.5

config.yolo.burn_in = 1000
config.yolo.lr = 0.0025
config.yolo.decay = 0.00025
config.yolo.momentum = 0.90
config.yolo.label_smoothing = 0.1

config.yolo.max_steps = 4000
config.yolo.map_after_steps = 500
config.yolo.map_on_step_mod = 20  # 50
config.yolo.validation_freq = 10 if utils.isme() else 10
config.yolo.n_workers = 12 if utils.isme() else 32
config.yolo.validation_steps = -1 if utils.isme() else 2

config.yolo.checkpoint_dir = Path("checkpoints")
config.yolo.preload_dataset = True
config.yolo.run_eagerly = (
    True
    if utils.isme()
    else False
    # False
)

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
    "text": (
        str(config.train_out_dir / "classes.txt"),
        str(config.weights_dir / "text.weights")
    )
}

# config.yolo.architecture_type = "edges"
# config.yolo.architecture_type = "stripped"
config.yolo.architecture_type = "text"
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
config.unet.input_size = 608  # 448  # 448 #224 #608 #416  #288
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
config.augment.yolo.img_params.resize = 1000  # resizes longest image axis to that size

config.augment.unet = EasyDict()
config.augment.unet.img_params = EasyDict()
config.augment.unet.img_params.channels = 1  # config.unet.channels
config.augment.unet.img_params.keep_ar = True
config.augment.unet.img_params.resize = 640  # resizes longest image axis to that size

# whether to perform flip and rotation on the dataset
config.augment.perform_rotation = False
config.augment.perform_flip = False
config.augment.include_merged = False
config.augment.augment_valid = False

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
    "arrow_left": "arrow_top",
    "arrow_top": "arrow_right",
    "arrow_right": "arrow_bot",
    "arrow_bot": "arrow_left",
    "text": "text",
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
    "arrow_right": "arrow_left",
    "arrow_left": "arrow_right",
    "arrow_top": "arrow_top",
    "arrow_bot": "arrow_bot",
    "text": "text",
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
    # "arrow_right",
    # "arrow_left",
    # "arrow_top",
    # "arrow_bot",
    # "text"


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
    # REMOVE FOR NOW
    # "arrow_right",
    # "arrow_left",
    # "arrow_top",
    # "arrow_bot",
    # "text",
    # OLD LABELS
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

## experiment
config.yolo.pexperiment = f"LR_{config.yolo.lr}"
