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

config.train.seeds = [42, 1337, 0xDEADBEEF]

########
# yolo #
########
config.yolo = EasyDict()


## architecture
config.yolo.input_size = 608  # 608, 736, 832, 928, 960, 1120, 1280, 1600
config.yolo.channels = 1
config.yolo.tiny = True
config.yolo.small = True
config.yolo.weights_type = "yolo"  # yolo, tf
config.yolo.backbone = "yolo"  # yolo, mobilenetv3-large, mobilenetv3-small
config.yolo.pretrained_weights = config.weights_dir / "yolov4-tiny-small.weights"

## training
config.yolo.loss_gamma = 0.0  # 0.5

config.yolo.burn_in = 1000
config.yolo.decay = 0.00025
config.yolo.momentum = 0.90
config.yolo.label_smoothing = 0.1

config.yolo.max_steps = 4000
# step > this == 0 => perform mAP
config.yolo.map_after_steps = 500
# step % this == 0 => perform mAP
config.yolo.map_on_step_mod = 10
config.yolo.validation_freq = 10 if utils.isme() else 10
config.yolo.n_workers = 12 if utils.isme() else 32
config.yolo.validation_steps = -1 if utils.isme() else 2

config.yolo.checkpoint_dir = Path("checkpoints")
config.yolo.preload_dataset = True
config.yolo.run_eagerly = (
    True
    # if utils.isme()
    # else False
    # False
)

config.yolo.augment = EasyDict()

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
        str(config.weights_dir / "best.weights"),
    ),
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
config.unet.input_size = 448 if utils.isme() else 448  # 448  # 448 #224 #608 #416  #288
config.unet.channels = 3

# training
config.unet.lr = 3e-3 #3e-4 #0.0025
config.unet.batch_size = 64 if utils.isme() else 64
config.unet.subdivision = 16 if utils.isme() else 8
config.unet.valid_batch_size = 24 if utils.isme() else 24
config.unet.valid_subdivision = 1 if utils.isme() else 3
# minibatch_size = batch_size / subdivision
config.unet.n_epochs = 1000
config.unet.burn_in = 100
config.unet.lr_decay_fixed = [300, 400]

# optimizers
config.unet.amsgrad = True
config.unet.decay = 0.000005
config.unet.betas = (0.95, 0.999)
config.unet.momentum = 0.95

# loss functions
config.unet.focal_alpha = 0.1 # 0.1 best
config.unet.focal_gamma = 2   # 2 best
config.unet.focal_reduction = "mean"


# priority[pretrained] > priority[checkpoint]
config.unet.pretrained_path = None
# config.unet.pretrained_path = Path("weights/mobilenet_v2_rgb.pth")
config.unet.checkpoint_path = None
# config.unet.checkpoint_path = Path("weights/checkpoint.pth")
# config.unet.output_dir = Path("outputs")
config.unet.weights = Path("mobile_unet") / config.weights_dir / "text.pth"

# utility
config.unet.n_workers = 1 if utils.isme() else 12

# experiments
config.unet.experiment_dir = Path("experiments_unet")

config.unet.experiment_name = "test"
config.unet.experiment_param = "test"

# lr init SGD

# lr init AMSGrad

# offline aug

# online aug
config.unet.augment = EasyDict()

config.unet.augment.random_scale = 0.4
config.unet.augment.rotate = 20
config.unet.augment.crop_size = 0.9
config.unet.augment.color_jitter = 0.2
config.unet.augment.blur = 3


# grid


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
config.yolo.experiment_dir = Path("experiments_yolo")

# config.yolo.experiment_name = "lr_init"
# config.yolo.experiment_param = f"LR_{config.yolo.lr}"

# whether to perform flip and rotation on the dataset
config.augment.perform_rotation = True
config.augment.perform_flip = True
config.augment.include_merged = True

# config.yolo.experiment_name = "offline_aug"
# config.yolo.experiment_param = f"offaug_P{int(config.augment.include_merged)}_F{int(config.augment.perform_flip)}_R{int(config.augment.perform_rotation)}"

# config.yolo.experiment_name = "offline_baseline"
# config.yolo.experiment_param = f"offline_baseline_P{int(config.augment.include_merged)}_F{int(config.augment.perform_flip)}_R{int(config.augment.perform_rotation)}"

config.yolo.augment.rotate = 20  # 10, 20, 30
# config.yolo.experiment_name = "rotate"
# config.yolo.experiment_param = f"rotate_{config.yolo.augment.rotate}"

config.yolo.augment.random_scale = 0.3  # 0.1, 0.2, 0.3, 0.4, 0.5
# config.yolo.experiment_name = "random_scale"
# config.yolo.experiment_param = f"random_scale_{config.yolo.augment.random_scale}"

config.yolo.augment.color_jitter = 0.2 # 0.1, 0.2, 0.3
# config.yolo.experiment_name = "color_jitter"
# config.yolo.experiment_param = f"color_jitter_{config.yolo.augment.color_jitter}"

config.yolo.augment.bbox_safe_crop = True
# config.yolo.experiment_name = "bbox_safe_crop"
# config.yolo.experiment_param = f"bbox_safe_crop_{int(config.yolo.augment.bbox_safe_crop)}"

# config.yolo.augment.gaussian_noise = True
# config.yolo.experiment_name = "gaussian_noise"
# config.yolo.experiment_param = f"gaussian_noise_{int(config.yolo.augment.gaussian_noise)}"

config.yolo.augment.blur = 3 # 5
# config.yolo.experiment_name = "blur"
# config.yolo.experiment_param = f"blur_{config.yolo.augment.blur}"

# all augs
# config.yolo.experiment_name = "all_augs_with_jitter_noise_blur"
# config.yolo.experiment_param = "all_augs"

config.yolo.experiment_name = "all_augs_without_jitter_noise_blur"
config.yolo.experiment_param = "all_augs"


################
#### grid ######
################


# activation
config.yolo.activation = None  # leaky, hswish

# batch
config.yolo.batch_size = 2 if utils.isme() else 16
config.yolo.accumulation_steps = (
    8 if utils.isme() else 2
)
config.yolo.real_batch_size = (
    config.yolo.batch_size * config.yolo.accumulation_steps
)

# lr
config.yolo.lr = None  # 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001

# loss
config.yolo.loss = None  # "ciou", "eiou", "diou"

#config.yolo.experiment_name = "grid"
#config.yolo.experiment_param = (
#    lambda a, bs, lr, l: f"grid_act_{a}_bs_{bs}_lr_{lr}_loss_{l}"
#)

# fmt: off
# params = [
#     ["Activation", config.yolo.activation],
#     ["BatchSize", config.yolo.real_batch_size],
#     ["LR", config.yolo.lr],
#     ["Loss", config.yolo.loss],
# ]
# fmt: on



config.yolo.experiment_name = "test"
config.yolo.experiment_param = "test"
