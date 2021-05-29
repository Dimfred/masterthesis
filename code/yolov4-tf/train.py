# augmentation
import albumentations as A
import click


def main():
    import cv2 as cv
    import numpy as np

    import tensorflow as tf
    import tensorflow.keras as K
    from tensorflow.python.keras.backend import backend
    import tensorflow_addons as tfa

    import numba as nb
    from torch.nn.modules import activation

    # has to be called right after tf import
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    from tensorflow.keras import callbacks, optimizers
    import time
    import sys

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # model
    from yolov4.tf import YOLOv4
    from yolov4.tf.train import SaveWeightsCallback
    from trainer import Trainer

    # utils
    import utils
    from config import config

    def print_parameters():
        from tabulate import tabulate

        # fmt: off
        pretty = [["LR", config.yolo.lr]]
        pretty += [["BS", config.yolo.real_batch_size]]
        pretty += [["Activation", config.yolo.activation]]
        pretty += [["Loss", config.yolo.loss]]
        pretty += [["Offline", f"P {config.augment.include_merged}, F {config.augment.perform_flip}, R {config.augment.perform_rotation}"]]
        pretty += [["Rotate", config.yolo.augment.rotate]]
        pretty += [["RandomScale", config.yolo.augment.random_scale]]
        pretty += [["ColorJitter", config.yolo.augment.color_jitter]]
        pretty += [["Crop", config.yolo.augment.bbox_safe_crop]]
        print(tabulate(pretty))
        # fmt: on

    def create_model():
        yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
        yolo.classes = config.yolo.classes
        yolo.input_size = config.yolo.input_size
        yolo.channels = config.yolo.channels
        yolo.batch_size = config.yolo.batch_size
        yolo.make_model(
            activation1=config.yolo.activation, backbone=config.yolo.backbone
        )
        yolo.model.summary()

        return yolo

    # fmt:off
    classes = utils.Yolo.parse_classes(config.train_out_dir / "classes.txt")
    pad_size = 1.2 * 1000

    def make_train_augmentation(aug):
        if aug == "all":
            pass
        elif aug == "color":
            this_aug = A.ColorJitter(
                brightness=config.yolo.augment.color_jitter,
                contrast=config.yolo.augment.color_jitter,
                saturation=config.yolo.augment.color_jitter,
                hue=config.yolo.augment.color_jitter,
                p=0.5
            )
        elif aug == "crop":
            crop_width=pad_size * config.yolo.augment.bbox_safe_crop
            crop_height=pad_size * config.yolo.augment.bbox_safe_crop
            this_aug = A.RandomSizedBBoxSafeCrop(
                width=crop_width,
                height=crop_height,
                p=0.5,
            )
        elif aug == "rot":
            this_aug = A.Rotate(
                limit=config.yolo.augment.rotate,
                border_mode=cv.BORDER_CONSTANT,
                p=0.5,
            )
        elif aug == "scale":
            this_aug = A.RandomScale(
                scale_limit=config.yolo.augment.random_scale,
                p=0.5
            )
        else:
            raise ValueError("Aug does not exist")

        _train_augmentations = A.Compose([
            A.PadIfNeeded(
                # min_height=int(pad_size),
                # min_width=int(pad_size),
                min_height=1000,
                min_width=1000,
                border_mode=cv.BORDER_CONSTANT,
                value=0,
                always_apply=True
            ),
            this_aug,
            A.Resize(
                width=config.yolo.input_size,
                height=config.yolo.input_size,
                always_apply=True
            ),
        ], bbox_params=A.BboxParams("yolo"))

        def train_augmentations(image, bboxes):
            augmented = _train_augmentations(image=image, bboxes=bboxes)
            return augmented["image"], augmented["bboxes"]

        return train_augmentations

    _valid_augmentations = A.Compose([
        A.PadIfNeeded(
            min_height=1000,
            min_width=1000,
            border_mode=cv.BORDER_CONSTANT,
            value=0,
            always_apply=True
        ),
        A.Resize(
            width=config.yolo.input_size,
            height=config.yolo.input_size,
            always_apply=True
        ),
    ], bbox_params=A.BboxParams("yolo"))

    def valid_augmentations(image, bboxes):
        augmented = _valid_augmentations(image=image, bboxes=bboxes)
        return augmented["image"], augmented["bboxes"]

    # fmt:on



    ####################################################################################
    ##### LR ###########################################################################
    ####################################################################################
    # lr, *runs = sys.argv[1:]
    # lr = float(lr)
    # runs = (int(r) for r in runs)

    # config.yolo.lr = lr
    # config.yolo.experiment_param = config.yolo.experiment_param(lr)

    ####################################################################################
    ##### OFFLINE AUG ##################################################################
    ####################################################################################
    # runs = sys.argv[1:]
    # runs = (int(r) for r in runs)

    ####################################################################################
    ##### ONLINE AUG ###################################################################
    ####################################################################################
    config.yolo.augment.rotate = None
    config.yolo.augment.random_scale = None
    config.yolo.augment.color_jitter = None
    config.yolo.augment.bbox_safe_crop = None #

    aug, param, *runs = sys.argv[1:]
    if aug == "rot":
        config.yolo.augment.rotate = int(param)
        config.yolo.experiment_name = "rotate"
        config.yolo.experiment_param = f"rotate_{param}"
    elif aug == "scale":
        config.yolo.augment.random_scale = float(param)  # 0.1, 0.2, 0.3
        config.yolo.experiment_name = "random_scale"
        config.yolo.experiment_param = f"random_scale_{param}"
    elif aug == "color":
        config.yolo.augment.color_jitter = float(param)  # 0.1, 0.2, 0.3
        config.yolo.experiment_name = "color_jitter"
        config.yolo.experiment_param = f"color_jitter_{param}"
    elif aug == "crop":
        config.yolo.augment.bbox_safe_crop = float(param) # 0.7, 0.8 0.9
        config.yolo.experiment_name = "bbox_safe_crop"
        config.yolo.experiment_param = f"bbox_safe_crop_{param}"

    runs = (int(r) for r in runs)
    train_augmentations = make_train_augmentation(aug)
    ####################################################################################
    ##### END EXPERIEMENTS #############################################################
    ####################################################################################

    ####################################################################################
    ##### GRID #########################################################################
    ####################################################################################
    # grid
    # activation, bs, lr, loss = sys.argv[1:5]

    # config.yolo.activation = activation

    # batch_size = int(bs)
    # config.yolo.batch_size = 16
    # config.yolo.accumulation_steps = batch_size // config.yolo.batch_size
    # config.yolo.real_batch_size = (
    #    config.yolo.batch_size * config.yolo.accumulation_steps
    # )

    # config.yolo.lr = float(lr)
    # config.yolo.loss = loss

    # config.yolo.experiment_param = config.yolo.experiment_param(
    #    config.yolo.activation,
    #    config.yolo.real_batch_size,
    #    config.yolo.lr,
    #    config.yolo.loss,
    # )


    for run in runs:
        while True:
            seed = config.train.seeds[run]
            utils.seed_all(seed)

            # model creation
            yolo = create_model()
            print("---------------------------------------------------")
            print("CLASSSES:")
            print(yolo.classes)
            print("---------------------------------------------------")

            # optimizer = optimizers.Adam(learning_rate=config.yolo.lr, amsgrad=False)
            # optimizer = optimizers.Adam(
            #     learning_rate=config.yolo.lr, momentum=config.yolo.momentum, amsgrad=True
            # )
            optimizer = optimizers.SGD(
                learning_rate=config.yolo.lr, momentum=config.yolo.momentum
            )
            # optimizer = tfa.optimizers.SGDW(
            #     learning_rate=config.yolo.lr,
            #     momentum=config.yolo.momentum,
            #     weight_decay=config.yolo.decay,
            # )

            yolo.compile(
                optimizer=optimizer,
                loss_iou_type=config.yolo.loss,
                loss_verbose=0,
                run_eagerly=config.yolo.run_eagerly,
                loss_gamma=config.yolo.loss_gamma,
            )

            # dataset creation
            train_dataset = yolo.load_tfdataset(
                dataset_path=config.train_out_dir / "labels.txt",
                dataset_type=config.yolo.weights_type,
                label_smoothing=config.yolo.label_smoothing,
                preload=config.yolo.preload_dataset,
                # preload=False,
                training=True,
                augmentations=train_augmentations,
                n_workers=config.yolo.n_workers,
            )

            valid_dataset = yolo.load_tfdataset(
                dataset_path=config.valid_out_dir / "labels.txt",
                dataset_type=config.yolo.weights_type,
                preload=config.yolo.preload_dataset,
                training=False,
                augmentations=valid_augmentations,
                n_workers=config.yolo.n_workers,
            )

            def lr_scheduler(step, lr, burn_in):
                # TODO cosine
                # cycle = 1000
                # mult = 2

                # ORIGINAL DARKNET
                if step < burn_in:
                    multiplier = (step / burn_in) ** 4
                    return lr * multiplier

                return lr

            experiment = utils.YoloExperiment(
                config.yolo.experiment_dir,
                config.yolo.experiment_name,
                config.yolo.experiment_param,
                run,
            )

            # gib ihm
            trainer = Trainer(
                yolo,
                max_steps=config.yolo.max_steps,
                validation_freq=config.yolo.validation_freq,
                batch_size=config.yolo.batch_size,
                accumulation_steps=config.yolo.accumulation_steps,
                map_after_steps=config.yolo.map_after_steps,
                map_on_step_mod=config.yolo.map_on_step_mod,
                lr_scheduler=lr_scheduler,
                # resize_model=resize_model,
                experiment=experiment,
                burn_in=config.yolo.burn_in,
                checkpoint_dir=config.yolo.checkpoint_dir,
            )
            try:
                print_parameters()
                trainer.train(train_dataset, valid_dataset)
                break
            except Exception as e:
                print("---------------------------------------------")
                print("---------------------------------------------")
                print(e)
                print("---------------------------------------------")
                print("---------------------------------------------")


if __name__ == "__main__":
    main()
