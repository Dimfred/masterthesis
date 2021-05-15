# augmentation
import albumentations as A

# SLOW AF and not a real bonus, but could be just wrong
# utils.TextProjection(
#     text_idx=classes.index("text"),
#     ground_idxs=[
#         classes.index("gr_left"),
#         classes.index("gr_right"),
#         classes.index("gr_bot"),
#         classes.index("gr_top")
#     ],
#     texts=utils.load_imgs(config.texts_dir, cv.IMREAD_GRAYSCALE),
#     classes=utils.Yolo.parse_classes(config.train_out_dir / "classes.txt"),
#     always_apply=True
# ),


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

        # yolo.load_weights(config.yolo.pretrained_weights, weights_type=config.yolo.weights_type)
        # yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

        return yolo

    # fmt:off
    base_augmentations = A.Compose([
        # A.Normalize(mean=config.train.mean, std=config.train.std, max_pixel_value=255),
    ])

    classes = utils.Yolo.parse_classes(config.train_out_dir / "classes.txt")

    def train_augmentations(image, bboxes):
        _train_augmentations = A.Compose([
            A.PadIfNeeded(
                min_height=1200,
                min_width=1200,
                border_mode=cv.BORDER_CONSTANT,
                value=0,
                always_apply=True
            ),
            A.RandomScale(
                scale_limit=config.yolo.augment.random_scale,
                p=0.5
            ),
            A.Rotate(
                limit=config.yolo.augment.rotate,
                border_mode=cv.BORDER_CONSTANT,
                p=0.5,
            ),
            # THIS DOES NOT RESIZE ANYMORE THE RESIZING WAS COMMENTED OUT
            A.RandomSizedBBoxSafeCrop(
                width=None, # unused
                height=None, # unused
                p=0.5,
            ),

            A.ColorJitter(
                brightness=config.yolo.augment.color_jitter,
                contrast=config.yolo.augment.color_jitter,
                saturation=config.yolo.augment.color_jitter,
                hue=config.yolo.augment.color_jitter,
                p=0.5
            ),
            A.Blur(
                blur_limit=config.yolo.augment.blur,
                p=0.5
            ),

            A.Resize(
                width=config.yolo.input_size,
                height=config.yolo.input_size,
                always_apply=True
            ),

            base_augmentations
        ], bbox_params=A.BboxParams("yolo"))

        augmented = _train_augmentations(image=image, bboxes=bboxes)
        return augmented["image"], augmented["bboxes"]

    def valid_augmentations(image, bboxes):
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

        augmented = _valid_augmentations(image=image, bboxes=bboxes)
        return augmented["image"], augmented["bboxes"]
    # fmt:on



    # grid
    activation, bs, lr, loss = sys.argv[1:]

    config.yolo.activation = activation

    batch_size = int(bs)
    config.yolo.batch_size = 16
    config.yolo.accumulation_steps = batch_size // config.yolo.batch_size
    config.yolo.real_batch_size = (
        config.yolo.batch_size * config.yolo.accumulation_steps
    )

    config.yolo.lr = float(lr)
    config.yolo.loss = loss

    config.yolo.experiment_param = config.yolo.experiment_param(
        config.yolo.activation,
        config.yolo.real_batch_size,
        config.yolo.lr,
        config.yolo.loss,
    )

    # load run and seed
    # run_ = sys.argv[1]
    # for run in (run_,):
    for run in (0, 1, 2):
        done = False
        while not done:
            seed = config.train.seeds[int(run)]
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
                trainer.train(train_dataset, valid_dataset)
                done = True
            except Exception as e:
                print("---------------------------------------------")
                print("---------------------------------------------")
                print(e)
                print("---------------------------------------------")
                print("---------------------------------------------")


if __name__ == "__main__":
    main()
