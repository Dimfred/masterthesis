# augmentation
import albumentations as A
import cv2 as cv

import tensorflow as tf

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import callbacks, optimizers

AUTOTUNE = tf.data.experimental.AUTOTUNE

# model
from yolov4.tf import YOLOv4
from yolov4.tf.train import SaveWeightsCallback
from trainer import Trainer

# utils
import utils
from config import config

utils.seed("tf", "np", "imgaug")


def create_model():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
    yolo.input_size = config.yolo.input_size
    yolo.channels = config.yolo.channels
    yolo.batch_size = config.yolo.batch_size
    yolo.make_model()

    # yolo.load_weights(config.yolo.pretrained_weights, weights_type=config.yolo.weights_type)
    # yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

    return yolo


# fmt:off
base_augmentations = A.Compose([
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
    )
])


def train_augmentations(image, bboxes):
    _train_augmentations = A.Compose([
        # TODO tune params
        # TODO bboxed still disappearing
        # has to happen before the crop if not can happen that bboxes disappear
        A.Rotate(
            limit=10,
            border_mode=cv.BORDER_REFLECT_101,
            p=0.3,
        ),
        # A.RandomScale(scale_limit=0.1, p=0.3),
        # THIS DOES NOT RESIZE ANYMORE THE RESIZING WAS COMMENTED OUT
        # A.RandomSizedBBoxSafeCrop(
        #     width=None, # unused
        #     height=None, # unused
        #     p=0.3,
        # ),
        A.OneOf([
            A.CLAHE(p=1),
            A.ColorJitter(p=1),
        ], p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.GaussNoise(p=0.3),
        base_augmentations
    ], bbox_params=A.BboxParams("yolo"))

    augmented = _train_augmentations(image=image, bboxes=bboxes)
    return augmented["image"], augmented["bboxes"]

def valid_augmentations(image, bboxes):
    _valid_augmentations = A.Compose([
        base_augmentations,
    ], bbox_params=A.BboxParams("yolo"))

    augmented = _valid_augmentations(image=image, bboxes=bboxes)
    return augmented["image"], augmented["bboxes"]
# fmt:on


if __name__ == "__main__":
    # model creation
    yolo = create_model()
    # optimizer = optimizers.Adam(learning_rate=config.yolo.lr)
    optimizer = optimizers.SGD(config.yolo.lr, momentum=config.yolo.momentum)
    yolo.compile(
        optimizer=optimizer,
        loss_iou_type=config.yolo.loss,
        loss_verbose=0,
        run_eagerly=config.yolo.run_eagerly,
        # steps_per_execution=config.yolo.accumulation_steps,
    )

    # dataset creation
    train_dataset = yolo.load_dataset(
        dataset_path=config.train_out_dir / "labels.txt",
        dataset_type=config.yolo.weights_type,
        label_smoothing=0.05,
        preload=config.yolo.preload_dataset,
        # preload=False,
        training=True,
        augmentations=train_augmentations,
    )

    valid_dataset = yolo.load_dataset(
        dataset_path=config.valid_out_dir / "labels.txt",
        dataset_type=config.yolo.weights_type,
        preload=config.yolo.preload_dataset,
        training=False,
        augmentations=valid_augmentations,
    )

    def lr_scheduler(step, lr):
        lr = config.yolo.lr
        max_steps = config.yolo.max_steps
        burn_in = config.yolo.burn_in

        if step < burn_in:
            return (lr / burn_in) * (step + 1)
        # if step < int(epochs * 0.5):
        #     return lr
        # if epoch < int(epochs * 0.8):
        #     return lr * 0.5
        # if epoch < int(epochs * 0.9):
        #     return lr * 0.1
        tf.print(lr)
        return lr

        # return lr * 0.01

    # gib ihm
    trainer = Trainer(
        yolo,
        max_steps=config.yolo.max_steps,
        validation_freq=config.yolo.validation_freq,
        accumulation_steps=config.yolo.accumulation_steps,
        map_after_steps=config.yolo.map_after_steps,
        map_on_step_mod=config.yolo.map_on_step_mod,
        lr_scheduler=lr_scheduler,
    )
    trainer.train(train_dataset, valid_dataset)
