#!/usr/bin/env python3.8

import albumentations as A
import imgaug

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
import numpy as np
import cv2 as cv

from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
from yolov4.tf.train import SaveWeightsCallback
from config import config


def seed():
    np.random.seed(1337)
    imgaug.random.seed(1337)
    tf.random.set_seed(1337)


yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
# yolo = YOLOv4()
# yolo = YOLOv4(tiny=config.yolo.tiny)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.batch_size = config.yolo.batch_size
yolo.subdivisions = config.yolo.subdivisions
# TODO check other params

yolo.make_model()
# yolo.load_weights(config.yolo.pretrained_weights, weights_type=config.yolo.weights_type)

##############
## TRAINING ##
##############

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
        A.RandomScale(scale_limit=0.1, p=0.3),
        # THIS DOES NOT RESIZE ANYMORE THE RESIZING WAS COMMENTED OUT
        A.RandomSizedBBoxSafeCrop(
            width=None, # unused
            height=None, # unused
            p=0.3,
        ),
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


train_dataset = yolo.load_dataset(
    dataset_path=config.train_out_dir / "labels.txt",
    dataset_type=config.yolo.weights_type,
    label_smoothing=0.05,
    preload=config.yolo.preload_dataset,
    # preload=False,
    training=True,
    augmentations=train_augmentations,
)
# test_dataset(yolo, dataset)

valid_dataset = yolo.load_dataset(
    dataset_path=config.valid_out_dir / "labels.txt",
    dataset_type=config.yolo.weights_type,
    preload=config.yolo.preload_dataset,
    training=False,
    augmentations=valid_augmentations,
)


optimizer = optimizers.Adam(learning_rate=config.yolo.lr)
yolo.compile(
    optimizer=optimizer,
    loss_iou_type=config.yolo.loss,
    loss_verbose=0,
    run_eagerly=config.yolo.run_eagerly,
)


def lr_scheduler(epoch):
    lr = config.yolo.lr
    epochs = config.yolo.epochs

    if epoch < int(epochs * 0.5):
        return lr
    if epoch < int(epochs * 0.8):
        return lr * 0.5
    if epoch < int(epochs * 0.9):
        return lr * 0.1

    return lr * 0.01


_callbacks = [
    callbacks.LearningRateScheduler(lr_scheduler),
    callbacks.TerminateOnNaN(),
    callbacks.TensorBoard(log_dir="./log"),
    SaveWeightsCallback(
        yolo=yolo,
        dir_path=config.yolo.checkpoint_dir,
        weights_type=config.yolo.weights_type,
        epoch_per_save=100,
    ),
]

seed()
yolo.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=config.yolo.epochs,
    callbacks=_callbacks,
    validation_steps=2,
    validation_freq=config.yolo.batch_size / 2,
    steps_per_epoch=config.yolo.batch_size,  # config.yolo.batch_size,
    workers=16,
)
