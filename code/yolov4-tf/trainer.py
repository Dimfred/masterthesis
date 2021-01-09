#!/usr/bin/env python3.8

import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras import optimizers, callbacks
from functools import partial
import albumentations as A

AUTOTUNE = tf.data.experimental.AUTOTUNE

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
from yolov4.tf.train import SaveWeightsCallback
from config import config

yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
# TODO normally 64 and subdivision 3
yolo.batch_size = 2
# TODO check other params

yolo.make_model()
# yolo.load_weights(config.yolo.pretrained_weights, weights_type=config.yolo.weights_type)

train_dataset = yolo.load_dataset(
    dataset_path=config.train_out_dir / "labels.txt",
    dataset_type=config.yolo.weights_type,
    label_smoothing=0.05,
    # This enables augmentation. Mosaic, cut_out, mix_up are used
    # training=True,
    training=False,
)
# test_dataset(yolo, dataset)

valid_dataset = yolo.load_dataset(
    dataset_path=config.valid_out_dir / "labels.txt",
    dataset_type=config.yolo.weights_type,
    training=False,
)

##############
## TRAINING ##
##############

# TODO config
epochs = 4000
lr = 1e-4


optimizer = optimizers.Adam(learning_rate=lr)
loss = "ciou"
yolo.compile(optimizer=optimizer, loss_iou_type=loss, loss_verbose=1)


def lr_scheduler(epoch):
    if epoch < int(epoch * 0.5):
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

yolo.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=_callbacks,
    validation_steps=-1,
    validation_freq=1,
    steps_per_epoch=35,
)
