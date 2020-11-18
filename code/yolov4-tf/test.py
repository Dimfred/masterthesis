#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import tensorflow as tf
import os

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config


# default 0.25, 0.3
# inference_params = {"score_threshold": 0.8, "iou_threshold": 0.8}

#         less cls    with edges     without edges
# values: stripped    label          safe
# TODO move config probably
model_type = "stripped"

weights = {
    "stripped": config.yolo.stripped_weights,
    "safe": config.yolo.safe_label_weights,
    "edges": config.yolo.label_weights,
}

classes = {
    "stripped": config.yolo.stripped_classes,
    "safe": config.yolo.safe_classes,
    "edges": config.yolo.classes,
}

# small will use yolov4 head with 3 yolo layers
yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = classes[model_type]
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.make_model()


# test dataset
test_dataset = len(sys.argv) > 1
if test_dataset:
    utils.test_dataset(yolo, config.label_dir)
    sys.exit()


yolo.load_weights(weights[model_type], weights_type=config.yolo.weights_type)

for file_ in os.listdir(config.preprocessed_valid_dir):
    if ".png" in file_ or ".jpg" in file_:
        print(file_)
        yolo.inference(str(config.preprocessed_valid_dir / file_))
