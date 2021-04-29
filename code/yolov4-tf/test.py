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

# from yolov4.model.backbone import CSPDarknet53Tiny
# tiny = CSPDarknet53Tiny(small=True)
# tiny.build(input_shape=(1, config.yolo.input_size, config.yolo.input_size, 1))
# tiny.summary()

# sys.exit()


# small will use yolov4 head with 3 yolo layers
yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.batch_size = 1
yolo.make_model()


# test dataset
test_dataset = len(sys.argv) > 1
if test_dataset:
    yolo.classes = config.yolo.full_classes
    yolo.make_model()

    to_test = sys.argv[1]
    if to_test == "train":
        utils.test_dataset(yolo, config.train_dir)
    elif to_test == "merged":
        utils.test_dataset(yolo, config.merged_dir)

    sys.exit()


yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

#dirs = [config.data / "tmp"]
# dirs = [config.valid_out_dir]
# dirs = [config.unlabeled_dir]
dirs = [config.unused_data_dir]
for dir_ in dirs:
    for img_path in utils.list_imgs(dir_):
        # print(str(img_path))
        # utils.show(cv.imread((str(img_path))))
        yolo.inference(str(img_path))
