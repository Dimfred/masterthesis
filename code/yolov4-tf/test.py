#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import tensorflow as tf

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


# small will use yolov4 head with 3 yolo layers
yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.make_model()


# test dataset
test_dataset = len(sys.argv) > 1
if test_dataset:
    utils.test_dataset(yolo, config.label_dir)
    sys.exit()


# small
yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)

# juli
# yolo.inference(media_path="data/labeled/06_00.jpg")
# yolo.inference(media_path="data/labeled/06_01.jpg")
# yolo.inference(media_path="data/labeled/06_02.jpg")

# grounds, sources, currents, inductors
# yolo.inference(media_path="data/labeled/00_08.jpg")
# yolo.inference(media_path="data/labeled/00_09.jpg")
# yolo.inference(media_path="data/labeled/00_10.jpg")

# jonas
yolo.inference(media_path=str(config.label_dir / "01_01.jpg"))
yolo.inference(media_path=str(config.label_dir / "01_02.jpg"))
yolo.inference(media_path=str(config.label_dir / "01_03.jpg"))
yolo.inference(media_path=str(config.label_dir / "01_04.jpg"))
yolo.inference(media_path=str(config.label_dir / "01_05.jpg"))
yolo.inference(media_path=str(config.label_dir / "01_06.jpg"))

# valid
yolo.inference(media_path="data/valid/00_11.jpg")
yolo.inference(media_path="data/valid/00_11_00.jpg")
yolo.inference(media_path="data/valid/00_11_01.jpg")
yolo.inference(media_path="data/valid/00_11_02.jpg")
yolo.inference(media_path="data/valid/00_11_03.jpg")
yolo.inference(media_path="data/valid/00_11_04.jpg")
yolo.inference(media_path="data/valid/00_11_05.jpg")
yolo.inference(media_path="data/valid/00_11_06.jpg")
