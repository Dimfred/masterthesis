#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import os
import shutil
import sys
from pathlib import Path
import tensorflow as tf
from config import config

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# usage
if len(sys.argv) < 3:
    print("Usage: ./label.py img_dir img_name1 img_name2...")
    sys.exit()

from yolov4.tf import YOLOv4

yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.make_model()

yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)

img_dir = Path(sys.argv[1])
label_dir = config.yolo_labeled_dir
for img_name in sys.argv[2:]:
    img_path = str(img_dir / img_name)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=2)

    # copy the original into the label directory labeled by yolo
    shutil.copy(img_path, label_dir / img_name)
    bounding_boxes = yolo.predict(img)

    # write labels
    with open(f"{os.path.splitext(label_dir / img_name)[0]}.txt", "w") as label_file:
        for x, y, w, h, cls_, _ in bounding_boxes:
            print(f"{int(cls_)} {x} {y} {w} {h}", file=label_file)
