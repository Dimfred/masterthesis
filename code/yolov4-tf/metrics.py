#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import tensorflow as tf
import os
from tabulate import tabulate
import numpy as np

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config


check_errors = True

# filter_ = lambda file_: ("00_20" in file_ or "00_19" in file_)
# is checkered
# filter_ = lambda file_: sum(
#     [(name in file_) for name in ("00_19", "00_20", "07_05", "07_06", "07_07", "07_08")]
# )
filter_ = lambda file_: True
if not filter_("asdf"):
    print("-----------------------------------------------------------------")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("CARE FILTER IS USED")

for input_size in (608, 832):
    print("--------------------------------------------------------------------------")
    print("For input_size:", input_size)

    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
    yolo.input_size = input_size
    yolo.channels = config.yolo.channels
    yolo.make_model()
    yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)


    dir_ = config.valid_preprocessed_dir
    metrics = utils.Metrics(yolo.classes, dir_, iou_thresh=0.2)

    errors = []

    for file_ in os.listdir(dir_):
        if utils.is_img(file_) and filter_(file_):
            # print(file_)
            img_path = str(dir_ / file_)
            ground_truth_path = utils.label_file_from_img(img_path)

            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            img = utils.resize_max_axis(img, 1000)
            img = np.expand_dims(img, axis=2)

            ground_truth = utils.load_ground_truth(ground_truth_path)
            predictions = yolo.predict(img)

            err = metrics.calculate(ground_truth, predictions, img.shape[:2])
            errors.append((img, img_path, *err))


    metrics.confusion()
    metrics.perform(["f1", "precision", "recall"], precision=4)

    if check_errors:
        has_errors = lambda x, y, z: (x or y or z)
        for img, img_path, wrong_prediction, unmatched_gt, unmatched_pred in errors:
            if not has_errors(wrong_prediction, unmatched_gt, unmatched_pred):
                continue

            print("Image:", img_path)
            for err_bbox in wrong_prediction:
                print("Wrong predictions")
                bbox = np.array([err_bbox.yolo()])
                tmp = img.copy()
                tmp = yolo.draw_bboxes(tmp, bbox)
                utils.show(tmp)

            # TODO
            for err_bbox in unmatched_gt:
                print("Unmatched Ground truth")
                bbox = np.array([err_bbox.yolo()])
                tmp = img.copy()
                tmp = yolo.draw_bboxes(tmp, bbox)
                utils.show(tmp)

            for err_bbox in unmatched_pred:
                print("Unmatched Prediction")
                bbox = np.array([err_bbox.yolo()])
                tmp = img.copy()
                tmp = yolo.draw_bboxes(tmp, bbox)
                utils.show(tmp)
