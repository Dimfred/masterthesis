#!/usr/bin/env python3

import sys
from pathlib import Path
import cv2 as cv
import tensorflow as tf
import os
from tabulate import tabulate
import numpy as np
from mean_average_precision import MeanAveragePrecision
from utils import YoloBBox

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

for input_size in (608,):  # 832):
    print("--------------------------------------------------------------------------")
    print("For input_size:", input_size)

    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
    yolo.input_size = input_size
    yolo.channels = config.yolo.channels
    yolo.make_model()
    yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)
    # yolo.load_weights(config.weights_dir / "yolov4-tiny-100.weights", weights_type=config.yolo.weights_type)

    dir_ = config.test_out_dir

    preds, gts = [], []
    for img_path in utils.list_imgs(dir_):
        if filter_(img_path):
            print(img_path)
            ground_truth_path = utils.Yolo.label_from_img(img_path)

            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            img = utils.resize_max_axis(img, 1000)
            img = np.expand_dims(img, axis=2)

            gt = utils.load_ground_truth(ground_truth_path)
            gts.append(gt)

            pred = yolo.predict(img)
            preds.append(pred)

    img_shape = (input_size, input_size)
    mAP = utils.MeanAveragePrecision(
        yolo.classes, img_shape, iou_threshs=[0.5, 0.6, 0.7, 0.8]
    )
    mAP.add(preds, gts)
    results = mAP.compute()

    pretty = mAP.prettify(results)
    print(pretty)

    maps = mAP.get_maps(results)
    print(maps)





            # bbox_gt = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]
            # bbox_gt = np.vstack([[*bbox.abs(), bbox.label, 0, 0] for bbox in bbox_gt])

            # print(bbox_gt.shape)

            # bbox_pred = [
            #     YoloBBox(img.shape).from_prediction(pred) for pred in predictions
            # ]
            # bbox_pred = np.vstack(
            #     [[*bbox.abs(), bbox.label, bbox.confidence] for bbox in bbox_pred]
            # )
            # # print(bbox_pred.shape)

            # mAP.add(bbox_pred, bbox_gt)

    # iou_threshs = np.arange(0.5, 1, 0.05)
    # results_095 = mAP.value(
    #     iou_thresholds=iou_threshs, recall_thresholds=None, mpolicy="greedy"
    # )
    # results_05 = mAP.value(iou_thresholds=0.5, recall_thresholds=None, mpolicy="greedy")

    # pretty = [["Class", "AP@0.5"]] #, "mAP@0.5", "mAP@0.5:0.95"]]
    # map05 = results_05["mAP"]
    # map095 = results_095["mAP"]
    # print(results_095)

    # results_05 = sorted(results_05[0.5].items())
    # results_095 = sorted(results_095[0.5].items())
    # for (cls_idx, data05), (cls_idx2, data095) in zip(results_05, results_095):
    #     assert cls_idx == cls_idx2

    #     ffloat = lambda f: "{:.3f}".format(f)

    #     cls_name = yolo.classes[cls_idx]
    #     # ap05, ap095  = data05["ap"], data095["ap"]
    #     ap05 = data05["ap"]
    #     pretty += [[cls_name, ffloat(ap05)]]

    # print(tabulate(pretty))

    # err = metrics.calculate(ground_truth, predictions, img.shape[:2])
    # errors.append((img, img_path, *err))

    # metrics.confusion()
    # metrics.perform(["f1", "precision", "recall"], precision=4)

    # if check_errors:
    #     has_errors = lambda x, y, z: (x or y or z)
    #     for img, img_path, wrong_prediction, unmatched_gt, unmatched_pred in errors:
    #         if not has_errors(wrong_prediction, unmatched_gt, unmatched_pred):
    #             continue

    #         print("Image:", img_path)
    #         for err_bbox in wrong_prediction:
    #             print("Wrong predictions")
    #             bbox = np.array([err_bbox.yolo()])
    #             tmp = img.copy()
    #             tmp = yolo.draw_bboxes(tmp, bbox)
    #             utils.show(tmp)

    #         # TODO
    #         for err_bbox in unmatched_gt:
    #             print("Unmatched Ground truth")
    #             bbox = np.array([err_bbox.yolo()])
    #             tmp = img.copy()
    #             tmp = yolo.draw_bboxes(tmp, bbox)
    #             utils.show(tmp)

    #         for err_bbox in unmatched_pred:
    #             print("Unmatched Prediction")
    #             bbox = np.array([err_bbox.yolo()])
    #             tmp = img.copy()
    #             tmp = yolo.draw_bboxes(tmp, bbox)
    #             utils.show(tmp)
