#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import tensorflow as tf
import torch as t


from numba import njit
from pathlib import Path
import sys
import enum
import itertools as it
from typing import List

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
from mobile_unet import MobileNetV2_unet
from mobile_unet.eval_unet import get_data_loaders

import utils
from utils import YoloBBox
from config import config

from postprocessing import Postprocessor
from ltbuilder import LTWriter
from ltbuilderadapter import LTBuilderAdapter


def init_yolo():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
    yolo.input_size = config.yolo.input_size
    yolo.channels = config.yolo.channels
    yolo.make_model()

    # yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)
    yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)
    # yolo.load_weights(
    #     config.yolo.stripped_weights, weights_type=config.yolo.weights_type
    # )

    return yolo


def init_unet():
    device = t.device("cuda")

    unet = MobileNetV2_unet(
        mode="eval",
        n_classes=config.unet.n_classes,
        input_size=config.unet.input_size,
        channels=config.unet.channels,
        pretrained=None,
    )
    unet.load_state_dict(t.load(config.unet.weights))
    unet.to(device)
    unet.eval()
    return unet


def imread(path):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=2)
    return img


# TODO grid normalizer class
# @njit
def get_min_dist_between_bounding_boxes(bounding_boxes):
    # for i in range(len(bounding_boxes)):
    #     for j in range(i + 1, len(bounding_boxes)):
    # b1, b2 = bounding_boxes[i], bounding_boxes[j]

    min_dist = sys.maxsize
    for b1, b2 in it.combinations(bounding_boxes, 2):
        b1x1, b1y1, b1x2, b1y2 = b1
        b2x1, b2y1, b2x2, b2y2 = b2

        b1m = np.array([int(b1y1 + 0.5 * b1y2), int(b1x1 + 0.5 * b1x2)])
        b2m = np.array([int(b2y1 + 0.5 * b2y2), int(b2x1 + 0.5 * b2x2)])

        min_dist = min(np.linalg.norm(b1m - b2m), min_dist)

    return min_dist


def get_min_bbox_coord(bounding_boxes):
    min_ = sys.maxsize
    for x1, y1, x2, y2 in bounding_boxes:
        min_ = min(min_, x2 - x1, y2 - y1)

    return min_


lt_file = "ltbuilder/circuits/g.asc"

if __name__ == "__main__":

    for img_name in [
        # valid
        # "07_05.png",
        # "07_06.png",
        # "07_07.png",
        "08_01.png",
        # "08_05.png",
        # "08_07.png",
        # "08_08.png",
        # train
        # "00_11.jpg",
        # "00_19.jpg",
        # "08_08.png",
        # "00_13.jpg",
    ]:
        img_path = config.valid_dir / img_name
        # img_path = config.train_dir / img_name
        print(img_path)

        img = imread(img_path)
        img = utils.resize_max_axis(img, 1000)
        # DEBUG
        # utils.show(img)

        # yolo predictions
        yolo = init_yolo()
        # DEBUG
        yolo.inference(str(img_path))

        prediction_params = {"iou_threshold": 0.3, "score_threshold": 0.25}
        yolo_prediction = yolo.predict(img)
        yolo.unload()

        # unet predictions
        unet = init_unet()
        output = unet.predict(img)
        unet.unload()

        segmentation = cv.resize(output, img.shape[:2][::-1])
        # DEBUG
        utils.show(segmentation[..., np.newaxis], img)

        bboxes = [YoloBBox(img.shape[:2]).from_prediction(p) for p in yolo_prediction]
        # DEBUG
        classes = utils.load_yolo_classes(config.yolo.classes)
        for idx, bbox in enumerate(bboxes):
            print(idx, classes[bbox.label])
        abs_bboxes = [bbox.abs() for bbox in bboxes]

        # close small holes in wire
        bin_img = cv.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
        # DEBUG
        utils.show(bin_img)

        segmentation[segmentation > 0] = 1

        segmentation = segmentation * bin_img
        # DEBUG
        utils.show(segmentation)

        # segmentation = cv.dilate(segmentation, kernel, iterations=2)



        postprocessor = Postprocessor(bboxes, segmentation)
        topology = postprocessor.make_topology()

        # TODO this sucks
        # smallest distance between bounding boxes as a grid normalizer
        min_dist = get_min_dist_between_bounding_boxes(abs_bboxes)
        grid_size = get_min_bbox_coord(abs_bboxes) / 3

        ltbuilder = LTBuilderAdapter(config.yolo.classes, grid_size)
        ltbuilder.make_ltcomponents(bboxes)
        ltbuilder.make_wires(topology, postprocessor.connected_components)

        writer = LTWriter()
        writer.write(lt_file, ltbuilder.ltcomponents)

        # # line follower
        # for label, bbox_idx_orientations in connected_bounding_box_idxs.items():
        #     label = 2
        #     print(bbox_idx_orientations)
        #     ccomp = np.uint8(connected_components.copy())
        #     ccomp[ccomp != label] = 0
        #     ccomp[ccomp == label] = 255

        #     utils.show(ccomp)

        #     bbox_idxs = bbox_idx_orientations.keys()

        #     # main_bbox = abs_bounding_boxes[bbox_idxs[0]]

        #     start = (100, 598)
        #     end = (388, 417)

        #     cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        #     cimg[:, :, :] = 0

        #     cimg[ccomp == 255] = (255, 255, 255)
        #     cimg[start] = (0, 0, 255)
        #     cimg[end] = (0, 0, 255)

        #     bfs = BFS(ccomp, 255)
        #     out = bfs.fit(start, end)

        #     if out is None:
        #         print("NO PATH DETECTED")
        #         sys.exit()

        #     for p in out:
        #         cimg[p] = (0, 0, 255)
        #     utils.show(cimg)

        #     # maybe fuck angle, just Fit the found coordinates to the grid
        #     edges = [start]
        #     out = np.vstack(out)
        #     init_angle = angle(out[0], out[5]path
        #     angle_th = 30
        #     for i, p in enumerate(out[:-5]):
        #         ang = angle(p, out[i + 5])

        #         if abs(ang - init_angle) > 30:
        #             init_angle = ang
        #             edges.append(p)

        #     if tuple(edges[-1]) != end:
        #         edges.append(end)
        #     # pop start and end
        #     # replace with LTCoordinates from corresponding component

        #     print(edges)
        #     for i in range(len(edges) - 1):
        #         cimg = cv.line(
        #             cimg, tuple(edges[i][::-1]), tuple(edges[i + 1][::-1]), (255, 0, 0), 1
        #         )
        #     utils.show(cimg)

        # ccomp = np.uint8(connected_components.copy())
        # ccomp[ccomp == 1] = 255
        # edges = cv.Canny(ccomp, 100, 200, None, 3)

        # utils.show(edges)

        # utils.show(cimg)
        # draw wires
        # for _label, bbox_idx_orientations in connected_bounding_box_idxs.items():
        #     drawn_paths = []

        #     bboxes_current_label = [
        #         ltcomponents[idx] for idx in bbox_idx_orientations.keys()
        #     ]

        #     # get the most left bounding box and the bounding box most bot
        #     most_left = min(
        #         bboxes_current_label, key=lambda ltcomponent: ltcomponent.x.value
        #     )
        #     most_bot = max(
        #         bboxes_current_label, key=lambda ltcomponent: ltcomponent.y.value
        #     )

        #     if len(bbox_idx_orientations.keys()) == 2:
        #         for (b1_idx, b1_orientation), (b2_idx, b2_orientation) in it.combinations(
        #             bbox_idx_orientations.items(), 2
        #         ):
        #             c1, c2 = ltcomponents[b1_idx], ltcomponents[b2_idx]
        #             b1, b2 = abs_bounding_boxes[b1_idx], abs_bounding_boxes[b2_idx]

        #             if (
        #                 b1_orientation == LTBuilder.CONNECTION_ORIENTATION.LEFT.value
        #                 and b2_orientation == LTBuilder.CONNECTION_ORIENTATION.RIGHT.value
        #             ):
        #                 (x1l, y1l), (x2r, y2r) = c1.left, c2.right
        #                 xmin, xmax = min(x1l, x2r), max(x1l, x2r)
        #                 ymin, ymax = min(y1l, y2r), max(y1l, y2r),

        #                 w1 = Wire((xmin, ymax), (xmax, ymax))
        #                 w2 = Wire((xmin, ymin), (xmax, ymax))

        #                 ltcomponents.append(w1)
        #                 ltcomponents.append(w2)

        #             elif(
        #                 b1_orientation == LTBuilder.CONNECTION_ORIENTATION.RIGHT.value
        #                 and b2_orientation == LTBuilder.CONNECTION_ORIENTATION.LEFT.value
        #             ):
        #                 w = Wire(c1.right, c2.left)
        #                 ltcomponents.append(w)

        #             elif(
        #                 b1_orientation == LTBuilder.CONNECTION_ORIENTATION.LEFT.value
        #                 and b2_orientation == LTBuilder.CONNECTION_ORIENTATION.LEFT.value
        #             ):
        #                 (x1l, y1l), (x2l, y2l) = c1.left, c2.left
        #                 xmin, xmax = min(x1l, x2l), max(x1l, x2l)
        #                 ymin, ymax = min(y1l, y2l), max(y1l, y2l),

        #                 w1 = Wire((xmin, ymax), (xmax, ymax))
        #                 w2 = Wire((xmax, ymin), (xmax, ymax))

        #                 ltcomponents.append(w1)
        #                 ltcomponents.append(w2)

        #             elif(
        #                 b1_orientation == LTBuilder.CONNECTION_ORIENTATION.RIGHT.value
        #                 and b2_orientation == LTBuilder.CONNECTION_ORIENTATION.RIGHT.value
        #             ):
        #                 (x1r, y1r), (x2r, y2r) = c1.right, c2.right
        #                 xmin, xmax = min(x1r, x2r), max(x1r, x2r)
        #                 ymin, ymax = min(y1r, y2r), max(y1r, y2r),

        #                 # c1 left
        #                 if x1r < x2r:
        #                     p1 = (xmin, ymin)
        #                     p2 = (xmax, ymin)
        #                     p3 = (xmax, ymax)
        #                 else:
        #                     p1 = (xmax, ymin)
        #                     p2 = (xmax, ymax)
        #                     p3 = (xmin, ymax)

        #                 ltcomponents.append(Wire(p1, p2))
        #                 ltcomponents.append(Wire(p2, p3))

        #     elif len(bbox_idx_orientations.keys()) == 3:
        #         pass

        # elif (
        #     b1_orientation == LTBuilder.CONNECTION_ORIENTATION.RIGHT.value
        #     and b2_orientation == LTBuilder.CONNECTION_ORIENTATION.LEFT.value
        # ):
        # # TODO only when edges active
        # if c1 is None or c2 is None:
        #     print("is none")
        #     continue

        #         b1, b2 = abs_bounding_boxes[b1_idx], abs_bounding_boxes[b2_idx]

        #         if c1.is_horizontal and c2.is_horizontal:
        #             # b1 if more left
        #             if b1[0] < b2[0]:
        #                 wire = Wire(c1.right, c2.left)
        #             else:
        #                 wire = Wire(c1.left, c2.right)

        #         elif c1.is_vertical and c2.is_vertical:
        #             # b1 if is above
        #             if b1[1] < b2[1]:
        #                 wire = Wire(c1.bottom, c2.top)
        #             else:
        #                 wire = Wire(c1.top, c2.bottom)

        #         elif c1.is_vertical and c2.is_horizontal:
        #             # b1 is above
        #             if b1[1] < b2[1]:
        #                 # b1 is more left
        #                 if b1[0] < b2[0]:
        #                     wire = Wire(c1.bottom, c2.left)
        #                 # b1 is more right
        #                 else:
        #                     wire = Wire(c1.bottom, c2.right)
        #             else:
        #                 if b1[0] < b2[0]:
        #                     wire = Wire(c1.top, c2.left)
        #                 # b1 is more right
        #                 else:
        #                     wire = Wire(c1.top, c2.right)

        #         elif c1.is_horizontal and c2.is_vertical:
        #             # b1 is above
        #             if b1[1] < b2[1]:
        #                 # b1 is more left
        #                 if b1[0] < b2[0]:
        #                     wire = Wire(c1.right, c2.top)
        #                 # b1 is more right
        #                 else:
        #                     wire = Wire(c1.left, c2.bottom)
        #             else:
        #                 # b1 is more left
        #                 if b1[0] < b2[0]:
        #                     wire = Wire(c1.right, c2.bottom)
        #                 # b1 is more right
        #                 else:
        #                     wire = Wire(c1.left, c2.top)
        #         else:
        #             print("c1.is_horizonal\n{}".format(c1.is_horizontal))
        #             print("c2.is_vertical\n{}".format(c2.is_vertical))
        #             wire = None
        #             print("THIS CASE SHOULD NOTHAPPEN")

        #         ltcomponents.append(wire)
