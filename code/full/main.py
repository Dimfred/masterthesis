#!/usr/bin/env python3

from pathlib import Path
import cv2 as cv
import numpy as np
from numba import njit
import sys
import itertools as it

import tensorflow as tf

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config

from ltbuilder import (
    Wire,
    Diode,
    Resistor,
    Capacitor,
    Inductor,
    Source,
    Current,
    LTWriter,
)


def init_yolo():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.safe_classes
    yolo.input_size = config.yolo.input_size
    yolo.channels = config.yolo.channels
    yolo.make_model()
    yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)

    return yolo


def imread(path):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=2)
    return img


def make_abs(bounding_boxes, img):
    # from: x, y, w, h, cls, conf
    # to: x1, y1, x2, y2, cls, conf
    h, w = img.shape[:2]

    abs_bounding_boxes = []
    for xmid, ymid, wrel, hrel, cls_, conf in bounding_boxes:
        wabs, habs = w * wrel, h * hrel
        xabs, yabs = w * xmid, h * ymid

        x1, y1 = int(xabs - 0.5 * wabs), int(yabs - 0.5 * habs)
        x2, y2 = int(xabs + 0.5 * wabs), int(yabs + 0.5 * habs)

        abs_bounding_boxes.append((x1, y1, x2, y2, cls_, conf))

    return abs_bounding_boxes


def remove_boxes(img, bounding_boxes, fill=0, debug=False):
    # bounding_boxes absolute not yolo
    for x1, y1, x2, y2, _, _ in bounding_boxes:
        img[y1:y2, x1:x2] = fill

    if debug:
        utils.show(img)

    return img


def preprocess(img, debug=False):
    # TODO move this shit to config!

    # blur the shit out of the image
    img = cv.GaussianBlur(img, (5, 5), 10)
    if debug:
        utils.show(img)

    # bin threshold
    img = cv.adaptiveThreshold(
        img,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY_INV,
        blockSize=11,
        C=2,
    )
    if debug:
        utils.show(img)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    # remove noise
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
    if debug:
        utils.show(img)

    return img


def close_wire_holes(img, debug=False):
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    # close holes in wire
    img = cv.morphologyEx(
        img,
        cv.MORPH_CLOSE,
        kernel,
        iterations=10,
    )
    img = cv.dilate(img, kernel, iterations=2)
    if debug:
        utils.show(img)

    return img


@njit
def colorize_connected_components(dst, n_labels, components):
    colors = np.array(
        [
            (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            for i in range(n_labels)
        ],
    )

    y = 0
    x = 0
    for row in components:
        for label in row:
            if label != 0:
                dst[y, x] = colors[label]
                x += 1
            else:
                dst[y, x] = (0, 0, 0)
                x += 1

        y += 1
        x = 0

    return dst


def connected_components(img, debug=False):
    n_labels, *components = cv.connectedComponents(img)

    if debug:
        cimg = img.copy()
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cimg[:, :] = (0, 0, 0)
        cimg = colorize_connected_components(cimg, n_labels, components[0])
        utils.show(cimg)

    return n_labels, components[0]


def make_bbox_mask(img, abs_bounding_boxes, debug=False):
    bbox_mask = img.copy()
    bbox_mask[:, :] = 0
    for x1, y1, x2, y2, _, _ in abs_bounding_boxes:
        bbox_mask[y1:y2, x1:x2] = 255

    if debug:
        utils.show(bbox_mask)

    return bbox_mask


def is_in_bounding_box(x, y, bounding_box):
    x1, y1, x2, y2, _, _ = bounding_box
    return x1 <= x and x <= x2 and y1 <= y and y <= y2


def get_intersection_idxs(wire_bbox_mask, debug=False):
    intersection_idxs = np.argwhere(wire_bbox_mask)

    if debug:
        intersections = img.copy()
        intersections[:, :] = 0
        for y, x in intersection_idxs:
            intersections[y, x] = 255
        # TODO why does
        # intersections[intersections_idxs] = 255 not work

        utils.show(intersections)

    return intersection_idxs


def get_connected_bounding_boxes(
    abs_bounding_boxes, connected_components, intersection_idxs, debug=False
):
    connected_bounding_box_idx = {}
    for y, x in intersection_idxs:
        label = connected_components[y, x]
        for bbox_idx, bbox in enumerate(abs_bounding_boxes):
            if is_in_bounding_box(x, y, bbox):
                if label not in connected_bounding_box_idx:
                    connected_bounding_box_idx[label] = set()

                connected_bounding_box_idx[label].add(bbox_idx)

    if debug:
        print(connected_bounding_box_idx)

    return connected_bounding_box_idx

# @njit
def get_min_dist_between_bounding_boxes(bounding_boxes):
    # for i in range(len(bounding_boxes)):
    #     for j in range(i + 1, len(bounding_boxes)):
            # b1, b2 = bounding_boxes[i], bounding_boxes[j]

    min_dist = sys.maxsize
    for b1, b2 in it.combinations(bounding_boxes, 2):
        b1x1, b1y1, b1x2, b1y2, _, _ = b1
        b2x1, b2y1, b2x2, b2y2, _, _ = b2

        b1m = np.array([int(b1y1 + 0.5 * b1y2), int(b1x1 + 0.5 * b1x2)])
        b2m = np.array([int(b2y1 + 0.5 * b2y2), int(b2x1 + 0.5 * b2x2)])

        min_dist = min(np.linalg.norm(b1m - b2m), min_dist)

    return min_dist


class LTBuilder:
    label_mapping = {
        0: Diode,
        1: Diode,
        2: Diode,
        3: Diode,
        4: Source,
        5: Source,
        6: Source,
        7: Source,
        8: Resistor,
        9: Resistor,
        10: Resistor,
        11: Resistor,
        12: Capacitor,
        13: Capacitor,
        14: Capacitor,
        15: Capacitor,
        16: Capacitor,
        17: Capacitor,
        18: Source,
        19: Source,
        20: Source,
        21: Source,
        22: Inductor,
        23: Inductor,
        24: Inductor,
        25: Inductor,
        26: Source,
        27: Source,
        28: Current,
        29: Current,
    }

    # TODO add all
    rotation_mapping = {
        "diode_left": 90,
        "diode_top": 180,
        "diode_right": 270,
        "diode_bot": 0,
        "capacitor_ver": 0,
        "capacitor_hor": 270,
        "inductor_de_ver": 0,
        "inductor_de_hor": 270,
        "resistor_de_ver": 0,
        "resistor_de_hor": 270,
        "ground_bot": 0,  # TODO better ground has no rotation
        "ground_top": 0,
        "ground_right": 0,
        "ground_left": 0,
        "source_ver": 0,
        "source_hor": 270,
        "current_ver": 0,
        "current_hor": 270,
    }

    label_names = {
        0: "diode_left",
        1: "diode_top",
        2: "diode_right",
        3: "diode_bot",
        4: "battery_left",
        5: "battery_top",
        6: "battery_right",
        7: "battery_bot",
        8: "resistor_de_hor",
        9: "resistor_de_ver",
        10: "resistor_usa_hor",
        11: "resistor_usa_ver",
        12: "capacitor_hor",
        13: "capacitor_ver",
        14: "ground_left",
        15: "ground_top",
        16: "ground_right",
        17: "ground_bot",
        18: "lamp_de_hor",
        19: "lamp_de_ver",
        20: "lamp_usa_hor",
        21: "lamp_usa_ver",
        22: "inductor_de_hor",
        23: "inductor_de_ver",
        24: "inductor_usa_hor",
        25: "inductor_usa_ver",
        26: "source_hor",
        27: "source_ver",
        28: "current_hor",
        29: "current_ver",
    }

    @staticmethod
    def build(name, x, y, label):
        label_name = LTBuilder.label_names[label]
        LTClass = LTBuilder.label_mapping[label]

        rotation = LTBuilder.rotation_mapping[label_name]
        return LTClass(name, x, y, rotation)


dorig = True
dprep = True
dremb = True
dclos = True
dconn = True
dmask = True
dinte = True
dbidx = True

lt_file = "ltbuilder/circuits/g.asc"

if __name__ == "__main__":
    imgs = ["00_11.jpg"]  # , "00_11_00.jpg"]
    imgs = [config.valid_dir / img for img in imgs]
    img = imgs[0]
    prediction_params = {"iou_threshold": 0.3, "score_threshold": 0.25}

    yolo = init_yolo()

    img = imread(img)
    img = utils.resize_max_axis(img, 1000)
    if dorig:
        utils.show(img)

    bounding_boxes = yolo.predict(img)
    abs_bounding_boxes = make_abs(bounding_boxes, img)

    # blur, bin threshold, rem salt
    img = preprocess(img, debug=dprep)

    # remove bounding boxes from the img
    img = remove_boxes(img, abs_bounding_boxes, debug=dremb)

    # dilate to close the holes
    img = close_wire_holes(img, debug=dclos)

    # find connected components
    n_labels, components = connected_components(img, debug=dconn)

    # create a mask where all boundings boxes are > 0
    bbox_mask = make_bbox_mask(img, abs_bounding_boxes, debug=dmask)

    # find intersection of bboxes and connected components
    wire_bbox_mask = np.logical_and(components, bbox_mask)

    # returns idx where a bounding box intersects a wire pixel
    intersection_idxs = get_intersection_idxs(wire_bbox_mask, debug=dinte)

    # {"connected_component_label": set(connected bounding box idxs)}
    connected_bounding_box_idxs = get_connected_bounding_boxes(
        abs_bounding_boxes, components, intersection_idxs, debug=dbidx
    )

    # smallest distance between bounding boxes as a normalizer
    min_dist = get_min_dist_between_bounding_boxes(abs_bounding_boxes)

    ltcomponents = []

    # draw all symbols in the grid
    for x1, y1, x2, y2, cls_, _ in abs_bounding_boxes:
        x, y = int(x1 / min_dist), int(y1 / min_dist)
        component = LTBuilder.build("TODO_NAME", x * 10, y * 10, cls_)
        ltcomponents.append(component)

    for _, connected_bbox_idxs in connected_bounding_box_idxs.items():
        for idx1, idx2 in it.combinations(connected_bbox_idxs, 2):
            c1, c2 = ltcomponents[idx1], ltcomponents[idx2]
            b1, b2 = abs_bounding_boxes[idx1], abs_bounding_boxes[idx2]

            if c1.is_horizontal and c2.is_horizontal:
                # b1 if more left
                if b1[0] < b2[0]:
                    wire = Wire(c1.end, c2.start)
                else:
                    wire = Wire(c1.start, c2.end)
            elif c1.is_vertical and c2.is_vertical:
                # b1 if is above
                if b1[1] < b2[1]:
                    wire = Wire(c1.end, c2.start)
                else:
                    wire = Wire(c1.start, c2.end)
            else:
                wire = Wire(c1.start, c2.start)

            ltcomponents.append(wire)

    writer = LTWriter()
    writer.write(lt_file, ltcomponents)
