#!/usr/bin/env python3

from pathlib import Path
import cv2 as cv
import numpy as np
from numba import njit

import tensorflow as tf

tf.get_logger().setLevel("INFO")

# has to be called right after tf import
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config


def init_yolo():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.classes
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



dorig = False
dprep = False
dremb = False
dclos = False
dconn = True
dmask = False
dinte = True

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
    connected = np.logical_and(components, bbox_mask)
    print("components.shape\n{}".format(components.shape))
    print("connected.shape\n{}".format(connected.shape))
    print("bbox_mask.shape\n{}".format(bbox_mask.shape))

    intersection_idxs = np.argwhere(connected)
    if dinte:
        intersections = img.copy()
        intersections[:, :] = 0
        for y, x in intersection_idxs:
            intersections[y, x] = 255
        # TODO why does
        # intersections[intersections_idxs] = 255 not work

        utils.show(intersections)

    connected_bounding_box_idx = {}
    for y, x in intersection_idxs:
        label = components[y, x]
        for bbox_idx, bbox in enumerate(abs_bounding_boxes):
            if is_in_bounding_box(x, y, bbox):
                if label not in connected_bounding_box_idx:
                    connected_bounding_box_idx[label] = set()

                connected_bounding_box_idx[label].add(bbox_idx)

    print(connected_bounding_box_idx)
