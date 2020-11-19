#!/usr/bin/env python3

from pathlib import Path
from sys import maxsize
import cv2 as cv
import cv2.ximgproc
import numpy as np
from numba import njit
import sys
import enum
import itertools as it
import math
import scipy as sp
import scipy.stats
import scipy.sparse
import sklearn as sk
import sklearn.linear_model
import sklearn.cluster
import hdbscan
import networkx as nx
from collections import deque

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
    Ground,
    LTWriter,
)


def init_yolo():
    yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
    yolo.classes = config.yolo.safe_classes
    # yolo.classes = config.yolo.classes
    # yolo.classes = config.yolo.stripped_classes
    yolo.input_size = config.yolo.input_size
    yolo.channels = config.yolo.channels
    yolo.make_model()

    # yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)
    yolo.load_weights(
        config.yolo.safe_label_weights, weights_type=config.yolo.weights_type
    )
    # yolo.load_weights(
    #     config.yolo.stripped_weights, weights_type=config.yolo.weights_type
    # )

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


def get_connected_components(img, debug=False):
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
        # TODO outofbounds check
        bbox_mask[y1 - 1 : y2 + 1, x1 - 1 : x2 + 1] = 255
        bbox_mask[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1] = 0

    if debug:
        utils.show(bbox_mask)

    return bbox_mask


def is_on_bounding_box(x, y, bounding_box):
    x1, y1, x2, y2, _, _ = bounding_box
    return x1 == x or x == x2 or y1 == y or y == y2


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


def get_connection_orientation(y, x, bounding_box):
    x1, y1, x2, y2, _, _ = bounding_box
    if x == x1:
        if y == y1 or y == y2:
            # undecided
            return None
        else:
            return LTBuilder.CONNECTION_ORIENTATION.LEFT
    elif x == x2:
        if y == y1 or y == y2:
            # undecided
            return None
        else:
            return LTBuilder.CONNECTION_ORIENTATION.RIGHT
    elif y == y1:
        return LTBuilder.CONNECTION_ORIENTATION.TOP
    elif y == y2:
        return LTBuilder.CONNECTION_ORIENTATION.BOTTOM

    return None


def get_connected_bounding_boxes(
    abs_bounding_boxes, connected_components, intersection_idxs, debug=False
):
    connected_bounding_box_idx_orientation = {}
    for y, x in intersection_idxs:
        label = connected_components[y, x]
        for bbox_idx, bbox in enumerate(abs_bounding_boxes):
            if label not in connected_bounding_box_idx_orientation:
                connected_bounding_box_idx_orientation[label] = {}

            connection_orientation = get_connection_orientation(y, x, bbox)
            if connection_orientation is None:
                continue

            if bbox_idx not in connected_bounding_box_idx_orientation[label]:
                connected_bounding_box_idx_orientation[label][bbox_idx] = [
                    0 for i in range(len(LTBuilder.CONNECTION_ORIENTATION))
                ]

            # increment the orientation index
            connected_bounding_box_idx_orientation[label][bbox_idx][
                connection_orientation.value
            ] += 1

    nms_bbox_idx_orientations = {}
    for label, bbox_idx_orientations in connected_bounding_box_idx_orientation.items():
        # intersection with just one bounding box (not a connection)
        if len(bbox_idx_orientations) <= 1:
            continue

        nms_bbox_idx_orientations[label] = {}
        # nms on orientation
        for bbox_idx, orientations in bbox_idx_orientations.items():
            nms_bbox_idx_orientations[label][bbox_idx] = orientations.index(
                max(orientations)
            )

    if debug:
        for label, bbox_idx_orientations in nms_bbox_idx_orientations.items():
            print(label, ":", bbox_idx_orientations)

    return nms_bbox_idx_orientations


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


def get_min_bbox_coord(bounding_boxes):
    min_ = sys.maxsize
    for x1, y1, x2, y2, _, _ in bounding_boxes:
        min_ = min(min_, x2 - x1, y2 - y1)

    return min_



def skeletonize(img, debug=False):
    # TODO algoname
    img = img.copy()
    skel = img.copy()
    skel[:, :] = 0
    tmp = skel.copy()
    cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

    while np.any(img):
        tmp = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=cross)
        tmp = cv.bitwise_not(tmp)
        tmp = cv.bitwise_and(img, tmp)
        skel = cv.bitwise_or(skel, tmp)
        img = cv.erode(img, kernel=cross)

    if debug:
        utils.show(skel)

    return skel


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
        14: Ground,
        15: Ground,
        16: Ground,
        17: Ground,
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
        "ground_bot": 0,  # TODO better; ground has no rotation
        "ground_top": 0,
        "ground_right": 0,
        "ground_left": 0,
        "source_ver": 0,
        "source_hor": 270,
        "current_ver": 0,
        "current_hor": 270,
        "battery_left": 0,
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

    class CONNECTION_ORIENTATION(enum.Enum):
        LEFT = 0
        RIGHT = 1
        TOP = 2
        BOTTOM = 3

    @staticmethod
    def build(name, x, y, label):
        if int(label) not in LTBuilder.label_names:
            print(label)
            return None

        label_name = LTBuilder.label_names[label]
        LTClass = LTBuilder.label_mapping[label]

        rotation = LTBuilder.rotation_mapping[label_name]
        return LTClass(name, x, y, rotation)


class BFS:
    _yneighbors = [-1, 0, 0, 1]
    _xneighbors = [0, -1, 1, 0]

    def __init__(self, binimg, value=255):
        self.binimg = binimg
        self.visited = np.zeros_like(binimg)
        self.queue = deque()
        self.value = 255

    def is_valid(self, p):
        return self.binimg[p] == self.value

    def fit(self, start: tuple, end: tuple):
        ys, xs = start
        self.visited[ys, xs] = True

        # 0 distance

        path = deque()
        self.queue.append([start])
        while self.queue:
            path = self.queue.popleft()

            p = path[-1]
            if p == end:
                return path

            for yoff, xoff in zip(self._yneighbors, self._xneighbors):
                y, x = p
                np = (y + yoff, x + xoff)

                if self.is_valid(np) and not self.visited[np]:
                    self.visited[np] = True
                    new_path = list(path)
                    new_path.append(np)
                    self.queue.append(new_path)

        return None


def angle(p1, p2):
    v = math.atan2(*(p2 - p1))
    angle = v * (180.0 / math.pi)

    # if angle < 0:
    #     angle += 360

    return angle


dinfe = True
dorig = True
dprep = True
dremb = True
dclos = True
dconn = True
dmask = True
dinte = True
dbidx = True
dskel = True

lt_file = "ltbuilder/circuits/g.asc"

if __name__ == "__main__":
    imgs = ["00_13.jpg"]  # , "00_11_00.jpg"]
    imgs = [config.unlabeled_dir / img for img in imgs]

    # imgs = ["07_00.png"]  # , "00_11_00.jpg"]
    # imgs = [config.valid_dir / img for img in imgs]
    img = imgs[0]
    print(img)
    prediction_params = {"iou_threshold": 0.3, "score_threshold": 0.25}

    yolo = init_yolo()
    if dinfe:
        yolo.inference(str(img))

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
    n_components, connected_components = get_connected_components(img, debug=dconn)

    # create a mask where all boundings boxes are > 0
    bbox_mask = make_bbox_mask(img, abs_bounding_boxes, debug=dmask)

    # find intersection of bboxes and connected components
    wire_bbox_mask = np.logical_and(connected_components, bbox_mask)

    # returns idx where a bounding box intersects a wire pixel
    intersection_idxs = get_intersection_idxs(wire_bbox_mask, debug=dinte)

    # {"connected_component_label": set(connected bounding box idxs)}
    connected_bounding_box_idxs = get_connected_bounding_boxes(
        abs_bounding_boxes, connected_components, intersection_idxs, debug=dbidx
    )

    # smallest distance between bounding boxes as a normalizer
    # min_dist = get_min_dist_between_bounding_boxes(abs_bounding_boxes)
    grid_size = get_min_bbox_coord(abs_bounding_boxes) / 2

    # TODO refactor this all into the LTBuilder
    ltcomponents = []

    # draw all symbols in the grid
    for x1, y1, x2, y2, cls_, _ in abs_bounding_boxes:
        xm, ym = int(x1 + 0.5 * x2), int(y1 + 0.5 * y2)

        x_grid, y_grid = int(xm / grid_size), int(ym / grid_size)

        component = LTBuilder.build("TODO_NAME", x_grid, y_grid, cls_)
        ltcomponents.append(component)

    # line follower
    for label, bbox_idx_orientations in connected_bounding_box_idxs.items():
        label = 2
        print(bbox_idx_orientations)
        ccomp = np.uint8(connected_components.copy())
        ccomp[ccomp != label] = 0
        ccomp[ccomp == label] = 255

        utils.show(ccomp)

        bbox_idxs = bbox_idx_orientations.keys()

        # main_bbox = abs_bounding_boxes[bbox_idxs[0]]

        start = (100, 598)
        end = (388, 417)

        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cimg[:, :, :] = 0

        cimg[ccomp == 255] = (255, 255, 255)
        cimg[start] = (0, 0, 255)
        cimg[end] = (0, 0, 255)

        bfs = BFS(ccomp, 255)
        out = bfs.fit(start, end)

        if out is None:
            print("NO PATH DETECTED")
            sys.exit()

        for p in out:
            cimg[p] = (0, 0, 255)
        utils.show(cimg)

        # maybe fuck angle, just Fit the found coordinates to the grid
        edges = [start]
        out = np.vstack(out)
        init_angle = angle(out[0], out[5])
        angle_th = 30
        for i, p in enumerate(out[:-5]):
            ang = angle(p, out[i + 5])

            if abs(ang - init_angle) > 30:
                init_angle = ang
                edges.append(p)

        if tuple(edges[-1]) != end:
            edges.append(end)


        # pop start and end
        # replace with LTCoordinates from corresponding component

        print(edges)
        for i in range(len(edges) - 1):
            cimg = cv.line(
                cimg, tuple(edges[i][::-1]), tuple(edges[i + 1][::-1]), (255, 0, 0), 1
            )
        utils.show(cimg)

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

    writer = LTWriter()
    writer.write(lt_file, ltcomponents)
