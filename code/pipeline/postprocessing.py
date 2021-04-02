import numpy as np


import numpy as np
import cv2 as cv

from typing import List
from numba import njit
from enum import Enum
from cached_property import cached_property
import sys

import utils
from utils import YoloBBox
from config import config

# TODO use evalPipiline
idx_to_classname = utils.Yolo.parse_classes(config.yolo.classes)


class ORIENTATION(Enum):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    BOTTOM = 3

    # TODO
    # def __eq__(self, other):
    #     if type(self) ==  type(other):
    #         raise ValueError("Can only compare to itself")

    #     return self.value == other.value


class Helpers:
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


class Utils:
    cross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    @staticmethod
    def connected_components(img, debug=False):
        n_labels, *components = cv.connectedComponents(img)

        if debug:
            cimg = img.copy()
            cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cimg[:, :] = (0, 0, 0)
            cimg = Helpers.colorize_connected_components(cimg, n_labels, components[0])
            utils.show(cimg)

        return n_labels, components[0]

    @staticmethod
    def rm_bboxes(img, bboxes, fill=0, debug=False):
        # bounding_boxes absolute not yolo
        for x1, y1, x2, y2 in bboxes:
            img[y1:y2, x1:x2] = fill

        if debug:
            utils.show(img)

        return img

    @staticmethod
    def make_bbox_mask(img, bboxes, debug=False):
        mask = np.zeros_like(img).astype(np.uint8)

        # TODO outofbounds check
        for x1, y1, x2, y2 in bboxes:
            mask[y1 - 2 : y2 + 2, x1 - 2 : x2 + 2] = 255
            mask[y1 + 2 : y2 - 2, x1 + 2 : x2 - 2] = 0

        if debug:
            utils.show(mask)

        return mask

    @staticmethod
    def close(img, iterations=4, debug=False):
        img = cv.morphologyEx(
            img,
            cv.MORPH_CLOSE,
            Utils.cross,
            iterations=iterations,
        )

        if debug:
            utils.show(img)

        return img

    @staticmethod
    def dilate(img, iterations=2, debug=False):
        img = cv.dilate(img, Utils.cross, iterations=iterations)

        if debug:
            utils.show(img)

        return img

    @staticmethod
    def erode(img, iterations=2, debug=False):
        img = cv.erode(img, Utils.cross, iterations=iterations)

        if debug:
            utils.show(img)

        return img


class BBoxConnection:
    def __init__(self, coords):
        self.coords = coords
        # after nms changes to orientation
        self.orientation = OrientationCounter()


class OrientationCounter:
    def __init__(self):
        self.count = [0 for _ in range(len(ORIENTATION))]

    def max(self):
        # returns the most present orientation
        return self.count.index(max(self.count))

    def __getitem__(self, idx):
        return self.count[idx]

    def __setitem__(self, idx, val):
        self.count[idx] = val


class BBoxConnector:
    def __init__(self, connected_components, bboxes, intersection_idxs):
        self.connected_components = connected_components
        self.bboxes = bboxes
        self.intersection_idxs = intersection_idxs

        self.topology = {}

    def make_topology(self, debug=False):
        # create each label where an intersection is generally present
        for y, x in self.intersection_idxs:
            label = self.connected_components[y, x]
            self.topology[label] = {}

        abs_bboxes = [bbox.abs for bbox in self.bboxes]

        for y, x in self.intersection_idxs:
            added = False
            label = self.connected_components[y, x]

            # DEBUG
            # print("LABEL", label)
            # cimg = np.uint8(self.connected_components.copy())
            # cimg[cimg > 0] = 255
            # cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)
            # cimg[y, x] = (0, 0, 255)
            # cv.circle(cimg, (x, y), 5, (0, 0, 255), 2)
            # utils.show(cimg)

            for bbox_idx, bbox in enumerate(abs_bboxes):
                orientation = self.get_connection_orientation(x, y, bbox)
                if orientation is None:
                    continue

                # if bbox_idx == 1 and label == 7:
                # DEBUG
                # print("LABEL", label)
                # cimg = np.uint8(self.connected_components.copy())
                # cimg[cimg > 0] = 255
                # cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)
                # cimg[y, x] = (0, 0, 255)
                # cv.circle(cimg, (x, y), 5, (0, 0, 255), 2)
                # utils.show(cimg)
                # print("DAFUUUQ")

                # create an orientation counter for this bounding box inside that label
                if bbox_idx not in self.topology[label]:
                    self.topology[label][bbox_idx] = BBoxConnection((x, y))

                # increment the current orientation
                self.topology[label][bbox_idx].orientation[orientation.value] += 1
                # TODO normally a point can only be associated with one bbox
                break

        self.topology = self.nms(self.topology)

        if debug:
            utils.Topology.print_dict(self.topology)

        return self.topology

    def get_connection_orientation(self, x, y, bbox):
        on_bbox_edge = lambda val, bbox_val: bbox_val - 2 <= val and val <= bbox_val + 2
        in_bbox = lambda val, low, high: low <= val and val <= high

        # TODO ugly
        x1, y1, x2, y2 = bbox
        if on_bbox_edge(x, x1) and in_bbox(y, y1, y2):
            return ORIENTATION.LEFT
        elif on_bbox_edge(x, x2) and in_bbox(y, y1, y2):
            return ORIENTATION.RIGHT
        elif on_bbox_edge(y, y1) and in_bbox(x, x1, x2):
            return ORIENTATION.TOP
        elif on_bbox_edge(y, y2) and in_bbox(x, x1, x2):
            return ORIENTATION.BOTTOM

        return None

    def nms(self, topology):
        suppressed = {}
        for label, sub_topology in topology.items():
            # delete sub topologies with only one component present, those are noise
            # TODO can't really do that
            # doing the one below instead
            if len(sub_topology) == 0:
                continue

            # if len(sub_topology) <= 1:
            #     continue

            suppressed[label] = {}

            # supress non-max orientations
            for bbox_idx, bbox_connection in sub_topology.items():
                bbox_connection.orientation = bbox_connection.orientation.max()

                bbox = self.bboxes[bbox_idx]

                # if *_hor or *_left or *_right => only left right possible
                # if *_ver or *_top or *_bot => only top bottom possible
                # print(idx_to_classname[bbox.label])
                classname = idx_to_classname[bbox.label]
                orientation = classname.split("_")[-1]

                bbox_or = bbox_connection.orientation
                if (
                    orientation == "hor"
                    or orientation == "left"
                    or orientation == "right"
                ):
                    if (
                        bbox_or == ORIENTATION.LEFT.value
                        or bbox_or == ORIENTATION.RIGHT.value
                    ):
                        suppressed[label][bbox_idx] = bbox_connection
                elif (
                    orientation == "ver" or orientation == "top" or orientation == "bot"
                ):
                    if (
                        bbox_or == ORIENTATION.TOP.value
                        or bbox_or == ORIENTATION.BOTTOM.value
                    ):
                        suppressed[label][bbox_idx] = bbox_connection
                # else nms

        removed_empty = {}
        for label, sub_topology in suppressed.items():
            if not sub_topology:
                continue

            removed_empty[label] = sub_topology

        return removed_empty


class WireTracer:
    def __init__(
        self, topology, connected_components, angle_threshold=10, angle_step=5
    ):
        self.topology = topology
        self.connected_components = connected_components

        self.angle_threshold = angle_threshold
        self.angle_step = angle_step

    def make_trace(self):
        pass


class NodeBuilder:
    # TODO creates fake nodes
    pass


class Postprocessor:
    # TODO topology builder
    def __init__(self, bboxes: List[YoloBBox], segmentation: np.ndarray):
        self.bboxes = bboxes
        self.segmentation_orig = segmentation

    def make_topology(self):
        self.segmentation = Utils.rm_bboxes(
            self.segmentation_orig, self.abs_bboxes, debug=False
        )

        self.segmentation = Utils.close(self.segmentation, debug=False)
        # self.segmentation = Utils.dilate(self.segmentation, iterations=2, debug=False)

        _, self.connected_components = Utils.connected_components(
            self.segmentation, debug=False
        )

        # split connected components in sub components create new bboxes and run
        # connected components again
        bbox_mask = Utils.make_bbox_mask(
            self.segmentation, self.abs_bboxes, debug=False
        )
        bbox_wire_mask = np.logical_and(bbox_mask, self.connected_components)
        # DEBUG
        # utils.show(np.uint8(np.logical_or(bbox_mask, self.connected_components)) * 255)
        # utils.show(np.uint8(bbox_wire_mask) * 255)
        # utils.show(np.uint8(np.logical_or(bbox_mask, self.connected_components)) * 255)

        bbox_wire_intersections = np.argwhere(bbox_wire_mask)

        connector = BBoxConnector(
            self.connected_components, self.bboxes, bbox_wire_intersections
        )
        topology = connector.make_topology()

        return topology

    @cached_property
    def abs_bboxes(self):
        return [bbox.abs for bbox in self.bboxes]
