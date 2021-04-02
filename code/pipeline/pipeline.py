#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import torch as t


from numba import njit
from pathlib import Path
import sys
import enum
import itertools as it
from typing import List

# import tensorflow as tf
# tf.get_logger().setLevel("INFO")
# has to be called right after tf import
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# from yolov4.tf import YOLOv4
from mobile_unet import MobileNetV2_unet

import utils
from config import config

import utils
from utils import YoloBBox
from config import config

from postprocessing import Postprocessor, ORIENTATION, BBoxConnection
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


def run_prediction(img_path, iou_thresh=0.25, conf_thresh=0.3, debug=False):
    import subprocess as sp

    prediction_file = "prediction.npy"

    sp.run(
        [
            "python3.8",
            "-c",
            f"""
import cv2 as cv
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config


yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.make_model()

# yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)
yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

def imread(path):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=2)
    return img

img = imread("{img_path}")
img = utils.resize_max_axis(img, 1000)

prediction = yolo.predict(
    img, iou_threshold={iou_thresh}, score_threshold={conf_thresh}
)

if {debug}:
    utils.show_bboxes(img, prediction, type_="pred")

np.save("{prediction_file}", prediction)
""",
        ],
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )
    return np.load(prediction_file)


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


class EvaluationItem:
    def __init__(self):
        self.classes = None
        self.img = None
        self.gt_bboxes = None
        self.pred_bboxes = None
        self.segmentation = None
        self.topology = None
        self.false_negative_gts = []

    def from_predict(self, classes, img, gt_bboxes, pred_bboxes, segmentation):
        self.classes = classes
        self.img = img
        self.gt_bboxes = gt_bboxes
        self.pred_bboxes = pred_bboxes
        self.segmentation = segmentation

        return self

    def from_postprocess(self, topology):
        self.topology = topology

        return self


def predict(img_path, iou_thresh, conf_thresh):
    img = imread(str(img_path))
    img = utils.resize_max_axis(img, 1000)
    # DEBUG
    # utils.show(img)

    yolo_prediction = run_prediction(
        img_path, iou_thresh=iou_thresh, conf_thresh=conf_thresh, debug=True
    )
    pred_bboxes = [YoloBBox(img.shape).from_prediction(p) for p in yolo_prediction]

    ground_truth = utils.Yolo.parse_labels(utils.Yolo.label_from_img(img_path))
    gt_bboxes = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]

    # unet predictions
    unet = init_unet()
    output = unet.predict(img)
    unet.unload()

    segmentation = cv.resize(output, img.shape[:2][::-1])
    # DEBUG
    utils.show(segmentation[..., np.newaxis], img)

    classes = utils.Yolo.parse_classes(config.yolo.classes)

    return EvaluationItem().from_predict(
        classes, img, gt_bboxes, pred_bboxes, segmentation
    )


def add_false_negatives(eval_item, threshold):
    pred_bboxes = eval_item.pred_bboxes
    gt_bboxes = eval_item.gt_bboxes

    matches = {idx: [] for idx, _ in enumerate(gt_bboxes)}

    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        for pred_idx, pred_bbox in enumerate(pred_bboxes):
            if utils.calc_iou(gt_bbox.abs, pred_bbox.abs) < threshold:
                continue

            matches[gt_idx].append(pred_idx)

    for gt_idx, match in matches.items():
        # unmatched gt
        if len(match) == 0:
            # we definetly have a false negative!
            if len(gt_bboxes) > len(pred_bboxes):
                # missing gt_bbox
                gt_bbox = gt_bboxes[gt_idx]
                # add the missing bbox to the predictions
                pred_bboxes.append(gt_bbox)
                # the idx of the fn prediction added to the predictions to match
                # the gt
                fn_pred_idx = len(pred_bboxes) - 1
                # add the fn to the gt_idx s.t. it can get synced with the gt
                # in the next step
                matches[gt_idx] = [fn_pred_idx]
                # remember that we have added this one
                eval_item.false_negative_gts.append(gt_idx)
                # add the fn to the topology of the prediciton

        elif len(match) > 1:
            raise ValueError("FALSE POSITIVE NOT YET HANDLED")
        # else == 1 correct

    gt_to_pred = {gt_idx: pred_idxs[0] for gt_idx, pred_idxs in matches.items()}
    pred_to_gt = {pred_idx: gt_idx for gt_idx, pred_idx in gt_to_pred.items()}

    # sync the gt with the pred
    new_pred_bboxes = []
    for gt_idx in range(len(gt_to_pred)):
        new_pred_idx = gt_to_pred[gt_idx]
        new_pred_bbox = pred_bboxes[new_pred_idx]
        new_pred_bboxes.append(new_pred_bbox)

    eval_item.pred_bboxes = new_pred_bboxes

    return eval_item


def find_unused_connected_component_label(topology, new_topology={}):
    for i in range(100000):
        if i not in topology and i not in new_topology:
            return i

    raise RuntimeError("Could not find unused connected component label.")


def remove_false_negatives_from_edges(eval_item):
    fn_idxs = eval_item.false_negative_gts
    topology = eval_item.topology

    new_topology = {}
    for cc_label, edge in topology.items():
        seperated_connections = []
        for fn_idx in fn_idxs:
            if len(edge) <= 1 or fn_idx not in edge:
                continue

            connection = edge.pop(fn_idx)
            seperated_connections.append((fn_idx, connection))

        new_topology[cc_label] = edge
        for fn_idx, connection in seperated_connections:
            new_cc_label = find_unused_connected_component_label(topology, new_topology)
            new_topology[new_cc_label] = {fn_idx: connection}

    eval_item.topology = new_topology

    return eval_item


def insert_missing_edges(eval_item):
    gt_bboxes = eval_item.gt_bboxes
    topology = eval_item.topology

    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        connection_l_or_t = None
        connection_r_or_b = None
        # count the occurences of the gt_edges
        for edge in topology.values():
            connection = edge.get(gt_idx, None)
            if connection is None:
                continue

            if (
                connection.orientation == ORIENTATION.LEFT.value
                or connection.orientation == ORIENTATION.TOP.value
            ):
                connection_l_or_t = connection
            elif (
                connection.orientation == ORIENTATION.RIGHT.value
                or connection.orientation == ORIENTATION.BOTTOM.value
            ):
                connection_r_or_b = connection

            if connection_l_or_t is not None and connection_r_or_b is not None:
                break

        if connection_l_or_t is None:
            cc_idx = find_unused_connected_component_label(topology)
            connection = BBoxConnection(None)
            connection.orientation = ORIENTATION.LEFT.value
            topology[cc_idx] = {gt_idx: connection}

        if connection_r_or_b is None:
            cc_idx = find_unused_connected_component_label(topology)
            connection = BBoxConnection(None)
            connection.orientation = ORIENTATION.RIGHT.value
            topology[cc_idx] = {gt_idx: connection}

    return eval_item


def topology(eval_item, threshold):
    img = eval_item.img
    segmentation = eval_item.segmentation
    pred_bboxes = eval_item.pred_bboxes

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
    # utils.show(bin_img)

    segmentation[segmentation > 0] = 1
    segmentation = segmentation * bin_img
    # segmentation = cv.dilate(segmentation, kernel, iterations=2)
    # DEBUG
    # utils.show(segmentation)

    # fill FNs from yolo with ground truth dummys
    eval_item = add_false_negatives(eval_item, threshold)
    # build topology
    postprocessor = Postprocessor(eval_item.pred_bboxes, segmentation)
    topology = postprocessor.make_topology()
    eval_item = eval_item.from_postprocess(topology)

    # convert all TPs produced by dummies to FNs
    if eval_item.false_negative_gts:
        eval_item = remove_false_negatives_from_edges(eval_item)

    # insert missing edges; edges are missing when there was no intersection of the
    # wire with the bbox, we just fake it to perform the evaluation correctly
    eval_item = insert_missing_edges(eval_item)

    utils.Topology.print_dict(eval_item.topology)

    return eval_item


def _perform(img_path, built_lt=False, eval_=True, eval_thresh=0.5):
    # img_path = config.valid_dir / img_name
    # img_path = config.train_dir / img_name
    # print(img_path)

    img = imread(img_path)
    img = utils.resize_max_axis(img, 1000)
    # DEBUG
    # utils.show(img)

    # DEBUG
    # yolo = init_yolo()
    # yolo.inference(str(img_path))
    # yolo.unload()

    # prediction_params = {"iou_threshold": 0.3, "score_threshold": 0.25}
    # yolo_prediction = yolo.predict(img)
    # print("yolo predicted")
    # yolo.unload()
    yolo_prediction = run_prediction(img_path)

    # unet predictions
    unet = init_unet()
    output = unet.predict(img)
    unet.unload()

    segmentation = cv.resize(output, img.shape[:2][::-1])
    # DEBUG
    # utils.show(segmentation[..., np.newaxis], img)

    bboxes = [YoloBBox(img.shape[:2]).from_prediction(p) for p in yolo_prediction]
    # DEBUG
    classes = utils.load_yolo_classes(config.yolo.classes)
    for idx, bbox in enumerate(bboxes):
        print(idx, classes[bbox.label])

    abs_bboxes = [bbox.abs for bbox in bboxes]

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
    # utils.show(bin_img)

    segmentation[segmentation > 0] = 1

    segmentation = segmentation * bin_img
    # DEBUG
    # utils.show(segmentation)

    # segmentation = cv.dilate(segmentation, kernel, iterations=2)

    postprocessor = Postprocessor(bboxes, segmentation)
    topology = postprocessor.make_topology()
    # print(topology)

    if built_lt:
        # TODO this sucks
        # smallest distance between bounding boxes as a grid normalizer
        min_dist = get_min_dist_between_bounding_boxes(abs_bboxes)
        grid_size = get_min_bbox_coord(abs_bboxes) / 3

        ltbuilder = LTBuilderAdapter(config.yolo.classes, grid_size)
        ltbuilder.make_ltcomponents(bboxes)
        ltbuilder.make_wires(
            topology, postprocessor.connected_components, segmentation, img
        )

        writer = LTWriter()
        writer.write(lt_file, ltbuilder.ltcomponents)

    return yolo_prediction, topology


if __name__ == "__main__":
    for img_name in [
        # valid
        # "07_00.png",
        # "07_05.png",
        # "07_06.png",
        # "07_07.png",
        # "08_01.png",
        # "08_02.png",
        # "08_05.png",
        # "08_07.png",
        # "08_08.png",
        # "08_09.png",
        # train
        # "00_11.jpg",
        # "00_19.jpg",
        # "08_08.png",
        # "00_13.jpg",
        # "00_20.jpg",
    ]:
        img_path = config.valid_dir / img_name
        perform(img_path)

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
