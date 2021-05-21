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
import math

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


def run_prediction(img_paths, iou_thresh=0.25, conf_thresh=0.3, debug=False):
    print("Run YOLO prediction...")

    import subprocess as sp

    prediction_dir = "predictions"

    sp.check_call(
        [
            "python3.8",
            "-c",
            f"""
import cv2 as cv
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from yolov4.tf import YOLOv4
import utils
from config import config


prediction_dir = Path("{prediction_dir}")

yolo = YOLOv4(tiny=config.yolo.tiny, small=config.yolo.small)
yolo.classes = config.yolo.classes
yolo.input_size = config.yolo.input_size
yolo.channels = config.yolo.channels
yolo.make_model()

# yolo.load_weights(config.yolo.label_weights, weights_type=config.yolo.weights_type)
yolo.load_weights(config.yolo.weights, weights_type=config.yolo.weights_type)

for file in os.listdir(prediction_dir):
    os.unlink(prediction_dir / file)

def imread(path):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    img = np.expand_dims(img, axis=2)
    return img

img_paths = {str(img_paths)}
for img_path in img_paths:
    img = imread(img_path)
    img = utils.resize_max_axis(img, 1000)

    prediction = yolo.predict(
        img, iou_threshold={iou_thresh}, score_threshold={conf_thresh}
    )

    if {debug}:
        utils.show_bboxes(img, prediction, type_="pred")

    if np.any(prediction == 42):
        print("WTF!!: ", prediction)

    prediction_dir = "{prediction_dir}"
    np.save(f"{{prediction_dir}}/{{Path(img_path).stem}}.npy", prediction)
""",
        ],
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )
    import os
    import time

    time.sleep(1)

    prediction_files = sorted(os.listdir(prediction_dir))
    predictions = [np.load(Path(prediction_dir) / pred) for pred in prediction_files]
    print("DONE")

    return predictions


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
        self.unmatched_gt = []

        # [(ecc1, arrow1), ...]
        self.matched_arrows = []
        # [(ecc1, text1), ...]
        self.matched_texts = []

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


def predict(img_paths, iou_thresh, conf_thresh):
    # DEBUG
    # utils.show(img)

    yolo_predictions = run_prediction(
        img_paths, iou_thresh=iou_thresh, conf_thresh=conf_thresh, debug=False
    )

    # unet predictions
    unet = init_unet()

    eval_items = []
    for img_path, yolo_pred in zip(img_paths, yolo_predictions):
        img_path = Path(img_path)

        img = imread(str(img_path))
        img = utils.resize_max_axis(img, 1000)

        pred_bboxes = [YoloBBox(img.shape).from_prediction(p) for p in yolo_pred]

        ground_truth = utils.Yolo.parse_labels(utils.Yolo.label_from_img(img_path))
        gt_bboxes = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]

        output = unet.predict(img)

        segmentation = cv.resize(output, img.shape[:2][::-1])
        # DEBUG
        # utils.show(segmentation[..., np.newaxis], img)

        classes = utils.Yolo.parse_classes(config.yolo.classes)

        eval_item = EvaluationItem().from_predict(
            classes, img, gt_bboxes, pred_bboxes, segmentation
        )
        eval_items.append(eval_item)

    unet.unload()

    return eval_items


def add_false_negatives_and_false_positive(eval_item, threshold):
    pred_bboxes = eval_item.pred_bboxes
    gt_bboxes = eval_item.gt_bboxes

    matches = {idx: [] for idx, _ in enumerate(gt_bboxes)}

    # match every gt against a prediction
    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        for pred_idx, pred_bbox in enumerate(pred_bboxes):
            if utils.calc_iou(gt_bbox.abs, pred_bbox.abs) < threshold:
                continue

            matches[gt_idx].append(pred_idx)

    # safe unmatched gts aka false positives
    unmatched_gt = []
    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        has_gt = False
        for gt_idx, match in matches.items():
            if pred_idx not in match:
                continue
            has_gt = True
            break

        if has_gt:
            continue

        unmatched_gt.append(pred_bbox)

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
            raise ValueError("FALSE POSITIVE WITH GROUND TRUTH NOT YET HANDLED")
            # print("FALSE POSITIVE OCCURED CARE!!!! TRYING TO HANDLE")

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
    eval_item.unmatched_gt = unmatched_gt
    eval_item.pred_bboxes.extend(unmatched_gt)


    # DEBUG
    if unmatched_gt:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Found FPs:")
        for bbox in unmatched_gt:
            cls_name = eval_item.classes[bbox.label]
            print(cls_name)
            cls_name = cls_name.split("_")[0]
            if cls_name != "arrow" and cls_name != "text":
                # raise RuntimeError("UNMATCHED GT WHAT TO DO!?!?!")
                print("TODO NORMALLY RUNTIME ERROR IN add_false_negatives")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

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


class ArrowAndTextMatcher:
    def __init__(self, eval_item):
        self.eval_item = eval_item
        self.classes = eval_item.classes

        self.left_arrow_idx = self.classes.index("arrow_left")
        self.right_arrow_idx = self.classes.index("arrow_right")
        self.top_arrow_idx = self.classes.index("arrow_top")
        self.bot_arrow_idx = self.classes.index("arrow_bot")
        self.text_idx = self.classes.index("text")

        # first seperate them
        self.arrows = {}
        self.texts = {}
        self.eccs = {}

        # TODO ??? can I really do that
        # self.eval_item.pred_bboxes.extend(self.eval_item.unmatched_gt)

        for idx, bbox in enumerate(self.eval_item.pred_bboxes):
            if self.is_arrow(bbox.label):
                self.arrows[idx] = bbox
            elif self.is_text(bbox.label):
                self.texts[idx] = bbox
            else:
                self.eccs[idx] = bbox

    def is_arrow(self, label_idx):
        return (
            label_idx == self.left_arrow_idx
            or label_idx == self.right_arrow_idx
            or label_idx == self.top_arrow_idx
            or label_idx == self.bot_arrow_idx
        )

    def is_text(self, label_idx):
        return label_idx == self.text_idx

    def match(self, distance_algorithm="nearest_neighbor", threshold="TODO"):
        if distance_algorithm == "nearest_neighbor":
            distance_algorithm = self.calc_nearest_neighbor
        else:
            raise ValueError(
                f"Distance Algorithm: '{distance_algorithm}' not supported"
            )

        arrow_distances = distance_algorithm(self.eccs, self.arrows)
        text_distances = distance_algorithm(self.eccs, self.texts)

        self.eval_item.matched_arrows = arrow_distances
        self.eval_item.matched_texts = text_distances

        return self.eval_item

    def calc_nearest_neighbor(self, eccs, to_match):
        def euclidean(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        results = {}
        for to_match_idx, to_match_bbox in to_match.items():
            result_per_match_idx = []
            for ecc_idx, ecc_bbox in eccs.items():
                to_match_mid = to_match_bbox.abs_mid
                ecc_mid = ecc_bbox.abs_mid

                distance = euclidean(to_match_mid, ecc_mid)
                result_per_match_idx.append((distance, ecc_idx))

            shortest_distance_first = lambda x: x[0]
            result_per_match_idx = sorted(
                result_per_match_idx, key=shortest_distance_first
            )

            results[to_match_idx] = result_per_match_idx

        return results


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


def fuse_textboxes(eval_item, fuse_textbox_iou):
    text_label_idx = eval_item.classes.index("text")

    pred_bboxes = eval_item.pred_bboxes

    done = False
    while not done:
        changed = False

        # print("ROUND")
        for idx1 in range(0, len(pred_bboxes) - 1):
            for idx2 in range(idx1 + 1, len(pred_bboxes)):
                # DEBUG
                # print(len(pred_bboxes), idx1, idx2)

                bbox1 = pred_bboxes[idx1]
                bbox2 = pred_bboxes[idx2]

                # when both boxes are text
                if bbox1.label != text_label_idx or bbox2.label != text_label_idx:
                    continue

                # when they have a certain iou then fuse them
                if utils.calc_iou(bbox1.abs, bbox2.abs) < fuse_textbox_iou:
                    continue

                # DEBUG
                # print("Fused Text Before")
                # utils.show_bboxes(
                #     eval_item.img, pred_bboxes, type_="utils", classes=eval_item.classes
                # )

                #
                # first sort them so we can pop first the most right idx
                idx1, idx2 = sorted((idx1, idx2))
                pred_bboxes.pop(idx2)
                pred_bboxes.pop(idx1)

                # now fuse them like in giou, biggest convex box around both

                bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2 = bbox1.abs
                bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2 = bbox2.abs

                new_x1, new_y1 = min(bbox1_x1, bbox2_x1), min(bbox1_y1, bbox2_y1)
                new_x2, new_y2 = max(bbox1_x2, bbox2_x2), max(bbox1_y2, bbox2_y2)

                new_bbox = YoloBBox(bbox1.img_dim).from_abs(
                    new_x1, new_y1, new_x2, new_y2, text_label_idx
                )
                new_bbox.confidence = max(bbox1.confidence, bbox2.confidence)

                pred_bboxes.append(new_bbox)

                # DEBUG
                # print("Fused Text After")
                # utils.show_bboxes(
                #     eval_item.img, pred_bboxes, type_="utils", classes=eval_item.classes
                # )

                changed = True
                break

            if changed:
                break

        if not changed:
            done = True

    eval_item.pred_bboxes = pred_bboxes
    return eval_item


def apply_occlusion_nms(eval_item, occlusion_iou):
    pred_bboxes = eval_item.pred_bboxes
    bboxes_to_remove = []

    for idx1 in range(len(pred_bboxes) - 1):
        for idx2 in range(idx1 + 1, len(pred_bboxes)):
            bbox1 = pred_bboxes[idx1]
            bbox2 = pred_bboxes[idx2]

            # when a certain thresh is reached between both
            if utils.calc_iou(bbox1.abs, bbox2.abs) < occlusion_iou:
                continue

            # remove the one with the lower confidence
            if bbox1.confidence < bbox2.confidence:
                bboxes_to_remove.append(idx1)
            else:
                bboxes_to_remove.append(idx2)

    bboxes_to_remove = sorted(bboxes_to_remove, reverse=True)
    for idx in bboxes_to_remove:
        # DEBUG
        # print("Occlusion NMS Before")
        # utils.show_bboxes(
        #     eval_item.img, pred_bboxes, type_="utils", classes=eval_item.classes
        # )

        pred_bboxes.pop(idx)

        # DEBUG
        # print("Occlusion NMS After")
        # utils.show_bboxes(
        #     eval_item.img, pred_bboxes, type_="utils", classes=eval_item.classes
        # )

    eval_item.pred_bboxes = pred_bboxes
    return eval_item


def topology(eval_item, fn_threshold, fuse_textbox_iou, occlusion_iou, debug=False):
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

    # first fuse text then remove occlusions, because text could also be occulusions
    # but are a special case
    eval_item = fuse_textboxes(eval_item, fuse_textbox_iou)
    eval_item = apply_occlusion_nms(eval_item, occlusion_iou)

    # fill FNs from yolo with ground truth dummys
    eval_item = add_false_negatives_and_false_positive(eval_item, fn_threshold)
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

    if debug:
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
