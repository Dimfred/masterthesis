import cv2 as cv
import numpy as np
import sys
import os
import math
from collections import deque
import mean_average_precision
import albumentations as AA
import random
from concurrent.futures import ThreadPoolExecutor

from ensemble_boxes import weighted_boxes_fusion

import numba as nb
import numba.typed as nt
import numba.core.types as nct

from pathlib import Path
from cached_property import cached_property
from tabulate import tabulate
import time

try:
    from numba import njit
except:

    def njit(f):
        def decorator(*args, **kwargs):
            return f(*args, **kwargs)

        return decorator


WINDOW_NAME = "img"


def show(*imgs, size=1000, max_axis=True):
    for i in range(len(imgs)):
        cv.namedWindow(str(i))
        cv.moveWindow(str(i), 0, 0)

    imgs = list(imgs)
    for i, img in enumerate(imgs):
        if max_axis:
            # imgs[i] = resize_max_axis(img, size)
            pass
        else:
            imgs[i] = resize(img, width=size)

    for i, img in enumerate(imgs):
        cv.imshow(str(i), img)

    while not ord("q") == cv.waitKey(200):
        pass
    cv.destroyAllWindows()


def show_bboxes(img, bboxes, orig=None, type_="gt", classes=None, others=[], gt=None):
    # type = gt | class_to_front | pred
    if type_ == "class_to_front":
        bboxes = [list(bbox) for bbox in bboxes]
        bboxes = A.class_to_front(bboxes)
        bboxes = [YoloBBox(img.shape).from_ground_truth(bbox) for bbox in bboxes]
    elif type_ == "pred":
        bboxes = [YoloBBox(img.shape).from_prediction(bbox) for bbox in bboxes]
    elif type_ == "gt":
        bboxes = [YoloBBox(img.shape).from_ground_truth(bbox) for bbox in bboxes]
    elif type_ == "utils":
        pass
    else:
        raise ValueError(f"Unknown bbox type '{type_}' in show_bboxes.")

    cimg = img.copy()
    if cimg.shape[2] == 1:
        cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.abs
        cv.rectangle(cimg, (x1, y1), (x2, y2), (0, 0, 255), 1)
        if classes is not None:
            cls_name = classes[bbox.label]
            cv.putText(
                cimg,
                cls_name,
                (x1 - 5, y1 - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    if gt is not None:
        gt = [YoloBBox(img.shape).from_ground_truth(bbox) for bbox in gt]
        for bbox in gt:
            x1, y1, x2, y2 = bbox.abs
            cv.rectangle(cimg, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if orig is None:
        show(cimg, *others)
    else:
        show(cimg, orig, *others)


def resize(img, width: int = None, height: int = None, interpolation=cv.INTER_CUBIC):
    h, w = img.shape[:2]

    if width is None and height is None:
        raise ValueError("Specify either width or height.")

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    len_shape_before = len(img.shape)
    img = cv.resize(img, dim, interpolation=interpolation)

    if len(img.shape) != len_shape_before:
        img = np.expand_dims(img, axis=2)

    return img


def pad_equal(img, size):
    img = np.squeeze(img)
    img_h, img_w = img.shape[:2]
    assert img_h == size or img_w == size

    if img_h < img_w:
        pad_size = size - img_h
        top = pad_size // 2
        bot = pad_size - top

        y_slice = slice(top, img_h + top)
        x_slice = slice(0, img_w)

        pad_top = np.zeros((top, img_w))
        pad_bot = np.zeros((bot, img_w))

        img = np.append(pad_top, img, axis=0)
        img = np.append(img, pad_bot, axis=0)
    else:
        pad_size = size - img_w
        left = pad_size // 2
        right = pad_size - left

        y_slice = slice(0, img_h)
        x_slice = slice(left, img_w + left)

        pad_left = np.zeros((img_h, left))
        pad_right = np.zeros((img_h, right))

        img = np.append(pad_left, img, axis=1)
        img = np.append(img, pad_right, axis=1)

    return img, y_slice, x_slice


def resize_max_axis(img, size):
    h, w = img.shape[:2]
    if h > w:
        if h == size:
            return img

        return resize(img, height=size)
    else:
        if w == size:
            return img

        return resize(img, width=size)


import socket

_is_me = socket.gethostname() == "dimfred-schlap"


def isme():
    return _is_me


def uniquecolors(n):
    from unique_color import unique_color as uc

    return uc.unique_color_rgb()[:n]


def is_img(path: str):
    return ".jpg" in path or ".png" in path


def color(s: str, color: int):
    CSI = "\x1B["
    return f"{CSI}31;{color}m{s}{CSI}0m"


def red(s: str):
    return color(s, 31)


def green(s: str):
    return color(s, 32)


def white(s: str):
    return color(s, 37)


def angle(p1, p2):
    v = math.atan2(*(p2 - p1))
    angle = v * (180.0 / math.pi)

    # if angle < 0:
    #     angle += 360

    return angle


# @njit
def calc_iou(b1, b2):
    # b1: x1, y1, x2, y2
    # b2: x1, y1, x2, y2

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    b1Area = abs((b1[2] - b1[0]) * (b1[3] - b1[1]))
    b2Area = abs((b2[2] - b2[0]) * (b2[3] - b2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(b1Area + b2Area - interArea)

    # return the intersection over union value
    return iou


# TODO rename load_yolo_labels
def load_ground_truth(file_):
    with open(file_, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    ground_truth = [line.split(" ") for line in lines]
    ground_truth = [
        (int(label), float(x), float(y), float(w), float(h))
        for label, x, y, w, h in ground_truth
    ]
    return ground_truth


def load_yolo_classes(path):
    with open(path, "r") as f:
        lines = f.readlines()

    return [line.strip() for line in lines]


def list_imgs(path: Path):
    # TODO unified pattern
    jpgs = path.glob("**/*.jpg")
    pngs = path.glob("**/*.png")

    img_paths = list(jpgs)
    img_paths.extend(pngs)

    by_path = lambda path: str(path)
    return sorted(img_paths, key=by_path)


def has_mask(mask_dir, img_file):
    name, ext = os.path.splitext(img_file)
    mask_name = f"{name}_fg_mask{ext}"
    return mask_name in os.listdir(mask_dir)


def segmentation_label_from_img(img_file: Path) -> Path:
    # e.g. img_file = /a/b/img.jpg
    # /a/b / img.npy"
    label_path = img_file.parent / f"{img_file.stem}.npy"
    return label_path


def img_from_fg(img_path: Path, fg_path: Path) -> Path:
    name = fg_path.name
    name = str(name).replace("_fg_mask", "")
    return img_path / name


def merged_name(img_path, bg_path):
    return f"{img_path.stem}_{bg_path.stem}{img_path.suffix}"


def pairwise(iterable, offset=1):
    return zip(iterable[:-offset], iterable[offset:])


def has_annotation(path):
    return "_a" in str(path)


def hough_inter(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])

    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


class YoloBBox:
    def __init__(self, img_dim, x=None, y=None, w=None, h=None, label=None):
        self.img_dim = img_dim[:2]
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.label = label

    def from_prediction(self, prediction):
        x, y, w, h, label, confidence = prediction
        self.label = int(label)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.confidence = float(confidence)

        return self

    def from_ground_truth(self, ground_truth):
        label, x, y, w, h = ground_truth
        self.label = int(label)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)

        return self

    def from_abs(self, x1, y1, x2, y2, label):
        self.label = int(label)
        ih, iw = self.img_dim

        h, w = (y2 - y1), (x2 - x1)
        ym = y1 + 0.5 * h
        xm = x1 + 0.5 * w

        self.x = xm / iw
        self.y = ym / ih
        self.w = w / iw
        self.h = h / ih

        return self

    @cached_property
    def abs(self):
        ih, iw = self.img_dim

        y_abs, x_abs = self.y * ih, self.x * iw
        h_abs, w_abs = self.h * ih, self.w * iw

        y1, x1 = int(y_abs - 0.5 * h_abs), int(x_abs - 0.5 * w_abs)
        y2, x2 = int(y_abs + 0.5 * h_abs), int(x_abs + 0.5 * w_abs)

        return x1, y1, x2, y2

    @cached_property
    def abs_mid(self):
        ih, iw = self.img_dim
        xm, ym = self.x * iw, self.y * ih

        return xm, ym

    @cached_property
    def abs_dim(self):
        x1, y1, x2, y2 = self.abs
        h, w = (y2 - y1), (x2 - x1)

        return w, h

    def yolo(self):
        return np.array((self.x, self.y, self.w, self.h, self.label))

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return (self.label, self.x, self.y, self.w, self.h)

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            raise ValueError("Wrong type used for equal.")

        return hash(self) == hash(other)


class AlbumentationsBBox:
    def __init__(self, img_dim):
        self.img_dim = img_dim[:2]

    def from_rel(self, bbox):
        x1, y1, x2, y2, label = bbox
        self.label = int(label)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

        return self

    def from_abs(self, x1, y1, x2, y2, label):
        self.label = int(label)
        ih, iw = self.img_dim

        self.x1 = x1 / iw
        self.y1 = y1 / ih
        self.x2 = x2 / iw
        self.y2 = y2 / ih

        return self

    @cached_property
    def abs(self):
        ih, iw = self.img_dim

        y1, x1 = self.y1 * ih, self.x1 * iw
        y2, x2 = self.y2 * ih, self.x2 * iw

        return int(x1), int(y1), int(x2), int(y2)

    @cached_property
    def abs_mid(self):
        x1, y1, x2, y2 = self.abs
        xm = x1 + (x2 - x1) // 2
        ym = y1 + (y2 - y1) // 2

        return xm, ym

    @cached_property
    def abs_dim(self):
        x1, y1, x2, y2 = self.abs
        h, w = (y2 - y1), (x2 - x1)

        return w, h

    def yolo(self):
        return np.array((self.x, self.y, self.w, self.h, self.label))

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return (self.label, self.x, self.y, self.w, self.h)

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            raise ValueError("Wrong type used for equal.")

        return hash(self) == hash(other)


class MeanAveragePrecision:
    def __init__(
        self,
        class_names,
        img_shape,
        iou_threshs=[0.5, 0.7],
        policy="greedy",
        same_img_shape=True,
    ):
        self.same_img_shape = same_img_shape
        self.img_shape = img_shape

        self.iou_threshs = iou_threshs
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.policy = policy

        self._map = mean_average_precision.MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=self.n_classes
        )

    def add(self, pred_batch, gt_batch, inverted_gt=False, absolute=True):
        if absolute:
            for batch_idx, (pred, gt) in enumerate(zip(pred_batch, gt_batch)):
                batch_idx = -1 if self.same_img_shape else batch_idx

                pred = self._convert_prediction(pred, batch_idx)
                if inverted_gt:
                    gt = self._convert_inverted_ground_truth(gt, batch_idx)
                else:
                    gt = self._convert_ground_truth(gt, batch_idx)

                self._map.add(pred, gt)

        else:
            for pred, gt in zip(pred_batch, gt_batch):
                pred = np.array([self.pred_to_rel(p) for p in pred])
                gt = np.array([self.gt_to_rel(g) for g in gt])

                self._map.add(pred, gt)

    def pred_to_rel(self, pred):
        x, y, w, h, label, conf = pred

        w_half, h_half = w / 2, h / 2
        x1, x2 = x - w_half, x + w_half
        y1, y2 = y - h_half, y + h_half

        return (x1, y1, x2, y2, label, conf)

    def gt_to_rel(self, gt):
        label, x, y, w, h = gt

        w_half, h_half = w / 2, h / 2
        x1, x2 = x - w_half, x + w_half
        y1, y2 = y - h_half, y + h_half

        # ..., difficult, crowd
        return (x1, y1, x2, y2, label, 0, 0)

    def compute(self, show=True):
        if show:
            start = time.perf_counter()

        results = []
        for idx, iou_thresh in enumerate(self.iou_threshs):
            if isinstance(iou_thresh, list) or isinstance(iou_thresh, tuple):
                threshs = np.arange(*iou_thresh)
                # print(threshs)

                results_ = [
                    self._map.value(iou_thresholds=iou_thresh_, mpolicy=self.policy)
                    for iou_thresh_ in threshs
                ]

                mAP = np.array([res_["mAP"] for res_ in results_]).mean()
                # cls_idx, {"ap": val, bla}
                aps_for_all_ious = [
                    sorted(res_[iou_thresh_].items())
                    for res_, iou_thresh_ in zip(results_, threshs)
                ]
                # n_classes = len(aps_for_all_ious)
                aps_for_all_ious = [
                    np.vstack([res_dict["ap"] for cls_idx, res_dict in aps_at_iou])
                    for aps_at_iou in aps_for_all_ious
                ]
                aps_for_all_ious = np.hstack(aps_for_all_ious)
                mean_aps = aps_for_all_ious.mean(axis=1)

                combined_results = {
                    iou_thresh: {
                        cls_idx: {"ap": ap} for cls_idx, ap in enumerate(mean_aps)
                    }
                }
                combined_results["mAP"] = mAP
                results.append(combined_results)
            else:
                results.append(
                    self._map.value(iou_thresholds=iou_thresh, mpolicy=self.policy)
                )

        if show:
            end = time.perf_counter()
            print("mAP computation took: ", "{:.3f}s".format(end - start))

        return results

    def reset(self):
        self._map.reset()
        self.img_shape = []

    def get_maps(self, results):
        return [
            (iou_thresh, res["mAP"])
            for res, iou_thresh in zip(results, self.iou_threshs)
        ]

    def prettify(self, results):
        ffloat = lambda f: "{:.6f}".format(f)

        map_header = ["mAP@{}".format(iou_thresh) for iou_thresh in self.iou_threshs]
        ap_header = ["AP@{}".format(iou_thresh) for iou_thresh in self.iou_threshs]

        pretty = [["Class", *ap_header, " ", *map_header]]
        # " ", "AP@0", ..., "AP@n", " ", "mAP@0", ..., "mAP@n"
        pretty_row = [
            [
                " ",
                *[" " for _ in range(len(ap_header))],
                " ",
                *[ffloat(res["mAP"]) for res in results],
            ]
        ]
        pretty += pretty_row

        all_aps = [
            sorted(res[iou_thresh].items())
            for res, iou_thresh in zip(results, self.iou_threshs)
        ]
        pretty_rows = [[cls_name] for _, cls_name in sorted(self.class_names.items())]

        for aps in all_aps:
            for cls_idx, cls_data in sorted(aps):
                cls_name = self.class_names[cls_idx]
                cls_ap = cls_data["ap"]

                pretty_rows[cls_idx].append(ffloat(cls_ap))

        pretty += pretty_rows

        return tabulate(pretty)

    def _convert_prediction(self, pred, batch_idx=-1):
        if batch_idx == -1:
            pred = [YoloBBox(self.img_shape).from_prediction(pred) for pred in pred]
        else:
            pred = [
                YoloBBox(self.img_shape[batch_idx]).from_prediction(pred)
                for pred in pred
            ]

        pred = np.vstack([[*bbox.abs, bbox.label, bbox.confidence] for bbox in pred])
        return pred

    def _convert_ground_truth(self, gt, batch_idx=-1):
        difficult, crowd = 0, 0

        if batch_idx == -1:
            gt = [YoloBBox(self.img_shape).from_ground_truth(gt) for gt in gt]
        else:
            gt = [
                YoloBBox(self.img_shape[batch_idx]).from_ground_truth(gt) for gt in gt
            ]
        gt = np.vstack([[*bbox.abs, bbox.label, difficult, crowd] for bbox in gt])
        return gt

    def _convert_inverted_ground_truth(self, gt, batch_idx=-1):
        gt = A.class_to_front(gt)
        return self._convert_ground_truth(gt, batch_idx=batch_idx)


class Metrics:
    def __init__(self, classes, label_dir, iou_thresh=0.5):
        self.classes = classes
        self.label_dir = label_dir
        self.iou_thresh = iou_thresh

        self._labels = [self.classes[cls_idx] for cls_idx in range(len(self.classes))]

        # ground truth on y axis, prediction on x axis
        self._confusion = np.zeros((len(classes), len(classes)))
        self._tps = np.zeros(len(classes))
        self._fns = np.zeros(len(classes))
        self._fps = np.zeros(len(classes))

        self._metric_mapping = {
            "f1": self.f1,
            "recall": self.recall,
            "precision": self.precision,
        }

    def calculate(self, ground_truth, predictions, img_dim):
        gt_bboxes = [YoloBBox(img_dim).from_ground_truth(gt) for gt in ground_truth]
        pred_bboxes = [YoloBBox(img_dim).from_prediction(pred) for pred in predictions]

        gt_to_preds = {gt_idx: [] for gt_idx, _ in enumerate(gt_bboxes)}

        # initial matching round
        for gt_idx, gt in enumerate(gt_bboxes):
            for pred_idx, pred in enumerate(pred_bboxes):
                iou = calc_iou(gt.abs, pred.abs)
                if iou > self.iou_thresh:
                    gt_to_preds[gt_idx].append((pred_idx, iou))

        # check that every prediction was matched only once to a gt
        all_matched_preds = []
        for matched_preds in gt_to_preds.values():
            all_matched_preds.extend((pred_idx for pred_idx, _ in matched_preds))

        # check that no prediction was matched twice to different gts
        if len(set(all_matched_preds)) != len(all_matched_preds):
            raise ValueError("You fucked!")

        # some predictions can't be matched to a gt hence they are false positives
        all_pred_idxs = set(range(len(pred_bboxes)))
        unmatched_preds = set(all_matched_preds) ^ all_pred_idxs
        if unmatched_preds:
            for unmatched_pred_idx in unmatched_preds:
                pred_bbox = pred_bboxes[unmatched_pred_idx]
                self._fps[pred_bbox.label] += 1

        # get all errors n stuff
        for gt_idx, matched_preds in gt_to_preds.items():
            gt_bbox = gt_bboxes[gt_idx]

            if len(matched_preds) == 0:
                self._fns[gt_bbox.label] += 1

            elif len(matched_preds) == 1:
                pred_idx, _ = matched_preds[0]
                pred_bbox = pred_bboxes[pred_idx]

                if gt_bbox.label == pred_bbox.label:
                    self._tps[gt_bbox.label] += 1
                else:
                    self._fns[gt_bbox.label] += 1
                    self._fps[pred_bbox.label] += 1

            elif len(matched_preds) > 1:
                tp_found = False
                for pred_idx, _ in matched_preds:
                    pred_bbox = pred_bboxes[pred_idx]

                    if pred_bbox.label == gt_bbox.label:
                        if not tp_found:
                            self._tps[gt_bbox.label] += 1
                            tp_found = True
                        else:
                            self._fps[gt_bbox.label] += 1

                    else:
                        self._fps[pred_bbox.label] += 1

            else:
                raise ValueError("LoL WTF!")

        # used_gt = set()
        # used_pred = set()

        # error_wrong_label = []

        # # first find pairs of matching iou
        # for gt in gt_bboxes:
        #     for pred in pred_bboxes:
        #         iou = calc_iou(gt.abs, pred.abs)
        #         if iou > self.iou_thresh:
        #             self._confusion[gt.label][pred.label] += 1
        #             used_gt.add(gt)
        #             used_pred.add(pred)

        #             if gt.label != pred.label:
        #                 error_wrong_label.append(pred)

        # # left bboxes are those, either not appearing in pred, or not appearing in the
        # # ground truth
        # unmatched_gt = set(gt_bboxes) ^ used_gt
        # for um in unmatched_gt:
        #     self._false_negatives[um.label] += 1

        # unmatched_pred = set(pred_bboxes) ^ used_pred
        # for um in unmatched_pred:
        #     self._false_positives[um.label] += 1

        # return error_wrong_label, unmatched_gt, unmatched_pred

    def confusion(self):
        pretty = [["GT/PR"] + list(range(len(self._labels)))]
        for idx, (label, row) in enumerate(zip(self._labels, self._confusion)):
            pretty_row = [f"{idx}: {label}"] + [
                green(str(int(n))) if int(n) != 0 else white(str(int(n))) for n in row
            ]
            pretty.append(pretty_row)

        print(tabulate(pretty))

        fns = self._false_negatives.reshape(-1, 1)
        fps = self._false_positives.reshape(-1, 1)
        combined = np.append(fns, fps, axis=1).astype(np.uint16)

        pretty = [["Label", "FPs", "FNs"]]
        for label, row in zip(self._labels, combined):
            pretty_row = [label] + list(row)
            pretty.append(pretty_row)

        print("------------------------------------------------")
        print("Unmatched FP & FN")
        print(tabulate(pretty))

    def recall(self):
        r = self._tps / (self._tps + self._fns)
        overall = r.mean()
        return np.concatenate((r, [overall]))

    def precision(self):
        p = self._tps / (self._tps + self._fps)
        overall = p.mean()
        return np.concatenate((p, [overall]))

    def f1(self):
        # 2 * precision * recall / (precision + recall)
        p = self.precision()
        r = self.recall()

        return 2 * p * r / (p + r)

    def label_stats(self):
        from .augment import YoloAugmentator

        label_counter = [0 for _ in range(len(self.classes))]

        img_label_paths = YoloAugmentator.fileloader(self.label_dir)

        for img_path, label_path in img_label_paths:
            labels = load_ground_truth(label_path)

            for label, *_ in labels:
                label_counter[label] += 1

        return np.array(label_counter)

    def perform(self, metrics):
        pretty = [["Label", "NLabels"] + list(metrics)]

        # combine all metrics, each col is one metric
        overall = None
        for metric in metrics:
            calced = self._metric_mapping[metric]()
            calced = calced.reshape((len(calced), 1))
            if overall is None:
                overall = calced
            else:
                overall = np.append(overall, calced, axis=1)

        return overall

    def prettify(self, metrics, results, iou=None):
        ffloat = lambda x: f"{x:.5f}"

        label_stats = self.label_stats()
        label_stats = list(label_stats) + [label_stats.sum()]

        pretty = [["Label", "Num", *metrics]]
        if iou is not None:
            pretty += [[f"@{iou}"]]
        for label, n_labels, row in zip(
            self._labels + ["Overall"], label_stats, results
        ):
            pretty_vals = [ffloat(val) for val in row]
            pretty += [[label, n_labels, *pretty_vals]]

        print(tabulate(pretty))


class BFS:
    _yneighbors = [-1, 0, 0, 1]
    _xneighbors = [0, -1, 1, 0]

    def __init__(self, mat, value=255, early_stop=sys.maxsize):
        self.mat = mat
        self.visited = np.zeros_like(mat)
        self.queue = deque()
        self.value = value
        self.early_stop = early_stop

    def is_valid(self, p):
        return self.mat[p] == self.value

    # @njit
    def fit(self, start: tuple, end: tuple):
        xs, ys = start
        self.visited[ys, xs] = True

        # 0 distance

        path = deque()
        self.queue.append([start])
        while self.queue:
            path = self.queue.popleft()

            if len(path) > self.early_stop:
                return None

            p = path[-1]
            if p == end:
                return path

            for yoff, xoff in zip(self._yneighbors, self._xneighbors):
                x, y = p
                neighbor = (y + yoff, x + xoff)

                if self.is_valid(neighbor) and not self.visited[neighbor]:
                    self.visited[neighbor] = True
                    new_path = list(path)
                    new_path.append(neighbor[::-1])
                    self.queue.append(new_path)

        return None


def stopwatchtf(name):
    import tensorflow as tf

    def _stopwatch(f):
        def _deco(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            end = time.perf_counter()
            tf.print(f"{green(name)}::took:", "{:4f}s".format(end - start))

            return ret

        return _deco

    return _stopwatch


def stopwatch(name):
    def _stopwatch(f):
        def _deco(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            end = time.perf_counter()
            print(f"{green(name)}::took:", "{:4f}s".format(end - start))

            return ret

        return _deco

    return _stopwatch


class A:
    @staticmethod
    def class_to_back(bboxes):
        to_back = lambda bbox: list(bbox[1:]) + [bbox[0]]
        return [to_back(bbox) for bbox in bboxes]

    @staticmethod
    def class_to_front(bboxes):
        to_front = lambda bbox: [bbox[-1]] + list(bbox[:-1])
        return [to_front(bbox) for bbox in bboxes]


def seed_all(seed, all_=True):
    import tensorflow as tf
    import torch
    import numpy as np
    import imgaug
    import random

    tf.random.set_seed(seed)
    np.random.seed(seed)
    imgaug.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print(f"Seeding {seed}: tf, torch, np, imgaug, python")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")

    # fmt: on


class Yolo:
    @staticmethod
    def parse_classes(path: Path, result_type=list):
        # result_type: arr | dict
        with open(path) as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        if not lines[-1]:
            lines = lines[:-1]

        if result_type == list:
            return lines
        if result_type == dict:
            return {i: cls_name for i, cls_name in enumerate(lines)}

        raise ValueError(f"Unknown 'result_type': {result_type}")

    def parse_labels(label_path: Path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        labels = []
        for line in lines:
            line = line.strip()
            l, x, y, w, h = line.split(" ")
            labels.append((int(l), float(x), float(y), float(w), float(h)))

        return labels

    @staticmethod
    def load_dataset(path):
        img_paths = list_imgs(path)

        img_label_paths = []
        for img_path in img_paths:
            label_path = Yolo.label_from_img(img_path)
            if label_path.exists():
                img_label_paths.append((img_path, label_path))

        return img_label_paths

    @staticmethod
    def label_from_img(img_path):
        label_path = img_path.parent / f"{img_path.stem}.txt"
        return label_path


class EvalTopology:
    @staticmethod
    def parse(path):
        with open(path, "r") as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        start, end = EvalTopology._get_bounds(lines)
        lines = lines[start:end]
        lines = EvalTopology._clean(lines)

        arr = EvalTopology._as_array(lines)
        arr = np.array(arr)

        return arr

    @staticmethod
    def _get_bounds(lines):
        return lines.index("## START") + 1, lines.index("## END")

    @staticmethod
    def _clean(lines):
        lines = [line for line in lines if line]
        # remove comments
        lines = [line.split("//")[0] for line in lines]
        # first line is just to help me build the topology => remove it
        lines = lines[1:]

        return lines

    @staticmethod
    def _as_array(lines):
        # when we have > 9 components we put two space to still better read the gt
        lines = [line.replace("  ", " ") for line in lines]
        lines = [line.replace(" ", ",") for line in lines]
        lines = ["[{}]".format(line) for line in lines]
        lines = [eval(line) for line in lines]

        return lines


class EvalArrowAndTextMatching:
    @staticmethod
    def parse(path):
        with open(path, "r") as f:
            lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]

        arrow_or_text_idx_to_ecc = [
            (int(arrow_or_text_idx), int(ecc_idx))
            for arrow_or_text_idx, ecc_idx in lines
        ]

        return arrow_or_text_idx_to_ecc


class Topology:
    @staticmethod
    def print_dict(topology):
        print("----------------------------------------------------------")
        print(red("TOPOLOGY:"))
        for i, edge in enumerate(topology.values()):
            pretty = f"{str(i).zfill(2)}: "
            pretty_edge = [
                f"{green(bbox_idx)}:{Orientation.to_str(connection.orientation)}"
                for bbox_idx, connection in edge.items()
            ]
            pretty_edge = ", ".join(pretty_edge)
            pretty += pretty_edge
            print(pretty_edge)


class Orientation:
    @staticmethod
    def to_str(orientation):
        if orientation == 0:
            return "l"
        if orientation == 1:
            return "r"
        if orientation == 2:
            return "t"
        if orientation == 3:
            return "b"


@nb.njit
def project(src, dst, y, x):
    where_src = np.argwhere(src != 255)
    y, x = int(y), int(x)
    for y_src, x_src in where_src:
        # print(y_src, x_src)
        y_dst, x_dst = y_src + y, x_src + x
        # print(y_dst, x_dst)
        dst[y_dst, x_dst, 0] = src[y_src, x_src]

    return dst


class TextProjection(AA.core.transforms_interface.DualTransform):
    def __init__(self, text_idx, ground_idxs, texts, classes, *args, **kwargs):
        super(TextProjection, self).__init__(*args, **kwargs)
        self.classes = classes
        self.text_idx = text_idx
        self.ground_idxs = ground_idxs
        self.texts = texts
        self.annotation_probability = 0.5

        self.scale_margin = 0.1  # => [0.9, 1.1]

    def apply(self, img, new_img, new_bboxes, **params):
        return img

    def apply_to_bbox(self, bbox, new_img, new_bboxes, **params):
        return bbox

    # @stopwatch("main")
    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        bboxes = params["bboxes"]

        if self.has_text(bboxes):
            return {"new_img": None, "new_bboxes": None}

        alb_bboxes = [AlbumentationsBBox(img.shape).from_rel(bbox) for bbox in bboxes]

        new_bboxes, new_texts = [], []
        for bbox in alb_bboxes:
            if self.is_ground(bbox.label):
                continue

            # change to perform an annotation on this bbox
            if np.random.uniform() < self.annotation_probability:
                continue

            orientation = self.get_orientation(bbox.label)
            h_annotation, w_annotation = self.calc_annotation_size(bbox, orientation)
            text = random.choice(self.texts).copy()
            text = resize(text, w_annotation, h_annotation)
            h_annotation, w_annotation = text.shape[:2]

            annotation_start = self.calc_start_coordinates(
                bbox, h_annotation, w_annotation, orientation
            )
            # when we are out of bounds by the calculation
            if annotation_start is None:
                continue

            y_start_bbox, x_start_bbox = annotation_start

            h_text_bbox, w_text_bbox = text.shape[:2]
            h_text_bbox += 4
            w_text_bbox += 4

            y_end_bbox, x_end_bbox = (
                y_start_bbox + h_text_bbox,
                x_start_bbox + w_text_bbox,
            )

            text_bbox = AlbumentationsBBox(img.shape).from_abs(
                x_start_bbox, y_start_bbox, x_end_bbox, y_end_bbox, self.text_idx
            )

            if not self.is_bbox_valid(text_bbox, new_bboxes, alb_bboxes):
                continue
            new_bboxes.append(text_bbox)

            img = project(text, img, *annotation_start)
            bboxes.append(
                (
                    text_bbox.x1,
                    text_bbox.y1,
                    text_bbox.x2,
                    text_bbox.y2,
                    text_bbox.label,
                )
            )

        # DEBUG
        # alb_bboxes = [AlbumentationsBBox(img.shape).from_rel(bbox) for bbox in bboxes]
        # # show_bboxes(img, alb_bboxes, type_="utils")
        # show(img)

        return {"new_img": img, "new_bboxes": bboxes}

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("text_idx",)

    # @stopwatch("is_bbox_valid")
    def is_bbox_valid(self, new_bbox, new_bboxes, present_bboxes):
        for present_bbox in present_bboxes:
            iou = calc_iou(new_bbox.abs, present_bbox.abs)
            if iou > 0.03:
                return False

        for present_bbox in new_bboxes:
            iou = calc_iou(new_bbox.abs, present_bbox.abs)
            if iou > 0.03:
                return False

        return True

    # @stopwatch("get_orientation")
    def get_orientation(self, cls_idx):
        cls_name = self.classes[cls_idx]
        # print(cls_name)
        _, cls_orientation = cls_name.rsplit("_", 1)

        if (
            cls_orientation == "left"
            or cls_orientation == "right"
            or cls_orientation == "hor"
        ):
            return "hor"
        else:
            return "ver"

    # @stopwatch("calc_annotation_size")
    def calc_annotation_size(self, bbox, orientation):
        w_bbox, h_bbox = bbox.abs_dim
        # print("-----------------------------")
        # print("AbsDim", bbox.abs_dim)
        # print("RelDim", bbox.h, bbox.w)
        scaler = np.random.uniform(0.6, 1.0)
        # print("Scaler", scaler)

        if orientation == "hor":
            h_annotation = int(scaler * h_bbox)
            w_annotation = None
        else:
            h_annotation = int(scaler * w_bbox)
            w_annotation = None
        # print("AbsH", h_annotation)

        return h_annotation, w_annotation

    # @stopwatch("calc_start_coords")
    def calc_start_coordinates(self, bbox, h_annotation, w_annotation, orientation):
        ih, iw = bbox.img_dim
        mx_bbox, my_bbox = bbox.abs_mid
        x1, y1, x2, y2 = bbox.abs

        if orientation == "hor":
            x_annotation_start = int(mx_bbox) - (w_annotation // 2)
            if x_annotation_start < 0:
                return None

            # top
            if random.uniform(0, 1) <= 0.5:
                y_annotation_start = y1 - 2 - h_annotation
                if y_annotation_start < 0:
                    return None
            # bot
            else:
                y_annotation_start = y2 + 2
        else:
            y_annotation_start = int(my_bbox) - (h_annotation // 2)
            if y_annotation_start < 0:
                return None

            # left
            if random.uniform(0, 1) <= 0.5:
                x_annotation_start = x1 - 2 - w_annotation
                if x_annotation_start < 0:
                    return None
            # right
            else:
                x_annotation_start = x2 + 2

        # aditionally verify positive img bounds
        x_annotation_end = x_annotation_start + w_annotation
        if x_annotation_end > iw:
            return None

        y_annotation_end = y_annotation_start + h_annotation
        if y_annotation_end > ih:
            return None

        return y_annotation_start, x_annotation_start

    # @stopwatch("has_text")
    def has_text(self, bboxes):
        for _, _, _, _, cls_ in bboxes:
            if int(cls_) == self.text_idx:
                return True

        return False

    # @stopwatch("is_ground")
    def is_ground(self, cls_):
        return cls_ in self.ground_idxs


def load_imgs(dir_, read_type=cv.IMREAD_ANYCOLOR):
    imgs = []

    def read(path):
        img = cv.imread(str(path), read_type)
        imgs.append(img)

    img_paths = list_imgs(dir_)
    with ThreadPoolExecutor(max_workers=32) as executor:
        for path in img_paths:
            executor.submit(read, path)

    return imgs


# - experiments
#   - init_lr
#       - lr1
#           - run1
#               - best_weights
#               - results
#               - tensorboard
#               -
#           - run2
#       - lr2
#   - offline_augs
#   - online_augs
#   - grid


class YoloExperiment:
    def __init__(self, experiment_dir, experiment_name, experiment_param, run):
        self.experiment_dir = experiment_dir
        self.experiment_name = self.experiment_dir / experiment_name
        self.experiment_param = self.experiment_name / experiment_param
        self.run = self.experiment_param / f"run{run}"

        self.weights = self.run / "best.weights"
        self.results = self.run / "results_raw.txt"
        self.tb_train_dir = str(self.run / "train")
        self.tb_valid_dir = str(self.run / "valid")

    def init(self):
        self.create_if_not_exists(self.experiment_dir)
        self.create_if_not_exists(self.experiment_name)
        self.create_if_not_exists(self.experiment_param)
        self.clean_path(self.run)
        self.create_if_not_exists(self.run)

    def create_if_not_exists(self, path):
        if not path.exists():
            path.mkdir()

    def clean_path(self, path):
        if not path.exists():
            return

        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                self.clean_path(child)

        path.rmdir()


class UnetExperiment:
    def __init__(self, experiment_dir, experiment_name, experiment_param, run):
        self.experiment_dir = experiment_dir
        self.experiment_name = self.experiment_dir / experiment_name
        self.experiment_param = self.experiment_name / experiment_param
        self.run = self.experiment_param / f"run{run}"

        self.weights = self.run / "best.pth"
        self.results = self.run / "results_raw.txt"
        self.tb_train_dir = str(self.run / "train")
        self.tb_valid_dir = str(self.run / "valid")

    def init(self):
        self.create_if_not_exists(self.experiment_dir)
        self.create_if_not_exists(self.experiment_name)
        self.create_if_not_exists(self.experiment_param)
        self.clean_path(self.run)
        self.create_if_not_exists(self.run)

    def create_if_not_exists(self, path):
        if not path.exists():
            path.mkdir()

    def clean_path(self, path):
        if not path.exists():
            return

        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                self.clean_path(child)

        path.rmdir()


@nb.njit
def nb_tta_perform(
    prediction, flip_rotation, classes, flip_table, rotation_table, index_by_name
):
    flip, rotation = flip_rotation
    if rotation:
        n_its = int(rotation / 90)
        n_its = 4 - n_its

        for it in range(n_its):
            prediction = [
                nb_tta_calc_rotation(p, classes, rotation_table, index_by_name)
                for p in prediction
            ]

    if flip:
        prediction = [
            nb_tta_calc_flip(p, classes, flip_table, index_by_name) for p in prediction
        ]

    return np.vstack(tuple(prediction))


@nb.njit
def nb_tta_calc_rotation(prediction, classes, rotation_table, index_by_name):
    # calculates +90deg
    x, y, w, h, l, conf = prediction

    cur_cls_name = classes[l]
    new_cls_name = rotation_table[cur_cls_name]

    new_l = index_by_name(new_cls_name, classes)
    new_x = 1 - y
    new_y = x
    new_w = h
    new_h = w

    return (new_x, new_y, new_w, new_h, new_l, conf)


@nb.njit
def nb_tta_calc_flip(prediction, classes, flip_table, index_by_name):
    # flip over y (ab => ba)
    x, y, w, h, l, conf = prediction
    cur_cls_name = classes[l]
    new_cls_name = flip_table[cur_cls_name]

    new_l = index_by_name(new_cls_name)
    new_x = 1 - x
    return (new_x, y, w, h, new_l, conf)


@nb.njit
def nb_tta_index_by_name(name, classes):
    for idx, cls_name in classes.items():
        if cls_name == name:
            return idx

    # raise ValueError("Unknown cls_name:", name)


class YoloTTA:
    def __init__(self, classes, flip_table, rotation_table):
        self.classes = classes
        self.flip_table = flip_table
        self.rotation_table = rotation_table

        # self.classes = nt.Dict.empty(key_type=nct.int64, value_type=nct.string)
        # for label_idx, cls_name in classes.items():
        #     self.classes[label_idx] = cls_name

        # self.flip_table = nt.Dict.empty(key_type=nct.string, value_type=nct.string)
        # for from_, to_ in flip_table.items():
        #     self.flip_table[from_] = to_

        # self.rotation_table = nt.Dict.empty(key_type=nct.string, value_type=nct.string)
        # for from_, to_ in rotation_table.items():
        #     self.rotation_table[from_] = to_

    def perform(self, prediction, flip_rotation):
        # NUMBA
        # return nb_tta_perform(
        #     prediction,
        #     flip_rotation,
        #     self.classes,
        #     self.flip_table,
        #     self.rotation_table,
        #     nb_tta_index_by_name,
        # )

        flip, rotation = flip_rotation
        if rotation:
            n_its = rotation // 90
            n_its = 4 - n_its

            for idx in range(len(prediction)):
                for it in range(n_its):
                    prediction[idx] = self.calc_rotation(prediction[idx])

        if flip:
            for idx in range(len(prediction)):
                prediction[idx] = self.calc_flip(prediction[idx])

        return np.vstack(tuple(prediction))

    def calc_rotation(self, prediction):
        # calculates +90deg
        x, y, w, h, l, conf = prediction

        cur_cls_name = self.classes[l]
        new_cls_name = self.rotation_table[cur_cls_name]

        new_l = self.index_by_name(new_cls_name)
        new_x = 1.0 - y
        new_y = x
        new_w = h
        new_h = w

        return np.array([new_x, new_y, new_w, new_h, new_l, conf])

    def calc_flip(self, prediction):
        # flip over y (ab => ba)
        x, y, w, h, l, conf = prediction
        cur_cls_name = self.classes[l]
        new_cls_name = self.flip_table[cur_cls_name]

        new_l = self.index_by_name(new_cls_name)
        new_x = 1.0 - x
        return np.array([new_x, y, w, h, new_l, conf])

    def augment(self, img, times=8):
        img = img.copy()
        oimg = img.copy()

        img_combs = [(img, (False, 0))]
        if times == 8:
            for rot in (90, 180, 270):
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                img_combs.append((np.expand_dims(img, axis=2), (False, rot)))

            img = cv.flip(oimg, +1)
            img_combs.append((np.expand_dims(img, axis=2), (True, 0)))

            for rot in (90, 180, 270):
                img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                img_combs.append((np.expand_dims(img, axis=2), (True, rot)))
        else:
            img = cv.rotate(img, cv.ROTATE_180)
            img_combs.append((np.expand_dims(img, axis=2), (False, 180)))

            img = cv.flip(oimg, +1)
            img_combs.append((np.expand_dims(img, axis=2), (True, 0)))

            img = cv.rotate(img, cv.ROTATE_180)
            img_combs.append((np.expand_dims(img, axis=2), (True, 180)))

        return img_combs


    def index_by_name(self, name):
        for idx, cls_name in self.classes.items():
            if cls_name == name:
                return idx

        raise ValueError("Unknown cls_name:", name)

    # def combinations(self):
    #     combinations = []
    #     for flip in (False, True):
    #         for rot in (0, 90, 180, 270):
    #             combinations.append((flip, rot))

    #     return combinations


class WBF:
    def __init__(
        self, iou_thresh=0.5, score_thresh=0.25, conf_type="avg", vote_thresh=1
    ):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.conf_type = conf_type
        self.vote_thresh = vote_thresh

    def perform(self, predictions):
        bboxes, scores, labels = self.adapt(predictions)
        bboxes, scores, labels = weighted_boxes_fusion(
            bboxes,
            scores,
            labels,
            iou_thr=self.iou_thresh,
            skip_box_thr=self.score_thresh,
            conf_type=self.conf_type,
            vote_thr=self.vote_thresh,
        )
        unadapted = self.unadapt(bboxes, scores, labels)

        return unadapted

    def adapt(self, predictions):
        # x, y, w, h, l, score => [x1, y1, x2, y2], [labels], [scores]
        bboxes, scores, labels = [], [], []

        for model_prediction in predictions:
            model_bboxes, model_scores, model_labels = [], [], []
            for x, y, w, h, l, score in model_prediction:
                half_w = w / 2
                half_h = h / 2

                x1, x2 = x - half_w, x + half_w
                if x1 < 0.0:
                    x1 = 0.0
                if x2 > 1.0:
                    x2 = 1.0

                y1, y2 = y - half_h, y + half_h
                if y1 < 0.0:
                    y1 = 0.0
                if y2 > 1.0:
                    y2 = 1.0

                model_bboxes.append((x1, y1, x2, y2))
                model_labels.append(l)
                model_scores.append(score)

            bboxes.append(model_bboxes)
            scores.append(model_scores)
            labels.append(model_labels)

        return bboxes, scores, labels

    def unadapt(self, bboxes, scores, labels):
        predictions = []
        for (x1, y1, x2, y2), score, label in zip(bboxes, scores, labels):
            w, h = x2 - x1, y2 - y1
            half_w, half_h = w / 2, h / 2

            x, y = x1 + half_w, y1 + half_h
            pred = np.array([x, y, w, h, label, score])
            predictions.append(pred)

        return predictions


@nb.njit
def nb_calc_iou_yolo(b1, b2):
    # b1: x1, y1, x2, y2
    # b2: x1, y1, x2, y2
    b1 = nb_convert_from_yolo_bbox(b1)
    b2 = nb_convert_from_yolo_bbox(b2)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    b1Area = abs((b1[2] - b1[0]) * (b1[3] - b1[1]))
    b2Area = abs((b2[2] - b2[0]) * (b2[3] - b2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(b1Area + b2Area - interArea)

    # return the intersection over union value
    return iou


@nb.njit
def nb_convert_from_yolo_bbox(bbox):
    x, y, w, h, l, c = bbox

    w_half = w / 2
    h_half = h / 2

    x1, x2 = x - w_half, x + w_half
    y1, y2 = y - h_half, y + h_half

    return x1, y1, x2, y2, l, c


@nb.njit
def nb_convert_to_yolo_bbox(bbox):
    x1, y1, x2, y2, l, c = bbox

    w = x2 - x1
    h = y2 - y1

    w_half = w / 2
    h_half = h / 2

    x = x1 + w_half
    y = y1 + h_half

    return x, y, w, h, l, c
