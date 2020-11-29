import cv2 as cv
import numpy as np
import os
from numba import njit
from tabulate import tabulate
import math


WINDOW_NAME = "img"


def show(*imgs, size=1000, max_axis=True):
    for i in range(len(imgs)):
        cv.namedWindow(str(i))
        cv.moveWindow(str(i), 100, 100)

    imgs = list(imgs)
    for i, img in enumerate(imgs):
        if max_axis:
            imgs[i] = resize_max_axis(img, size)
        else:
            imgs[i] = resize(img, width=size)

    for i, img in enumerate(imgs):
        cv.imshow(str(i), img)

    while not ord("q") == cv.waitKey(200):
        pass
    cv.destroyAllWindows()


def resize(img, width: int = None, height: int = None, interpolation=cv.INTER_AREA):
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


def resize_max_axis(img, size):
    h, w = img.shape[:2]
    if h > w:
        return resize(img, height=size)
    else:
        return resize(img, width=size)


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


def label_file_from_img(img_file):
    name, ext = os.path.splitext(img_file)
    return f"{name}.txt"


def has_mask(mask_dir, img_file):
    name, ext = os.path.splitext(img_file)
    mask_name = f"{name}_fg_mask{ext}"
    return mask_name in os.listdir(mask_dir)


def img_from_mask(dir_, mask):
    name, _ = os.path.splitext(mask)
    name = name.replace("_fg_mask", "")

    img_names = [name for name in os.listdir(dir_) if ".txt" not in name]

    for img_name in img_names:
        if name in img_name:
            return dir_ / img_name


def merged_name(img_name, bg_name):
    img_name, ext = os.path.splitext(img_name)
    bg_name, _ = os.path.splitext(bg_name)
    return f"{img_name}_{bg_name}{ext}"


class YoloBBox:
    def __init__(self, img_dim):
        self.img_dim = img_dim

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

    def abs(self):
        ih, iw = self.img_dim

        y_abs, x_abs = self.y * ih, self.x * iw
        h_abs, w_abs = self.h * ih, self.w * iw

        y1, x1 = int(y_abs - 0.5 * h_abs), int(x_abs - 0.5 * w_abs)
        y2, x2 = int(y_abs + 0.5 * h_abs), int(x_abs + 0.5 * w_abs)

        return x1, y1, x2, y2

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


class Metrics:
    def __init__(self, classes, label_dir, iou_thresh=0.5):
        self.classes = classes
        self.iou_thresh = iou_thresh
        self.label_dir = label_dir

        self._labels = [self.classes[cls_idx] for cls_idx in range(len(self.classes))]

        # ground truth on y axis, prediction on x axis
        self._confusion = np.zeros((len(classes), len(classes)))
        self._false_positives = np.zeros(len(classes))
        self._false_negatives = np.zeros(len(classes))

        self._metric_mapping = {
            "f1": self.f1,
            "recall": self.recall,
            "precision": self.precision,
        }

    def calculate(self, ground_truth, predictions, img_dim):
        gt_bboxes = [YoloBBox(img_dim).from_ground_truth(gt) for gt in ground_truth]
        pred_bboxes = [YoloBBox(img_dim).from_prediction(pred) for pred in predictions]

        used_gt = set()
        used_pred = set()

        error_wrong_label = []

        # first find pairs of matching iou
        for gt in gt_bboxes:
            for pred in pred_bboxes:
                iou = calc_iou(gt.abs(), pred.abs())
                if iou > self.iou_thresh:
                    self._confusion[gt.label][pred.label] += 1
                    used_gt.add(gt)
                    used_pred.add(pred)

                    if gt.label != pred.label:
                        error_wrong_label.append(pred)

        # left bboxes are those, either not appearing in pred, or not appearing in the
        # ground truth
        unmatched_gt = set(gt_bboxes) ^ used_gt
        for um in unmatched_gt:
            self._false_negatives[um.label] += 1

        unmatched_pred = set(pred_bboxes) ^ used_pred
        for um in unmatched_pred:
            self._false_positives[um.label] += 1

        return error_wrong_label, unmatched_gt, unmatched_pred

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

    def recall(self, show=True):
        # TP / (TP + FN + unmatched_FN)

        tp = np.diagonal(self._confusion)
        tpfn = self._confusion.sum(axis=1) + self._false_negatives

        recall = tp / tpfn

        pretty = [(label, rc) for label, rc in zip(self._labels, recall)]
        pretty += [("Overall", recall.sum() / len(self._labels))]

        if show:
            print("------------------------------------------------")
            print("RECALL")
            print(tabulate(pretty))

        return np.array([val for _, val in pretty])

    def precision(self, show=True):
        # TP / (TP + FP + unmatched_FP)

        tp = np.diagonal(self._confusion)
        tpfp = self._confusion.sum(axis=0) + self._false_positives

        precision = tp / tpfp

        pretty = [(label, pr) for label, pr in zip(self._labels, precision)]
        pretty += [("Overall", precision.sum() / len(self._labels))]

        if show:
            print("------------------------------------------------")
            print("PRECISION")
            print(tabulate(pretty))

        return np.array([val for _, val in pretty])

    def f1(self, show=True):
        # 2 * precision * recall / (precision + recall)

        precision = self.precision(show=False)
        recall = self.recall(show=False)

        f1 = 2 * precision * recall / (precision + recall)

        pretty = [(label, f1c) for label, f1c in zip(self._labels + ["Overall"], f1)]

        if show:
            print("------------------------------------------------")
            print("F1")
            print(tabulate(pretty))

        return np.array([val for _, val in pretty])

    def label_stats(self):
        label_counter = [0 for _ in range(len(self.classes))]
        for file_ in os.listdir(self.label_dir):
            if not is_img(file_):
                continue

            label_file = label_file_from_img(self.label_dir / file_)
            labels = load_ground_truth(label_file)

            for label, *_ in labels:
                label_counter[label] += 1

        return np.array(label_counter)

    def perform(self, metrics, show=True, precision=5):
        pretty = [["Label", "NLabels"] + metrics]

        # combine all metrics, each col is one metric
        overall = None
        for metric in metrics:
            calced = self._metric_mapping[metric](show=False)
            calced = calced.reshape((len(calced), 1))
            if overall is None:
                overall = calced
            else:
                overall = np.append(overall, calced, axis=1)

        float_format = "{{:.{}f}}".format(precision)

        label_stats = self.label_stats()
        label_stats = list(label_stats) + [label_stats.sum()]

        for label, n_labels, row in zip(
            self._labels + ["Overall"], label_stats, overall
        ):
            formatted_nums = [float_format.format(val) for val in row]
            pretty_row = [label, n_labels] + formatted_nums
            pretty.append(pretty_row)

        if show:
            print(tabulate(pretty))

        return pretty
