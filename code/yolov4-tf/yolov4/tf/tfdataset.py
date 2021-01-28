"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
Copyright (c) 2020 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from concurrent.futures.thread import ThreadPoolExecutor
from os import path
import random
from typing import Union

import cv2
import numpy as np

from . import train
from ..common import media

import tensorflow as tf
import tensorflow.keras as K
import utils
import time

from numba import njit


def bboxes_to_ground_truth_njit(
    bboxes, num_classes, grid_size, grid_xy, label_smoothing, anchors_ratio
):
    """
    @param bboxes: [[b_x, b_y, b_w, b_h, class_id], ...]

    @return [s, m, l] or [s, l]
        Dim(1, grid_y, grid_x, anchors,
                            (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
    """
    ground_truth = [
        np.zeros(
            (
                1,
                _size[0],
                _size[1],
                3,
                5 + num_classes,
            ),
            dtype=np.float32,
        )
        for _size in grid_size
    ]

    for i, _grid in enumerate(grid_xy):
        ground_truth[i][..., 0:2] = _grid

    for bbox in bboxes:
        # [b_x, b_y, b_w, b_h, class_id]
        xywh = np.array(bbox[:4], dtype=np.float32)
        class_id = int(bbox[4])

        # smooth_onehot = [0.xx, ... , 1-(0.xx*(n-1)), 0.xx, ...]
        onehot = np.zeros(num_classes, dtype=np.float32)
        onehot[class_id] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
        smooth_onehot = (
            1 - label_smoothing
        ) * onehot + label_smoothing * uniform_distribution

        ious = []
        exist_positive = False
        for i in range(len(grid_xy)):
            # Dim(anchors, xywh)
            anchors_xywh = np.zeros((3, 4), dtype=np.float32)
            anchors_xywh[:, 0:2] = xywh[0:2]
            anchors_xywh[:, 2:4] = anchors_ratio[i]
            iou = train.bbox_iou(xywh, anchors_xywh)
            ious.append(iou)
            iou_mask = iou > 0.3

            if np.any(iou_mask):
                exist_positive = True

                xy_grid = xywh[0:2] * (
                    grid_size[i][1],
                    grid_size[i][0],
                )
                xy_index = np.floor(xy_grid)

                for j, mask in enumerate(iou_mask):
                    if mask:
                        _x, _y = int(xy_index[0]), int(xy_index[1])
                        ground_truth[i][0, _y, _x, j, 0:4] = xywh
                        ground_truth[i][0, _y, _x, j, 4:5] = 1.0
                        ground_truth[i][0, _y, _x, j, 5:] = smooth_onehot

        if not exist_positive:
            index = np.argmax(np.array(ious))
            i = index // 3
            j = index % 3

            xy_grid = xywh[0:2] * (
                grid_size[i][1],
                grid_size[i][0],
            )
            xy_index = np.floor(xy_grid)

            _x, _y = int(xy_index[0]), int(xy_index[1])
            ground_truth[i][0, _y, _x, j, 0:4] = xywh
            ground_truth[i][0, _y, _x, j, 4:5] = 1.0
            ground_truth[i][0, _y, _x, j, 5:] = smooth_onehot

    return ground_truth


class TFDataset:
    def __init__(
        self,
        anchors: np.ndarray = None,
        batch_size: int = 2,
        dataset_path: str = None,
        dataset_type: str = "converted_coco",
        input_size: Union[list, tuple] = None,
        label_smoothing: float = 0.1,
        num_classes: int = None,
        image_path_prefix: str = None,
        strides: np.ndarray = None,
        xyscales: np.ndarray = None,
        channels: int = 3,
        data_augmentation: bool = False,
        augmentations=None,
        preload=False,
    ):
        # anchors / width
        self.anchors_ratio = anchors / input_size[0]
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        # "yolo", "converted_coco"
        self.dataset_type = dataset_type
        # (height, width)
        self.grid_size = (input_size[1], input_size[0]) // np.stack(
            (strides, strides), axis=-1
        )
        self.input_size = input_size
        self.channels = channels

        self.label_smoothing = label_smoothing
        self.image_path_prefix = image_path_prefix
        self.num_classes = num_classes
        self.xysclaes = xyscales

        self.grid_xy = [
            np.tile(
                np.reshape(
                    np.stack(
                        np.meshgrid(
                            (np.arange(_size[0]) + 0.5) / _size[0],
                            (np.arange(_size[1]) + 0.5) / _size[1],
                        ),
                        axis=-1,
                    ),
                    (1, _size[0], _size[1], 1, 2),
                ),
                (1, 1, 1, 3, 1),
            ).astype(np.float32)
            for _size in self.grid_size  # (height, width)
        ]

        self.preload = preload
        self.data_augmentation = data_augmentation
        self.dataset = self.load_dataset()
        self.augmentations = augmentations

        self.count = 0

    def load_dataset(self):
        """
        @return [[image_path, [[x, y, w, h, class_id], ...]], ...]
        """
        _dataset = []

        with open(self.dataset_path, "r") as fd:
            txt = fd.readlines()
            if self.dataset_type == "converted_coco":
                for line in txt:
                    # line: "<image_path> class_id,x,y,w,h ..."
                    bboxes = line.strip().split()
                    image_path = bboxes[0]
                    if self.image_path_prefix:
                        image_path = path.join(self.image_path_prefix, image_path)
                    xywhc_s = np.zeros((len(bboxes) - 1, 5))
                    for i, bbox in enumerate(bboxes[1:]):
                        # bbox = class_id,x,y,w,h
                        bbox = list(map(float, bbox.split(",")))
                        xywhc_s[i, :] = (
                            *bbox[1:],
                            bbox[0],
                        )
                    _dataset.append([image_path, xywhc_s])

            elif self.dataset_type == "yolo":
                for line in txt:
                    # line: "<image_path>"
                    image_path = line.strip()
                    if self.image_path_prefix:
                        image_path = path.join(self.image_path_prefix, image_path)
                    root, _ = path.splitext(image_path)
                    with open(root + ".txt") as fd2:
                        bboxes = fd2.readlines()
                        xywhc_s = np.zeros((len(bboxes), 5))
                        for i, bbox in enumerate(bboxes):
                            # bbox = class_id x y w h
                            bbox = bbox.strip()
                            bbox = list(map(float, bbox.split(" ")))
                            xywhc_s[i, :] = (
                                *bbox[1:],
                                bbox[0],
                            )
                        _dataset.append([image_path, xywhc_s])

        if len(_dataset) == 0:
            raise FileNotFoundError("Failed to find images")

        if self.preload:

            def _read_and_store(path, idx):
                img = self._imread(path)
                _dataset[idx][0] = img

            print("Preloading dataset...")
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=16) as executor:
                for idx, item in enumerate(_dataset):
                    img_path = item[0]
                    executor.submit(_read_and_store, img_path, idx)

            end = time.perf_counter()
            print("Dataset preloaded. Took:", end - start)

        return _dataset

    def _imread(self, path):
        if self.channels == 3:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=2)

        return image

    def load_img_and_labels(self, dataset):
        try:
            if self.preload:
                img = dataset[0]
            else:
                img = self._imread(dataset[0])
        except:
            return None

        labels = dataset[1]

        return img, labels

    def _next_data(self):
        _dataset = self.dataset[self.count]

        self.count += 1
        if self.count == len(self.dataset):
            self.count = 0
            return None

        ret = self.load_img_and_labels(_dataset)
        if ret is not None:
            return ret

        raise FileNotFoundError("Failed to find images")

    def _next(self):
        x, y = self._next_data()
        if self.augmentations is not None:
            x, y = self.augmentations(x, y)

        # x = np.expand_dims(x / 255.0, axis=0).astype(np.float64)
        x = x / 255.0
        # x = tf.Tensor(x)

        y = bboxes_to_ground_truth_njit(
            y,
            self.num_classes,
            self.grid_size,
            self.grid_xy,
            self.label_smoothing,
            self.anchors_ratio,
        )
        # y = [l.astype(np.float64) for l in y]
        # y = [tf.Tensor(l) for l in y]
        # TODO make better
        y = [np.squeeze(l, axis=0) for l in y]
        l1, l2, l3 = y

        return x, l1, l2, l3

    def _next_batch(self):
        batch_x = []
        _batch_y = [[] for _ in range(len(self.grid_size))]
        self.orig_labels = []

        augmentations = self.augmentations
        next_data = self._next_data

        for batch_idx in range(self.batch_size):
            x, y = next_data()
            if augmentations is not None:
                x, y = augmentations(x, y)

            if not self.data_augmentation:
                self.orig_labels.append(y)

            x = np.expand_dims(x / 255.0, axis=0)
            y = bboxes_to_ground_truth_njit(
                y,
                self.num_classes,
                self.grid_size,
                self.grid_xy,
                self.label_smoothing,
                self.anchors_ratio,
            )

            batch_x.append(x)
            for i, _y in enumerate(y):
                _batch_y[i].append(_y)

        batch_x = np.concatenate(batch_x, axis=0)
        batch_y = [np.concatenate(b_y, axis=0) for b_y in _batch_y]

        batch_l1, batch_l2, batch_l3 = batch_y
        return batch_x, batch_l1, batch_l2, batch_l3

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def generator(self):
        # TODO shuffle here
        while True:
            # item = self._next()
            item = self._next_batch()
            if item is None:
                break
                # return

            # x, l1, l2, l3 = item
            # x = tf.convert_to_tensor(x)
            # l1 = tf.convert_to_tensor(l1)
            # l2 = tf.convert_to_tensor(l2)
            # l3 = tf.convert_to_tensor(l3)

            # yield x, l1, l2, l3
            yield item
            # yield item
