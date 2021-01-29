from utils import Yolo
import cv2 as cv
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from config import config
import utils
from utils import YoloBBox

import time
import random


class MNistLoader:
    def __init__(self):
        self.data = [[] for _ in range(10)]

    @utils.stopwatch("Mnist::load")
    def load(self, mnist_dir, worker=16, roi_only=False):
        # start_load = time.perf_counter()

        def load_and_store(number, path):
            img = self._imread(path)
            img = 255 - img
            self.data[number].append(img)

        with ThreadPoolExecutor(max_workers=worker) as executor:
            for number in range(10):
                number_dir = mnist_dir / str(number)
                pngs = number_dir.glob("**/*.png")

                for png in pngs:
                    executor.submit(load_and_store, number, png)

        end_load = time.perf_counter()
        # print("MNist loading took:", end_load - start_load)

        if roi_only:
            self.extract_rois()

    def extract_rois(self):
        data = self.data
        for number in range(10):
            number_imgs = data[number]
            for idx, img in enumerate(number_imgs):
                grey_idxs = np.argwhere(img < 255)
                tl, br = self._bbox_from_number(grey_idxs)
                (y1, x1), (y2, x2) = tl, br

                roi = img[y1:y2, x1:x2]

                # replace original with roi only
                data[number][idx] = roi

                # DEBUG
                # utils.show(img, roi)

    def _bbox_from_number(self, grey_idxs):
        ys, xs = grey_idxs[:, 0], grey_idxs[:, 1]

        top, bot = np.min(ys), np.max(ys)
        left, right = np.min(xs), np.max(xs)

        return (top, left), (bot, right)

    def _imread(self, path):
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        return img


class MNistProjector:
    def __init__(self, mnist_dataset, classes):
        self.dataset = mnist_dataset
        self.classes = classes

    def project(self, img, labels, number_str):

        for label in labels:
            bbox = YoloBBox(img.shape).from_ground_truth(label)
            cls_name = self.classes[bbox.label]

            if "ver" in cls_name:
                continue

            numbers = self._numbers_from_str(number_str)
            x1, y1, x2, y2 = bbox.abs()
            self._project_number(img, numbers[0], y1, x1)

            utils.show(img)

    def _project_number(self, img, number, y, x):
        nh, nw = number.shape
        yp = y - nh
        print(number.shape)
        not_white = np.argwhere(number < 180)
        print(not_white.shape)

        # img[to_img_idxs] = number[not_white]
        for yn, xn in not_white:
            img[yp + yn, x + xn] = number[yn, xn]

    def _random_number(self, number):
        return random.choice(self.dataset[int(number)])

    def _numbers_from_str(self, number_str):
        return [self._random_number(n) for n in number_str]


if __name__ == "__main__":
    loader = MNistLoader()
    loader.load(config.mnist_train_dir, roi_only=True)

    classes = utils.Yolo.parse_classes(config.yolo.classes)
    dataset = utils.Yolo.load_dataset(config.train_out_dir)

    projector = MNistProjector(loader.data, classes)

    number = "0"
    for img_path, label_path in dataset:
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        labels = utils.Yolo.parse_labels(label_path)

        projector.project(img, labels, number)

        # utils.show(img)

    for i in range(20):
        utils.show(projector.dataset[0][i])
