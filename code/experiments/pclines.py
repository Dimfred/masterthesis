#!/usr/bin/env python3

import cv2 as cv
import numpy as np

import utils
from config import config

from numba import njit


def imgrad(img):
    # TODO move utils
    gx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    gy = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    orientation = 180 * np.arctan2(gy, gx) / np.pi
    # replace negative values by their positive pendant
    orientation[orientation < 0] = 180 + orientation[orientation < 0]

    return gx, gy, magnitude, orientation


# @njit
def histogram(data, bins):
    bin_idxs = [[] for _ in range(len(bins))]

    # for y in range(Y):
    for x in range(len(data)):
        val = data[x]

        # enumerate
        bin_counter = 0
        for bin_low, bin_high in bins:
            if bin_low <= val and val < bin_high:
                bin_idxs[bin_counter].append(x)
                break

            bin_counter += 1

    return bin_idxs


if __name__ == "__main__":

    for img_name in [
        "00_19.jpg",
        # "00_20.jpg",
        # "07_05.png",
        # "07_06.png",
        # "07_07.png",
        # "07_08.png",
        # "08_07.png",
        # "08_08.png",
        # "08_09.png",
        # "08_10.png",
    ]:

        img_path = str(config.valid_dir / img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = utils.resize_max_axis(img, 1000)

        cimg = img.copy()
        cimg = cv.cvtColor(cimg, cv.COLOR_GRAY2BGR)

        img = cv.GaussianBlur(img, (5, 5), sigmaX=1.2, sigmaY=1.2)
        # utils.show(img)

        gx, gy, magnitude, orientation = imgrad(img)
        # utils.show(gx, gy)

        # supress gradients with a too small magnitude
        magnitude_thresh = 300
        magnitude[magnitude < magnitude_thresh] = 0

        gx[magnitude == 0] = 0
        gy[magnitude == 0] = 0
        utils.show(gx, gy)

        where_relevant = magnitude > 0
        relevant_idxs = np.argwhere(where_relevant)

        magnitude = magnitude.flatten()
        orientation = orientation.flatten()

        where_relevant = where_relevant.flatten()
        relevant_orientations = orientation[where_relevant]

        n_bins = 12
        step = int(180.0 / n_bins)
        bins = [
            (low, high)
            for low, high in zip(range(0, 180, step), range(step, 180 + step, step))
        ]

        bin_idxs = histogram(relevant_orientations, bins)
        bin_idxs = [(i, bin_) for i, bin_ in enumerate(bin_idxs)]

        bin_idxs = sorted(bin_idxs, key=lambda bin: len(bin[1]), reverse=True)

        # print bins
        print("Bin with length")
        for i, bin_idx in bin_idxs:
            print(i, bins[i], len(bin_idx))

        # visualize most relevant bins
        relevant_bins = [5, 6, 0, 11]
        bin_show = cimg.copy()
        for bin_idx in relevant_bins:
            real_idxs = [relevant_idxs[idx] for idx in bin_idxs[bin_idx][1]]
            for y, x in real_idxs:
                bin_show[y, x] = (0, 0, 255)

        utils.show(bin_show)

        #






        # sort by binsize
        # orientation_hist = sorted(orientation_hist, key=lambda x: x[0], reverse=True)
        # print(orientation_hist)

        # # 75° 165°
        # # TODO find here the 90 degree pendants
        # dominant_bins = [0, 2]

        # bin_size = 15
        # bin_values = [orientation_hist[i][1] for i in dominant_bins]

        # bin_idxs = []
        # for bin_val in bin_values:
        #     idxs = np.argwhere(
        #         (relevant_orientations > bin_val)
        #         & (relevant_orientations < bin_val + bin_size)
        #     )
        #     print(idxs)
        #     bin_idxs.extend(list(zip(*idxs)))

        # # print(bin_idxs)
