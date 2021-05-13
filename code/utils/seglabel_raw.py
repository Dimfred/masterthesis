#!/usr/bin/env python3

import cv2 as cv
import numpy as np

import os

import utils
from config import config

label_dir = config.train_dir
# label_dir = config.valid_dir
# label_dir = config.test_dir

for img_name in [
    ###########
    ## TRAIN ##
    ###########
    # "00_06.jpg",
    # "00_09.jpg",
    # "00_10.jpg",
    # "00_11.jpg",
    # "00_12.jpg",
    # "00_13.jpg",
    # "00_14.jpg",
    # "00_15.jpg",
    # "00_16.jpg",
    # "00_17.jpg",
    # "00_18.jpg",
    # "00_20_c.jpg",
    # "00_21_a.jpg",
    # "00_23_a.jpg",
    # "00_24_a.jpg",
    # "00_25_a.jpg",
    # "00_26_a.jpg",
    # "00_27_a.jpg",
    # "00_28_a.jpg",
    # "00_29_a.jpg",
    # "01_00.jpg", # NO
    # "01_01.jpg",
    # "01_02.jpg",
    # "01_03.jpg",
    # "01_05.jpg",
    # "01_06.jpg",
    # "01_07.jpg",
    # "01_08.jpg",
    # "01_09_a.jpg",
    # "01_10_a.jpg",
    # "02_00.jpg", # NO
    # "02_01.jpg",
    # "02_02.jpg",
    # "02_03.jpg", # NO
    # "03_00.png",
    # "03_01.png",
    # "03_02.png",
    # "03_03.png",
    # "03_05.png",
    # "03_06.png",
    # "03_07.png",
    # "03_08.png",
    # "03_09.png",
    # "04_00.png",
    # "04_01.png",
    # "04_02.png",
    # "05_00.jpg",
    # "05_01.jpg",
    # "05_02.jpg",
    # "06_00.jpg",
    # "06_01.jpg",
    # "06_02.jpg",
    # "06_04_a.jpg",
    # "06_05_a.jpg",
    # "06_06_a.jpg",
    # "06_07_a.jpg",
    # "09_00.jpg",
    # "09_02.jpg",
    # "09_03.jpg",
    # "12_00.jpg",
    # "12_01.jpg",
    # "12_02.jpg",
    # "12_03.jpg",
    # "12_04.jpg",
    # "12_05.jpg",
    # "12_06.jpg",
    # "12_07.jpg",
    # "12_08.jpg",
    # "13_00.jpg",
    # "13_02.jpg",
    # "13_03.jpg",
    # "14_00.jpg",
    # "14_01.jpg",
    # "14_02.jpg",
    # "14_03.jpg",
    # "14_04.jpg",
    # "14_05.jpg",
    # "14_06.jpg",
    # "14_07.jpg",
    # "14_08.jpg",
    # "15_00.jpg",
    # "15_01.jpg",
    # "15_02.jpg",
    # "15_03.jpg",
    # "15_06_a.jpg",
    # "15_07_a.jpg",
    # "15_08_a.jpg",
    # "15_09_a.jpg",
    # "15_10_a.jpg",
    # "15_11_a.jpg",
    # "15_12_a.jpg",
    # "15_13_a.jpg",
    # "15_14_a.jpg",
    # "15_15_a.jpg",
    # "16_00.jpg",
    # "16_01.jpg",
    # "16_02.jpg",
    # "16_03.jpg",
    # "16_04.jpg",
    # "16_05.jpg",
    # "16_07.jpg",
    # "16_08.jpg",
    # "16_09.jpg",
    # "17_00.jpg",
    # "17_01.jpg",
    # "17_02.jpg",
    # "17_03.jpg",
    # "17_04.jpg",
    # "19_00.jpg",
    # "19_01.jpg",
    # "19_02.jpg",
    # "24_00_c.jpg",
    # "24_01_c.jpg",
    # "24_02_c_a.jpg",
    # "24_03_c_a.jpg",
    "27_00_a.png",
    "27_01_a.png",
    "27_02_a.png",
    "27_03_a.png",
    "27_04_a.png",
    "27_05_a.png",
    "27_06_a.png",
    "27_07_a.png",
    "27_08_a.png",
    "29_00_a.png",
    "29_01_a.png",
    "29_02_a.png",
    "29_03_a.png",
    "29_04_a.png",
    "29_05_a.png",
    "29_06_a.png",
    "29_07_a.png",
    "29_08_a.png",
    "32_00_a.jpg",
    "32_01_a.jpg",
    "32_02_a.jpg",
    "32_03_a.jpg",
    "32_04_a.jpg",
    "33_00_a.jpg",
    "33_01_a.jpg",
    "33_02_a.jpg",
    "33_03_a.jpg",
    "33_04_a.jpg",
    "33_05_a.jpg",
    "33_07_a.jpg",
    "33_08_a.jpg",
    "33_09_a.jpg",
    "33_10_a.jpg",
    "33_12_a.jpg",
    "33_13_a.jpg",
    "33_14_a.jpg",
    "33_15_a.jpg",
    "33_16_a.jpg",
    "33_17_a.jpg",
    "33_18_a.jpg",
    "33_19_a.jpg",
    "33_20_a.jpg",
    "33_21_a.jpg",
    "33_22_a.jpg",
    "33_27_a.jpg",
    "33_29_a.jpg",
    ###########
    ## VALID ##
    ###########
    # "00_08.jpg",
    # "00_19_c.jpg",
    # "00_22_a.jpg",
    # "00_30_a.jpg",
    # "00_31_a.jpg",
    # "01_04.jpg",
    # "01_11_a.jpg",
    # "03_04.png",
    # "04_03.png",
    # "06_03_c_a.jpg",
    # "09_01.jpg",
    # "09_04_c_a.png",
    # "13_01.jpg",
    # "15_04_c.jpg",
    # "15_05_c_a.jpg",
    # "16_06.jpg",
    # "24_04_c_a.jpg",
    # "31_00_a.jpg",
    # "31_01_a.jpg",
    # "33_06_a.jpg",
    # "33_23_a.jpg",
    # "33_26_a.jpg",
    # "33_28_a.jpg",
    ###########
    ## TEST ##
    ###########
    # "07_00.png",
    # "07_01.png",
    # "07_02.png",
    # "07_03.png",
    # "07_04.png",
    # "07_05_c.png",
    # "07_06_c.png",
    # "07_07_c.png",
    # "07_08_c.png",
    # "07_09_c_a.png",
    # "07_10_c_a.png",
    # "07_11_c_a.png",
    # "07_12_c_a.png",
    # "07_13_c_a.png",
    # "08_00.png",
    # "08_01.png",
    # "08_02.png",
    # "08_03.png",
    # "08_04.png",
    # "08_05.png",
    # "08_06.png",
    # "08_07_c.png",
    # "08_08_c.png",
    # "08_09_c.png",
    # "08_10_c.png",
    # "08_11_c_a.png",
    # "08_12_c_a.png",
    # "08_13_c_a.png",
    # "08_14_c_a.png",
    # "08_15_c_a.png",
    # "10_00.png",
    # "11_00.jpg",
    # "11_01.jpg",
    # "11_02_a.jpg",
    # "11_03_a.jpg",
    # "11_04_a.jpg",
    # "11_05_a.jpg",
    # "20_00_a.jpg",
    # "20_01_a.jpg",
    # "20_02_a.jpg",
    # "20_03.jpg",
    # "21_00_a.jpg",
    # "21_01_c_a.jpg",
    # "21_02_c_a.jpg",
    # "22_00.jpg",
    # "22_01.jpg",
    # "22_02.jpg",
    # "22_03.jpg",
    # "22_04_c.jpg",
    # "22_05_c.jpg",
    # "22_06_c.jpg",
    # "25_00_c_a.jpg",
    # "25_01_c_a.jpg",
    # "25_02_a.jpg",
    # "26_01_c.jpg",
    # "28_00_c_a.png",
]:
    print(img_name)
    low, blur, dilations = 60, 3, 5

    high = 2 * low
    blur_size = (blur, blur)

    path = str(label_dir / img_name)
    img = cv.imread(path)
    img = utils.resize_max_axis(img, 1000)

    orig = img.copy()

    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, blur_size, sigmaX=1.0, sigmaY=1.0)

    img = cv.Canny(img, low, high)
    img = cv.dilate(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=5)
    img = cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=dilations - 1)
    # img = cv.erode(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2)
    # img = cv.morphologyEx(
    #     img,
    #     cv.MORPH_CLOSE,
    #     cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
    #     iterations=2,
    # )

    img[img > 0] = 1

    show = orig.copy()
    show *= img[..., np.newaxis]
    for row in show:
        for val in row:
            if val[0] == 0 and val[1] == 0 and val[2] == 0:
                val[1] = 128

    utils.show(orig, show)

    name, _ = os.path.splitext(path)
    label_name = f"{name}.npy"

    label_mask = img
    np.save(label_name, label_mask)


# VERIFY DONE
#     loaded_mask = np.load(label_name)

#     show = orig.copy()
#     utils.show(orig, show * loaded_mask[..., np.newaxis])
