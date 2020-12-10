#!/usr/bin/env python3

import cv2 as cv
import numpy as np

import os

import utils
from config import config

# label_dir = config.valid_dir
label_dir = config.train_dir

for img_name in [
    # valid
    # "00_11.jpg",
    # "00_18.jpg",
    # "00_19.jpg"
    # "00_20.jpg"

    # "07_00.png",
    # "07_01.png",
    # "07_02.png",
    # "07_03.png",
    # "07_04.png",
    # "07_05.png",
    # "07_06.png",
    # "07_07.png",
    # "07_08.png",

    # "08_00.png",
    # "08_01.png",
    # "08_02.png",
    # "08_03.png",
    # "08_04.png",
    # "08_05.png",
    # "08_06.png",
    # "08_07.png",
    # "08_08.png",
    # "08_09.png",
    # "08_10.png",

    # "10_00.png",

    # "15_00.jpg",
    # "15_01.jpg",
    # "15_02.jpg",
    # "15_03.jpg",

    #train
    # "00_00.jpg", #no
    # "00_01.jpg", #no
    # "00_02.jpg", #no
    # "00_03.jpg", #no
    # "00_04.jpg", #no
    # "00_05.jpg", #no
    # "00_06.jpg",
    # "00_07.jpg", #no
    # "00_08.jpg",
    # "00_09.jpg",
    # "00_10.jpg",

    # "00_12.jpg",
    # "00_13.jpg",
    # "00_14.jpg",
    # "00_15.jpg",
    # "00_16.jpg",
    # "00_17.jpg",

    # "01_00.jpg", #no
    # "01_01.jpg",
    # "01_02.jpg",
    # "01_03.jpg",
    # "01_04.jpg",
    # "01_05.jpg",
    # "01_06.jpg",
    # "01_07.jpg",
    # "01_08.jpg",

    # "02_00.jpg", #no
    # "02_01.jpg",
    # "02_02.jpg",
    # "02_03.jpg", #no

    # "03_00.png",
    # "03_01.png",
    # "03_02.png",
    # "03_03.png",
    # "03_04.png",
    # "03_05.png",
    # "03_06.png",
    # "03_07.png",
    # "03_08.png",
    # "03_09.png",

    # "04_00.png",
    # "04_01.png",
    # "04_02.png",
    # "04_03.png",

    # "05_00.jpg",
    # "05_01.jpg",
    # "05_02.jpg",

    # "06_00.jpg",
    # "06_01.jpg",
    # "06_02.jpg",

    # "09_00.jpg",
    # "09_01.jpg",
    # "09_02.jpg",
    # "09_03.jpg",

    # "11_00.jpg",
    # "11_01.jpg",

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
    # "13_01.jpg",
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

    # "16_00.jpg",
    # "16_01.jpg",
    # "16_02.jpg",
    # "16_03.jpg",
    # "16_04.jpg",
    # "16_05.jpg",
    # "16_06.jpg",
    # "16_07.jpg",
    # "16_08.jpg",
    # "16_09.jpg",

    # "17_00.jpg"
    # "17_01.jpg"
    # "17_02.jpg"
    # "17_03.jpg"
    # "17_04.jpg"
]:
    low = 30
    high = 2 * low
    blur_size = (3, 3)


    path = str(label_dir / img_name)
    img = cv.imread(path)
    img = utils.resize_max_axis(img, 1000)

    orig = img.copy()


    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, blur_size, sigmaX=1.2, sigmaY=1.2)

    img = cv.Canny(img, low, high)
    img = cv.dilate(img, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=2)

    img[img > 0] = 1

    show = orig.copy()
    show *= img[..., np.newaxis]

    utils.show(orig, show)

    name, _ = os.path.splitext(path)
    label_name = f"{name}.npy"

    label_mask = img
    np.save(label_name, label_mask)


# VERIFY DONE
#     loaded_mask = np.load(label_name)

#     show = orig.copy()
#     utils.show(orig, show * loaded_mask[..., np.newaxis])
