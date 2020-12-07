#!/usr/bin/env python3

from PIL import Image
import requests
from io import BytesIO
import cv2
import cv2 as cv
from matplotlib import pyplot as plt

import utils
from config import config
import numpy as np

img = cv.imread(str(config.valid_dir / "00_20.jpg"))
img = utils.resize_max_axis(img, 1000)

A = np.array(img)
A2 = A.copy()
A_gray = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)


# Do some rough edge detection to find the grid
sX = cv2.Sobel(A_gray, cv2.CV_64F, 1, 0, ksize=3)
sY = cv2.Sobel(A_gray, cv2.CV_64F, 0, 1, ksize=3)
sX[sX<0] = 0
sY[sY<0] = 0

# plt.subplot(221)
# plt.imshow(sX)

utils.show(sX)
utils.show(sY)
# plt.subplot(222)
# plt.imshow(sY)

# plt.subplot(223)
# the sum operation projects the edges to the X or Y-axis.
# The 0.2 damps the high peaks a little
eX = (sX**.2).sum(axis=0)
eX = np.roll(eX, -1) # correct for the 1-pixel offset due to Sobel filtering
# plt.plot(eX)
# utils.show(eX)

# plt.subplot(224)
eY = (sY**.2).sum(axis=1)
eY = np.roll(eY, -1)
# plt.plot(eY)
# utils.show(eY)

mask = np.zeros(A2.shape[:2], dtype=np.uint8)
mask[eY>480,:] = 1
mask[:, eX>390] = 1


A2[mask.astype(bool),:] = 255
# plt.figure()
# plt.subplot(221)
# plt.imshow(A)

# plt.subplot(222)
# plt.imshow((A2))
utils.show(A)
utils.show(A2)

restored = cv2.inpaint(A, mask, 1, cv2.INPAINT_NS)
# plt.subplot(223)
# plt.imshow(restored)
utils.show(restored)
