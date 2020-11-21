#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from config import config
import utils


img = cv.imread(str(config.label_dir / "05_00.jpg"))  # , cv.IMREAD_GRAYSCALE)
img = utils.resize_max_axis(img, 1000)
utils.show(img)
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (170, 20, 582, 937)
#rect = (50, 50, 450, 290)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
img = img * mask2[:, :, np.newaxis]

plt.imshow(img), plt.colorbar(), plt.show()
