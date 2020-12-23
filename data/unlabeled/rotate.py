#!/usr/bin/env python3

import cv2 as cv
import sys

for img_name in sys.argv[1:]:
    img = cv.imread(img_name)
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
#    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    cv.imwrite(img_name, img)
