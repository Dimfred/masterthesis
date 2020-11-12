#!/usr/bin/env python3

import cv2 as cv
import sys  

img = cv.imread(sys.argv[1])
img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imwrite(sys.argv[1], img)
