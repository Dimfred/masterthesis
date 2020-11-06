from random import randint
import utils

import cv2 as cv
import numpy as np
import math
import random
from numba import njit
import os


# @njit
# def colorize_connected_components(dst, n_labels, components):
#     colors = np.array(
#         [
#             (
#                 np.random.randint(0, 255),
#                 np.random.randint(0, 255),
#                 np.random.randint(0, 255),
#             )
#             for i in range(n_labels)
#         ],
#     )

#     y = 0
#     x = 0
#     for row in components:
#         for label in row:
#             dst[y, x] = colors[label]
#             x += 1

#         y += 1
#         x = 0

#     return dst


# # path = "data/raw/example_circuit4.jpg"
# path = "data/raw/example_circuit.jpg"
# # path = "data/raw/github_resistor_usa_horizontal.jpg"

# img = cv.imread(path, cv.IMREAD_GRAYSCALE)
# img = utils.resize(img, 1000)

# blured = cv.GaussianBlur(img, (5, 5), 10)
# utils.show(blured, 1000)


# thimg = cv.adaptiveThreshold(
#     blured,
#     maxValue=255,
#     adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#     thresholdType=cv.THRESH_BINARY_INV,
#     blockSize=11,
#     C=2,
# )
# # utils.show(thimg, 1000)


# def remsq(img):
#     # for example_circuit.jpg
#     black = 0
#     img[173:255, 506:575] = black
#     img[298:381, 713:837] = black
#     img[413:513, 630:680] = black
#     img[441:513, 340:518] = black
#     return img


# rem = remsq(thimg)
# # utils.show(rem, 1000)
# morph_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

# salt_pepper = cv.morphologyEx(thimg, cv.MORPH_OPEN, morph_kernel, iterations=1)
# # utils.show(salt_pepper, 1000)

# close = salt_pepper
# # close = cv.morphologyEx(
# #     close,
# #     cv.MORPH_CLOSE,
# #     cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)),
# #     iterations=10,
# # )

# close = cv.dilate(close, morph_kernel, iterations=10)
# close = cv.erode(close, morph_kernel, iterations=10)
# utils.show(close, 1000)

# labels, *components = cv.connectedComponents(close)

# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# cimg[:, :] = (0, 0, 0)
# cimg = colorize_connected_components(cimg, labels, components[0])
# utils.show(cimg)

# HOUGH
# edges = cv.Canny(img, 100, 200, None, 3)
## utils.show(edges)
#
# lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
## utils.show(lines)
#
# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#
# for line in lines:
#    rho, theta = line[0]
#
#    a, b = math.cos(theta), math.sin(theta)
#    x0, y0 = a * rho, b * rho
#    pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
#    pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
#    cv.line(cimg, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
#
## utils.show(cimg)
