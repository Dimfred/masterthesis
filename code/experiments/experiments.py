#!/usr/bin/env python3

import cv2 as cv
import numpy as np

from random import randint
import math
import random
from numba import njit
import os
from pathlib import Path
import sys

import sklearn as sk
import sklearn.cluster

from config import config
import utils
from utils import YoloBBox
import itertools as it




import torch as t
import torch.nn as nn


arr = np.array([
    [
        [1, 1],
        [0, 1],
    ],
    [
        [0, 1],
        [0, 1],
    ]
])
print(arr.shape)


arr = t.Tensor(arr)

print(nn.Softmax(dim=0)(arr))



# def nms(img):
#     Y, X = img.shape[:2]

#     img = img.copy()
#     for y in range(1, Y):
#         for x in range(1, X):
#             val = img[y, x]
#             ry2 = range(max(y - 1, 1), min(y + 1, Y))
#             rx2 = range(max(x - 1, 1), min(x + 1, X))
#             for y2, x2 in zip(ry2, rx2):
#                 if img[y2, x2] > val:
#                     img[y, x] = 0
#                     break

#     return img


# def intersection(l1, l2):
#     # l1, l2 in hough form = (rho, theta)
#     rho1, theta1 = l1[0]
#     rho2, theta2 = l2[0]

#     # fmt: off
#     A = np.array([
#         [np.cos(theta1), np.sin(theta1)],
#         [np.cos(theta2), np.sin(theta2)]
#     ])
#     b = np.array([
#         [rho1],
#         [rho2]
#     ])
#     # fmt: on

#     try:
#         x0, y0 = np.linalg.solve(A, b)
#         x0, y0 = int(np.round(x0)), int(np.round(y0))
#     except:
#         return None, None

#     return x0, y0


# for img_name in [
#     "00_19.jpg",
#     "00_20.jpg",
#     "07_05.png",
#     "07_06.png",
#     "07_07.png",
#     "07_08.png",
#     "08_07.png",
#     "08_08.png",
#     "08_09.png",
#     "08_10.png",
# ]:
#     print(img_name)
#     img_name = config.valid_dir / img_name
#     img = cv.imread(str(img_name), cv.IMREAD_GRAYSCALE)
#     img = utils.resize_max_axis(img, 1000)

#     show = img.copy()
#     show = cv.cvtColor(show, cv.COLOR_GRAY2BGR)

#     # img = cv.GaussianBlur(img, (3, 3), 1) #, sigmaX=1.2, sigmaY=1.2)

#     img = cv.adaptiveThreshold(
#         img,
#         maxValue=255,
#         adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv.THRESH_BINARY_INV,
#         # TODO
#         blockSize=11,
#         # TODO
#         C=2,
#     )
#     # utils.show(img)

#     cross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     square = np.ones((2, 2))

#     kernel = square
#     img = cv.morphologyEx(
#         img,
#         cv.MORPH_OPEN,
#         kernel,
#         iterations=1,
#     )
#     utils.show(img)

#     ground_truth = utils.load_ground_truth(utils.label_file_from_img(img_name))
#     bboxes = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]

#     all_xs = []
#     all_ys = []
#     for bbox in bboxes:
#         x1, y1, x2, y2 = bbox.abs()
#         img[y1:y2, x1:x2] = 0

#         all_xs.append(x1)
#         all_xs.append(x2)
#         all_ys.append(y1)
#         all_ys.append(y2)

#     # utils.show(img)

#     xmin, ymin = min(all_xs), min(all_ys)
#     xmax, ymax = max(all_xs), max(all_ys)
#     # remove everything outside those coords
#     roi = img[ymin:ymax, xmin:xmax]
#     img = np.zeros_like(img)
#     img[ymin:ymax, xmin:xmax] = roi
#     utils.show(img)


#     # ORB for hough
#     # orb = cv.ORB_create(nfeatures=1000, edgeThreshold=100)
#     # kp = orb.detect(img, None)

#     # orb_show = cv.drawKeypoints(show.copy(), kp, None, color=(0, 255, 0), flags=0)
#     # utils.show(orb_show)

#     # points = [p.pt for p in kp]

#     # generate only with the ORB points
#     # hough = np.zeros_like(img)
#     # for x, y in points:
#     #     x, y = int(x), int(y)
#     #     hough[y, x] = 255

#     # utils.show(hough)

#     hough = img
#     lines = cv.HoughLines(hough, 1, np.pi / 180, 150, None, 0, 0)
#     print("n_lines:", len(lines))

#     # show initial hough lines
#     stored_lines = []
#     line_img = show.copy()

#     for line in lines:
#         rho, theta = line[0]

#         a, b = math.cos(theta), math.sin(theta)
#         x0, y0 = a * rho, b * rho
#         p1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
#         p2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
#         stored_lines.append((p1, p2))
#         cv.line(line_img, p1, p2, (0, 0, 255), 1, cv.LINE_AA)

#     utils.show(line_img)

#     def histogram(data, bins):
#         bin_idxs = [[] for _ in range(len(bins))]

#         # for y in range(Y):
#         for x in range(len(data)):
#             val = data[x]

#             # enumerate
#             bin_counter = 0
#             for bin_low, bin_high in bins:
#                 if bin_low <= val and val < bin_high:
#                     bin_idxs[bin_counter].append(x)
#                     break

#                 bin_counter += 1

#         return bin_idxs

#     get_theta = lambda line: line[0][1]
#     thetas = [180 * get_theta(l) / np.pi for l in lines]
#     # print(thetas)

#     step = 0.5
#     bins = [
#         (low, high)
#         for low, high in zip(np.arange(0, 180, step), np.arange(step, 180 + step, step))
#     ]

#     binned_idxs = histogram(thetas, bins)
#     # for i, bin_ in enumerate(binned_idxs):
#     #     if len(bin_) != 0:
#     #         print(i, bins[i], len(bin_))

#     binned_idxs = [(i, bin_) for i, bin_ in enumerate(binned_idxs)]
#     binned_idxs = sorted(binned_idxs, key=lambda idx_bin: len(idx_bin[1]), reverse=True)

#     best = binned_idxs[0]
#     bin_angle = bins[best[0]][0]
#     print(bin_angle)

#     pendant = None
#     if bin_angle < 90:
#         second_angle = bin_angle + 90
#         for i, bin_ in binned_idxs:
#             if bins[i][0] == second_angle:
#                 pendant = (i, bin_)
#                 break
#     elif bin_angle >= 90:
#         second_angle = bin_angle - 90
#         for i, bin_ in binned_idxs:
#             if bins[i][0] == second_angle:
#                 pendant = (i, bin_)
#                 break


#     line_img = show.copy()
#     for idx in it.chain(best[1], pendant[1]):
#         line = lines[idx]
#         rho, theta = line[0]

#         a, b = math.cos(theta), math.sin(theta)
#         x0, y0 = a * rho, b * rho
#         p1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
#         p2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
#         stored_lines.append((p1, p2))
#         cv.line(line_img, p1, p2, (0, 0, 255), 1, cv.LINE_AA)

#     utils.show(line_img)




    # does not work
    # count = {}
    # for l1, l2 in it.combinations(lines, 2):
    #     ix, iy = intersection(l1, l2)
    #     if ix is None or iy is None:
    #         continue

    #     ip = (iy, ix)
    #         count[ip] = 0

    #     count[ip] += 1

    # print(count)

    # scount = sorted(count.items(), key=lambda x: x[1])

    ### APPROX SQUARES ###

    # img = cv.dilate(img, cross)
    # utils.show(img)

    # contours, hierarchy = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cnt_len = cv.arcLength(cnt, True)
    #     cnt = cv.approxPolyDP(cnt, 0.04 * cnt_len, True)

    #     if len(cnt) == 4 and  cv.contourArea(cnt) > 40 and cv.isContourConvex(cnt):
    #         lens = []
    #         for a, b in zip(cnt, [*cnt[1:], cnt[0]]):
    #             side_len = np.linalg.norm(a[0] - b[0])
    #             lens.append(side_len)
    #         print(lens)

    #         almost_equal = lambda l1, l2: abs(l1 - l2) < 3
    #         all_equal = [almost_equal(l1, l2) for l1, l2 in it.combinations(lens, 2)]
    #         if sum(all_equal) != 6:
    #             print(sum(all_equal))
    #             continue

    #         show = cv.drawContours(show, [cnt], 0, (0, 0, 255), 2)
    #         utils.show(show)

    #         # lens = [np.linalg.norm(np.array(b[0]) - np.array(a[0])) for a, b in zip(cnt, [*cnt[1:], cnt[0]])]

    # erode = cv.erode(thresh, cross, iterations=1)

    # close2 = cv.morphologyEx(
    #     close1,
    #     cv.MORPH_CLOSE,
    #     cross,
    #     iterations=5,
    # )

    # close = cv.dilate(close, cross, iterations=10)
    # close = cv.erode(close, cross, iterations=10)

    # utils.show(thresh, erode) #, close2)

    ### APPROX SQUARES  END ###

    # # JULIA PHD BEGIN
    # # TODO try to understand see julia code, base on the PhD thesis
    # x_sigma, y_sigma = 1.2, 1.2

    # img = cv.GaussianBlur(img, (3, 3), sigmaX=x_sigma, sigmaY=y_sigma)
    # utils.show(img)

    # x_grad = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=7)
    # y_grad = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=7)
    # # utils.show(x_grad * y_grad, x_grad **2, y_grad **2)

    # diff = x_grad ** 2 - y_grad ** 2
    # both = 2 * x_grad * y_grad

    # blurred_diff = cv.GaussianBlur(diff, (11, 11), sigmaX=x_sigma, sigmaY=y_sigma)
    # blurred_both = cv.GaussianBlur(both, (11, 11), sigmaX=x_sigma, sigmaY=y_sigma)

    # ksize = 36
    # halfk = int(ksize / 2)
    # sqsize = int(np.sqrt(ksize))

    # h1 = np.array([
    #     (i ** 2 - j ** 2) / max(1, i ** 2 + j ** 2)
    #     for i, j in zip(range(-halfk, halfk), range(-halfk, halfk))
    # ])
    # # TODO centered?????
    # h1 = h1.reshape(sqsize, sqsize)

    # h2 = np.array([
    #     (2 * i * j) / max(1, i ** 2 + j ** 2)
    #     for i, j in zip(range(-halfk, halfk), range(-halfk, halfk))
    # ])
    # h2 = h2.reshape(sqsize, sqsize)

    # res = cv.filter2D(blurred_both, -1, h2) - cv.filter2D(blurred_diff, -1, h1)
    # res = nms(res)

    # where_cross = res > 0
    # print(where_cross.shape)
    # show[where_cross] = (0, 0, 255)
    # show = np.uint8(show)

    # utils.show(show)

    #### JULIA PHD END

    # GRAD in BBOX for CANNY

    # img = cv.GaussianBlur(img, (3, 3), sigmaX=1.2, sigmaY=1.2)

    # ground_truth = utils.load_ground_truth(utils.label_file_from_img(img_name))
    # bboxes = [YoloBBox(img.shape).from_ground_truth(gt) for gt in ground_truth]

    # orig = img.copy()
    # orig = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)

    # # blurred = img
    # blurred = cv.GaussianBlur(img, (3, 3), 0)
    # # # utils.show(blurred)

    # grad = np.abs(cv.Laplacian(img, cv.CV_64F))
    # grad[grad < 50] = 0
    # utils.show(orig, np.abs(grad))

    # # x_grad = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    # # y_grad = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    # # utils.show(x_grad, y_grad)

    # bbox_grads = []
    # for bbox in bboxes:
    #     x1, y1, x2, y2 = bbox.abs()
    #     bbox_grads.extend(grad[y1:y2, x1:x2].flatten())

    # bbox_grads = np.array(bbox_grads)
    # bbox_grads = np.abs(bbox_grads)
    # # remove small gradients
    # # bbox_grads = bbox_grads[bbox_grads > 20]
    # bbox_grads = bbox_grads.reshape(-1, 1)

    # n_clusters = 3
    # kmeans = sk.cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(bbox_grads)

    # thresholds = []
    # for label in range(n_clusters):
    #     labeled_grad = bbox_grads[kmeans.labels_ == label]
    #     mean = np.mean(labeled_grad)
    #     thresholds.append(mean)

    # print(thresholds)
    # high, mid, *_ = sorted(thresholds, reverse=True)

    # low_thresh = high
    # high_thresh = 2 * high
    # edges = cv.Canny(blurred, low_thresh, high_thresh, None, 3)
    # utils.show(edges)
    # img = cv.GaussianBlur(img, (5, 5), sigmaX=1.2, sigmaY=1.2)

    ### GRAD in BBOX END

    # HOUGH + buckets
    # thresh = cv.adaptiveThreshold(
    #     img,
    #     maxValue=255,
    #     adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     thresholdType=cv.THRESH_BINARY_INV,
    #     blockSize=11,
    #     C=2,
    # )
    # utils.show(thresh)

    # thresh = cv.erode(thresh, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
    # utils.show(thresh)

    # lines = cv.HoughLines(thresh, 1, np.pi / 180, 150, None, 0, 0)
    # stored_lines = []
    # line_img = orig.copy()
    # for line in lines:
    #     rho, theta = line[0]

    #     a, b = math.cos(theta), math.sin(theta)
    #     x0, y0 = a * rho, b * rho
    #     p1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
    #     p2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
    #     stored_lines.append((p1, p2))
    #     cv.line(line_img, p1, p2, (0, 0, 255), 3, cv.LINE_AA)

    # utils.show(line_img)

    # filtered_img = orig.copy()

    # # filter non perpendicular or non parralel lines
    # filtered = []
    # bucket = {}
    # for (p11, p12), (p21, p22) in it.combinations(stored_lines, 2):
    #     l1 = np.array(p11) - np.array(p12)
    #     l2 = np.array(p21) - np.array(p22)

    #     angle = utils.angle(l1, l2)

    #     are_parrallel = lambda angle: angle <= 2 and angle >= -2
    #     are_perpendicular = lambda angle: angle <= 92 and angle >= 88
    #     if are_parrallel(angle) or are_perpendicular(angle):
    #         real_angle = utils.angle(np.array(p11), np.array(p12))
    #         print(real_angle)
    #         if real_angle not in bucket:
    #             bucket[real_angle] = set()

    #         bucket[real_angle].add((p11, p12))
    #         bucket[real_angle].add((p21, p22))

    # # getbiggest bucked
    # max_bucket_size = 0
    # biggest_bucket = None
    # for b in bucket.values():
    #     if len(b) > max_bucket_size:
    #         max_bucket_size = len(b)
    #         biggest_bucket = b

    # # print(biggest_bucket)
    # # print("len(bucket)\n{}".format(len(bucket)))
    # # print([len(b) for b in bucket.values()])

    # for p1, p2 in biggest_bucket:
    #     cv.line(filtered_img, p1, p2, (0, 0, 255), 3, cv.LINE_AA)

    # utils.show(filtered_img)

    # harris = cv.cornerHarris(img, 2, 3, 0.04)
    # harris = cv.cornerHarris(edges, 2, 3, 0.04)

    # orig[harris > 0.04 * harris.max()] = (0, 0, 255)
    # utils.show(orig)


### ORB orb #########
# orb = cv.ORB_create(nfeatures=10000, edgeThreshold=60)
# kp = orb.detect(thresh, None)

# orig = cv.drawKeypoints(orig, kp, None, color=(0, 255, 0), flags=0)
# utils.show(orig)


# utils.show(norm_scaled, 1000)
# removes the pattern
# kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
# img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
# utils.show(img)

# contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# print("contours\n{}".format(contours))
# orig = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
# for contour in contours:
# pass
# (x, y, w, h) = cv.boundingRect(contour)
# cv.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
# area = cv.contourArea(contour)
# poly = cv.approxPolyDP(contour, 0.04 * cv.arcLength(contour, True), True)
# print(poly)
# print(len(poly))
# if len(poly) == 4:
# cv.drawContours(orig, [poly], (0, 0, 255), 5)
# utils.show(orig)


# cv.putText(
#     img,
#     ("width = {}, height = {}".format(w, h)),
#     (x + 30, y + 30),
#     cv.FONT_HERSHEY_SIMPLEX,
#     1,
#     (0, 255, 0),
#     2,
#     cv.LINE_AA,
# )

# utils.show(orig)

### MEH
# edges = cv.Canny(img, 100, 200, None, 3)
# ## utils.show(edges)
# #
# lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
# #
# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# for line in lines:
#    rho, theta = line[0]

#    a, b = math.cos(theta), math.sin(theta)
#    x0, y0 = a * rho, b * rho
#    pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
#    pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
#    cv.line(cimg, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

# utils.show(cimg)

# img = cv.cornerHarris(img, 2, 3, 0.04)

# #TODO
# norm = np.empty(img.shape, dtype=np.float32)
# cv.normalize(img, norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
# norm_scaled = cv.convertScaleAbs(norm)

# thresh = 255
# h, w = norm.shape[:2]
# for y in range(h):
#     for x in range(w):
#         if int(norm[y, x]) > thresh:
#             cv.circle(norm_scaled, (x, y), 5, (0), 2)

# utils.show(norm_scaled, 1000)


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

############ TEST WITH line fitting

# TODO try to split lines
# F: idx 10
# utils.show(np.uint8(connected_components))
# for c in range(100):
#     print(c)
#     concomp = np.uint8(connected_components.copy())
#     concomp[concomp != c] = 0
#     concomp[concomp == c] = 255

#     utils.show(concomp)

# ccomp_idx = 10
# ccomp = np.uint8(connected_components.copy())
# # TODO do in 1 step
# ccomp[ccomp != 10] = 0
# ccomp[ccomp == 10] = 255
# # ccomp = cv.dilate(ccomp, )
# # utils.show(ccomp)

# # skeleton = skeletonize(ccomp, debug=dskel)

# # canny

# edges = cv.Canny(ccomp, 100, 200, None, 3)
# # utils.show(edges)

# # fmt:off
# vertical_kernels = np.array([
#     np.array([
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0]
#     ]),
#     np.array([
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0]
#     ]),
#     np.array([
#         [0, 0, 1],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0]
#     ]),
#     np.array([
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [1, 0, 0]
#     ]),
#     np.array([
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ]),
# ])
# horizontal_kernels = np.vstack([vk.T for vk in vertical_kernels])

# # fmt:on

# binarized = edges.copy()
# binarized[binarized > 0] = 1

# out = np.zeros_like(edges)

# vidxs = []
# for vk in vertical_kernels:
#     conv = cv.filter2D(binarized, -1, vk)
#     ys, xs = np.where(conv == 4)
#     vidxs += [(y, x) for y, x in zip(ys, xs)]

# vidxs = np.vstack(vidxs)
# mean = vidxs.mean(axis=0)
# std = vidxs.std(axis=0)

# # filter all outliers where x is too far away
# ymean, xmean = mean
# vidxs = np.vstack([(y, x) for y, x in vidxs if abs(x - xmean) < 30])

# for y, x in vidxs:
#     out[y, x] = 255
# utils.show(out)

# ys, xs = vidxs[:, 0], vidxs[:, 1]

# xmin, xmax = np.min(xs), np.max(xs)
# ymin, ymax = np.min(ys), np.max(ys)

# xmin, xmax = int(xmean), int(xmean)

# out = np.zeros_like(edges)
# cv.line(out, (xmin, ymin), (xmax, ymax), 255, 3, cv.LINE_8)
# utils.show(out)

# out = np.zeros_like(edges)
# for vk in vertical_kernels:
#     hk = vk.T
#     conv = cv.filter2D(binarized, -1, hk)
#     out[np.where(conv == 4)] == 255
# utils.show(out)


# HOUGH

# edge_coords = np.where(edges != 0)
# print(edge_coords)

# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# cimg[:, :, :] = 0

# utils.show(cimg)

# HOUGH
# lines = cv.HoughLines(skeleton, 1, np.pi / 180, 30, None, 0, 0)
# print("lines", lines)

# cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# for line in lines:
#    rho, theta = line[0]

#    a, b = math.cos(theta), math.sin(theta)
#    x0, y0 = a * rho, b * rho
#    pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
#    pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
#    cv.line(cimg, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
# utils.show(cimg)

### wire drawing old
# draw all symbols in the grid
#     for x1, y1, x2, y2, cls_, _ in abs_bounding_boxes:
#         xm, ym = int(x1 + 0.5 * x2), int(y1 + 0.5 * y2)

#         x_grid, y_grid = int(xm / grid_size), int(ym / grid_size)

#         component = LTBuilder.build("TODO_NAME", x_grid, y_grid, cls_)
#         ltcomponents.append(component)

#     # draw wires
#     for _, connected_bbox_idxs in connected_bounding_box_idxs.items():
#         for idx1, idx2 in it.combinations(connected_bbox_idxs, 2):
#             c1, c2 = ltcomponents[idx1], ltcomponents[idx2]
#             # TODO only when edges active
#             if c1 is None or c2 is None:
#                 print("is none")
#                 continue

#             b1, b2 = abs_bounding_boxes[idx1], abs_bounding_boxes[idx2]

#             if c1.is_horizontal and c2.is_horizontal:
#                 # b1 if more left
#                 if b1[0] < b2[0]:
#                     wire = Wire(c1.right, c2.left)
#                 else:
#                     wire = Wire(c1.left, c2.right)

#             elif c1.is_vertical and c2.is_vertical:
#                 # b1 if is above
#                 if b1[1] < b2[1]:
#                     wire = Wire(c1.bottom, c2.top)
#                 else:
#                     wire = Wire(c1.top, c2.bottom)

#             elif c1.is_vertical and c2.is_horizontal:
#                 # b1 is above
#                 if b1[1] < b2[1]:
#                     # b1 is more left
#                     if b1[0] < b2[0]:
#                         wire = Wire(c1.bottom, c2.left)
#                     # b1 is more right
#                     else:
#                         wire = Wire(c1.bottom, c2.right)
#                 else:
#                     if b1[0] < b2[0]:
#                         wire = Wire(c1.top, c2.left)
#                     # b1 is more right
#                     else:
#                         wire = Wire(c1.top, c2.right)

#             elif c1.is_horizontal and c2.is_vertical:
#                 # b1 is above
#                 if b1[1] < b2[1]:
#                     # b1 is more left
#                     if b1[0] < b2[0]:
#                         wire = Wire(c1.right, c2.top)
#                     # b1 is more right
#                     else:
#                         wire = Wire(c1.left, c2.bottom)
#                 else:
#                     # b1 is more left
#                     if b1[0] < b2[0]:
#                         wire = Wire(c1.right, c2.bottom)
#                     # b1 is more right
#                     else:
#                         wire = Wire(c1.left, c2.top)
#             else:
#                 print("c1.is_horizonal\n{}".format(c1.is_horizontal))
#                 print("c2.is_vertical\n{}".format(c2.is_vertical))
#                 wire = None
#                 print("THIS CASE SHOULD NOTHAPPEN")

#             ltcomponents.append(wire)

#     # hough lines p
#     lines = cv.HoughLinesP(edges, 1, math.pi/100, 2, None, 20, 3)

#     for line in lines:
#         line = line[0]
#         p1 = (line[0], line[1])
#         p2 = (line[2], line[3])
#         print(p1, p2)
#         cimg = cv.line(cimg, p1, p2, (0, 0, 255), 1)
#         utils.show(cimg)

#     # HARRIS
#     # harris = cv.cornerHarris(np.float32(ccomp), 2, 3, 0.04)
#     # utils.show(harris)

# ### Clustering


# class SubLineSegmentator:
#     def __init__(self, *args, clusterer="kmeans", **kwargs):
#         self._clusterer = clusterer
#         if clusterer == "kmeans":
#             self.clusterer = sk.cluster.KMeans(*args, **kwargs)
#         elif clusterer == "hdbscan":
#             self.clusterer = hdbscan.HDBSCAN(*args, **kwargs)

#     def fit_predict(self, edges):
#         self.descriptors = self.create_descriptors(edges)
#         self.labels_ = self.clusterer.fit_predict(self.descriptors)
#         return self.labels_

#     def filter_diagonal_lines(self, edges):
#         # fmt:off
#         vertical_kernels = np.array([
#             np.array([
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0]
#             ]),
#             np.array([
#                 [1, 0, 0],
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0]
#             ]),
#             np.array([
#                 [0, 0, 1],
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0]
#             ]),
#             np.array([
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [1, 0, 0]
#             ]),
#             np.array([
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [0, 0, 1]
#             ]),
#         ])
#         horizontal_kernels = np.vstack([[vk.T] for vk in vertical_kernels])
#         # fmt:on

#         binarized = edges.copy()
#         binarized[binarized == 255] = 1

#         idxs = []
#         for vk in vertical_kernels:
#             corr = cv.filter2D(binarized, -1, vk)
#             ys, xs = np.where(corr == 4)
#             idxs += [(y, x) for y, x in zip(ys, xs)]

#         for hk in horizontal_kernels:
#             corr = cv.filter2D(binarized, -1, hk)
#             ys, xs = np.where(corr == 4)
#             idxs += [(y, x) for y, x in zip(ys, xs)]

#         idxs = np.vstack(idxs)
#         ys, xs = idxs[:, 0], idxs[:, 1]

#         filtered = np.zeros_like(edges)
#         filtered[ys, xs] = 255

#         return filtered

#     def create_descriptors(self, edges):
#         # filtered = self.filter_diagonal_lines(edges)
#         filtered = edges
#         utils.show(filtered)

#         edge_idxs = np.where(filtered == 255)
#         # TODO slow
#         descriptors = []
#         self.coordinates = []
#         for y, x in zip(*edge_idxs):
#             # TODO outofbounds
#             # extract 3x3 around target point
#             orientation_desc = edges[y - 1 : y + 2, x - 1 : x + 2]
#             # desc = np.append([y, x], orientation_desc.flatten())
#             self.coordinates.append((y, x))
#             desc = np.array([y, x])
#             # desc = np.array([y**2, x**2])
#             # desc = [np.linalg.norm(np.array([y, x]))]
#             desc = np.append(desc, orientation_desc.flatten())
#             descriptors.append(desc)

#         return np.vstack(descriptors)

#     def show_clusters(self, rgb_img):
#         rgb_img = rgb_img.copy()

#         if self._clusterer == "kmeans":
#             self.n_clusters = self.clusterer.n_clusters
#         elif self._clusterer == "hdbscan":
#             self.n_clusters = len(set(self.labels_)) - 1

#         print("CLUSTERS:", self.n_clusters)
#         colors = utils.uniquecolors(self.n_clusters)
#         for i, label in np.ndenumerate(self.labels_):
#             # desc = self.descriptors[i]
#             # y, x = desc[:2]
#             # print(i)
#             if label != -1:
#                 y, x = self.coordinates[i[0]]
#                 rgb_img[y, x] = colors[label]

#         utils.show(rgb_img)

# # main
#     cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#     cimg[:, :, :] = 0

#     for label, bbox_idx_orientations in connected_bounding_box_idxs.items():
#         n_connections = len(bbox_idx_orientations.keys())
#         print(n_connections)
#         # kmeans
#         # segmentator = SubLineSegmentator(n_clusters=n_connections, clusterer="kmeans")
#         # hdbscan

#         ccomp = np.uint8(connected_components.copy())
#         ccomp[ccomp != label] = 0
#         ccomp[ccomp == label] = 255

#         # TODO optimize
#         edges = cv.Canny(ccomp, 100, 200, None, 3)
#         # utils.show(edges)

#         segmentator = SubLineSegmentator(
#             cluster_selection_epsilon=50,  # max distance to count point to cluster
#             min_cluster_size=30,  # min num of clusters
#             # min_samples=50, # min samples to form a cluster
#             clusterer="hdbscan",
#         )
#         segmentator.fit_predict(edges)
#         segmentator.show_clusters(cimg)

# graph
# build graph
# graph = np.zeros((len(lookup), len(lookup)))
# for y, n1 in np.ndenumerate(lookup):
#     for x, n2 in np.ndenumerate(lookup):
#         if np.linalg.norm(n1 - n2) == 1:
#             graph[y, x] = 1
#             graph[x, y] = 1

# graph = sp.sparse.csr_matrix(graph)
# print("created graph")
# dist_matrix, predecessors = sp.sparse.csgraph.dijkstra(
#     csgraph=graph, directed=False, indices=0
# )

# print(ang)

# pos = k
# while


# bbox_idxs = bbox_idx_orientations.keys()
# main_bbox = bbox_idxs[0]
# for idx in bbox_idxs[1:]:
# find path between 0 and idx

# csr = sp.sparse.csr_matrix(ccomp)
# dist_matrix, predecessors = sp.sparse.csgraph.dijkstra(csgraph=csr, directed=False, indices=0)

# fast line detector
# fld = cv.ximgproc.createFastLineDetector()
# lines = fld.detect(np.uint8(connected_components))

# fld_img = fld.drawSegments(img, lines)
# utils.show(fld_img)

# SKELETONIZE
# def skeletonize(img, debug=False):
#     # TODO algoname
#     img = img.copy()
#     skel = img.copy()
#     skel[:, :] = 0
#     tmp = skel.copy()
#     cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))

#     while np.any(img):
#         tmp = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=cross)
#         tmp = cv.bitwise_not(tmp)
#         tmp = cv.bitwise_and(img, tmp)
#         skel = cv.bitwise_or(skel, tmp)
#         img = cv.erode(img, kernel=cross)

#     if debug:
#         utils.show(skel)

#     return skel

# def preprocess(img, debug=False):
#     # TODO move this shit to config!

#     # blur the shit out of the image
#     img = cv.GaussianBlur(img, (3, 3), 1)
#     if debug:
#         utils.show(img)

#     # bin threshold
#     # img = cv.adaptiveThreshold(
#     #     img,
#     #     maxValue=255,
#     #     adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#     #     thresholdType=cv.THRESH_BINARY_INV,
#     #     blockSize=11,
#     #     C=2,
#     # )
#     # if debug:
#     #     utils.show(img)

#     # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

#     low_thresh = 80
#     high_thresh = 2 * low_thresh
#     img = cv.Canny(img, low_thresh, high_thresh, None, 3)
#     if debug:
#         utils.show(img)

#     # # remove noise
#     # img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
#     # if debug:
#     #     utils.show(img)

#     return img


# def close_wire_holes(img, debug=False):
#     kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

#     # close holes in wire
#     img = cv.morphologyEx(
#         img,
#         cv.MORPH_CLOSE,
#         kernel,
#         iterations=3,
#     )
#     img = cv.dilate(img, kernel, iterations=1)
#     if debug:
#         utils.show(img)

#     return img
