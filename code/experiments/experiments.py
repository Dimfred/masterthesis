from random import randint
import utils

import cv2 as cv
import numpy as np
import math
import random
from numba import njit
import os
from pathlib import Path

data = Path("data")
labeled = data / "labeled"

img_path = labeled / "03_02.png"
img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)

utils.show(img, 416)

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

    ccomp_idx = 10
    ccomp = np.uint8(connected_components.copy())
    # TODO do in 1 step
    ccomp[ccomp != 10] = 0
    ccomp[ccomp == 10] = 255
    # ccomp = cv.dilate(ccomp, )
    # utils.show(ccomp)

    # skeleton = skeletonize(ccomp, debug=dskel)

    # canny

    edges = cv.Canny(ccomp, 100, 200, None, 3)
    # utils.show(edges)

    # fmt:off
    vertical_kernels = np.array([
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]),
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0]
        ]),
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
    ])
    horizontal_kernels = np.vstack([vk.T for vk in vertical_kernels])

    # fmt:on

    binarized = edges.copy()
    binarized[binarized > 0] = 1

    out = np.zeros_like(edges)

    vidxs = []
    for vk in vertical_kernels:
        conv = cv.filter2D(binarized, -1, vk)
        ys, xs = np.where(conv == 4)
        vidxs += [(y, x) for y, x in zip(ys, xs)]

    vidxs = np.vstack(vidxs)
    mean = vidxs.mean(axis=0)
    std = vidxs.std(axis=0)

    # filter all outliers where x is too far away
    ymean, xmean = mean
    vidxs = np.vstack([(y, x) for y, x in vidxs if abs(x - xmean) < 30])

    for y, x in vidxs:
        out[y, x] = 255
    utils.show(out)

    ys, xs = vidxs[:, 0], vidxs[:, 1]

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    xmin, xmax = int(xmean), int(xmean)

    out = np.zeros_like(edges)
    cv.line(out, (xmin, ymin), (xmax, ymax), 255, 3, cv.LINE_8)
    utils.show(out)

    out = np.zeros_like(edges)
    for vk in vertical_kernels:
        hk = vk.T
        conv = cv.filter2D(binarized, -1, hk)
        out[np.where(conv == 4)] == 255
    utils.show(out)









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
    for x1, y1, x2, y2, cls_, _ in abs_bounding_boxes:
        xm, ym = int(x1 + 0.5 * x2), int(y1 + 0.5 * y2)

        x_grid, y_grid = int(xm / grid_size), int(ym / grid_size)

        component = LTBuilder.build("TODO_NAME", x_grid, y_grid, cls_)
        ltcomponents.append(component)

    # draw wires
    for _, connected_bbox_idxs in connected_bounding_box_idxs.items():
        for idx1, idx2 in it.combinations(connected_bbox_idxs, 2):
            c1, c2 = ltcomponents[idx1], ltcomponents[idx2]
            # TODO only when edges active
            if c1 is None or c2 is None:
                print("is none")
                continue

            b1, b2 = abs_bounding_boxes[idx1], abs_bounding_boxes[idx2]

            if c1.is_horizontal and c2.is_horizontal:
                # b1 if more left
                if b1[0] < b2[0]:
                    wire = Wire(c1.right, c2.left)
                else:
                    wire = Wire(c1.left, c2.right)

            elif c1.is_vertical and c2.is_vertical:
                # b1 if is above
                if b1[1] < b2[1]:
                    wire = Wire(c1.bottom, c2.top)
                else:
                    wire = Wire(c1.top, c2.bottom)

            elif c1.is_vertical and c2.is_horizontal:
                # b1 is above
                if b1[1] < b2[1]:
                    # b1 is more left
                    if b1[0] < b2[0]:
                        wire = Wire(c1.bottom, c2.left)
                    # b1 is more right
                    else:
                        wire = Wire(c1.bottom, c2.right)
                else:
                    if b1[0] < b2[0]:
                        wire = Wire(c1.top, c2.left)
                    # b1 is more right
                    else:
                        wire = Wire(c1.top, c2.right)

            elif c1.is_horizontal and c2.is_vertical:
                # b1 is above
                if b1[1] < b2[1]:
                    # b1 is more left
                    if b1[0] < b2[0]:
                        wire = Wire(c1.right, c2.top)
                    # b1 is more right
                    else:
                        wire = Wire(c1.left, c2.bottom)
                else:
                    # b1 is more left
                    if b1[0] < b2[0]:
                        wire = Wire(c1.right, c2.bottom)
                    # b1 is more right
                    else:
                        wire = Wire(c1.left, c2.top)
            else:
                print("c1.is_horizonal\n{}".format(c1.is_horizontal))
                print("c2.is_vertical\n{}".format(c2.is_vertical))
                wire = None
                print("THIS CASE SHOULD NOTHAPPEN")

            ltcomponents.append(wire)

    # hough lines p
    lines = cv2.HoughLinesP(edges, 1, math.pi/100, 2, None, 20, 3)

    for line in lines:
        line = line[0]
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        print(p1, p2)
        cimg = cv.line(cimg, p1, p2, (0, 0, 255), 1)
        utils.show(cimg)

    # HARRIS
    # harris = cv.cornerHarris(np.float32(ccomp), 2, 3, 0.04)
    # utils.show(harris)

### Clustering


class SubLineSegmentator:
    def __init__(self, *args, clusterer="kmeans", **kwargs):
        self._clusterer = clusterer
        if clusterer == "kmeans":
            self.clusterer = sk.cluster.KMeans(*args, **kwargs)
        elif clusterer == "hdbscan":
            self.clusterer = hdbscan.HDBSCAN(*args, **kwargs)

    def fit_predict(self, edges):
        self.descriptors = self.create_descriptors(edges)
        self.labels_ = self.clusterer.fit_predict(self.descriptors)
        return self.labels_

    def filter_diagonal_lines(self, edges):
        # fmt:off
        vertical_kernels = np.array([
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ]),
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ]),
            np.array([
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ]),
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0]
            ]),
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
        ])
        horizontal_kernels = np.vstack([[vk.T] for vk in vertical_kernels])
        # fmt:on

        binarized = edges.copy()
        binarized[binarized == 255] = 1

        idxs = []
        for vk in vertical_kernels:
            corr = cv.filter2D(binarized, -1, vk)
            ys, xs = np.where(corr == 4)
            idxs += [(y, x) for y, x in zip(ys, xs)]

        for hk in horizontal_kernels:
            corr = cv.filter2D(binarized, -1, hk)
            ys, xs = np.where(corr == 4)
            idxs += [(y, x) for y, x in zip(ys, xs)]

        idxs = np.vstack(idxs)
        ys, xs = idxs[:, 0], idxs[:, 1]

        filtered = np.zeros_like(edges)
        filtered[ys, xs] = 255

        return filtered

    def create_descriptors(self, edges):
        # filtered = self.filter_diagonal_lines(edges)
        filtered = edges
        utils.show(filtered)

        edge_idxs = np.where(filtered == 255)
        # TODO slow
        descriptors = []
        self.coordinates = []
        for y, x in zip(*edge_idxs):
            # TODO outofbounds
            # extract 3x3 around target point
            orientation_desc = edges[y - 1 : y + 2, x - 1 : x + 2]
            # desc = np.append([y, x], orientation_desc.flatten())
            self.coordinates.append((y, x))
            desc = np.array([y, x])
            # desc = np.array([y**2, x**2])
            # desc = [np.linalg.norm(np.array([y, x]))]
            desc = np.append(desc, orientation_desc.flatten())
            descriptors.append(desc)

        return np.vstack(descriptors)

    def show_clusters(self, rgb_img):
        rgb_img = rgb_img.copy()

        if self._clusterer == "kmeans":
            self.n_clusters = self.clusterer.n_clusters
        elif self._clusterer == "hdbscan":
            self.n_clusters = len(set(self.labels_)) - 1

        print("CLUSTERS:", self.n_clusters)
        colors = utils.uniquecolors(self.n_clusters)
        for i, label in np.ndenumerate(self.labels_):
            # desc = self.descriptors[i]
            # y, x = desc[:2]
            # print(i)
            if label != -1:
                y, x = self.coordinates[i[0]]
                rgb_img[y, x] = colors[label]

        utils.show(rgb_img)

# main
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cimg[:, :, :] = 0

    for label, bbox_idx_orientations in connected_bounding_box_idxs.items():
        n_connections = len(bbox_idx_orientations.keys())
        print(n_connections)
        # kmeans
        # segmentator = SubLineSegmentator(n_clusters=n_connections, clusterer="kmeans")
        # hdbscan

        ccomp = np.uint8(connected_components.copy())
        ccomp[ccomp != label] = 0
        ccomp[ccomp == label] = 255

        # TODO optimize
        edges = cv.Canny(ccomp, 100, 200, None, 3)
        # utils.show(edges)

        segmentator = SubLineSegmentator(
            cluster_selection_epsilon=50,  # max distance to count point to cluster
            min_cluster_size=30,  # min num of clusters
            # min_samples=50, # min samples to form a cluster
            clusterer="hdbscan",
        )
        segmentator.fit_predict(edges)
        segmentator.show_clusters(cimg)

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
