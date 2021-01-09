import numpy as np
import cv2 as cv
from numpy.core.numeric import cross
import cv2.ximgproc
import sklearn as sk
import sklearn.cluster

import sys
import math
import itertools as it

from typing import List

from ltbuilder import (
    Wire,
    Diode,
    Resistor,
    Capacitor,
    Inductor,
    Source,
    Current,
    Ground,
)
import utils
from utils import YoloBBox

from postprocessing import ORIENTATION, Utils


class LTBuilderAdapter:
    _GENERAL_ROTATION_MAPPING = {
        "bot": 0,
        "left": 90,
        "top": 180,
        "right": 270,
        "hor": 270,
        "ver": 0,
    }

    _GENERAL_CLS_MAPPING = {
        "diode": Diode,
        "cap": Capacitor,
        "ind": Inductor,
        "res": Resistor,
        "gr": Ground,
        "source": Source,
        "current": Current,
        # "node": Node
    }

    def __init__(self, classes_path, grid_size):
        self.classes = utils.load_yolo_classes(classes_path)
        self.grid_size = grid_size

        self.rotation_mapping = self._make_rotation_mapping()
        self.ltcomponent_mapping = self._make_ltcomponent_mapping()

        self.ltcomponents = []

    def make_ltcomponents(self, bboxes: List[YoloBBox]):
        # draw all symbols in the grid
        for bbox in bboxes:
            xm, ym = bbox.abs_mid()

            x_grid, y_grid = int(xm / self.grid_size), int(ym / self.grid_size)

            component = self.make_ltcomponent("TODO_NAME", x_grid, y_grid, bbox.label)
            self.ltcomponents.append(component)

    def make_wires(self, topology, connected_components, segmentation):
        self._create_subnodes(topology, connected_components, segmentation)

        # use just the topology to create a clean trace
        for label, sub_topology in topology.items():
            # 2 nodes at first
            if len(sub_topology) > 2:
                continue

            b1_idx, b2_idx = list(sub_topology.keys())
            b1_con, b2_con = sub_topology[b1_idx], sub_topology[b2_idx]

            lt1, lt2 = self.ltcomponents[b1_idx], self.ltcomponents[b2_idx]

            def is_vertical(orientation):
                return (
                    orientation == ORIENTATION.TOP.value
                    or orientation == ORIENTATION.BOTTOM.value
                )

            def is_horizontal(orientation):
                return (
                    orientation == ORIENTATION.LEFT.value
                    or orientation == ORIENTATION.RIGHT.value
                )

            b1_or, b2_or = b1_con.orientation, b2_con.orientation

            if is_vertical(b1_or):
                if is_horizontal(b2_or):
                    plt1 = lt1.top if b1_or == ORIENTATION.TOP else lt1.bottom
                    plt2 = lt2.left if b2_or == ORIENTATION.LEFT else lt2.right

                    xstart, _ = plt1
                    _, yend = plt2
                    mid = (xstart, yend)

                    w1 = Wire(plt1, mid)
                    w2 = Wire(mid, plt2)

                    self.ltcomponents.append(w1)
                    self.ltcomponents.append(w2)
                else:  # is_vertical(b2)
                    plt1 = lt1.top if b1_or == ORIENTATION.TOP else lt1.bottom
                    plt2 = lt2.top if b2_or == ORIENTATION.TOP else lt2.bottom

                    xlt1, ylt1 = plt1
                    xlt2, ylt2 = plt2

                    if b1_or == b2_or:
                        # create 2 fake edges
                        if b1_or == ORIENTATION.TOP:
                            offset, TODO_NAME = -1, min
                        else:
                            offset, TODO_NAME = 1, max

                        y_corner = TODO_NAME(ylt1, ylt2)
                        corner1 = (xlt1, y_corner + offset)
                        corner2 = (xlt2, y_corner + offset)

                        w1 = Wire(plt1, corner1)
                        w2 = Wire(corner1, corner2)
                        w3 = Wire(corner2, plt2)

                        self.ltcomponents.append(w1)
                        self.ltcomponents.append(w2)
                        self.ltcomponents.append(w3)
                    else:
                        if xlt1 == xlt2:
                            w = Wire(plt1, plt2)
                            self.ltcomponents.append(w)
                        else:
                            midy = min(ylt1, ylt2) + abs(int((ylt1 - ylt2) / 2))
                            mid1 = (xlt1, midy)
                            mid2 = (xlt2, midy)

                            w1 = Wire(plt1, mid1)
                            w2 = Wire(mid1, mid2)
                            w3 = Wire(mid2, plt2)

                            self.ltcomponents.append(w1)
                            self.ltcomponents.append(w2)
                            self.ltcomponents.append(w3)

            else:
                if is_vertical(b2_con.orientation):
                    plt1 = (
                        lt1.left
                        if b1_con.orientation == ORIENTATION.LEFT
                        else lt1.right
                    )
                    plt2 = (
                        lt2.top if b2_con.orientation == ORIENTATION.TOP else lt2.bottom
                    )

                    _, yend = plt1
                    xstart, _ = plt2
                    mid = (xstart, yend)

                    w1 = Wire(plt1, mid)
                    w2 = Wire(mid, plt2)

                    self.ltcomponents.append(w1)
                    self.ltcomponents.append(w2)
                else:
                    if b1_or == b2_or:
                        pass
                    else:
                        plt1 = lt1.left if b1_or == ORIENTATION.LEFT else lt1.right
                        plt2 = lt2.left if b2_or == ORIENTATION.LEFT else lt2.right

                        xlt1, ylt1 = plt1
                        xlt2, ylt2 = plt2

                        if ylt1 == ylt2:
                            w = Wire(plt1, plt2)
                            self.ltcomponents.append(w)
                        else:
                            midx = min(xlt1, xlt2) + abs(int((xlt1 - xlt2) / 2))
                            mid1 = (midx, ylt1)
                            mid2 = (midx, ylt2)

                            w1 = Wire(plt1, mid1)
                            w2 = Wire(mid1, mid2)
                            w3 = Wire(mid2, plt2)

                            self.ltcomponents.append(w1)
                            self.ltcomponents.append(w2)
                            self.ltcomponents.append(w3)

    def _create_subnodes(self, topology, connected_components, segmentation):
        # splits sub_topologies > 2 into pieces with len = 2
        new_topology = {}
        for label, sub_topology in topology.items():
            if len(sub_topology) == 2:
                new_topology[label] = sub_topology
                continue

            connected_component_orig = np.uint8(connected_components == label) * 255
            # DEBUG
            # utils.show(connected_component)

            connected_component = segmentation * (connected_component_orig == 255)
            connected_component = Utils.close(connected_component, 2)
            # utils.show(connected_component)

            # connected_component = connected_component_orig.copy()
            connected_component = cv.ximgproc.thinning(connected_component)
            # DEBUG
            # utils.show(connected_component)

            lines = cv.HoughLines(connected_component, 1, np.pi / 180, 20)

            suppressed_lines = []
            cimg = cv.cvtColor(connected_component, cv.COLOR_GRAY2BGR)
            corig = cv.cvtColor(connected_component_orig, cv.COLOR_GRAY2BGR)

            def make_line(rho, theta):
                a, b = math.cos(theta), math.sin(theta)
                x0, y0 = a * rho, b * rho
                p1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
                p2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))

                return p1, p2

            # fmt:off
            hor_kernel = np.array([
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0]~/scripts/run/rofipass
            ])
            min_line_len = max(*hor_kernel.shape)
            # fmt:on
            ver_kernel = hor_kernel.T

            hor_lines = cv.filter2D(
                np.uint8(connected_component == 255),
                -1,
                hor_kernel,
                borderType=cv.BORDER_CONSTANT,
            )

            ver_lines = cv.filter2D(
                np.uint8(connected_component == 255),
                -1,
                ver_kernel,
                borderType=cv.BORDER_CONSTANT,
            )
            hor_lines = np.uint8(hor_lines == min_line_len)
            ver_lines = np.uint8(ver_lines == min_line_len)

            # fmt:off
            ver_inc = [
                np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0]
                ]),
                np.array([
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ]),
            ]

            hor_inc = [
                np.array([
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0]
                ]),
                np.array([
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0]
                ]),
            ]
            # fmt:on

            for i in range(20):
                for inc in hor_inc:
                    hor_ext = cv.filter2D(
                        np.uint8(hor_lines), -1, inc, borderType=cv.BORDER_CONSTANT
                    )

                    hor_ext = np.argwhere(hor_ext == 2)
                    for y, x in hor_ext:
                        hor_lines[y, x - 1 : x + 2] = 1

                for inc in ver_inc:
                    ver_ext = cv.filter2D(
                        np.uint8(ver_lines), -1, inc, borderType=cv.BORDER_CONSTANT
                    )

                    ver_ext = np.argwhere(ver_ext == 2)
                    for y, x in ver_ext:
                        ver_lines[y - 1 : y + 2, x] = 1

                # DEBUG
                # utils.show(hor_lines * 255, ver_lines * 255)

            # hor_lines = Utils.dilate(hor_lines)
            # ver_lines = Utils.dilate(ver_lines)

            horver_inter = np.uint8(np.logical_and(hor_lines, ver_lines))
            inter_idxs = np.argwhere(horver_inter)

            nms_inter_groups = []
            for inter1, inter2 in it.combinations(inter_idxs, 2):
                if np.linalg.norm(inter1 - inter2) > 10:
                    continue

                inter1, inter2 = tuple(inter1), tuple(inter2)

                appended = False
                for group in nms_inter_groups:
                    if inter1 in group:
                        group.append(inter2)
                        appended = True
                        break
                    elif inter2 in group:
                        group.append(inter1)
                        appended = True
                        break

                if not appended:
                    nms_inter_groups.append([inter1, inter2])

            for inter in inter_idxs:
                inter = tuple(inter)
                exists = False
                for group in nms_inter_groups:
                    if inter in group:
                        exists = True
                        break

                if not exists:
                    nms_inter_groups.append([inter])

            nms_inter_idxs = []
            for group in nms_inter_groups:
                # TODO maybe mean or something
                nms_inter_idxs.append(group[0])
            inter_idxs = nms_inter_idxs

            # DEBUG
            for y, x in inter_idxs:
                cv.circle(cimg, (x, y), 5, (0, 0, 255))
            #     cv.circle(corig, (x, y), 5, (0, 0, 255))
            utils.show(cimg)

            node_bboxes = []
            # grow a bounding box around each point
            for y, x in inter_idxs:
                for i in range(100):
                    # TODO maybe but actually components are perfectly seperated
                    offset = 3

                    # TODO make cleverer; grow a square for now
                    tl, tr = (y - i, x - i), (y - i, x + i)
                    bl, br = (y + i, x - i), (y + i, x + i)

                    # if every coord is outside of the component
                    if (
                        not connected_component_orig[tl]
                        and not connected_component_orig[tr]
                        and not connected_component_orig[bl]
                        and not connected_component_orig[br]
                    ):
                        y, x = tl
                        tl = (x - offset, y - offset)

                        y, x = br
                        br = (x + offset, y + offset)
                        node_bboxes.append((tl, br))
                        break

            # DEBUG
            for p1, p2 in node_bboxes:
                corig = cv.rectangle(corig, p1, p2, (0, 0, 255), 1)
            utils.show(corig)

            # utils.show(connected_component, hor_lines, ver_lines, horver_inter)

            # inter_img = cimg.copy()
            # mean_img = cimg.copy()
            # for line in lines:
            #     rho, theta = line[0]

            #     # supressing non hor / ver lines
            #     angle = theta * 180 / np.pi
            #     if not ((-2 < angle and angle < 2) or (88 < angle and angle < 92)):
            #         continue

            #     suppressed_lines.append((rho, theta))

            #     p1, p2 = make_line(rho, theta)
            #     # DEBUG
            #     cv.line(cimg, p1, p2, (0, 0, 255), 1)

            # print(len(suppressed_lines))
            # thetas = np.array([theta for _, theta in suppressed_lines]).reshape(-1, 1)

            # # split into two groups
            # segmented_lines = sk.cluster.KMeans(n_clusters=2, random_state=0).fit(
            #     thetas
            # )

            # segment1, segment2 = [], []
            # for idx, (label, line) in enumerate(
            #     zip(segmented_lines.labels_, suppressed_lines)
            # ):
            #     _segment = segment1 if label else segment2
            #     _segment.append(line)

            # segment1, segment2 = np.vstack(segment1), np.vstack(segment2)

            # # CRAP
            # # segment1[:, 1] = np.mean(segment1[:, 1])
            # # segment2[:, 1] = np.mean(segment2[:, 1])

            # # for line1, line2 in zip(segment1, segment2):
            # #     rho, theta = line1
            # #     p1, p2 = make_line(rho, theta)
            # #     # DEBUG
            # #     cv.line(mean_img, p1, p2, (0, 0, 255), 1)

            # #     rho, theta = line2
            # #     p1, p2 = make_line(rho, theta)
            # #     # DEBUG
            # #     cv.line(mean_img, p1, p2, (0, 0, 255), 1)

            # # utils.show(mean_img)

            # w_img, h_img = connected_component.shape[:2]
            # inters = []
            # for line1 in segment1:
            #     for line2 in segment2:
            #         x, y = utils.hough_inter(line1, line2)
            #         # if points not in the img ignore them
            #         # if x < 0 or x > w_img or y < 0 or y > h_img:
            #         #     continue

            #         # supress the point if it is not inside connected_components
            #         # if connected_component[y, x] == 0:
            #         #     continue

            #         inters.append((x, y))

            # print("len(inters)\n{}".format(len(inters)))
            # # DEBUG
            # for inter in inters:
            #     cv.circle(inter_img, inter, 5, (0, 0, 255), 1)
            # utils.show(connected_component, cimg, inter_img)

            # harris = cv.cornerHarris(connected_component, 5, 3, 0.1)
            # harris = np.argwhere(harris > 0)

            # DEBUG
            # harris_show = cv.cvtColor(connected_component, cv.COLOR_GRAY2BGR)
            # for y, x in harris:
            #     cv.circle(harris_show, (x, y), 6, (0, 0, 255), 1)

            # DEBUG
            # utils.show(connected_component, cimg, harris_show)

    # def __create_subnodes(self, topology, connected_components):
    #     for label, sub_topology in topology.items():
    #         if len(sub_topology) == 2:
    #             continue

    #         bbox_idxs = list(sub_topology.keys())

    #         full_path = []
    #         first = True
    #         for b1_idx, b2_idx in utils.pairwise(bbox_idxs):
    #             angle_step = 20
    #             angle_threshold = 60

    #             b1_con, b2_con = sub_topology[b1_idx], sub_topology[b2_idx]

    #             if first:
    #                 # are in x, y
    #                 start, end = b1_con.coords, b2_con.coords

    #                 # trace the path
    #                 bfs = utils.BFS(connected_components, value=label)
    #                 # returns in x, y
    #                 path = bfs.fit(start, end)

    #             else:
    #                 # when we already have a path select the shortest by starting at
    #                 # every point in the old graph

    #                 # reduce search space to 100 euclidean nearest vertices
    #                 end = b2_con.coords
    #                 dist = lambda p1, p2: np.linalg.norm(np.array(np.array(p1) - np.array(p2)))
    #                 distances = [(dist(start, end), start) for start in full_path]
    #                 distances = sorted(distances, key=lambda tup: tup[0])
    #                 search_space = [start for _, start in distances[:80]]

    #                 shortest_start = None
    #                 shortest_path = None
    #                 shortest_len = sys.maxsize
    #                 for x, y in search_space:
    #                     start = (x, y)

    #                     bfs = utils.BFS(
    #                         connected_components, value=label, early_stop=shortest_len
    #                     )
    #                     path = bfs.fit(start, end)

    #                     # TODO this can fuck me
    #                     if path is not None and len(path) < shortest_len:
    #                         shortest_path = list(path)
    #                         shortest_len = len(path)
    #                         shortest_start = start

    #                 start = shortest_start
    #                 path = shortest_path

    #             if path is None:
    #                 print("PATH COULD NOT BE FOUND")
    #                 continue

    #             print("len(path)\n{}".format(len(path)))
    #             if len(path) <= angle_step:
    #                 print("ANGLE_STEP OUT OF BOUNDS")
    #                 angle_step = 1

    #             # extend the overall path
    #             full_path.extend(path)

    #             # make the path clean, extracting points of interest
    #             edges = [start]
    #             path = np.vstack(path)

    #             # walks along the wire checking angle between point and n points in the
    #             # future if angle changes > threshold we take a "snapshot"
    #             last_angle = utils.angle(path[0], path[angle_step])
    #             for p1, p2 in utils.pairwise(path, offset=angle_step):
    #                 current_angle = utils.angle(p1, p2)
    #                 if abs(current_angle - last_angle) > angle_threshold:
    #                     last_angle = current_angle
    #                     edges.append(p1)

    #             # the end point can go missing sometimes
    #             # if tuple(edges[-1]) != end[::-1]:
    #             #     # TODO useless will be replaced with real LTCoords
    #             #     edges.append(end[::-1])

    #             # normalize wire coords to LTCoords
    #             edges = [
    #                 (int(x / self.grid_size), int(y / self.grid_size)) for x, y in edges
    #             ]

    #             def get_lt_connection(lt_component, orientation):
    #                 if orientation == ORIENTATION.LEFT.value:
    #                     return lt_component.left
    #                 elif orientation == ORIENTATION.RIGHT.value:
    #                     return lt_component.right
    #                 elif orientation == ORIENTATION.TOP.value:
    #                     return lt_component.top
    #                 elif orientation == ORIENTATION.BOTTOM.value:
    #                     return lt_component.bottom

    #             if first:
    #                 edges.pop(0)

    #                 lt1_component, lt2_component = (
    #                     self.ltcomponents[b1_idx],
    #                     self.ltcomponents[b2_idx],
    #                 )

    #                 b1_orientation, b2_orientation = (
    #                     b1_con.orientation,
    #                     b2_con.orientation,
    #                 )
    #                 lt1_con = get_lt_connection(lt1_component, b1_orientation)
    #                 lt2_con = get_lt_connection(lt2_component, b2_orientation)

    #                 edges.insert(0, lt1_con)
    #                 edges.append(lt2_con)
    #             else:
    #                 lt2_component = self.ltcomponents[b2_idx]
    #                 b2_orientation = b2_con.orientation
    #                 lt2_con = get_lt_connection(lt2_component, b2_orientation)
    #                 edges.append(lt2_con)

    #             for (x1, y1), (x2, y2) in utils.pairwise(edges):
    #                 wire = Wire((x1, y1), (x2, y2))
    #                 self.ltcomponents.append(wire)

    #             first = False

    # INITIAL UGLY TRACE
    # for label, sub_topology in topology.items():
    #     bbox_idxs = list(sub_topology.keys())

    #     full_path = []
    #     first = True
    #     for b1_idx, b2_idx in utils.pairwise(bbox_idxs):
    #         angle_step = 10
    #         angle_threshold = 5

    #         b1_con, b2_con = sub_topology[b1_idx], sub_topology[b2_idx]

    #         if first:
    #             # are in x, y
    #             start, end = b1_con.coords, b2_con.coords

    #             # trace the path
    #             bfs = utils.BFS(connected_components, value=label)
    #             # returns in x, y
    #             path = bfs.fit(start, end)

    #         else:
    #             # when we already have a path select the shortest by starting at
    #             # every point in the old graph

    #             # reduce search space to 100 euclidean nearest vertices
    #             end = b2_con.coords
    #             dist = lambda p1, p2: np.linalg.norm(np.array(np.array(p1) - np.array(p2)))
    #             distances = [(dist(start, end), start) for start in full_path]
    #             distances = sorted(distances, key=lambda tup: tup[0])
    #             search_space = [start for _, start in distances[:80]]

    #             shortest_start = None
    #             shortest_path = None
    #             shortest_len = sys.maxsize
    #             for x, y in search_space:
    #                 start = (x, y)

    #                 bfs = utils.BFS(
    #                     connected_components, value=label, early_stop=shortest_len
    #                 )
    #                 path = bfs.fit(start, end)

    #                 # TODO this can fuck me
    #                 if path is not None and len(path) < shortest_len:
    #                     shortest_path = list(path)
    #                     shortest_len = len(path)
    #                     shortest_start = start

    #             start = shortest_start
    #             path = shortest_path

    #         if path is None:
    #             print("PATH COULD NOT BE FOUND")
    #             continue

    #         print("len(path)\n{}".format(len(path)))
    #         if len(path) <= angle_step:
    #             print("ANGLE_STEP OUT OF BOUNDS")
    #             angle_step = 1

    #         # extend the overall path
    #         full_path.extend(path)

    #         # make the path clean, extracting points of interest
    #         edges = [start]
    #         path = np.vstack(path)

    #         # walks along the wire checking angle between point and n points in the
    #         # future if angle changes > threshold we take a "snapshot"
    #         last_angle = utils.angle(path[0], path[angle_step])
    #         for p1, p2 in utils.pairwise(path, offset=angle_step):
    #             current_angle = utils.angle(p1, p2)
    #             if abs(current_angle - last_angle) > angle_threshold:
    #                 last_angle = current_angle
    #                 edges.append(p1)

    #         # the end point can go missing sometimes
    #         # if tuple(edges[-1]) != end[::-1]:
    #         #     # TODO useless will be replaced with real LTCoords
    #         #     edges.append(end[::-1])

    #         # normalize wire coords to LTCoords
    #         edges = [
    #             (int(x / self.grid_size), int(y / self.grid_size)) for x, y in edges
    #         ]

    #         def get_lt_connection(lt_component, orientation):
    #             if orientation == ORIENTATION.LEFT.value:
    #                 return lt_component.left
    #             elif orientation == ORIENTATION.RIGHT.value:
    #                 return lt_component.right
    #             elif orientation == ORIENTATION.TOP.value:
    #                 return lt_component.top
    #             elif orientation == ORIENTATION.BOTTOM.value:
    #                 return lt_component.bottom

    #         if first:
    #             edges.pop(0)

    #             lt1_component, lt2_component = (
    #                 self.ltcomponents[b1_idx],
    #                 self.ltcomponents[b2_idx],
    #             )

    #             b1_orientation, b2_orientation = (
    #                 b1_con.orientation,
    #                 b2_con.orientation,
    #             )
    #             lt1_con = get_lt_connection(lt1_component, b1_orientation)
    #             lt2_con = get_lt_connection(lt2_component, b2_orientation)

    #             edges.insert(0, lt1_con)
    #             edges.append(lt2_con)
    #         else:
    #             lt2_component = self.ltcomponents[b2_idx]
    #             b2_orientation = b2_con.orientation
    #             lt2_con = get_lt_connection(lt2_component, b2_orientation)
    #             edges.append(lt2_con)

    #         for (x1, y1), (x2, y2) in utils.pairwise(edges):
    #             wire = Wire((x1, y1), (x2, y2))
    #             self.ltcomponents.append(wire)

    #         first = False

    def make_ltcomponent(self, name, x, y, label_idx):
        if label_idx >= len(self.classes):
            print("LABEL OUT OF RANGE")
            print(label_idx)
            return None

        label_name = self.classes[label_idx]

        LTClass = self.ltcomponent_mapping[label_name]
        rotation = self.rotation_mapping[label_name]

        return LTClass(name, x, y, rotation)

    def _make_rotation_mapping(self):
        rotation_mapping = {}
        for cls_ in self.classes:
            *_, orientation = cls_.rsplit("_", 1)
            rotation = self._GENERAL_ROTATION_MAPPING[orientation]
            rotation_mapping[cls_] = rotation

        return rotation_mapping

    def _make_ltcomponent_mapping(self):
        ltcomponent_mapping = {}
        for cls_ in self.classes:
            name, *_ = cls_.split("_", 1)
            LTClass = self._GENERAL_CLS_MAPPING[name]
            ltcomponent_mapping[cls_] = LTClass

        return ltcomponent_mapping
