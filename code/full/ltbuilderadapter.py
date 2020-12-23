import numpy as np

import sys
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

from postprocessing import ORIENTATION


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

    def make_wires(self, topology, connected_components):

        for label, sub_topology in topology.items():
            bbox_idxs = list(sub_topology.keys())

            full_path = []
            first = True
            for b1_idx, b2_idx in utils.pairwise(bbox_idxs):
                angle_step = 10
                angle_threshold = 30

                b1_con, b2_con = sub_topology[b1_idx], sub_topology[b2_idx]

                if first:
                    # are in x, y
                    start, end = b1_con.coords, b2_con.coords

                    # trace the path
                    bfs = utils.BFS(connected_components, value=label)
                    # returns in x, y
                    path = bfs.fit(start, end)

                else:
                    # when we already have a path select the shortest by starting at
                    # every point in the old graph

                    # reduce search space to 100 euclidean nearest vertices
                    end = b2_con.coords
                    dist = lambda p1, p2: np.linalg.norm(np.array(np.array(p1) - np.array(p2)))
                    distances = [(dist(start, end), start) for start in full_path]
                    distances = sorted(distances, key=lambda tup: tup[0])
                    search_space = [start for _, start in distances[:80]]

                    shortest_start = None
                    shortest_path = None
                    shortest_len = sys.maxsize
                    for x, y in search_space:
                        start = (x, y)

                        bfs = utils.BFS(
                            connected_components, value=label, early_stop=shortest_len
                        )
                        path = bfs.fit(start, end)

                        # TODO this can fuck me
                        if path is not None and len(path) < shortest_len:
                            shortest_path = list(path)
                            shortest_len = len(path)
                            shortest_start = start

                    start = shortest_start
                    path = shortest_path

                if path is None:
                    print("PATH COULD NOT BE FOUND")
                    continue


                print("len(path)\n{}".format(len(path)))
                if len(path) <= angle_step:
                    print("ANGLE_STEP OUT OF BOUNDS")
                    angle_step = 1


                # extend the overall path
                full_path.extend(path)

                # make the path clean, extracting points of interest
                edges = [start]
                path = np.vstack(path)

                # walks along the wire checking angle between point and n points in the
                # future if angle changes > threshold we take a "snapshot"
                last_angle = utils.angle(path[0], path[angle_step])
                for p1, p2 in utils.pairwise(path, offset=angle_step):
                    current_angle = utils.angle(p1, p2)
                    if abs(current_angle - last_angle) > angle_threshold:
                        last_angle = current_angle
                        edges.append(p1)

                # the end point can go missing sometimes
                # if tuple(edges[-1]) != end[::-1]:
                #     # TODO useless will be replaced with real LTCoords
                #     edges.append(end[::-1])

                # normalize wire coords to LTCoords
                edges = [
                    (int(x / self.grid_size), int(y / self.grid_size)) for x, y in edges
                ]

                def get_lt_connection(lt_component, orientation):
                    if orientation == ORIENTATION.LEFT.value:
                        return lt_component.left
                    elif orientation == ORIENTATION.RIGHT.value:
                        return lt_component.right
                    elif orientation == ORIENTATION.TOP.value:
                        return lt_component.top
                    elif orientation == ORIENTATION.BOTTOM.value:
                        return lt_component.bottom

                if first:
                    edges.pop(0)

                    lt1_component, lt2_component = (
                        self.ltcomponents[b1_idx],
                        self.ltcomponents[b2_idx],
                    )

                    b1_orientation, b2_orientation = (
                        b1_con.orientation,
                        b2_con.orientation,
                    )
                    lt1_con = get_lt_connection(lt1_component, b1_orientation)
                    lt2_con = get_lt_connection(lt2_component, b2_orientation)

                    edges.insert(0, lt1_con)
                    edges.append(lt2_con)
                else:
                    lt2_component = self.ltcomponents[b2_idx]
                    b2_orientation = b2_con.orientation
                    lt2_con = get_lt_connection(lt2_component, b2_orientation)
                    edges.append(lt2_con)

                for (x1, y1), (x2, y2) in utils.pairwise(edges):
                    wire = Wire((x1, y1), (x2, y2))
                    self.ltcomponents.append(wire)

                first = False

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
