import numpy as np
import itertools as it

from pipeline.postprocessing import ORIENTATION, BBoxConnection
from config import config
import utils

classes = {
    0: "diode_left",
    1: "diode_top",
    2: "diode_right",
    3: "diode_bot",
    4: "res_de_hor",
    5: "res_de_ver",
    6: "cap_hor",
    7: "cap_ver",
    8: "gr_left",
    9: "gr_top",
    10: "gr_right",
    11: "gr_bot",
    12: "ind_de_hor",
    13: "ind_de_ver",
    14: "source_hor",
    15: "source_ver",
    16: "current_hor",
    17: "current_ver",
}

# idx: label line


class GroundTruth:
    def __init__(self, bboxes, adjacency_path):
        self.bboxes = bboxes
        # as adjacency matrix
        # pair of two cols = idx in self.labels
        self.adjacency = utils.EvalTopology.parse(adjacency_path)

        self.false_negatives = []

    def match(self, pred, threshold):
        matches = {idx: [] for idx, _ in enumerate(self.bboxes)}

        for gt_idx, gt_bbox in enumerate(self.bboxes):
            for pred_idx, pred_bbox in enumerate(pred.bboxes):
                if utils.calc_iou(gt_bbox.abs, pred_bbox.abs) < threshold:
                    continue

                matches[gt_idx].append(pred_idx)

        for gt_idx, match in matches.items():
            # unmatched gt
            if len(match) == 0:
                # we definetly have a false negative!
                if len(self.bboxes) > len(pred.bboxes):
                    # missing gt_bbox
                    gt_bbox = self.bboxes[gt_idx]
                    # add the missing bbox to the predictions
                    pred.bboxes.append(gt_bbox)
                    # the idx of the fn prediction added to the predictions to match
                    # the gt
                    fn_pred_idx = len(pred.bboxes) - 1
                    # add the fn to the gt_idx s.t. it can get synced with the gt
                    # in the next step
                    matches[gt_idx] = [fn_pred_idx]
                    # remember that we have added this one
                    self.false_negatives.append(gt_idx)
                    # add the fn to the topology of the prediciton

                    gt_topology_left_top = 2 * gt_idx
                    gt_topology_right_bot = gt_topology_left_top + 1

                    if np.any(self.adjacency[:, gt_topology_left_top]):
                        pred.add_false_negative(fn_pred_idx, ORIENTATION.LEFT.value)

                    if np.any(self.adjacency[:, gt_topology_right_bot]):
                        pred.add_false_negative(fn_pred_idx, ORIENTATION.RIGHT.value)

            elif len(match) > 1:
                raise ValueError("FALSE POSITIVE NOT YET HANDLED")
            # else == 1 correct

        gt_to_pred = {gt_idx: pred_idxs[0] for gt_idx, pred_idxs in matches.items()}
        pred_to_gt = {pred_idx: gt_idx for gt_idx, pred_idx in gt_to_pred.items()}

        # sync the prediction idxs with the idxs of the ground truth
        new_bbox_idxs = [gt_to_pred[gt_idx] for gt_idx in range(len(gt_to_pred))]
        pred.bboxes = [pred.bboxes[idx] for idx in new_bbox_idxs]

        new_topology = {}
        for edge_idx, edge in pred.topology.items():
            new_edge = {}
            for pred_idx, pred_component in edge.items():
                new_pred_idx = pred_to_gt[pred_idx]
                new_edge[new_pred_idx] = pred_component

            new_topology[edge_idx] = new_edge

        pred.topology = new_topology

        return pred


class Prediction:
    def __init__(self, bboxes, topology):
        self.bboxes = bboxes
        self.topology = topology
        self.added_false_negatives = []

    @property
    def adjacency(self):
        rows = []

        for edge_idx, connections in self.topology.items():
            edge = np.zeros((2 * len(self.bboxes),), dtype=np.uint8)
            for bbox_idx, connection in connections.items():
                orientation = connection.orientation
                if (
                    orientation == ORIENTATION.LEFT.value
                    or orientation == ORIENTATION.TOP.value
                ):
                    adjacency_idx = 2 * bbox_idx
                elif (
                    orientation == ORIENTATION.RIGHT.value
                    or orientation == ORIENTATION.BOTTOM.value
                ):
                    adjacency_idx = 2 * bbox_idx + 1
                else:
                    raise RuntimeError(f"Unknown Orientation {orientation}")

                edge[adjacency_idx] = 1

            rows.append(edge)

        # TODO why do I have to do that?
        # adjacency_mat = np.vstack([row for row in rows if np.any(row)])
        _adjacency = np.vstack(rows)

        return _adjacency

    def add_false_negative(self, fn_pred_idx, fn_orientation):
        # find an unused edge_idx
        for i in range(100000):
            if i not in self.topology:
                edge_idx = i
                break

        connection = BBoxConnection(None)
        connection.orientation = fn_orientation
        self.topology[edge_idx] = {fn_pred_idx: connection}


class Match:
    def __init__(self, gt_idx, pred_idx):
        self.gt_idx = gt_idx
        self.pred_idx = pred_idx


class MatchedIdxs(list):
    def __init__(self):
        super().__init__()

    def has_gt_idx(self, gt_idx):
        for match in self:
            if match.gt_idx == gt_idx:
                return True

        return False

    def has_pred_idx(self, pred_idx):
        for match in self:
            if match.pred_idx == pred_idx:
                return True

        return False


class Result:
    def __init__(self, TPS=0, TNS=0, FPS=0, FNS=0):
        self.TPS = TPS
        self.FPS = FPS
        self.TNS = TNS
        self.FNS = FNS

    def __add__(self, other):
        new_res = Result()
        new_res.TPS = self.TPS + other.TPS
        new_res.TNS = self.TNS + other.TNS
        new_res.FPS = self.FPS + other.FPS
        new_res.FNS = self.FNS + other.FNS

        return new_res

    def __str__(self):
        return "TPS: {}\nTNS: {}\nFPS: {}\nFNS: {}".format(
            self.TPS, self.TNS, self.FPS, self.FNS
        )


class Evaluator:
    def __init__(self, gt, pred):
        self.gt = gt.copy()
        self.pred = pred.copy()

    def evaluate(self):
        res = Result()

        matched_idxs = self.find_perfect_matches(self.gt, self.pred)

        unmatched_gt_idxs, unmatched_pred_idxs = self._get_unmatched(matched_idxs)

        # all matched correct
        res += self.count_perfect_matches(matched_idxs)
        if not unmatched_gt_idxs:
            return res

        while unmatched_gt_idxs:
            perm_res, matched_idxs = self.count_permutation_matches(matched_idxs)
            res += perm_res

            split_res, matched_idxs = self.count_split_matches(matched_idxs)
            res += split_res

            unmatched_gt_idxs, unmatched_pred_idxs = self._get_unmatched(matched_idxs)

        # TODO probably due to some FPs in detection which won't be able to be matched
        # against
        if unmatched_pred_idxs:
            print("Unmatched:", unmatched_pred_idxs)

        return res

    def find_perfect_matches(self, gt, pred):
        matched_idxs = MatchedIdxs()
        for gt_edge_idx, gt_edge in enumerate(gt):
            if matched_idxs.has_gt_idx(gt_edge_idx):
                continue

            for pred_edge_idx, pred_edge in enumerate(pred):
                if matched_idxs.has_pred_idx(pred_edge_idx):
                    continue

                if self._is_perfect_match(gt_edge, pred_edge):
                    matched_idxs.append(Match(gt_edge_idx, pred_edge_idx))
                    break

        return matched_idxs

    def count_perfect_matches(self, matched_idxs):
        TPS, TNS = 0, 0

        # count TPS, TNS in the perfect match edges
        for match in matched_idxs:
            TPS += (self.gt[match.gt_idx] == 1).sum() - 1
            TNS += (self.gt[match.gt_idx] == 0).sum() - 1

        return Result(TPS=TPS, TNS=TNS)

    def count_permutation_matches(self, matched_idxs):
        res = Result()
        unmatched_gt_idxs, unmatched_pred_idxs = self._get_unmatched(matched_idxs)

        # TODO put this in the inner clause
        # create all permutations of length 2 : len(unmatched_preds)

        found = False
        for r in range(2, len(unmatched_pred_idxs) + 1):
            if r == 7:
                raise ValueError(
                    "The length of the combinations exceeded 6. This should not happen."
                )

            idxs = it.combinations(unmatched_pred_idxs, r=r)
            for idx_comb in idxs:
                combination = np.vstack([self.pred[idx] for idx in idx_comb])
                combination = combination.sum(axis=0)
                combination[combination > 1] = 1

                for unmatched_gt_idx in unmatched_gt_idxs:
                    unmatched_gt = self.gt[unmatched_gt_idx]
                    if self._is_perfect_match(unmatched_gt, combination):
                        res.TPS += (unmatched_gt == 1).sum()
                        res.TNS += (unmatched_gt == 0).sum()

                        n_components_needed = len(idx_comb)
                        penalty = n_components_needed - 1

                        res.TPS -= penalty
                        res.FNS += penalty

                        # add to matched pairs
                        for idx in idx_comb:
                            matched_idxs.append(Match(unmatched_gt_idx, idx))

                        found = True
                        # TODO also break the above
                        break

                if found:
                    break

            if found:
                break

        return res, matched_idxs

    def count_split_matches(self, matched_idxs):
        res = Result()
        unmatched_gt_idxs, unmatched_pred_idxs = self._get_unmatched(matched_idxs)

        for unmatched_gt_idx in unmatched_gt_idxs:
            gt_edge = self.gt[unmatched_gt_idx]
            for unmatched_pred_idx in unmatched_pred_idxs:
                pred_edge = self.pred[unmatched_pred_idx]
                # check if the gt_edge is a sub graph of the prediction
                if self._is_sub_edge(gt_edge, pred_edge):
                    # we have a perfect submatch
                    # split the pred_edge into two parts
                    split_pred = pred_edge.copy()
                    split_pred[gt_edge == 1] = 0

                    # TODO replace the current prediction egde with the new one
                    self.pred[unmatched_pred_idx] = split_pred

                    res.TPS += (gt_edge == 1).sum()
                    res.TNS += (gt_edge == 0).sum()
                    res.FPS += 1

                    # TODO how to insert a partial match?
                    matched_idxs.append(Match(unmatched_gt_idx, -1))

                    unmatched_gt_idxs, unmatched_pred_idxs = self._get_unmatched(
                        matched_idxs
                    )

                    for unmatched_gt_idx in unmatched_gt_idxs:
                        gt_edge = self.gt[unmatched_gt_idx]
                        if self._is_perfect_match(gt_edge, split_pred):
                            res.TPS += (gt_edge == 1).sum()
                            res.TNS += (gt_edge == 0).sum()
                            matched_idxs.append(
                                Match(unmatched_gt_idx, unmatched_pred_idx)
                            )
                            return res, matched_idxs

                    rec_res, matched_idxs = self.count_split_matches(matched_idxs)
                    res += rec_res
                    return res, matched_idxs

        return res, matched_idxs

    def _is_sub_edge(self, e1, e2):
        # is e1 in e2?
        e1_idxs = np.argwhere(e1 == 1)
        e1_in_e2 = np.array([e2[e1_idx] for e1_idx in e1_idxs])
        return np.all(e1_in_e2)

    def _get_unmatched(self, matched_idxs):
        unmatched_gt_idxs = [
            idx for idx in range(len(self.gt)) if not matched_idxs.has_gt_idx(idx)
        ]
        unmatched_pred_idxs = [
            idx for idx in range(len(self.pred)) if not matched_idxs.has_pred_idx(idx)
        ]

        return unmatched_gt_idxs, unmatched_pred_idxs

    def _is_perfect_match(self, gt_edge, pred_edge):
        return np.all(gt_edge == pred_edge)


# OLD
# if len(unmatched_gt_idxs) == 1 and len(unmatched_pred_idxs) == 1:
#     partial_matched_idxs = MatchedIdxs()
#     partial_matched_idxs.append(
#         Match(unmatched_gt_idxs[0], unmatched_pred_idxs[0])
#     )
#     res += self.count_partial_matches(partial_matched_idxs)

# # prediction for an edge is missing
# elif unmatched_gt_idxs and not unmatched_pred_idxs:
#     for unmatched_gt_idx in unmatched_gt_idxs:
#         unmatched_gt = self.gt[unmatched_gt_idx]
#         res.TNS += (unmatched_gt == 0).sum()
#         res.FNS += (unmatched_gt == 1).sum()

# # gt for an edge is missing
# elif not unmatched_gt_idxs and unmatched_pred_idxs:
#     for unmatched_pred_idx in unmatched_pred_idxs:
#         unmatched_pred = self.pred[unmatched_pred_idx]
#         raise NotImplementedError("Case [no gt but pred] not implemented.")

# len(gt) < len(pred) => e.g. H bridge split in two
# test all remaining permutations
# elif unmatched_gt_idxs and unmatched_pred_idxs:

# def count_partial_matches(self, partial_matched_idxs):
#     res = Result()

#     for match in partial_matched_idxs:
#         unmatched_gt = self.gt[match.gt_idx]
#         unmatched_pred = self.pred[match.pred_idx]

#         eq = unmatched_gt == unmatched_pred

#         # count FPS, FNS in the edge with errors
#         neq_idxs = np.argwhere(eq == 0)
#         for neq_idx in neq_idxs:
#             neq_idx = neq_idx[0]
#             if unmatched_gt[neq_idx] == 1:  # => pred[neq_idx] == 0 !!
#                 res.FNS += 1
#             else:  # gt[neq_idx] == 0 => pred[neq_idx] == 1 !!
#                 res.FPS += 1

#         # count TPS, TNS in the edge with erros
#         eq_idxs = np.argwhere(eq == 1)
#         for eq_idx in eq_idxs:
#             eq_idx = eq_idx[0]
#             if unmatched_gt[eq_idx] == 1:
#                 res.TPS += 1
#             else:
#                 res.TNS += 1

#     return res
