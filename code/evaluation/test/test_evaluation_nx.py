import pytest
import numpy as np
import itertools as it
import networkx as nx

from evaluation import Evaluator


def p(*args):
    print("")
    print("-----------------------------")
    print(*args)


def build_permutations(gt):
    rows = len(gt)
    permutation_idxs = it.permutations(range(rows))
    permutations = []
    for idxs in permutation_idxs:
        p = np.vstack([gt[idx] for idx in idxs])
        permutations.append(p)

    return permutations


# GT1:
#        -----------
#       |          |
#      C1t         |
#      C1b         |
#      |           |
#   e2-------      | e1
#   |      |       |
#    C3t   C2t     |
#    C3b   C2b     |
#    |     |       |
#    ---------------


@pytest.fixture
def GT1():
    #     C1t, C1b, C2t, C2b, C3t, C3b
    #  e1
    #  e2

    #  c1con  (edge between top and bottom dunno whether it is needed)
    #  c2con
    #  c3con
    # fmt:off
    gt = nx.Graph()
    gt.add_edges_from(
        [
            # ("c1t", "c2b", "c3b"),
            # ("c1b", "c2t", "c3t")
            # graph
            # ("c1t", "c3b"), ("c1t", "c2b"), ("c3b", "c2b"), # e1
            # ("c1b", "c3t"), ("c1b", "c2t"), ("c3t", "c2t"), # e2
            # bipartite
            ("e1", "c1t"), ("e1", "c2b"), ("e1", "c3b"),
            ("e2", "c1b"), ("e2", "c2t"), ("e2", "c3t")
        ]
    )
    # fmt:on

    return gt


def test_gt1_perfect_match(GT1):
    d = nx.graph_edit_distance(GT1, GT1)
    assert d == 0.0


# 1 error
@pytest.fixture
def PRED1_C3t_missing(GT1):
    pred = nx.Graph()
    pred.add_nodes_from(["e1", "e2", "e3"], bipartite=0)
    pred.add_nodes_from(["c1t", "c1b", "c2t", "c2b", "c3t", "c3b"])

    # fmt: off
    pred.add_edges_from(
        [
            # normal graph
            # ("c1t", "c3b"), ("c1t", "c2b"), ("c3b", "c2b"), # e1
            # ("c1b", "c2t"), # e2
            # bipartite
            ("e1", "c1t"), ("e1", "c2b"), ("e1", "c3b"),
            ("e2", "c1b"), ("e2", "c2t"),
            ("e3", "c3t")
        ]
    )
    # fmt: on
    return pred


def test_gt1_C3t_missing(GT1, PRED1_C3t_missing):
    # d = nx.graph_edit_distance(GT1, PRED1_C3t_missing)
    # p(d)

    # e = nx.optimal_edit_paths(GT1, PRED1_C3t_missing)
    e = nx.optimize_edit_paths(GT1, PRED1_C3t_missing)
    for e_ in e:
        for e__ in e_:
            p(e__)
    # for e_ in e:
    #     for e__ in e_:
    #         p("MIN:", e__)

    # d = nx.graph_edit_distance(GT1, PRED1_C3t_missing)
    # p(d)


@pytest.fixture
def PRED1_C1bC2tC3t_missing(GT1):
    pred = nx.Graph()
    # fmt: off
    pred.add_edges_from(
        [
            ("c1t", "c3b"), ("c1t", "c2b"), ("c3b", "c2b"), # e1
        ]
    )
    # fmt: on
    return pred


def test_gt1_C1bC2tC3t_missing(GT1, PRED1_C1bC2tC3t_missing):
    # d = nx.graph_edit_distance(GT1, PRED1_C1bC2tC3t_missing)
    # p(d)
    pass


# GT2:
#                    e1
#            |--------|-----|
#           C4t      C1t    |
#           C4b  e2  C1b    |
#            |-------|      |
#           C3t     C2t     C5t
#           C3b     C2b     C5b
#            |-------|------|
#                    e3
#
@pytest.fixture
def GT2():
    # C1t, C1b ... C5t, C5b
    gt = np.array(
        [
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],  # e2
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    gt = build_permutations(gt)
    return gt


# def test_gt2_perfect_match(GT2):
#     for gt1, gt2 in it.product(GT2, GT2):
#         evaluator = Evaluator(gt1, gt2)
#         res = evaluator.evaluate()

#         assert res.TPS == 10
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 0

# def test_gt2_perfect_match(GT2):


@pytest.fixture
def PRED2_H_bridge_split():
    pred = np.array(
        [  # 1  1  2  2  3  3  4  4  5  5
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # e2.1
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.2
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    pred = build_permutations(pred)
    return pred


# def test_gt2_H_bridge_split(GT2, PRED2_H_bridge_split):
#     for gt, pred in it.product(GT2, PRED2_H_bridge_split):
#         evaluator = Evaluator(gt, pred)
#         res = evaluator.evaluate()

#         assert res.TPS == 8
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 2


@pytest.fixture
def PRED2_C4bC1b_missing():
    pred = np.array(
        #    t  b  t  b  t  b  t  b  t  b
        [  # 1  1  2  2  3  3  4  4  5  5
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # e2.3 | C3t & C2t
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    pred = build_permutations(pred)
    return pred


# def test_gt2_C4bC1b_missing(GT2, PRED2_C4bC1b_missing):
#     for gt, pred in it.product(GT2, PRED2_C4bC1b_missing):
#         evaluator = Evaluator(gt, pred)
#         res = evaluator.evaluate()

#         assert res.TPS == 7
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 3


@pytest.fixture
def PRED2_C4bC1bC3tC2t_missing():
    pred = np.array(
        #    t  b  t  b  t  b  t  b  t  b
        [  # 1  1  2  2  3  3  4  4  5  5
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    pred = build_permutations(pred)
    return pred


# e2 missing
# def test_gt2_C4bC1bC3tC2t_missing(GT2, PRED2_C4bC1bC3tC2t_missing):
#     for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_missing):
#         evaluator = Evaluator(gt, pred)
#         res = evaluator.evaluate()

#         assert res.TPS == 6
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 4


# e2 and C5t
@pytest.fixture
def PRED2_C4bC1bC3tC2t_and_C5t_missing():
    pred = np.array(
        #    t  b  t  b  t  b  t  b  t  b
        [  # 1  1  2  2  3  3  4  4  5  5
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # e1.1
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # e1.2 | C5t
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    pred = build_permutations(pred)
    return pred


# def test_gt2_C4bC1bC3tC2t_and_C5t_missing(GT2, PRED2_C4bC1bC3tC2t_and_C5t_missing):
#     for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_and_C5t_missing):
#         evaluator = Evaluator(gt, pred)
#         res = evaluator.evaluate()

#         assert res.TPS == 5
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 5


# e2 and C5t and C1t
@pytest.fixture
def PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing():
    pred = np.array(
        #    t  b  t  b  t  b  t  b  t  b
        [  # 1  1  2  2  3  3  4  4  5  5
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # e1.1 | C4t
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # e1.2 | C5t
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # e1.3 | C1t
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
        ]
    )
    pred = build_permutations(pred)
    return pred


# def test_gt2_C4bC1bC3tC2t_and_C5tC1tC4t_missing(
#     GT2, PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing
# ):
#     for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing):
#         evaluator = Evaluator(gt, pred)
#         res = evaluator.evaluate()

#         assert res.TPS == 3
#         assert res.TNS == 20
#         assert res.FPS == 0
#         assert res.FNS == 7
