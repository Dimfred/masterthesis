import pytest
import numpy as np
import itertools as it
import networkx as nx

from evaluation import Evaluator


def build_permutations(gt):
    rows = len(gt)
    permutation_idxs = it.permutations(range(rows))
    permutations = []
    for idxs in permutation_idxs:
        p = np.vstack([gt[idx] for idx in idxs])
        permutations.append(p)

    return permutations[:300]


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
    # fmt: off
    gt = np.array( [
    #    t  b  t  b  t  b
    #    1  1  2  2  3  3
        [1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
    ])
    # fmt: on
    gt = build_permutations(gt)
    return gt


# 1
def test_gt1_perfect_match(GT1):
    for gt1, gt2 in it.product(GT1, GT1):
        evaluator = Evaluator(gt1, gt2)
        res = evaluator.evaluate()

        assert res.TPS == 6
        assert res.TNS == 6
        assert res.FPS == 0
        assert res.FNS == 0


@pytest.fixture
def PRED1_C3t_missing(GT1):
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b
    #    1  1  2  2  3  3
        [1, 0, 0, 1, 0, 1],  # e1
        [0, 1, 1, 0, 0, 0],  # e2 without C3t
        [0, 0, 0, 0, 1, 0],  # C3t
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 2
def test_gt1_C3t_missing(GT1, PRED1_C3t_missing):
    for gt, pred in it.product(GT1, PRED1_C3t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 5
        assert res.TNS == 6
        assert res.FPS == 0
        assert res.FNS == 1


@pytest.fixture
def PRED1_C1bC2tC3t_missing(GT1):
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b
    #    1  1  2  2  3  3
        [1, 0, 0, 1, 0, 1],  # e1
        [0, 1, 0, 0, 0, 0],  # C1b
        [0, 0, 1, 0, 0, 0],  # C2t
        [0, 0, 0, 0, 1, 0],  # C3t
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 3
def test_gt1_C1bC2tC3t_missing(GT1, PRED1_C1bC2tC3t_missing):
    for (
        gt,
        pred,
    ) in it.product(GT1, PRED1_C1bC2tC3t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 4
        assert res.TNS == 6
        assert res.FPS == 0
        assert res.FNS == 2


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
    # fmt: off
    gt = np.array([
    #    t  b  t  b  t  b  t  b  t  b
    #    1  1  2  2  3  3  4  4  5  5
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],  # e2
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    gt = build_permutations(gt)
    return gt


# 4
def test_gt2_perfect_match(GT2):
    for gt1, gt2 in it.product(GT2, GT2):
        evaluator = Evaluator(gt1, gt2)
        res = evaluator.evaluate()

        assert res.TPS == 10
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 0


@pytest.fixture
def PRED2_H_bridge_split():
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b  t  b  t  b
    #    1  1  2  2  3  3  4  4  5  5
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # e2.1
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.2
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 5
def test_gt2_H_bridge_split(GT2, PRED2_H_bridge_split):
    for gt, pred in it.product(GT2, PRED2_H_bridge_split):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 9
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 1


@pytest.fixture
def PRED2_C4bC1b_missing():
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b  t  b  t  b
    #    1  1  2  2  3  3  4  4  5  5
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],  # e2.3 | C3t & C2t
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 6
def test_gt2_C4bC1b_missing(GT2, PRED2_C4bC1b_missing):
    for gt, pred in it.product(GT2, PRED2_C4bC1b_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 8
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 2


@pytest.fixture
def PRED2_C4bC1bC3tC2t_missing():
    # fmt: off
    pred = np.array([
        #    t  b  t  b  t  b  t  b  t  b
        #    1  1  2  2  3  3  4  4  5  5
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # e1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 7
# e2 missing
def test_gt2_C4bC1bC3tC2t_missing(GT2, PRED2_C4bC1bC3tC2t_missing):
    for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 7
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 3


# e2 and C5t
@pytest.fixture
def PRED2_C4bC1bC3tC2t_and_C5t_missing():
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b  t  b  t  b
    #    1  1  2  2  3  3  4  4  5  5
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # e1.1
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # e1.2 | C5t
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 8
def test_gt2_C4bC1bC3tC2t_and_C5t_missing(GT2, PRED2_C4bC1bC3tC2t_and_C5t_missing):
    for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_and_C5t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 6
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 4


# e2 and C5t and C1t
@pytest.fixture
def PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing():
    # fmt: off
    pred = np.array([
    #    t  b  t  b  t  b  t  b  t  b
    #    1  1  2  2  3  3  4  4  5  5
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # e1.1 | C4t
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # e1.2 | C5t
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # e1.3 | C1t
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # e2.1 | C1b
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # e2.2 | C4b
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # e2.3 | C2t
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2.4 | C3t
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],  # e3
    ])
    # fmt: on
    pred = build_permutations(pred)
    return pred


# 9
def test_gt2_C4bC1bC3tC2t_and_C5tC1tC4t_missing(
    GT2, PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing
):
    for gt, pred in it.product(GT2, PRED2_C4bC1bC3tC2t_and_C5tC1tC4t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 5
        assert res.TNS == 20
        assert res.FPS == 0
        assert res.FNS == 5
        # break


# TODO FP cases take an edge and split it up => use the permutations to rematch it
# remaining 1's are FPs

# TODO mixed case FP FN


# GT3:
#                    e1
#            |-------|-------|---------|
#           C1t     C3t     C5t        |
#           C1b     C3b     C5b        |
#        e2  |   e3  |   e4  |         |
#           C2t     C4t     C6t        |
#           C2b     C4b     C6b        |
#            |-------|-------|---------|
#                    e
#
@pytest.fixture
def GT3():
    # fmt: off
    gt = np.array([
    #    t, b, t, b, t, b, t, b, t, b, t, b
    #    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # e1
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # e2
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # e3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # e4
    ])
    # fmt: on

    gt = build_permutations(gt)
    return gt


# 10
def test_gt3_perfect(GT3):
    for gt1, gt2 in it.product(GT3, GT3):
        evaluator = Evaluator(gt1, gt2)
        res = evaluator.evaluate()

        assert res.TPS == 12
        assert res.TNS == 36
        assert res.FPS == 0
        assert res.FNS == 0


@pytest.fixture
def PRED3_e2e3_connected():
    # fmt: off
    pred = np.array([
    #    t, b, t, b, t, b, t, b, t, b, t, b
    #    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # e1
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # e2 and e3
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # e4
    ])
    # fmt: on

    pred = build_permutations(pred)
    return pred


# 11
def test_gt3_e2e3_connected(GT3, PRED3_e2e3_connected):
    for gt, pred in it.product(GT3, PRED3_e2e3_connected):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 12
        assert res.TNS == 36
        assert res.FPS == 1
        assert res.FNS == 0


@pytest.fixture
def PRED3_e2e3e4_connected():
    # fmt: off
    pred = np.array([
    #    t, b, t, b, t, b, t, b, t, b, t, b
    #    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # e1
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # e2 and e3 and e4
    ])
    # fmt: on

    pred = build_permutations(pred)
    return pred


# 12
def test_gt3_e2e3e4_connected(GT3, PRED3_e2e3e4_connected):
    for gt, pred in it.product(GT3, PRED3_e2e3e4_connected):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 12
        assert res.TNS == 36
        assert res.FPS == 2
        assert res.FNS == 0


@pytest.fixture
def PRED3_e2e3_connected_C3t_missing():
    # fmt: off
    gt = np.array([
    #    t, b, t, b, t, b, t, b, t, b, t, b
    #    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],  # e1
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # e2 and e3 without c3b
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # c3b
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # e4
    ])
    # fmt: on

    gt = build_permutations(gt)
    return gt

# 13
def test_gt3_e2e3_connected_C3t_missing(GT3, PRED3_e2e3_connected_C3t_missing):
    for gt, pred in it.product(GT3, PRED3_e2e3_connected_C3t_missing):
        evaluator = Evaluator(gt, pred)
        res = evaluator.evaluate()

        assert res.TPS == 11
        assert res.TNS == 36
        assert res.FPS == 1
        assert res.FNS == 1
