"""Unit tests for the recall function.

The assignment asks specifically for tests on the recall implementation, and
the case that earns them is the tie at the k-th neighbour: the naive
set-intersection formula gets it wrong, ours should not.
"""

import numpy as np
import pytest

from annbench.recall import recall_at_k, true_distances


# ----------------------------------------------------------------------
# the basics
# ----------------------------------------------------------------------
def test_perfect_match_is_one():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    assert recall_at_k(gt_ids, gt_ids, gt_d, k=10) == 1.0


def test_order_within_top_k_does_not_matter():
    """recall@k is a set measure: a correct set in the wrong order is correct."""
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    shuffled = gt_ids[:, ::-1].copy()
    assert recall_at_k(shuffled, gt_ids, gt_d, k=10) == 1.0


def test_disjoint_is_zero():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    found = np.arange(100, 110)[None, :]
    found_d = np.full((1, 10), 1e6)
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == 0.0


def test_half_correct_is_half():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    found = np.array([[0, 1, 2, 3, 4, 100, 101, 102, 103, 104]])
    found_d = np.array([[1, 2, 3, 4, 5, 1e6, 1e6, 1e6, 1e6, 1e6]], dtype=float)
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == pytest.approx(0.5)


def test_mean_over_queries():
    gt_ids = np.tile(np.arange(10), (2, 1))
    gt_d = np.tile(np.arange(1, 11, dtype=float), (2, 1))
    found = np.vstack([np.arange(10), np.arange(100, 110)])
    found_d = np.vstack([np.arange(1, 11, dtype=float), np.full(10, 1e6)])
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == pytest.approx(0.5)
    per_q = recall_at_k(found, gt_ids, gt_d, found_d, k=10, per_query=True)
    assert per_q.tolist() == [1.0, 0.0]


# ----------------------------------------------------------------------
# ties: the reason this function is not a one-liner
# ----------------------------------------------------------------------
def test_tie_at_kth_neighbour_is_not_a_miss():
    """Ground truth ends with a tie; the index returns the other tied vector.

    The naive set intersection scores 0.9. It should be 1.0: both vectors are
    at exactly the 10th distance, so either is a correct 10th neighbour.
    """
    gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    gt_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 9.0]])
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])  # 99 ties with id 9
    found_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 9.0]])

    naive = len(set(found[0].tolist()) & set(gt_ids[0].tolist())) / 10
    assert naive == pytest.approx(0.9)
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == 1.0


def test_all_distances_tied_gives_full_recall():
    """Every database vector equidistant: any 10 are a correct answer."""
    gt_ids = np.arange(10)[None, :]
    gt_d = np.full((1, 10), 5.0)
    found = np.arange(50, 60)[None, :]
    found_d = np.full((1, 10), 5.0)
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == 1.0


def test_float_noise_within_rtol_counts_as_a_tie():
    """float32 accumulation jitters distances; rtol must absorb that."""
    gt_ids = np.arange(10)[None, :]
    gt_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]])
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 77]])
    found_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0 * (1 + 1e-7)]])
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10, rtol=1e-5) == 1.0


def test_just_outside_rtol_is_a_miss():
    """The tolerance must not be so loose that a genuine miss slips through."""
    gt_ids = np.arange(10)[None, :]
    gt_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]])
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 77]])
    found_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10.1]])
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10, rtol=1e-5) == pytest.approx(0.9)


def test_tie_test_ignored_when_no_true_distances_given():
    """Without exact distances the function must fall back to set recall,
    not quietly accept the index's own (possibly approximate) distances."""
    gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    gt_d = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 9.0]])
    found = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])
    assert recall_at_k(found, gt_ids, gt_d, None, k=10) == pytest.approx(0.9)


# ----------------------------------------------------------------------
# degenerate output from a real index
# ----------------------------------------------------------------------
def test_faiss_minus_one_padding_is_not_a_hit():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    found = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1]])
    found_d = np.array([[1, 2, 3, 4, 5, np.inf, np.inf, np.inf, np.inf, np.inf]])
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == pytest.approx(0.5)


def test_duplicate_ids_count_once():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    found = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    found_d = np.array([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]], dtype=float)
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == pytest.approx(0.2)


def test_recall_never_exceeds_one_when_zero_distance_duplicates_exist():
    """Near-duplicate databases can make many vectors tie at distance 0."""
    gt_ids = np.arange(10)[None, :]
    gt_d = np.zeros((1, 10))
    found = np.arange(1000, 1010)[None, :]
    found_d = np.zeros((1, 10))
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == 1.0


def test_fewer_than_k_results_returned():
    gt_ids = np.arange(10)[None, :]
    gt_d = np.arange(1, 11, dtype=float)[None, :]
    found = np.array([[0, 1, 2]])
    found_d = np.array([[1.0, 2.0, 3.0]])
    assert recall_at_k(found, gt_ids, gt_d, found_d, k=10) == pytest.approx(0.3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 10), int), np.zeros((3, 10), int), np.zeros((3, 10)), k=10)


# ----------------------------------------------------------------------
# true_distances
# ----------------------------------------------------------------------
def test_true_distances_matches_brute_force():
    rng = np.random.default_rng(0)
    train = rng.standard_normal((50, 7)).astype(np.float32)
    queries = rng.standard_normal((4, 7)).astype(np.float32)
    ids = rng.integers(0, 50, size=(4, 5))
    got = true_distances(queries, train, ids)
    want = np.array(
        [[float(((train[j] - queries[i]) ** 2).sum()) for j in ids[i]] for i in range(4)]
    )
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6)


def test_true_distances_marks_missing_ids_as_infinite():
    train = np.zeros((5, 3), dtype=np.float32)
    queries = np.ones((1, 3), dtype=np.float32)
    out = true_distances(queries, train, np.array([[0, -1]]))
    assert out[0, 0] == pytest.approx(3.0)
    assert np.isinf(out[0, 1])


def test_recall_against_exact_search_is_one():
    """End-to-end sanity: exact search must score 1.0 under our own metric."""
    faiss = pytest.importorskip("faiss")
    rng = np.random.default_rng(1)
    train = rng.standard_normal((2000, 16)).astype(np.float32)
    queries = rng.standard_normal((50, 16)).astype(np.float32)
    index = faiss.IndexFlatL2(16)
    index.add(train)
    d, i = index.search(queries, 10)
    exact = true_distances(queries, train, i)
    assert recall_at_k(i, i, d.astype(np.float64), exact, k=10) == 1.0
