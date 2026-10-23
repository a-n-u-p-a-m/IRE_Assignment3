"""Tests for the pieces between the measurement and the report.

These are cheap guards against the failure modes that would silently corrupt
numbers: a Pareto frontier that keeps dominated points, an interpolation that
extrapolates past the sweep, a results file that loses a row, a synthetic
generator that is not actually seeded, or an index wrapper that returns ids in
the wrong frame of reference.
"""

import json
import os

import numpy as np
import pytest

from annbench.datasets import Dataset, compute_ground_truth, split_queries, subsample
from annbench.frontier import pareto_mask, qps_at_recall
from annbench.indexes import divisors, make_index
from annbench.recall import recall_at_k, true_distances
from annbench.results import COLUMNS, ResultsWriter, make_row
from annbench.synthetic import make_subspace_dataset, relative_contrast


# ----------------------------------------------------------------------
# frontier
# ----------------------------------------------------------------------
def test_pareto_drops_dominated_points():
    recall = np.array([0.5, 0.7, 0.7, 0.9])
    qps = np.array([1000.0, 800.0, 500.0, 100.0])
    # (0.7, 500) is dominated by (0.7, 800); everything else is on the curve.
    assert pareto_mask(recall, qps).tolist() == [True, True, False, True]


def test_pareto_handles_a_single_point():
    assert pareto_mask([0.9], [10.0]).tolist() == [True]


def test_qps_at_recall_interpolates_between_bracketing_points():
    recall = np.array([0.90, 0.96])
    qps = np.array([1000.0, 100.0])
    q = qps_at_recall(recall, qps, 0.95)
    assert 100.0 < q < 1000.0
    # log-linear: 5/6 of the way from 0.90 to 0.96
    assert q == pytest.approx(10 ** (3 - (5 / 6) * 1), rel=1e-6)


def test_qps_at_recall_returns_nan_when_target_never_reached():
    assert np.isnan(qps_at_recall([0.5, 0.8], [100.0, 50.0], 0.95))


def test_qps_at_recall_floor_is_a_measured_point():
    q = qps_at_recall([0.90, 0.96], [1000.0, 100.0], 0.95, method="floor")
    assert q == 100.0


# ----------------------------------------------------------------------
# results store
# ----------------------------------------------------------------------
def test_results_roundtrip_and_resume(tmp_path):
    path = str(tmp_path / "results.csv")
    w = ResultsWriter(path)
    w.write(experiment="part2", dataset="sift", index_type="HNSW",
            params_json=json.dumps({"M": 16}), search_param=64, qps_median=123.0)
    assert w.has(experiment="part2", dataset="sift", index_type="HNSW",
                 params_json=json.dumps({"M": 16}), search_param=64)
    assert not w.has(experiment="part2", dataset="sift", index_type="HNSW",
                     params_json=json.dumps({"M": 16}), search_param=128)

    w2 = ResultsWriter(path)  # fresh writer sees the existing row
    assert w2.has(experiment="part2", dataset="sift", index_type="HNSW",
                  params_json=json.dumps({"M": 16}), search_param=64)

    from annbench.results import load_results
    df = load_results(path)
    assert len(df) == 1 and df["qps_median"].iloc[0] == 123.0


def test_force_truncates(tmp_path):
    path = str(tmp_path / "r.csv")
    ResultsWriter(path).write(experiment="part2", dataset="sift", index_type="HNSW")
    w = ResultsWriter(path, force=True)
    assert not w.has(experiment="part2", dataset="sift", index_type="HNSW")


def test_unknown_column_is_rejected():
    with pytest.raises(KeyError):
        make_row(not_a_column=1)
    assert set(make_row().keys()) == set(COLUMNS)


# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------
def _toy_dataset(n=500, d=8, nq=20, seed=0):
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, d)).astype(np.float32)
    queries = rng.standard_normal((nq, d)).astype(np.float32)
    ids, dists = compute_ground_truth(train, queries, k=10)
    return Dataset(name="toy", train=train, queries=queries, gt_ids=ids, gt_dists=dists)


def test_split_queries_takes_first_and_last_halves():
    ds = _toy_dataset(nq=20)
    tune = split_queries(ds, "tune", half=10)
    test = split_queries(ds, "test", half=10)
    assert np.array_equal(tune.queries, ds.queries[:10])
    assert np.array_equal(test.queries, ds.queries[10:])
    # the ground truth must be sliced with the queries, not left whole
    assert np.array_equal(test.gt_ids, ds.gt_ids[10:])


def test_subsample_recomputes_ground_truth_in_the_subset_frame():
    ds = _toy_dataset(n=500)
    sub = subsample(ds, 100, seed=0)
    assert sub.n == 100
    assert sub.gt_ids.max() < 100  # ids index into the subset, not the original
    # and the recomputed ground truth is genuinely exact for the subset
    exact = true_distances(sub.queries, sub.train, sub.gt_ids)
    assert np.allclose(exact[:, :10], sub.gt_dists[:, :10], rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------
# indexes
# ----------------------------------------------------------------------
def test_divisors_rejects_illegal_pq_m_for_glove():
    assert 16 in divisors(128) and 32 in divisors(128)
    assert 16 not in divisors(100)      # the GloVe trap
    assert {4, 10, 20, 25, 50} <= set(divisors(100))


def test_ivfpq_rejects_m_that_does_not_divide_d():
    ds = _toy_dataset(n=400, d=100)
    idx = make_index("IVF-PQ", 100, nlist=8, m=16, build_threads=1)
    with pytest.raises(ValueError, match="divide"):
        idx.build(ds.train)


@pytest.mark.parametrize(
    "kind,params,param_value",
    [
        ("Flat", {}, None),
        ("HNSW", {"M": 8, "ef_construction": 60}, 200),
        ("IVF-Flat", {"nlist": 16}, 16),
        ("IVF-PQ", {"nlist": 16, "m": 4, "nbits": 6}, 16),
        ("IVF-PQ", {"nlist": 16, "m": 4, "nbits": 6, "refine": True,
                    "refine_k": 100, "refine_at_k": 10}, 16),
    ],
)
def test_every_index_finds_its_own_neighbours(kind, params, param_value):
    """With the search parameter wide open, every index should be close to
    exact on easy low-dimensional data. This catches id-mapping bugs, metric
    mix-ups and wrappers that silently return nothing."""
    ds = _toy_dataset(n=1000, d=8, nq=25)
    idx = make_index(kind, ds.dim, seed=0, build_threads=1, **params)
    idx.build(ds.train)
    if param_value is not None:
        idx.set_search_param(param_value)
    ids, _ = idx.search(ds.queries, 10, threads=1)
    exact = true_distances(ds.queries, ds.train, ids)
    r = recall_at_k(ids, ds.gt_ids, ds.gt_dists, exact, k=10)
    floor = 0.99 if kind in ("Flat", "HNSW", "IVF-Flat") else 0.5
    if params.get("refine"):
        # re-ranking with the original vectors removes the quantisation
        # ceiling, so this configuration has no excuse for low recall
        floor = 0.95
    assert r >= floor, f"{kind} only reached recall {r:.3f}"


def test_search_one_and_batch_agree():
    ds = _toy_dataset(n=800, d=8, nq=10)
    idx = make_index("HNSW", ds.dim, M=8, ef_construction=60, build_threads=1)
    idx.build(ds.train)
    idx.set_search_param(100)
    batch_ids, _ = idx.search(ds.queries, 10, threads=1)
    for i in range(ds.n_queries):
        one_ids, _ = idx.search_one(ds.queries[i], 10)
        assert set(one_ids.tolist()) == set(batch_ids[i].tolist())


def test_ivfpq_refine_beats_plain_ivfpq():
    """Re-ranking exists to lift the quantisation ceiling; check it does.

    This also guards the construction order: the vectors must be added through
    the IndexRefineFlat wrapper, not to the base index before wrapping.
    """
    ds = _toy_dataset(n=2000, d=16, nq=50)
    scores = {}
    for refine in (False, True):
        params = {"nlist": 32, "m": 4, "nbits": 6}
        if refine:
            params |= {"refine": True, "refine_k": 100, "refine_at_k": 10}
        idx = make_index("IVF-PQ", ds.dim, seed=0, build_threads=1, **params)
        idx.build(ds.train)
        idx.set_search_param(32)
        ids, _ = idx.search(ds.queries, 10, threads=1)
        exact = true_distances(ds.queries, ds.train, ids)
        scores[refine] = recall_at_k(ids, ds.gt_ids, ds.gt_dists, exact, k=10)
    assert scores[True] > scores[False]
    assert scores[True] >= 0.99


def test_index_reports_plausible_size():
    ds = _toy_dataset(n=1000, d=8)
    flat = make_index("Flat", 8, build_threads=1)
    flat.build(ds.train)
    # 8 float32 per vector = 32 bytes, plus a small header
    assert 32 <= flat.bytes_per_vector() < 64


# ----------------------------------------------------------------------
# synthetic generator (Part 4)
# ----------------------------------------------------------------------
def test_generator_is_seeded():
    a = make_subspace_dataset(n=200, n_queries=10, dim=16, k_intrinsic=4, seed=7, kgt=5)
    b = make_subspace_dataset(n=200, n_queries=10, dim=16, k_intrinsic=4, seed=7, kgt=5)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.queries, b.queries)


def test_generator_varies_with_seed():
    a = make_subspace_dataset(n=200, n_queries=10, dim=16, k_intrinsic=4, seed=1, kgt=5)
    b = make_subspace_dataset(n=200, n_queries=10, dim=16, k_intrinsic=4, seed=2, kgt=5)
    assert not np.array_equal(a.train, b.train)


def test_ambient_dimension_is_constant_across_conditions():
    """The whole point of the design: only intrinsic dimension changes."""
    shapes = {
        k: make_subspace_dataset(n=100, n_queries=5, dim=32, k_intrinsic=k, seed=0, kgt=5).train.shape
        for k in (2, 8, 32)
    }
    assert len(set(shapes.values())) == 1


def test_signal_power_is_constant_across_intrinsic_dimension():
    """The invariant the whole Part 4 design rests on.

    Drawing coefficients as N(0, I_k) makes signal power grow as k, so sweeping
    k_intrinsic would also sweep the signal-to-noise ratio and the data scale -
    three variables, not one. Coefficients are drawn with variance 1/k instead,
    which pins E||signal||^2 at 1. This test is what catches a regression to
    the broken version.

    The invariant is on the mean *squared* norm (the power). The mean norm
    itself is slightly below 1 at small k by Jensen's inequality -- for k=2 it
    is Gamma(1.5) = 0.886 -- which is a property of the chi distribution, not a
    scaling bug.
    """
    power = {}
    for k in (2, 8, 32, 128):
        ds = make_subspace_dataset(n=4000, n_queries=50, dim=128, k_intrinsic=k,
                                   noise_frac=0.0, seed=0, kgt=10)
        power[k] = float((ds.train.astype(np.float64) ** 2).sum(axis=1).mean())
    assert max(power.values()) / min(power.values()) < 1.05, power
    assert all(0.95 < v < 1.05 for v in power.values()), power


def test_noise_frac_is_a_fraction_of_the_signal_norm():
    """noise_frac must mean the same thing in every condition."""
    for k in (2, 32, 128):
        clean = make_subspace_dataset(n=3000, n_queries=20, dim=128, k_intrinsic=k,
                                      noise_frac=0.0, seed=1, kgt=5)
        noisy = make_subspace_dataset(n=3000, n_queries=20, dim=128, k_intrinsic=k,
                                      noise_frac=0.25, seed=1, kgt=5)
        added = np.linalg.norm(noisy.train - clean.train, axis=1).mean()
        signal = np.linalg.norm(clean.train, axis=1).mean()
        assert abs(added / signal - 0.25) < 0.05, (k, added / signal)


def test_contrast_falls_monotonically_with_intrinsic_dimension():
    """The mechanism under test, and the prediction in hypothesis.md.

    With signal power held constant, more effective directions means distances
    concentrate and the nearest neighbour stands out less. The broken
    generator produced a non-monotonic curve peaking at k=8, so this asserts
    the whole ordering rather than just the endpoints.
    """
    contrasts = []
    for k in (2, 8, 32, 128):
        ds = make_subspace_dataset(n=20000, n_queries=200, dim=128, k_intrinsic=k,
                                   noise_frac=0.05, seed=0, kgt=10)
        contrasts.append(float(np.median(relative_contrast(ds, sample=5000, seed=0))))
    assert contrasts == sorted(contrasts, reverse=True), contrasts
    assert contrasts[0] > 2 * contrasts[-1], contrasts


def test_part4_respects_the_500k_cap():
    """The brief caps synthetic data at 500K vectors; the config must comply."""
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("full", "quick"):
        with open(os.path.join(here, "configs", f"{name}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["part4"]["n"] <= 500_000


def test_part4_configs_use_the_renamed_noise_key():
    """`noise` and `noise_frac` are different quantities; a stale config must
    not be silently reinterpreted."""
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("full", "quick"):
        with open(os.path.join(here, "configs", f"{name}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "noise_frac" in cfg["part4"], name
        assert "noise" not in cfg["part4"], name


def test_part4_has_a_control_condition():
    """The brief requires an unchanged control: k_intrinsic == dim."""
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("full", "quick"):
        with open(os.path.join(here, "configs", f"{name}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["part4"]
        assert cfg["dim"] in cfg["k_intrinsic"], name
        assert len(cfg["k_intrinsic"]) >= 4, name


def test_configs_use_legal_pq_m_values():
    """Guards the GloVe d=100 trap at config level, before any long run."""
    import yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dims = {"sift": 128, "glove": 100}
    for name in ("full", "quick"):
        with open(os.path.join(here, "configs", f"{name}.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for dataset, ms in cfg["part2"]["ivfpq"]["m"].items():
            for m in ms:
                assert dims[dataset] % m == 0, f"{name}: m={m} illegal for {dataset}"
