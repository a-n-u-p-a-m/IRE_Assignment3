"""Seeded synthetic data for the Part 4 experiment.

The question studied in Part 4
------------------------------
*How does the **intrinsic** dimensionality of the data — the number of
directions the vectors actually vary in — change what each index can do, at a
fixed ambient dimension?*

Why that framing rather than "vary the dimension"
-------------------------------------------------
The obvious experiment is to generate uniform random vectors at d = 8, 32, 128,
512 and watch recall fall as the distances concentrate.  It is confounded three
ways: changing d also changes bytes per vector, the cost of one distance
computation, and the legal PQ configurations.  A speed difference between d = 8
and d = 512 cannot be attributed to geometry, because the arithmetic changed
too.

So instead the **ambient dimension stays at 128 for every condition** and the
data is generated on a random k-dimensional linear subspace of that space, plus
a small isotropic noise floor.

Normalising the signal power — and why the obvious version is wrong
-------------------------------------------------------------------
The natural way to write this generator is

    X = C B^T + sigma * E,     C ~ N(0, I_k),  B orthonormal d x k    (WRONG)

and it is subtly broken.  Because ``B`` is orthonormal, ``E||C B^T||^2 = k``,
so the **signal power grows with k** while the noise stays at ``sigma^2 d``.
Sweeping k then changes three things at once: the number of effective
directions (intended), the signal-to-noise ratio (an 8x swing from k=2 to
k=128), and the overall scale of the data.  That breaks the brief's
one-variable-at-a-time rule, and it is not academic — it inverts the result.
At k=2 the true in-subspace neighbour separation falls *below* the noise floor,
so the nearest neighbour is decided by the noise rather than by the subspace.
Relative contrast then comes out non-monotonic (it peaks at k=8), and IVF-PQ
scores *worse* at k=2 than at k=8, because the sub-quantisers faithfully encode
the 2-d structure and discard exactly the noise that carried the ranking.

So the coefficients are drawn with variance ``1/k``:

    X = C B^T + (eta / sqrt(d)) * E,   C ~ N(0, I_k / k),  E ~ N(0, I_d)

which gives ``E||C B^T||^2 = 1`` for every k, and makes ``eta`` the norm of the
noise **as a fraction of the signal norm** — a dimensionless quantity that
means the same thing in every condition.  Every condition now has identical
vector size, distance cost, index memory, PQ layout, signal power and
signal-to-noise ratio.  The only thing that changes is how many directions the
data lives in.  ``k_intrinsic == dim`` is the **control**: the subspace is the
whole space, so the generator degenerates to isotropic Gaussian data.

The noise floor itself matters: at ``eta = 0`` and k = 2 the data is a
genuinely rank-2 sheet, and the comparison would be about rank rather than
about *effective* dimensionality.  A small ``eta`` keeps every condition
full-rank.

One honest caveat, which the report should state rather than hide: at the
lowest k the noise floor still bounds the nearest-neighbour distance, because
200 000 points packed into 2 effective dimensions sit far closer together than
any noise floor that leaves the data full-rank.  At ``eta = 0.05``, k = 2 and
n = 200 000, the mean NN distance is ~0.9x the noise floor; by k = 8 it is
~3.9x and the signal dominates.  This is a property of low-dimensional data
with many points, not an artefact that can be tuned away.

Everything is driven by a ``numpy`` ``default_rng(seed)``, so a given seed
reproduces byte-identical data on any machine.
"""

from __future__ import annotations

import numpy as np

from .datasets import Dataset, compute_ground_truth

__all__ = ["make_subspace_dataset", "relative_contrast"]


def _orthonormal_basis(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random d x k matrix with orthonormal columns (Haar distributed)."""
    a = rng.standard_normal((d, k))
    q, r = np.linalg.qr(a)
    # Fix the sign convention so the basis is a deterministic function of the
    # seed across LAPACK versions.
    q *= np.sign(np.diag(r))
    return q[:, :k]


def make_subspace_dataset(
    n: int = 200_000,
    n_queries: int = 1_000,
    dim: int = 128,
    k_intrinsic: int = 32,
    noise_frac: float = 0.05,
    seed: int = 0,
    kgt: int = 100,
    name: str | None = None,
) -> Dataset:
    """Vectors on a random ``k_intrinsic``-dimensional subspace of ``dim``.

    Signal power is held at 1 for every ``k_intrinsic`` (see the module
    docstring), so ``noise_frac`` is the noise norm as a fraction of the signal
    norm and means the same thing in every condition.

    Queries are drawn from the *same* distribution as the database (this
    experiment varies geometry, not drift) but are not members of it.  Ground
    truth is computed exactly by brute force.
    """
    if not 1 <= k_intrinsic <= dim:
        raise ValueError(f"k_intrinsic must be in [1, {dim}], got {k_intrinsic}")
    if noise_frac < 0:
        raise ValueError(f"noise_frac must be >= 0, got {noise_frac}")

    rng = np.random.default_rng(seed)
    basis = _orthonormal_basis(dim, k_intrinsic, rng)

    total = n + n_queries
    # Variance 1/k per coefficient, so E||signal||^2 = 1 regardless of k.
    coeffs = (
        rng.standard_normal((total, k_intrinsic)) / np.sqrt(k_intrinsic)
    ).astype(np.float32)
    x = coeffs @ basis.T.astype(np.float32)
    if noise_frac > 0:
        # Scaled by 1/sqrt(d) so E||noise||^2 = noise_frac^2, i.e. the noise
        # norm is noise_frac times the signal norm in every condition.
        x += (
            (noise_frac / np.sqrt(dim)) * rng.standard_normal((total, dim))
        ).astype(np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)

    train = x[:n]
    queries = np.ascontiguousarray(x[n:])
    gt_ids, gt_dists = compute_ground_truth(train, queries, k=kgt)

    return Dataset(
        name=name or f"subspace-d{dim}-k{k_intrinsic}",
        train=train,
        queries=queries,
        gt_ids=gt_ids,
        gt_dists=gt_dists,
        normalized=False,
        note=(
            f"synthetic: n={n}, dim={dim}, k_intrinsic={k_intrinsic}, "
            f"noise_frac={noise_frac}, unit signal power, seed={seed}"
            + (" (CONTROL: isotropic)" if k_intrinsic == dim else "")
        ),
    )


def relative_contrast(
    ds: Dataset,
    sample: int = 10_000,
    seed: int = 0,
    k: int = 1,
) -> np.ndarray:
    """Per-query relative contrast: mean distance / distance to the k-th NN.

    A standard hardness measure.  Values near 1 mean "everything is about
    equally far away", which is the regime where partition-based indexes stop
    being able to prune.  Reported in Part 4 to show that the generator really
    did move the data into that regime, and reusable for the Part 3
    query-difficulty study.

    Distances are compared in the same squared-L2 unit as everything else; the
    square root is taken so the ratio has the usual scale.
    """
    rng = np.random.default_rng(seed)
    m = min(sample, ds.n)
    idx = rng.choice(ds.n, size=m, replace=False)
    ref = ds.train[idx].astype(np.float64)

    q = ds.queries.astype(np.float64)
    # ||q - r||^2 = ||q||^2 - 2 q.r + ||r||^2
    qn = (q * q).sum(1)[:, None]
    rn = (ref * ref).sum(1)[None, :]
    d2 = np.maximum(qn - 2.0 * q @ ref.T + rn, 0.0)
    mean_d = np.sqrt(d2).mean(axis=1)
    nn_d = np.sqrt(ds.gt_dists[:, k - 1])
    return mean_d / np.maximum(nn_d, 1e-12)
