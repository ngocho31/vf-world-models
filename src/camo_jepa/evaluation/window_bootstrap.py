"""Window-level aggregation and bootstrap confidence intervals (WM-12).

Scores are stored per window as ``[sum, count]`` pairs. A pooled value is the
token-weighted ratio ``sum(sums) / sum(counts)``. Intervals resample *windows*,
never tokens: tokens of one window share the scene, and neighbouring windows
share frames, so two resampling schemes are reported:

* ``ci95``      - circular block bootstrap over consecutive windows
  (``block`` windows per block), which keeps that serial dependence;
* ``ci95_iid``  - windows resampled independently, for reference only.

A paired difference is the difference of the two pooled (token-weighted)
means, the same weighting as the pooled table, with both sides drawn from the
same resampled windows. An interval is ``None`` when there are too few windows
to resample (a block bootstrap needs at least two blocks).
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_indices(n_windows: int, block: int, n_boot: int, seed: int) -> np.ndarray | None:
    """``[n_boot, n_windows]`` window indices from a circular block bootstrap (block=1: i.i.d.).

    Returns ``None`` below two blocks' worth of windows: every resample would then
    be a rotation of the same windows and the interval would have zero width.
    """
    if n_windows < 1 or block < 1 or n_boot < 1:
        raise ValueError("n_windows, block and n_boot must be positive")
    if n_windows < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_windows / block)
    starts = rng.integers(0, n_windows, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)) % n_windows
    return indices.reshape(n_boot, -1)[:, :n_windows]


def _interval(samples: np.ndarray) -> list[float] | None:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [2.5, 97.5])
    return [float(low), float(high)]


def ratio_summary(pairs: np.ndarray, resamples: dict[str, np.ndarray]) -> dict:
    """Pooled ratio of ``pairs`` ``[n_windows, 2]`` (sum, count) with bootstrap intervals."""
    sums, counts = pairs[:, 0].astype(np.float64), pairs[:, 1].astype(np.float64)
    total = counts.sum()
    out = {
        "mean": float(sums.sum() / total) if total > 0 else None,
        "n_tokens": int(total),
        "n_windows": int((counts > 0).sum()),
    }
    for name, idx in resamples.items():
        if idx is None:
            out[name] = None
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
        out[name] = _interval(boot)
    return out


def window_means(pairs: np.ndarray) -> np.ndarray:
    """Per-window mean ``sum / count`` (NaN where the window has no token)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pairs[:, 1] > 0, pairs[:, 0] / pairs[:, 1], np.nan)


def contrast_summary(terms: list[tuple[float, np.ndarray]], resamples: dict[str, np.ndarray | None]) -> dict:
    """Linear contrast ``sum_k c_k * pooled_k`` of token-weighted pooled means over the same windows.

    ``terms`` is ``[(c_k, pairs_k)]``; each resample draws one set of windows for every term.
    """
    sums = [pairs[:, 0].astype(np.float64) for _, pairs in terms]
    counts = [pairs[:, 1].astype(np.float64) for _, pairs in terms]
    coefs = [coef for coef, _ in terms]
    complete = all(count.sum() > 0 for count in counts)
    out = {
        "mean": float(sum(c * s.sum() / n.sum() for c, s, n in zip(coefs, sums, counts))) if complete else None,
        "n_windows": int(np.all([n > 0 for n in counts], axis=0).sum()),
    }
    for name, idx in resamples.items():
        if idx is None:
            out[name] = None
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = sum(c * s[idx].sum(axis=1) / n[idx].sum(axis=1) for c, s, n in zip(coefs, sums, counts))
        out[name] = _interval(boot)
    return out


def paired_summary(pairs_a: np.ndarray, pairs_b: np.ndarray, resamples: dict[str, np.ndarray | None]) -> dict:
    """Pooled mean of ``a`` minus pooled mean of ``b`` over the same windows.

    ``windows_a_higher`` counts windows whose own mean is higher under ``a``;
    it is descriptive only, the estimate and intervals are token-weighted.
    """
    out = contrast_summary([(1.0, pairs_a), (-1.0, pairs_b)], resamples)
    both = (pairs_a[:, 1] > 0) & (pairs_b[:, 1] > 0)
    per_window = window_means(pairs_a) - window_means(pairs_b)
    out["windows_a_higher"] = int((per_window[both] > 0).sum())
    return out


class WindowSeries:
    """Collect ``[sum, count]`` per window under ``/``-joined keys."""

    def __init__(self) -> None:
        self.rows: list[dict[str, list[float]]] = []

    def add_window(self, row: dict[str, list[float]]) -> None:
        self.rows.append(row)

    def pairs(self, key: str) -> np.ndarray:
        return np.array([row.get(key, [0.0, 0]) for row in self.rows], dtype=np.float64).reshape(-1, 2)

    def keys(self) -> list[str]:
        return sorted({key for row in self.rows for key in row})


def flatten_scores(prefix: str, nested: dict) -> dict[str, list[float]]:
    """``{a: {b: [s, c]}}`` -> ``{"prefix/a/b": [s, c]}``."""
    flat = {}
    for key, value in nested.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_scores(path, value))
        else:
            flat[path] = value
    return flat


def nest(flat: dict[str, dict]) -> dict:
    """Inverse of ``flatten_scores`` for summary dicts."""
    tree: dict = {}
    for path, value in flat.items():
        node = tree
        *parents, leaf = path.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return tree
