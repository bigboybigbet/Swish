"""
Model metrics as data: the numbers behind the old fig_feature_importance and
fig_calibration charts (the Model page draws them with ECharts).
"""
import math

import numpy as np

from server.jsonutil import r4


def feature_importance(feat_imp: dict | None) -> list:
    """{feature: importance} → [{feature, importance}], most important first."""
    rows = [{"feature": str(k), "importance": r4(v)} for k, v in (feat_imp or {}).items()]
    return sorted(rows, key=lambda r: r["importance"] or 0, reverse=True)


def wilson(k: int, n: int, z: float = 1.96) -> list | None:
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [r4(max(0.0, centre - half)), r4(min(1.0, centre + half))]


def calibration_bins(proba, y, n_bins: int = 10, with_ci: bool = False) -> list:
    """Reliability bins over [0, 1]: [{lo, hi, n, mean_pred, actual}] for non-empty bins.

    with_ci adds gap (actual − mean_pred) and a 95% Wilson interval on `actual`.
    """
    p = np.asarray(proba, dtype=float)
    t = np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        n = int(mask.sum())
        if not n:
            continue
        row = {"lo": r4(lo), "hi": r4(hi), "n": n,
               "mean_pred": r4(p[mask].mean()), "actual": r4(t[mask].mean())}
        if with_ci:
            row["gap"] = r4(t[mask].mean() - p[mask].mean())
            row["ci95"] = wilson(int(t[mask].sum()), n)
        out.append(row)
    return out
