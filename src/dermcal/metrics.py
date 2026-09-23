"""Discrimination, calibration and clinical utility, with bootstrap intervals.

All functions take y (0/1 outcomes) and p (predicted risks) as 1-D arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 1234
N_BOOT = 2000
MIN_EVENTS_SLOPE = 50     # calibration slope and decile plots need at least this many events


def auc(y, p) -> float:
    """Area under the ROC curve (Mann-Whitney statistic)."""
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def oe_ratio(y, p) -> float:
    """Observed over expected events. Above 1: risk underestimated; below 1: overestimated."""
    e = np.sum(p)
    return np.sum(y) / e if e > 0 else np.nan


def calibration_slope(y, p) -> float:
    """Slope of a logistic regression of the outcome on logit(p), fitted by Newton-Raphson."""
    eps = 1e-6
    q = np.clip(p, eps, 1 - eps)
    z = np.log(q / (1 - q))
    X = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ b))
        W = np.clip(mu * (1 - mu), 1e-8, None)
        try:
            step = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return np.nan
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b[1]


def net_benefit(y, p, t: float = 0.10) -> float:
    """Net benefit of biopsying lesions with predicted risk >= t."""
    pred = p >= t
    n = len(y)
    return np.sum(pred & (y == 1)) / n - np.sum(pred & (y == 0)) / n * (t / (1 - t))


def net_benefit_biopsy_all(y, t: float = 0.10) -> float:
    """Net benefit of biopsying every lesion."""
    prev = np.mean(y)
    return prev - (1 - prev) * (t / (1 - t))


def bootstrap_ci(fn, y, p, n_boot: int = N_BOOT, seed: int = SEED):
    """95% percentile interval from resampling images within the stratum.

    Resamples containing only one outcome class are skipped.
    """
    y = np.asarray(y, float); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v = fn(y[i], p[i])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def stratum_summary(name: str, d: pd.DataFrame, threshold: float = 0.10) -> dict:
    """All performance measures for one stratum (one row of Table 2)."""
    y = d["label_malignant"].to_numpy(float)
    p = d["risk"].to_numpy(float)
    ev = int(y.sum())
    a_lo, a_hi = bootstrap_ci(auc, y, p)
    o_lo, o_hi = bootstrap_ci(oe_ratio, y, p)
    if ev >= MIN_EVENTS_SLOPE:
        s = calibration_slope(y, p)
        s_lo, s_hi = bootstrap_ci(calibration_slope, y, p)
    else:
        s = s_lo = s_hi = np.nan
    n_lo, n_hi = bootstrap_ci(lambda yy, pp: net_benefit(yy, pp, threshold), y, p)
    return {"stratum": name, "n": len(d), "events": ev, "prev": y.mean(),
            "AUC": auc(y, p), "AUC_lo": a_lo, "AUC_hi": a_hi,
            "OE": oe_ratio(y, p), "OE_lo": o_lo, "OE_hi": o_hi,
            "slope": s, "slope_lo": s_lo, "slope_hi": s_hi,
            "NB": net_benefit(y, p, threshold), "NB_lo": n_lo, "NB_hi": n_hi,
            "NB_all": net_benefit_biopsy_all(y, threshold)}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def decile_calibration(y, p) -> pd.DataFrame:
    """Mean predicted risk and observed proportion, with Wilson intervals, by decile of risk."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    rows = []
    for b in np.unique(bins):
        m = bins == b
        lo, hi = wilson(y[m].sum(), m.sum())
        rows.append({"decile": int(b) + 1, "n": int(m.sum()), "mean_predicted": p[m].mean(),
                     "observed": y[m].mean(), "observed_lo": lo, "observed_hi": hi})
    return pd.DataFrame(rows)


def decision_curve(y, p, thresholds) -> pd.DataFrame:
    """Net benefit of the model, of biopsying all and of biopsying none across thresholds."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    return pd.DataFrame({"threshold": thresholds,
                         "model": [net_benefit(y, p, t) for t in thresholds],
                         "biopsy_all": [net_benefit_biopsy_all(y, t) for t in thresholds],
                         "biopsy_none": 0.0})
