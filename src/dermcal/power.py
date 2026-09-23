"""Precision attainable in each evaluation stratum.

Calibration needs more outcome events than discrimination. Riley et al.
(Stat Med 2021) recommend at least 100 events and 100 non-events for external
validation of a binary prediction model, and more before a calibration slope is
estimated with useful precision. These functions give approximate interval
widths from the number of events in a stratum, which were used to decide which
strata are analysed and which are reported descriptively.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

Z95 = 1.959963985


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------
def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of the c-statistic (Hanley & McNeil 1982)."""
    if n_pos < 1 or n_neg < 1:
        return np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def auc_ci_halfwidth(n_pos: int, n_neg: int, auc: float = 0.85) -> float:
    return Z95 * hanley_mcneil_se(auc, n_pos, n_neg)


def min_detectable_auc_gap(n1_pos, n1_neg, n2_pos, n2_neg,
                           auc: float = 0.85, power: float = 0.80,
                           alpha: float = 0.05) -> float:
    """Smallest AUC difference between two subgroups detectable at 80% power."""
    se1 = hanley_mcneil_se(auc, n1_pos, n1_neg)
    se2 = hanley_mcneil_se(auc, n2_pos, n2_neg)
    if not np.isfinite(se1) or not np.isfinite(se2):
        return np.nan
    se = math.sqrt(se1 ** 2 + se2 ** 2)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * se


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def oe_ratio_ci_width(events: int, prevalence: float) -> float:
    """Multiplicative width of the 95% CI for the observed/expected ratio.

    SE(log O/E) ~= sqrt((1 - prev) / events). A width of 2.0 means the upper
    limit is twice the lower limit.
    """
    if events < 1 or not (0 < prevalence < 1):
        return np.nan
    se = math.sqrt((1 - prevalence) / events)
    return math.exp(Z95 * se) / math.exp(-Z95 * se)


def calibration_slope_se(events: int) -> float:
    """Approximate SE of the calibration slope.

    Riley et al. give SE(slope) ~ sqrt(1 / (events * Var(LP))) for a linear
    predictor with variance Var(LP). With Var(LP) fixed at 1.5 this depends on
    the number of events only.
    """
    if events < 2:
        return np.nan
    return math.sqrt(1.0 / (events * 1.5))


def wilson_halfwidth(k: int, n: int) -> float:
    """Half-width of the Wilson interval for one calibration bin."""
    if n < 1:
        return np.nan
    p = k / n
    denom = 1 + Z95 ** 2 / n
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n ** 2)) / denom
    return half


# --------------------------------------------------------------------------
# Decision curve analysis
# --------------------------------------------------------------------------
def net_benefit_se(n: int, prevalence: float, threshold: float,
                   sensitivity: float = 0.85, specificity: float = 0.80) -> float:
    """Approximate SE of net benefit at one threshold probability.

    NB = TP/n - (FP/n) * (pt / (1 - pt)). Treating TP and FP counts as binomial
    and independent gives a first-order variance.
    """
    if n < 1 or not (0 < threshold < 1):
        return np.nan
    w = threshold / (1 - threshold)
    p_tp = prevalence * sensitivity
    p_fp = (1 - prevalence) * (1 - specificity)
    var = (p_tp * (1 - p_tp) + w ** 2 * p_fp * (1 - p_fp)) / n
    return math.sqrt(var)


# --------------------------------------------------------------------------
# Per-stratum summary
# --------------------------------------------------------------------------
def feasibility_table(df: pd.DataFrame,
                      group_cols=("source", "fst_group"),
                      label_col: str = "label_malignant",
                      assumed_auc: float = 0.85,
                      dca_threshold: float = 0.10,
                      min_events: int = 25) -> pd.DataFrame:
    """One row per evaluation stratum, with the precision it can support."""
    rows = []
    d = df[df[label_col].notna()].copy()
    for keys, g in d.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        pos = int((g[label_col] == 1).sum())
        neg = int((g[label_col] == 0).sum())
        prev = pos / n if n else np.nan

        auc_hw = auc_ci_halfwidth(pos, neg, assumed_auc)
        oe_w = oe_ratio_ci_width(pos, prev) if pos and 0 < prev < 1 else np.nan
        slope_se = calibration_slope_se(pos)
        # a 10-bin calibration curve puts ~n/10 in each bin
        bin_hw = wilson_halfwidth(max(1, int(round(prev * n / 10))), max(1, n // 10))
        nb_se = net_benefit_se(n, prev, dca_threshold) if 0 < prev < 1 else np.nan

        rows.append({
            **dict(zip(group_cols, keys)),
            "n": n, "malignant": pos, "benign": neg,
            "prevalence": round(prev, 4) if np.isfinite(prev) else np.nan,
            "AUC_95CI_halfwidth": round(auc_hw, 4) if np.isfinite(auc_hw) else np.nan,
            "OE_95CI_fold_width": round(oe_w, 2) if np.isfinite(oe_w) else np.nan,
            "calib_slope_SE": round(slope_se, 3) if np.isfinite(slope_se) else np.nan,
            "calib_bin_95CI_halfwidth": round(bin_hw, 3) if np.isfinite(bin_hw) else np.nan,
            f"netbenefit_SE_at_pt{dca_threshold}": round(nb_se, 5) if np.isfinite(nb_se) else np.nan,
            "meets_Riley_min_100": (pos >= 100 and neg >= 100),
            "reportable_calibration": (pos >= min_events and neg >= min_events),
            "verdict": _verdict(pos, neg, auc_hw, slope_se),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["n"], ascending=False).reset_index(drop=True)


def _verdict(pos, neg, auc_hw, slope_se) -> str:
    if pos < 10 or neg < 10:
        return "NOT EVALUABLE - report counts only"
    if pos < 25 or neg < 25:
        return "descriptive only - no calibration curve"
    if pos < 100 or neg < 100:
        return "underpowered - report interval estimates, no hypothesis test"
    if np.isfinite(auc_hw) and auc_hw > 0.05:
        return "adequate for calibration, imprecise AUC"
    return "adequate"
