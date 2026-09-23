"""Eligibility criteria, attrition and verification tables.

In several sources whether a lesion was biopsied depends on whether it was
malignant:

  ISIC 2020      all malignant lesions are histopathology-confirmed; benign
                 lesions may be confirmed by expert agreement or follow-up
  PAD-UFES-20    all BCC, SCC and melanoma are biopsy-proven; naevi, seborrhoeic
                 and actinic keratoses are usually diagnosed clinically
  HAM10000       about half of images are histopathology-confirmed, unevenly
                 across classes
  DDI            every lesion biopsied

Restricting to histopathology-confirmed images therefore removes benign images
at a higher rate than malignant ones (differential verification) and raises
the apparent prevalence. verification_bias_table() and prevalence_shift()
quantify this for each source.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
def verification_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """Histopathology confirmation rate among malignant and benign lesions, by source."""
    rows = []
    d = df[df["label_malignant"].notna()]
    for s, g in d.groupby("source"):
        for lbl, name in ((1.0, "malignant"), (0.0, "benign")):
            sub = g[g["label_malignant"] == lbl]
            if sub.empty:
                continue
            rows.append({
                "source": s, "class": name, "n": len(sub),
                "histo_confirmed": int((sub["confirm_histo"] == 1).sum()),
                "histo_%": round(100 * (sub["confirm_histo"] == 1).mean(), 1),
                "confirm_unknown_%": round(100 * sub["confirm_histo"].isna().mean(), 1),
            })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    piv = t.pivot_table(index="source", columns="class",
                        values="histo_%", aggfunc="first")
    piv["verification_gap_pp"] = (piv.get("malignant", np.nan)
                                  - piv.get("benign", np.nan)).round(1)
    piv["bias_risk"] = piv["verification_gap_pp"].map(
        lambda g: "SEVERE" if pd.notna(g) and g >= 50 else
                  "HIGH" if pd.notna(g) and g >= 20 else
                  "moderate" if pd.notna(g) and g >= 5 else
                  "none detected" if pd.notna(g) else "cannot assess")
    return piv.reset_index()


def prevalence_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Apparent malignancy prevalence before vs after the histopathology filter."""
    rows = []
    d = df[df["label_malignant"].notna()]
    for s, g in d.groupby("source"):
        full = g["label_malignant"].mean()
        conf = g.loc[g["confirm_histo"] == 1, "label_malignant"]
        rows.append({
            "source": s,
            "n_all": len(g),
            "prevalence_all": round(full, 4),
            "n_histo_confirmed": len(conf),
            "prevalence_histo_only": round(conf.mean(), 4) if len(conf) else np.nan,
            "prevalence_inflation_x": (round(conf.mean() / full, 2)
                                       if len(conf) and full > 0 else np.nan),
        })
    return pd.DataFrame(rows).sort_values("n_all", ascending=False)


# --------------------------------------------------------------------------
def build_cohort(df: pd.DataFrame, cfg,
                 require_histo: bool | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the inclusion criteria in order, logging attrition at each step."""
    require_histo = cfg.require_histopath if require_histo is None else require_histo
    log: List[Dict] = []
    d = df.copy()

    def step(name, mask, note=""):
        nonlocal d
        before = len(d)
        d = d[mask(d)].copy()
        log.append({"step": name, "n_before": before, "n_removed": before - len(d),
                    "n_after": len(d), "note": note})

    log.append({"step": "0. all records loaded", "n_before": np.nan,
                "n_removed": np.nan, "n_after": len(d), "note": "raw union of releases"})

    step("1. drop exact duplicate image_id within source",
         lambda x: ~x.duplicated(["source", "image_id"]),
         "same file listed twice in one release")

    step("2. resolvable binary label",
         lambda x: x["label_malignant"].notna(),
         "drops UNK / OTHER classes that cannot be scored malignant vs benign")

    if require_histo:
        step("3. histopathology-confirmed only",
             lambda x: x["confirm_histo"] == 1,
             "primary analysis restriction; see verification_bias_table()")
    else:
        log.append({"step": "3. histopathology restriction NOT applied",
                    "n_before": len(d), "n_removed": 0, "n_after": len(d),
                    "note": "sensitivity arm: all confirmation methods retained"})

    return d, pd.DataFrame(log)


def cohort_summary(cohort: pd.DataFrame) -> pd.DataFrame:
    """Size, prevalence and demographics by role and source (basis of Table 1)."""
    rows = []
    for (role, src), g in cohort.groupby(["role", "source"], dropna=False):
        pos = int((g["label_malignant"] == 1).sum())
        rows.append({
            "role": role, "source": src, "modality": g["modality"].mode().iat[0]
            if not g["modality"].isna().all() else np.nan,
            "n": len(g), "malignant": pos, "benign": len(g) - pos,
            "prevalence": round(pos / len(g), 4) if len(g) else np.nan,
            "has_fitzpatrick_%": round(100 * g["fitzpatrick"].notna().mean(), 1),
            "median_age": round(g["age_approx"].median(), 1)
            if g["age_approx"].notna().any() else np.nan,
            "female_%": round(100 * g["sex"].astype(str).str.lower()
                              .str.startswith("f").mean(), 1)
            if g["sex"].notna().any() else np.nan,
        })
    return (pd.DataFrame(rows)
            .sort_values(["role", "n"], ascending=[True, False])
            .reset_index(drop=True))


# --------------------------------------------------------------------------
def modality_confound_table(df: pd.DataFrame) -> pd.DataFrame:
    """Image modality by role.

    Development images are dermoscopic and all external images are clinical
    photographs, so skin-tone groups are compared within each external dataset
    rather than against the development data.
    """
    t = (df.groupby(["role", "modality"])
           .agg(n=("image_id", "size"),
                sources=("source", lambda s: ", ".join(sorted(set(s)))))
           .reset_index())
    return t


def tone_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Fitzpatrick label availability and counts per skin-tone group, by source."""
    rows = []
    for s, g in df.groupby("source"):
        rec = {"source": s, "n": len(g), "role": g["role"].iat[0],
               "fitz_labelled_%": round(100 * g["fitzpatrick"].notna().mean(), 1)}
        for grp in ["I-II", "III-IV", "V-VI"]:
            sub = g[g["fst_group"] == grp]
            pos = int((sub["label_malignant"] == 1).sum())
            rec[f"{grp}_n"] = len(sub)
            rec[f"{grp}_mal"] = pos
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
