"""Overlap between releases and the grouping structure used for splitting.

The ISIC releases are nested. ISIC 2018 Task 3 is HAM10000; ISIC 2019 was
assembled from HAM10000, BCN20000 and MSK images; ISIC 2017 draws on the MSK
and UDA collections that also feed ISIC 2019; and HAM10000 contains several
images of the same lesion. Pooling the releases without de-duplication would
place the same image, or the same lesion, in both the training and the
held-out data.

Images are de-duplicated on the ISIC identifier, and the development split is
made by lesion, then patient, then image.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Identifier-level overlap
# --------------------------------------------------------------------------
def id_overlap_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise count of shared image_ids between sources."""
    sources = sorted(df["source"].dropna().unique())
    sets = {s: set(df.loc[df["source"] == s, "image_id"]) for s in sources}
    m = pd.DataFrame(0, index=sources, columns=sources, dtype=int)
    for a in sources:
        for b in sources:
            m.loc[a, b] = len(sets[a] & sets[b])
    return m


def dedup_by_id(df: pd.DataFrame, priority: Optional[List[str]] = None
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one row per image_id, preferring the release with the most complete labels.

    Default priority:
      ISIC2019 (eight classes including SCC)
        > ISIC2018 (HAM10000, carries dx_type)
        > ISIC2020 (binary target; diagnosis unknown for most rows)
        > ISIC2017 (three classes)
    External sets are not de-duplicated here; overlap between external and
    development images is checked by contamination_report().
    """
    priority = priority or ["ISIC2019", "ISIC2018", "ISIC2020", "ISIC2017"]
    rank = {s: i for i, s in enumerate(priority)}
    d = df.copy()
    d["_rank"] = d["source"].map(lambda s: rank.get(s, 99))
    # prefer rows that actually carry a confirmation field
    d["_has_confirm"] = (d["confirm_type"] != "not_recorded").astype(int)
    d = d.sort_values(["image_id", "_rank", "_has_confirm"],
                      ascending=[True, True, False])
    keep = d.drop_duplicates("image_id", keep="first")
    dropped = d.loc[~d.index.isin(keep.index)]
    return (keep.drop(columns=["_rank", "_has_confirm"]),
            dropped.drop(columns=["_rank", "_has_confirm"]))


def contamination_report(df: pd.DataFrame) -> pd.DataFrame:
    """Number of external images whose identifier also appears in the development data."""
    dev = set(df.loc[df["role"] == "development", "image_id"])
    rows = []
    for s in sorted(df.loc[df["role"] == "external_test", "source"].unique()):
        ext = set(df.loc[df["source"] == s, "image_id"])
        rows.append({"external_set": s, "n": len(ext),
                     "shared_ids_with_development": len(dev & ext)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Grouping structure
# --------------------------------------------------------------------------
def grouping_audit(df: pd.DataFrame) -> pd.DataFrame:
    """How many images per lesion / per patient, by source.

    A source with more than one image per lesion is split by lesion, and a
    source with patient identifiers is split by patient.
    """
    rows = []
    for s, g in df.groupby("source"):
        n_img = len(g)
        n_les = g["lesion_id"].nunique(dropna=True)
        n_pat = g["patient_id"].nunique(dropna=True)
        rows.append({
            "source": s,
            "images": n_img,
            "lesion_id_present_%": round(100 * g["lesion_id"].notna().mean(), 1),
            "unique_lesions": n_les if n_les else np.nan,
            "images_per_lesion": round(n_img / n_les, 2) if n_les else np.nan,
            "patient_id_present_%": round(100 * g["patient_id"].notna().mean(), 1),
            "unique_patients": n_pat if n_pat else np.nan,
            "images_per_patient": round(n_img / n_pat, 2) if n_pat else np.nan,
            "required_split_unit": ("patient" if n_pat else
                                    "lesion" if n_les else "image"),
        })
    return pd.DataFrame(rows).sort_values("images", ascending=False)
