"""One loader per dataset. Each returns a DataFrame with the columns below,
so later steps never use a source-specific column name.

CANONICAL SCHEMA
----------------
image_id          str   unique within source; ISIC ids are globally unique
source            str   'ISIC2017' ... 'PAD-UFES-20'
source_split      str   'train' / 'val' / 'test'
role              str   'development' | 'external_test'
modality          str   'dermoscopic' | 'clinical'
dx_raw            str   the source's own label string
dx                str   canonical class (see config.DX_CANONICAL)
label_malignant   float 1 / 0 / NaN   (NaN = cannot be determined)
confirm_type      str   'histopathology' | 'follow_up' | 'consensus' |
                        'confocal' | 'not_recorded'
confirm_histo     float 1 / 0 / NaN
confirm_evidence  str   how confirm_type was obtained — 'field' (read from a
                        real column) or 'inferred:<rule>' (derived from
                        another field)
patient_id        str or NaN
lesion_id         str or NaN
fitzpatrick       float 1..6 or NaN
fst_group         str   'I-II' | 'III-IV' | 'V-VI' | NaN
age_approx        float
sex               str
anatom_site       str
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import (
    Config, DX_ALIASES, MALIGNANT_CORE, BENIGN_CORE, FST_GROUPS, DDI_TONE_MAP,
)

CANONICAL_COLS = [
    "image_id", "source", "source_split", "role", "modality",
    "dx_raw", "dx", "label_malignant",
    "confirm_type", "confirm_histo", "confirm_evidence",
    "patient_id", "lesion_id", "fitzpatrick", "fst_group",
    "age_approx", "sex", "anatom_site",
]

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalise_dx(raw) -> str:
    """Map a free-text label onto the canonical vocabulary.

    Exact alias first. Otherwise the longest alias that occurs in the label as a
    whole word or phrase, so that for example "melanocytic nevi" maps to NV
    through "nevi" and not to MEL through the prefix "mel". Labels with no
    matching alias are OTHER.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "UNK"
    s = _PUNCT.sub(" ", str(raw).strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    if s in DX_ALIASES:
        return DX_ALIASES[s]
    padded = f" {s} "
    hits = [k for k in DX_ALIASES if k and f" {k} " in padded]
    if hits:
        return DX_ALIASES[max(hits, key=len)]
    return "OTHER"


def malignancy(dx: str, akiec_is_malignant: bool) -> float:
    if dx in MALIGNANT_CORE:
        return 1.0
    if dx in BENIGN_CORE:
        return 0.0
    if dx == "AK":
        return 1.0 if akiec_is_malignant else 0.0
    return np.nan  # UNK / OTHER


def _pick(df: pd.DataFrame, *names, default=None):
    """Return the first column present, case/space insensitive."""
    lookup = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for n in names:
        k = n.lower().strip().replace(" ", "_")
        if k in lookup:
            return df[lookup[k]]
    if default is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def _blank(n: int) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([np.nan] * n) for c in CANONICAL_COLS})


def _finalise(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["dx"] = df["dx_raw"].map(normalise_dx)
    df["label_malignant"] = df["dx"].map(lambda d: malignancy(d, cfg.akiec_is_malignant))
    df["confirm_histo"] = df["confirm_type"].map(
        lambda t: 1.0 if t == "histopathology" else (np.nan if t == "not_recorded" else 0.0)
    )
    if "fitzpatrick" in df:
        df["fst_group"] = pd.to_numeric(df["fitzpatrick"], errors="coerce").map(
            lambda v: FST_GROUPS.get(int(v)) if pd.notna(v) and 1 <= v <= 6 else np.nan
        )
    for c in CANONICAL_COLS:
        if c not in df:
            df[c] = np.nan
    df["image_id"] = df["image_id"].astype(str).str.strip()
    return df[CANONICAL_COLS]


# ==========================================================================
# ISIC 2017
# ==========================================================================
def load_isic2017(path: Path, cfg: Config, split: str = "train",
                  metadata_path: Optional[Path] = None) -> pd.DataFrame:
    """Ground truth is two one-hot columns; everything else is 'nevus'."""
    gt = pd.read_csv(path)
    img = _pick(gt, "image_id", "image", "image_name").astype(str)
    mel = pd.to_numeric(_pick(gt, "melanoma"), errors="coerce").fillna(0)
    sk = pd.to_numeric(_pick(gt, "seborrheic_keratosis"), errors="coerce").fillna(0)

    dx_raw = np.where(mel > 0.5, "melanoma",
                      np.where(sk > 0.5, "seborrheic keratosis", "nevus"))

    out = pd.DataFrame({
        "image_id": img,
        "source": "ISIC2017",
        "source_split": split,
        "role": "development",
        "modality": "dermoscopic",
        "dx_raw": dx_raw,
        # The ISIC 2017 ground truth has no confirmation field; it is filled
        # from the ISIC Archive by enrich_confirmation_from_archive().
        "confirm_type": "not_recorded",
        "confirm_evidence": "field:absent",
    })

    if metadata_path is not None and Path(metadata_path).exists():
        md = pd.read_csv(metadata_path)
        md_id = _pick(md, "image_id", "image").astype(str)
        md = md.assign(_id=md_id)
        out = out.merge(
            md[["_id"]].assign(
                age_approx=pd.to_numeric(_pick(md, "age_approximate", "age_approx"), errors="coerce"),
                sex=_pick(md, "sex"),
            ),
            left_on="image_id", right_on="_id", how="left",
        ).drop(columns=["_id"])
    return _finalise(out, cfg)


# ==========================================================================
# ISIC 2018 Task 3  (= HAM10000)
# ==========================================================================
def load_isic2018(path: Path, cfg: Config, split: str = "train",
                  ham_metadata_path: Optional[Path] = None,
                  lesion_grouping_path: Optional[Path] = None) -> pd.DataFrame:
    """One-hot ground truth; dx_type and lesion_id come from the HAM10000 metadata.

    dx_type records how each diagnosis was established (histopathology, follow-up,
    consensus or confocal microscopy). It is distributed with the HAM10000
    metadata, not with the challenge ground-truth file.
    """
    gt = pd.read_csv(path)
    img = _pick(gt, "image", "image_id", "image_name").astype(str)
    onehot = [c for c in gt.columns if c.strip().upper() in
              {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC", "AK", "UNK"}]
    if onehot:
        vals = gt[onehot].apply(pd.to_numeric, errors="coerce").fillna(0)
        dx_raw = vals.idxmax(axis=1).str.strip()
    else:
        dx_raw = _pick(gt, "dx", "diagnosis").astype(str)

    out = pd.DataFrame({
        "image_id": img,
        "source": "ISIC2018",
        "source_split": split,
        "role": "development",
        "modality": "dermoscopic",
        "dx_raw": dx_raw.values,
        "confirm_type": "not_recorded",
        "confirm_evidence": "field:absent",
    })

    # Some mirrors ship HAM10000_metadata (which carries dx_type) instead of the
    # one-hot ground truth; in that case it is its own confirmation source.
    if ham_metadata_path is None and "dx_type" in {c.lower() for c in gt.columns}:
        ham_metadata_path = path

    if ham_metadata_path is not None and Path(ham_metadata_path).exists():
        ham = pd.read_csv(ham_metadata_path)
        ham = ham.assign(_id=_pick(ham, "image_id", "image").astype(str))
        dx_type = _pick(ham, "dx_type", "diagnosis_confirm_type").astype(str).str.lower()
        mapped = dx_type.map({
            "histo": "histopathology", "histopathology": "histopathology",
            "follow_up": "follow_up", "consensus": "consensus",
            "confocal": "confocal",
        }).fillna("not_recorded")
        side = pd.DataFrame({
            "_id": ham["_id"],
            "_confirm": mapped.values,
            "_lesion": _pick(ham, "lesion_id").values,
            "_age": pd.to_numeric(_pick(ham, "age"), errors="coerce").values,
            "_sex": _pick(ham, "sex").values,
            "_site": _pick(ham, "localization").values,
        }).drop_duplicates("_id")
        out = out.merge(side, left_on="image_id", right_on="_id", how="left")
        got = out["_confirm"].notna()
        out.loc[got, "confirm_type"] = out.loc[got, "_confirm"]
        out.loc[got, "confirm_evidence"] = "field:HAM10000.dx_type"
        out["lesion_id"] = out["_lesion"]
        out["age_approx"] = out["_age"]
        out["sex"] = out["_sex"]
        out["anatom_site"] = out["_site"]
        out = out.drop(columns=[c for c in out.columns if c.startswith("_")])

    if lesion_grouping_path is not None and Path(lesion_grouping_path).exists():
        lg = pd.read_csv(lesion_grouping_path)
        lg = lg.assign(_id=_pick(lg, "image", "image_id").astype(str),
                       _les=_pick(lg, "lesion_id", "lesion").astype(str))
        out = out.merge(lg[["_id", "_les"]].drop_duplicates("_id"),
                        left_on="image_id", right_on="_id", how="left")
        out["lesion_id"] = out["lesion_id"].fillna(out["_les"])
        out = out.drop(columns=["_id", "_les"])

    return _finalise(out, cfg)


# ==========================================================================
# ISIC 2019
# ==========================================================================
def load_isic2019(path: Path, cfg: Config, split: str = "train",
                  metadata_path: Optional[Path] = None) -> pd.DataFrame:
    gt = pd.read_csv(path)
    img = _pick(gt, "image", "image_id", "image_name").astype(str)
    onehot = [c for c in gt.columns if c.strip().upper() in
              {"MEL", "NV", "BCC", "AK", "AKIEC", "BKL", "DF", "VASC", "SCC", "UNK"}]
    vals = gt[onehot].apply(pd.to_numeric, errors="coerce").fillna(0)
    dx_raw = vals.idxmax(axis=1).str.strip()
    # a row that is all-zero is genuinely unlabelled, not class 0
    dx_raw = dx_raw.where(vals.max(axis=1) > 0, "unknown")

    out = pd.DataFrame({
        "image_id": img,
        "source": "ISIC2019",
        "source_split": split,
        "role": "development",
        "modality": "dermoscopic",
        "dx_raw": dx_raw.values,
        "confirm_type": "not_recorded",
        "confirm_evidence": "field:absent",
    })

    if metadata_path is not None and Path(metadata_path).exists():
        md = pd.read_csv(metadata_path)
        md = md.assign(_id=_pick(md, "image", "image_id").astype(str))
        side = pd.DataFrame({
            "_id": md["_id"],
            "_age": pd.to_numeric(_pick(md, "age_approx", "age"), errors="coerce").values,
            "_sex": _pick(md, "sex").values,
            "_site": _pick(md, "anatom_site_general", "anatom_site_general_challenge").values,
            "_les": _pick(md, "lesion_id").values,
        }).drop_duplicates("_id")
        out = out.merge(side, left_on="image_id", right_on="_id", how="left")
        out["age_approx"], out["sex"] = out["_age"], out["_sex"]
        out["anatom_site"], out["lesion_id"] = out["_site"], out["_les"]
        out = out.drop(columns=[c for c in out.columns if c.startswith("_")])

    return _finalise(out, cfg)


# ==========================================================================
# ISIC 2020 (SIIM-ISIC)
# ==========================================================================
def load_isic2020(path: Path, cfg: Config, split: str = "train") -> pd.DataFrame:
    """ISIC 2020 training set.

    Every malignant diagnosis in this release is histopathology-confirmed, while
    benign diagnoses may rest on expert agreement or follow-up. The diagnosis
    column is 'unknown' for about 82% of rows, most of them benign, so a known
    diagnosis does not imply histopathology. Confirmation derived here is marked
    'inferred' in confirm_evidence; the ISIC Archive field replaces it where
    available.
    """
    df = pd.read_csv(path)
    img = _pick(df, "image_name", "image_id", "image").astype(str)
    dx_raw = _pick(df, "diagnosis").astype(str).str.strip().str.lower()
    ben_mal = _pick(df, "benign_malignant").astype(str).str.strip().str.lower()
    target = pd.to_numeric(_pick(df, "target"), errors="coerce")

    is_mal = (ben_mal == "malignant") | (target == 1)
    specific = dx_raw.notna() & ~dx_raw.isin(["unknown", "nan", ""])

    confirm = np.where(
        is_mal, "histopathology",
        np.where(specific, "not_recorded", "not_recorded"),
    )
    evidence = np.where(
        is_mal,
        "inferred:isic2020_all_malignant_are_histo_confirmed",
        "inferred:benign_confirmation_method_not_distributed",
    )

    out = pd.DataFrame({
        "image_id": img,
        "source": "ISIC2020",
        "source_split": split,
        "role": "development",
        "modality": "dermoscopic",
        # keep the *specific* dx where present; otherwise fall back to the
        # binary field so the row is not discarded as UNK when we do know it
        # was benign on expert consensus.
        "dx_raw": np.where(specific, dx_raw, np.where(is_mal, "melanoma", "unknown")),
        "confirm_type": confirm,
        "confirm_evidence": evidence,
        "patient_id": _pick(df, "patient_id").values,
        "age_approx": pd.to_numeric(_pick(df, "age_approx"), errors="coerce").values,
        "sex": _pick(df, "sex").values,
        "anatom_site": _pick(df, "anatom_site_general_challenge", "anatom_site_general").values,
    })
    out["_benign_known"] = (~is_mal & specific).values
    res = _finalise(out, cfg)
    # a benign row with a specific diagnosis is a real benign label even though
    # its confirmation method is unrecorded
    res["label_malignant"] = np.where(is_mal.values, 1.0, res["label_malignant"].values)
    return res


# ==========================================================================
# Fitzpatrick17k (external evaluation only)
# ==========================================================================
def load_fitzpatrick17k(path: Path, cfg: Config) -> pd.DataFrame:
    """Fitzpatrick17k, used only for external evaluation.

    Labels come from dermatology atlases and are not biopsy-confirmed. The
    dataset is never used for training.
    """
    df = pd.read_csv(path)
    img = _pick(df, "md5hash", "image_id", "filename").astype(str)
    fitz = pd.to_numeric(_pick(df, "fitzpatrick_scale", "fitzpatrick"), errors="coerce")
    fitz = fitz.where(fitz.between(1, 6))  # -1 is the dataset's 'unknown' code
    three = _pick(df, "three_partition_label").astype(str).str.lower()
    label = _pick(df, "label").astype(str)

    out = pd.DataFrame({
        "image_id": img,
        "source": "Fitzpatrick17k",
        "source_split": "test",
        "role": "external_test",
        "modality": "clinical",
        "dx_raw": label.values,
        "confirm_type": "not_recorded",
        "confirm_evidence": "field:absent(atlas-sourced labels, not biopsy-confirmed)",
        "fitzpatrick": fitz.values,
    })
    res = _finalise(out, cfg)
    # trust the curated three-way partition over our alias map where they differ
    res["label_malignant"] = np.where(
        three.values == "malignant", 1.0,
        np.where(three.values == "benign", 0.0,
                 np.where(three.values == "non-neoplastic", 0.0, res["label_malignant"].values)),
    )
    res["dx_raw"] = res["dx_raw"] + " | " + three.values
    return res


# ==========================================================================
# DDI — Diverse Dermatology Images
# ==========================================================================
def load_ddi(path: Path, cfg: Config) -> pd.DataFrame:
    """Diverse Dermatology Images (DDI).

    Every lesion was biopsied, whatever the outcome, so verification is complete
    and independent of the diagnosis. Skin tone is released in three groups
    (12, 34, 56).
    """
    df = pd.read_csv(path)
    img = _pick(df, "DDI_file", "ddi_file", "filename", "image_id").astype(str)
    tone = _pick(df, "skin_tone", "skin tone")
    mal = _pick(df, "malignant")
    disease = _pick(df, "disease").astype(str)

    fst_group = tone.map(lambda v: DDI_TONE_MAP.get(str(v).strip(), np.nan))
    fitz = fst_group.map({"I-II": 1.5, "III-IV": 3.5, "V-VI": 5.5})

    mal_num = mal.map(lambda v: 1.0 if str(v).strip().lower() in {"true", "1", "1.0", "yes"}
                      else (0.0 if str(v).strip().lower() in {"false", "0", "0.0", "no"} else np.nan))

    out = pd.DataFrame({
        "image_id": img,
        "source": "DDI",
        "source_split": "test",
        "role": "external_test",
        "modality": "clinical",
        "dx_raw": disease.values,
        "confirm_type": "histopathology",
        "confirm_evidence": "field:all DDI lesions biopsy-proven by construction",
    })
    res = _finalise(out, cfg)
    res["fst_group"] = fst_group.values
    res["fitzpatrick"] = fitz.values
    res["label_malignant"] = np.where(mal_num.notna().values, mal_num.values,
                                      res["label_malignant"].values)
    return res


# ==========================================================================
# PAD-UFES-20
# ==========================================================================
def load_pad_ufes(path: Path, cfg: Config) -> pd.DataFrame:
    """PAD-UFES-20.

    All BCC, SCC and melanoma are biopsy-proven, whereas naevi, seborrhoeic
    keratoses and actinic keratoses are usually diagnosed clinically, so
    biopsied lesions are mostly malignant. The Fitzpatrick column is spelled
    'fitspatrick' in the released file.
    """
    df = pd.read_csv(path)
    img = _pick(df, "img_id", "image_id", "filename").astype(str)
    diag = _pick(df, "diagnostic", "diagnosis").astype(str)
    biop = _pick(df, "biopsed", "biopsied")
    fitz = pd.to_numeric(_pick(df, "fitspatrick", "fitzpatrick", "fitzpatrick_scale"),
                         errors="coerce")
    fitz = fitz.where(fitz.between(1, 6))

    biop_bool = biop.map(lambda v: True if str(v).strip().lower() in {"true", "1", "1.0", "yes"}
                         else (False if str(v).strip().lower() in {"false", "0", "0.0", "no"} else None))

    out = pd.DataFrame({
        "image_id": img,
        "source": "PAD-UFES-20",
        "source_split": "test",
        "role": "external_test",
        "modality": "clinical",
        "dx_raw": diag.values,
        "confirm_type": np.where(biop_bool.eq(True).values, "histopathology",
                                 np.where(biop_bool.eq(False).values, "consensus", "not_recorded")),
        "confirm_evidence": "field:PAD-UFES-20.biopsed",
        "fitzpatrick": fitz.values,
        "patient_id": _pick(df, "patient_id").values,
        "lesion_id": _pick(df, "lesion_id").values,
        "age_approx": pd.to_numeric(_pick(df, "age"), errors="coerce").values,
        "sex": _pick(df, "gender", "sex").values,
        "anatom_site": _pick(df, "region").values,
    })
    return _finalise(out, cfg)


# ==========================================================================
# Orchestrator
# ==========================================================================
LOADER_ROUTES = [
    # (canonical key, loader, kwargs referencing other discovered keys)
    ("isic2017_train", load_isic2017, dict(split="train")),
    ("isic2017_val",   load_isic2017, dict(split="val")),
    ("isic2017_test",  load_isic2017, dict(split="test")),
    ("isic2018_train", load_isic2018, dict(split="train")),
    ("isic2018_test",  load_isic2018, dict(split="test")),
    ("isic2019_train", load_isic2019, dict(split="train")),
    ("isic2019_test",  load_isic2019, dict(split="test")),
    ("isic2020_train", load_isic2020, dict(split="train")),
    ("fitzpatrick17k", load_fitzpatrick17k, {}),
    ("ddi",            load_ddi, {}),
    ("pad_ufes_20",    load_pad_ufes, {}),
]


def load_all(paths: Dict[str, Path], cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Load every discovered dataset and report the ones that were not found."""
    frames, missing = [], []
    for key, fn, kw in LOADER_ROUTES:
        if key not in paths:
            missing.append(key)
            continue
        kwargs = dict(kw)
        if key == "isic2018_train":
            kwargs["ham_metadata_path"] = paths.get("isic2018_train_meta") or paths.get("isic2018_train")
            kwargs["lesion_grouping_path"] = paths.get("isic2018_lesion_groupings")
        if key.startswith("isic2019"):
            kwargs["metadata_path"] = paths.get(key + "_meta")
        if key.startswith("isic2017"):
            kwargs["metadata_path"] = paths.get(key + "_meta")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = fn(paths[key], cfg, **kwargs)
            frames.append(df)
            if verbose:
                print(f"  loaded {key:<20} n={len(df):>7,}  ({paths[key].name})")
        except Exception as exc:  # continue with the remaining datasets
            print(f"  FAILED {key}: {type(exc).__name__}: {exc}")
    if missing and verbose:
        print(f"\n  not discovered (skipped): {', '.join(missing)}")
    if not frames:
        return _blank(0)
    return pd.concat(frames, ignore_index=True)


# ==========================================================================
# Cross-release field coalescing
# ==========================================================================
def coalesce_across_releases(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Merge complementary fields for the same image across releases.

    A HAM10000 image appears in both ISIC 2018 and ISIC 2019. Only the ISIC 2018
    distribution carries dx_type, and only ISIC 2019 separates SCC from AK.
    Fields are therefore pooled across releases before de-duplication, so that
    keeping one row per image does not discard the confirmation field of about
    10,000 images.
    """
    d = df.copy()
    donors = {
        "confirm_type": lambda s: s[s != "not_recorded"],
        "confirm_evidence": None,          # carried along with confirm_type
        "lesion_id": lambda s: s.dropna(),
        "patient_id": lambda s: s.dropna(),
        "fitzpatrick": lambda s: s.dropna(),
        "age_approx": lambda s: s.dropna(),
        "sex": lambda s: s.dropna(),
        "anatom_site": lambda s: s.dropna(),
    }
    filled = {}
    for col, sel in donors.items():
        if sel is None:
            continue
        src = d[["image_id", col]].copy()
        good = sel(src[col])
        src = src.loc[good.index].drop_duplicates("image_id")
        lut = dict(zip(src["image_id"], src[col]))
        before = d[col].isna().sum() if col != "confirm_type" else (d[col] == "not_recorded").sum()
        if col == "confirm_type":
            mask = d[col] == "not_recorded"
            d.loc[mask, col] = d.loc[mask, "image_id"].map(lut).fillna("not_recorded")
            after = (d[col] == "not_recorded").sum()
        else:
            d[col] = d[col].fillna(d["image_id"].map(lut))
            after = d[col].isna().sum()
        filled[col] = int(before - after)

    # evidence string must follow the confirmation it describes
    ev = d.loc[d["confirm_type"] != "not_recorded", ["image_id", "confirm_evidence"]]
    ev = ev[ev["confirm_evidence"].astype(str).str.startswith("field")].drop_duplicates("image_id")
    lut_ev = dict(zip(ev["image_id"], ev["confirm_evidence"]))
    m = (d["confirm_type"] != "not_recorded") & \
        (~d["confirm_evidence"].astype(str).str.startswith("field"))
    d.loc[m, "confirm_evidence"] = d.loc[m, "image_id"].map(lut_ev).fillna(
        d.loc[m, "confirm_evidence"])

    d["confirm_histo"] = d["confirm_type"].map(
        lambda t: 1.0 if t == "histopathology" else (np.nan if t == "not_recorded" else 0.0))
    d["fst_group"] = pd.to_numeric(d["fitzpatrick"], errors="coerce").map(
        lambda v: FST_GROUPS.get(int(v)) if pd.notna(v) and 1 <= v <= 6 else np.nan
    ).fillna(d["fst_group"])

    if verbose:
        print("  fields recovered by cross-release coalescing:")
        for k, v in filled.items():
            if v:
                print(f"    {k:<16} +{v:,} values")
    return d


# ==========================================================================
# ISIC Archive API enrichment
# ==========================================================================
ARCHIVE_FIELDS = ["isic_id", "diagnosis_1", "diagnosis_3", "diagnosis_confirm_type",
                  "melanocytic", "fitzpatrick_skin_type", "lesion_id", "patient_id",
                  "anatomical_site", "age_approx", "sex", "attribution"]


def enrich_confirmation_from_archive(df: pd.DataFrame,
                                     archive_metadata_csv: Optional[Path] = None,
                                     verbose: bool = True) -> pd.DataFrame:
    """Fill `confirm_type` from a full ISIC Archive metadata export.

    The ISIC 2017 and 2019 challenge ground-truth files have no confirmation
    field. The ISIC Archive records diagnosis_confirm_type for every image under
    the same ISIC identifier. The export is obtained with

        pip install isic-cli
        isic metadata download > isic_archive_metadata.csv
    """
    if archive_metadata_csv is None or not Path(archive_metadata_csv).exists():
        if verbose:
            print("  No ISIC Archive metadata export supplied; confirmation status\n"
                  "  of ISIC 2017 and 2019 images stays not_recorded.")
        return df

    arch = pd.read_csv(archive_metadata_csv, low_memory=False)
    key = _pick(arch, "isic_id", "image_id", "name").astype(str)
    ct = _pick(arch, "diagnosis_confirm_type").astype(str).str.lower().str.strip()
    mapped = ct.map(lambda v: "histopathology" if "histo" in v
                    else "confocal" if "confocal" in v
                    else "follow_up" if "follow" in v or "serial" in v
                    else "consensus" if "consensus" in v or "expert" in v
                    else "not_recorded")
    fitz = pd.to_numeric(
        _pick(arch, "fitzpatrick_skin_type").astype(str).str.extract(r"(\d)")[0],
        errors="coerce")

    side = (pd.DataFrame({"_id": key, "_ct": mapped.values, "_fitz": fitz.values,
                          "_les": _pick(arch, "lesion_id").values,
                          "_pat": _pick(arch, "patient_id").values})
            .drop_duplicates("_id"))
    d = df.merge(side, left_on="image_id", right_on="_id", how="left")

    mask = (d["confirm_type"] == "not_recorded") & d["_ct"].notna() & (d["_ct"] != "not_recorded")
    n = int(mask.sum())
    d.loc[mask, "confirm_type"] = d.loc[mask, "_ct"]
    d.loc[mask, "confirm_evidence"] = "field:ISIC_Archive.diagnosis_confirm_type"
    d["fitzpatrick"] = d["fitzpatrick"].fillna(d["_fitz"])
    d["lesion_id"] = d["lesion_id"].fillna(d["_les"])
    d["patient_id"] = d["patient_id"].fillna(d["_pat"])
    d = d.drop(columns=[c for c in d.columns if c.startswith("_")])
    d["confirm_histo"] = d["confirm_type"].map(
        lambda t: 1.0 if t == "histopathology" else (np.nan if t == "not_recorded" else 0.0))
    d["fst_group"] = pd.to_numeric(d["fitzpatrick"], errors="coerce").map(
        lambda v: FST_GROUPS.get(int(v)) if pd.notna(v) and 1 <= v <= 6 else np.nan
    ).fillna(d["fst_group"])
    if verbose:
        print(f"  ISIC Archive enrichment: confirmation recovered for {n:,} images")
    return d
