"""Paths, diagnosis vocabulary, analysis settings and dataset discovery.

Column names, class names and thresholds used elsewhere in the package are
defined here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------
# 1. Canonical diagnosis vocabulary
# --------------------------------------------------------------------------
# Every source's label set is mapped onto the ISIC 2019 eight-class vocabulary.
# UNK is kept as its own state so that the number of unresolved labels can be
# reported.

DX_CANONICAL: List[str] = [
    "MEL",    # melanoma
    "NV",     # melanocytic nevus
    "BCC",    # basal cell carcinoma
    "AK",     # actinic keratosis  (2018/HAM 'AKIEC' also covers Bowen's/SCCis)
    "BKL",    # benign keratosis (solar lentigo / seb. keratosis / lichen planus-like)
    "DF",     # dermatofibroma
    "VASC",   # vascular lesion
    "SCC",    # squamous cell carcinoma
    "UNK",    # present in the source, but no specific diagnosis recorded
    "OTHER",  # source-specific class with no ISIC-2019 equivalent
]

# Raw label string -> canonical. Keys are lower-cased and stripped of
# punctuation before lookup (see schemas.normalise_dx).
DX_ALIASES: Dict[str, str] = {
    # --- melanoma ---
    "melanoma": "MEL", "mel": "MEL", "malignant melanoma": "MEL",
    "melanoma metastasis": "MEL", "melanoma in situ": "MEL",
    "lentigo maligna": "MEL", "lentigo maligna melanoma": "MEL",
    "nodular melanoma": "MEL", "superficial spreading melanoma": "MEL",
    "acral lentiginous melanoma": "MEL",
    # --- nevus ---
    "nevus": "NV", "nv": "NV", "nev": "NV", "nevi": "NV", "melanocytic nevus": "NV",
    "nevus spilus": "NV", "blue nevus": "NV", "congenital nevus": "NV",
    "dysplastic nevus": "NV", "atypical nevus": "NV", "spitz nevus": "NV",
    "compound nevus": "NV", "junctional nevus": "NV", "dermal nevus": "NV",
    "halo nevus": "NV", "epidermal nevus": "NV",
    # --- BCC ---
    "basal cell carcinoma": "BCC", "bcc": "BCC",
    "basal cell carcinoma superficial": "BCC",
    "basal cell carcinoma nodular": "BCC",
    # --- SCC ---
    "squamous cell carcinoma": "SCC", "scc": "SCC",
    "squamous cell carcinoma in situ": "SCC", "sccis": "SCC",
    "bowen disease": "SCC", "bowens disease": "SCC",
    "keratoacanthoma": "SCC",
    # --- AK / AKIEC ---
    "actinic keratosis": "AK", "ak": "AK", "ack": "AK",
    "akiec": "AK", "actinic keratosis intraepithelial carcinoma": "AK",
    "solar keratosis": "AK",
    # --- benign keratosis ---
    "bkl": "BKL", "benign keratosis": "BKL",
    "seborrheic keratosis": "BKL", "sek": "BKL", "seborrheic keratoses": "BKL",
    "solar lentigo": "BKL", "lentigo simplex": "BKL", "lichenoid keratosis": "BKL",
    "lichen planus like keratosis": "BKL", "pigmented benign keratosis": "BKL",
    # --- DF ---
    "dermatofibroma": "DF", "df": "DF",
    # --- vascular ---
    "vasc": "VASC", "vascular lesion": "VASC", "hemangioma": "VASC",
    "angioma": "VASC", "glomangioma": "VASC", "angiokeratoma": "VASC", "pyogenic granuloma": "VASC",
    "cherry angioma": "VASC", "lymphangioma": "VASC",
    # --- unknown / not recorded ---
    "unknown": "UNK", "unk": "UNK", "none": "UNK", "nan": "UNK", "": "UNK",
    "indeterminate": "UNK", "other": "OTHER",
    "cafe au lait macule": "OTHER", "atypical melanocytic proliferation": "UNK",
    "verruca": "OTHER", "wart": "OTHER", "molluscum": "OTHER",
    "scar": "OTHER", "acrochordon": "OTHER", "clear cell acanthoma": "OTHER",
}

# --------------------------------------------------------------------------
# 2. Malignancy policy
# --------------------------------------------------------------------------
# Outcome: malignant (MEL, BCC, SCC) versus benign (NV, BKL, DF, VASC, and AK
# when akiec_is_malignant=False, the setting used in the study). Actinic
# keratosis is premalignant and is counted on the benign side. HAM10000 /
# ISIC 2018 combine actinic keratosis with Bowen's disease in one AKIEC class,
# but every ISIC 2018 image is also in ISIC 2019, whose labels separate AK from
# SCC, and the ISIC 2019 row is the one kept after de-duplication.

MALIGNANT_CORE = {"MEL", "BCC", "SCC"}
BENIGN_CORE = {"NV", "BKL", "DF", "VASC"}

# --------------------------------------------------------------------------
# 3. Skin tone
# --------------------------------------------------------------------------
# Fitzpatrick types I-VI grouped into the three strata used in the analysis.
FST_GROUPS = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}

# DDI ships skin tone pre-binned into these same three strata.
DDI_TONE_MAP = {"12": "I-II", "34": "III-IV", "56": "V-VI",
                12: "I-II", 34: "III-IV", 56: "V-VI"}


# --------------------------------------------------------------------------
# 4. Runtime configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Settings for one run. Written to outputs/run_config.json."""

    # --- where the data lives -------------------------------------------
    # Set data_root to the parent that contains the dataset folders. On Kaggle
    # that is normally '/kaggle/input'; locally it might be './data'.
    data_root: Path = Path(os.environ.get("DERM_DATA_ROOT", "/kaggle/input"))
    cache_dir: Path = Path(os.environ.get("DERM_CACHE", "./cache"))
    out_dir: Path = Path(os.environ.get("DERM_OUT", "./outputs"))

    # Explicit overrides. Anything set here wins over auto-discovery.
    # Keys must be one of DATASET_KEYS below.
    path_overrides: Dict[str, str] = field(default_factory=dict)

    # --- analysis policy -------------------------------------------------
    akiec_is_malignant: bool = False     # primary arm; True = sensitivity arm
    require_histopath: bool = True       # primary endpoint restriction

    # --- precision ---------------------------------------------------------
    min_events_per_subgroup: int = 25    # below this a stratum is descriptive only

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


DATASET_KEYS = [
    "isic2017_train", "isic2017_val", "isic2017_test",
    "isic2018_train", "isic2018_test", "isic2018_lesion_groupings",
    "isic2019_train", "isic2019_test",
    "isic2020_train", "isic2020_test",
    "fitzpatrick17k", "ddi", "pad_ufes_20",
]

# Ordered, *specific* discovery patterns. Order matters: the first pattern that
# matches wins, so the most distinctive filename goes first. Patterns are
# matched against the lower-cased full path, not just the basename — this is
# what stops '*2019*.csv' from swallowing an unrelated file.
DISCOVERY_PATTERNS: Dict[str, List[str]] = {
    "isic2017_train":  [r"isic-2017_training_part3_groundtruth", r"2017.*training.*groundtruth"],
    "isic2017_val":    [r"isic-2017_validation_part3_groundtruth", r"2017.*validation.*groundtruth"],
    "isic2017_test":   [r"isic-2017_test_v2_part3_groundtruth", r"2017.*test.*groundtruth"],
    "isic2018_train":  [r"isic2018_task3_training_groundtruth", r"2018.*task3.*training.*groundtruth", r"\bham10000_metadata"],
    "isic2018_test":   [r"isic2018_task3_test_groundtruth", r"2018.*task3.*test.*groundtruth"],
    "isic2018_lesion_groupings": [r"isic2018_task3_training_lesiongroupings", r"lesion.?grouping"],
    "isic2019_train":  [r"isic_2019_training_groundtruth", r"2019.*training.*groundtruth"],
    "isic2019_test":   [r"isic_2019_test_groundtruth", r"2019.*test.*groundtruth"],
    "isic2020_train":  [r"isic_2020_training_groundtruth", r"siim.*train\.csv$", r"(^|/)train\.csv$"],
    "isic2020_test":   [r"isic_2020_test_metadata", r"(^|/)test\.csv$"],
    "fitzpatrick17k":  [r"fitzpatrick17k", r"fitzpatrick.?17k"],
    "ddi":             [r"ddi_metadata", r"(^|/)ddi.*\.csv$"],
    "pad_ufes_20":     [r"pad.?ufes.?20.*metadata", r"pad.?ufes.*\.csv$"],
}

# Companion metadata files that carry age/sex/site but not the label.
METADATA_PATTERNS: Dict[str, List[str]] = {
    "isic2019_train": [r"isic_2019_training_metadata"],
    "isic2019_test":  [r"isic_2019_test_metadata"],
    "isic2018_train": [r"\bham10000_metadata"],
}


# --------------------------------------------------------------------------
# 5. Content-based discovery
# --------------------------------------------------------------------------
# Mirrors of the same release use different file names ("train.csv",
# "ISIC_2019_Training_GroundTruth.csv", ...) but keep the original column
# headers. Each file is therefore identified by a signature: a set of columns
# that must be present, and optionally a set that must be absent.

SIGNATURES: Dict[str, Dict[str, object]] = {
    # --- ISIC 2017: two one-hot columns, nothing else ---
    "isic2017_gt": {
        "require": {"image_id", "melanoma", "seborrheic_keratosis"},
        "score": 100,
    },
    # --- ISIC 2018 Task 3 == HAM10000 one-hot ground truth ---
    "isic2018_gt": {
        "require": {"image", "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"},
        "score": 100,
    },
    # --- HAM10000 metadata: carries dx_type ---
    "ham_metadata": {
        "require": {"lesion_id", "image_id", "dx", "dx_type"},
        "score": 110,
    },
    # --- ISIC 2019 ground truth: 8 classes + UNK, has SCC and AK (not AKIEC) ---
    "isic2019_gt": {
        "require": {"image", "MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"},
        "score": 100,
    },
    "isic2019_meta": {
        "require": {"image", "age_approx", "anatom_site_general"},
        "forbid": {"MEL", "target"},
        "score": 90,
    },
    # --- ISIC 2020 / SIIM: binary target ---
    "isic2020_train": {
        "require": {"image_name", "patient_id", "diagnosis", "benign_malignant", "target"},
        "score": 100,
    },
    "isic2020_test": {
        "require": {"image_name", "patient_id"},
        "forbid": {"target", "benign_malignant"},
        "score": 80,
    },
    # --- Fitzpatrick17k ---
    "fitzpatrick17k": {
        "require": {"md5hash", "fitzpatrick_scale"},
        "score": 100,
    },
    # --- DDI ---
    "ddi": {
        "require": {"skin_tone", "malignant"},
        "score": 100,
    },
    # --- PAD-UFES-20: note the dataset's own misspelling of 'fitzpatrick' ---
    "pad_ufes_20": {
        "require": {"img_id", "diagnostic", "biopsed"},
        "score": 100,
    },
}

# Looser signatures, used only when the strict ones above do not match (some
# mirrors drop columns, add an index column, or give HAM10000 labels as a
# single dx string instead of one-hot columns).
SIGNATURES.update({
    "ham_metadata_alt": {
        "require": {"image_id", "dx", "dx_type"},
        "score": 95,
    },
    "isic2018_gt_alt": {
        "require": {"image", "MEL", "NV", "BCC", "BKL"},
        "forbid": {"SCC"},
        "score": 85,
    },
    "isic2019_gt_alt": {
        "require": {"MEL", "NV", "BCC", "AK", "SCC"},
        "score": 90,
    },
    "fitzpatrick17k_alt": {
        "require": {"fitzpatrick_scale"},
        "score": 80,
    },
    "fitzpatrick17k_alt2": {
        "require": {"md5hash", "three_partition_label"},
        "score": 85,
    },
    "ddi_alt": {
        "require": {"DDI_file", "skin_tone"},
        "score": 95,
    },
    "ddi_alt2": {
        "require": {"skin_tone", "disease"},
        "score": 85,
    },
    "pad_ufes_alt": {
        "require": {"img_id", "diagnostic", "fitspatrick"},
        "score": 95,
    },
    "isic2017_gt_alt": {
        "require": {"melanoma", "seborrheic_keratosis"},
        "score": 90,
    },
})

# signature key -> canonical dataset key(s) it satisfies
SIG_TO_DATASET = {
    "isic2017_gt": "isic2017_train",
    "isic2018_gt": "isic2018_train",
    "ham_metadata": "isic2018_train_meta",
    "isic2019_gt": "isic2019_train",
    "isic2019_meta": "isic2019_train_meta",
    "isic2020_train": "isic2020_train",
    "isic2020_test": "isic2020_test",
    "fitzpatrick17k": "fitzpatrick17k",
    "ddi": "ddi",
    "pad_ufes_20": "pad_ufes_20",
    # loose fallbacks
    "ham_metadata_alt": "isic2018_train_meta",
    "isic2018_gt_alt": "isic2018_train",
    "isic2019_gt_alt": "isic2019_train",
    "fitzpatrick17k_alt": "fitzpatrick17k",
    "fitzpatrick17k_alt2": "fitzpatrick17k",
    "ddi_alt": "ddi",
    "ddi_alt2": "ddi",
    "pad_ufes_alt": "pad_ufes_20",
    "isic2017_gt_alt": "isic2017_train",
}

# --------------------------------------------------------------------------
# Kaggle datasets used in the reported run
# --------------------------------------------------------------------------
# Files inside each dataset are still identified by column signature. Kaggle
# mounts a dataset at /kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>,
# so both locations are checked.
PINNED_ROOTS: Dict[str, str] = {
    "ISIC 2017":       "/kaggle/input/datasets/johnchfr/isic-2017",
    "ISIC 2018 / HAM": "/kaggle/input/datasets/nightfury007/ham10000-isic2018-raw",
    "ISIC 2019":       "/kaggle/input/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification",
    "ISIC 2020":       "/kaggle/input/datasets/prashantjeswani/siimisic2020",
    "Fitzpatrick17k":  "/kaggle/input/datasets/nazmusresan/fitzpatrick17k",
    "DDI":             "/kaggle/input/datasets/souvikda/ddidiversedermatologyimages-multimodal-dataset",
    "PAD-UFES-20":     "/kaggle/input/datasets/maxjen/pad-ufes-20",
}


def resolve_pinned_roots(verbose: bool = True) -> Dict[str, Optional[Path]]:
    """Check each pinned dataset actually mounted, trying both Kaggle layouts."""
    out: Dict[str, Optional[Path]] = {}
    for name, raw in PINNED_ROOTS.items():
        p = Path(raw)
        cands = [p]
        # /kaggle/input/datasets/<user>/<slug>  <->  /kaggle/input/<slug>
        if "/datasets/" in raw:
            cands.append(Path("/kaggle/input") / p.name)
        else:
            cands.append(Path(raw.replace("/kaggle/input/", "/kaggle/input/datasets/")))
        hit = next((c for c in cands if c.exists()), None)
        out[name] = hit
        if verbose:
            if hit:
                n_csv = sum(1 for _ in hit.rglob("*.csv"))
                n_img = sum(1 for x in hit.rglob("*")
                            if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
                print(f"  [OK  ] {name:<16} {n_csv:>3} csv, {n_img:>7,} images   {hit}")
            else:
                print(f"  [MISS] {name:<16} not mounted at {raw}")
    return out

# Some releases ship train/val/test with IDENTICAL headers. Split them apart by
# looking at the path, and by size where the path is uninformative.
SPLIT_HINTS = [
    (r"valid", "val"), (r"test", "test"), (r"train", "train"),
]


def _read_header(path: Path) -> Optional[set]:
    """Read only the first line. Never loads the file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
        if not line:
            return None
        for sep in (",", ";", "\t"):
            cols = [c.strip().strip('"').strip("'") for c in line.rstrip("\r\n").split(sep)]
            if len(cols) > 1:
                return set(cols)
        return None
    except Exception:
        return None


def _n_rows(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return -1


def _split_of(path: Path) -> Optional[str]:
    s = str(path).lower().replace("\\", "/")
    for pat, name in SPLIT_HINTS:
        if re.search(pat, s):
            return name
    return None


def scan(cfg: Config) -> pd.DataFrame:
    """List every tabular file under data_root with its columns and best signature match."""
    rows = []
    if not cfg.data_root.exists():
        return pd.DataFrame(columns=["path", "rows", "n_cols", "match", "split", "columns"])

    IMG = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    SKIP = {".zip", ".gz", ".tar", ".7z", ".rar", ".h5", ".hdf5", ".npy", ".npz",
            ".pkl", ".pt", ".pth", ".ckpt", ".json", ".md", ".pdf", ".doc",
            ".docx", ".xlsx", ".ipynb", ".py", ".yml", ".yaml", ".cfg", ".ini"}
    # os.walk avoids a stat() call per file, which matters on network mounts
    # with ~90,000 images. Extensionless files are accepted because some
    # mirrors ship HAM10000_metadata without an extension.
    candidates = []
    for dirpath, _, filenames in os.walk(cfg.data_root, followlinks=False):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG or ext in SKIP:
                continue
            if ext and ext not in {".csv", ".tsv", ".txt", ".data"}:
                continue
            candidates.append(Path(dirpath) / fn)
    for p in sorted(candidates):
        cols = _read_header(p)
        if not cols or len(cols) < 2:
            continue
        best, best_score = None, 0
        for sig_key, sig in SIGNATURES.items():
            req = sig["require"]
            forbid = sig.get("forbid", set())
            if req.issubset(cols) and not (forbid & cols):
                sc = int(sig["score"]) + len(req)
                if sc > best_score:
                    best, best_score = sig_key, sc
        rows.append({
            "path": str(p), "rows": _n_rows(p), "n_cols": len(cols),
            "match": best or "-", "score": best_score, "split": _split_of(p),
            "columns": ", ".join(sorted(cols))[:180],
        })
    return pd.DataFrame(rows)


def discover(cfg: Config, verbose: bool = True) -> Dict[str, Path]:
    """Identify every dataset file by its column signature.

    Falls back to filename matching only for datasets that no signature matched.
    """
    found: Dict[str, Path] = {}

    # 1. explicit overrides always win
    for k, v in cfg.path_overrides.items():
        p = Path(v)
        if not p.exists():
            raise FileNotFoundError(f"override for {k!r} does not exist: {p}")
        found[k] = p
        if verbose:
            print(f"  [SET ] {k:<24} {p}  (explicit override)")

    if not cfg.data_root.exists():
        print(f"\n  WARNING: data_root does not exist: {cfg.data_root}")
        return found

    inv = scan(cfg)
    if inv.empty:
        print(f"\n  WARNING: no readable CSV files under {cfg.data_root}")
        return found

    matched = inv[inv["match"] != "-"].copy()

    # 2. assign each signature's best file.
    # Order by score so a strict signature always beats a loose fallback for the
    # same dataset key. Relying on alphabetical groupby order would be fragile.
    order = (matched.groupby("match")["score"].max()
             .sort_values(ascending=False).index.tolist())
    for sig_key in order:
        grp = matched[matched["match"] == sig_key]
        ds_key = SIG_TO_DATASET.get(sig_key)
        if ds_key is None or ds_key in found:
            continue

        # Sort largest-first. For train/val/test triples this makes the biggest
        # file the training set; for metadata files (e.g. HAM10000_metadata vs
        # the 1,512-row test GT that shares its columns) it makes the 10,015-row
        # training metadata win over the test file.
        grp = grp.sort_values("rows", ascending=False)
        base = ds_key
        if base.endswith("_meta"):
            # metadata: just take the largest; no split fan-out
            if base not in found:
                found[base] = Path(grp.iloc[0]["path"])
            continue
        for i, (_, r) in enumerate(grp.iterrows()):
            split = r["split"]
            if i == 0:
                key = base
            elif split in ("val", "test") and base.endswith("_train"):
                key = base.replace("_train", f"_{split}")
            else:
                continue
            if key not in found:
                found[key] = Path(r["path"])

    # 2b. HAM metadata doubles as the 2018 training ground truth (it carries both
    # `dx` and `dx_type`). If no separate one-hot GT was found, use the largest
    # HAM-signature file as the training source as well.
    if "isic2018_train" not in found:
        ham_files = matched[matched["match"].isin(["ham_metadata", "ham_metadata_alt"])]
        if len(ham_files):
            biggest = ham_files.sort_values("rows", ascending=False).iloc[0]
            found["isic2018_train"] = Path(biggest["path"])

    # 3. filename fallback for anything still missing
    for key, patterns in DISCOVERY_PATTERNS.items():
        if key in found:
            continue
        for p in cfg.data_root.rglob("*.csv"):
            s = str(p).lower().replace("\\", "/")
            if any(re.search(pat, s) for pat in patterns):
                found[key] = p
                break

    if verbose:
        print(f"  scanned {len(inv)} csv files under {cfg.data_root}")
        print(f"  identified {len(matched)} by column signature\n")
        core = ["isic2017_train", "isic2018_train", "isic2018_train_meta",
                "isic2019_train", "isic2019_train_meta", "isic2020_train",
                "fitzpatrick17k", "ddi", "pad_ufes_20"]
        for k in core + [x for x in sorted(found) if x not in core]:
            if k in found:
                n = _n_rows(found[k])
                print(f"  [OK  ] {k:<24} n={n:>7,}  {found[k]}")
            elif k in core:
                print(f"  [MISS] {k:<24} not found")

        unmatched = inv[inv["match"] == "-"]
        if len(unmatched):
            print(f"\n  {len(unmatched)} CSV file(s) not matched to a dataset:")
            for _, r in unmatched.head(12).iterrows():
                print(f"    {Path(r['path']).name:<44} cols: {r['columns'][:90]}")

    return found


