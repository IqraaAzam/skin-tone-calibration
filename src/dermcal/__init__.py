"""dermcal: data preparation and evaluation code for the study
"Calibration and net benefit of a skin cancer triage model in lighter and
darker skin".

    from dermcal import Config, discover, load_all
    cfg = Config(data_root="/kaggle/input")
    df = load_all(discover(cfg), cfg)

Modules
    config    paths, diagnosis vocabulary, dataset discovery
    schemas   one loader per dataset, all returning the same columns
    dedup     identifier overlap between releases and grouping structure
    cohort    eligibility criteria, attrition and verification tables
    power     precision attainable in each evaluation stratum
    metrics   discrimination, calibration and net benefit with bootstrap intervals
    plots     cohort description figures
"""

from .config import Config, discover, DATASET_KEYS, DX_CANONICAL, FST_GROUPS
from .schemas import (
    load_all, CANONICAL_COLS, normalise_dx,
    coalesce_across_releases, enrich_confirmation_from_archive,
)
from . import dedup, cohort, power, metrics, plots

__version__ = "1.0.0"
__all__ = [
    "Config", "discover", "load_all", "CANONICAL_COLS", "DATASET_KEYS",
    "DX_CANONICAL", "FST_GROUPS", "normalise_dx", "coalesce_across_releases",
    "enrich_confirmation_from_archive", "dedup", "cohort", "power", "metrics", "plots",
]
