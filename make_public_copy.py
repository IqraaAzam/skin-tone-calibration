"""Write a copy of this repository with the DDI rows removed.

The Diverse Dermatology Images research use agreement does not allow any part of
that dataset to be distributed, published or reproduced, so the per-image rows
that come from DDI cannot be published. This script copies everything else and
removes those rows from the cohort and prediction files. Aggregate results
(Tables 1-4, the figures and the logs) are kept, because they are summaries
rather than copies of the dataset.

Anyone with access to DDI can restore the rows by running notebooks 01 and 02.

    python make_public_copy.py ../skin-tone-calibration-public
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FILES_WITH_ROWS = [
    "results/cohorts/cohort_arm_a_endpoint.csv",
    "results/cohorts/cohort_arm_b_endpoint.csv",
    "results/predictions/predictions_external_armA.csv",
    "results/predictions/predictions_external_armB.csv",
]

dest = Path(sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "skin-tone-calibration-public").resolve()
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(HERE, dest, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", ".git"))

for rel in FILES_WITH_ROWS:
    p = dest / rel
    df = pd.read_csv(p, low_memory=False)
    kept = df[df["source"] != "DDI"]
    kept.to_csv(p, index=False)
    print(f"{rel}: removed {len(df) - len(kept):,} DDI rows, {len(kept):,} remain")

print(f"\nwritten to {dest}")
