# Calibration and net benefit of a skin cancer triage model in lighter and darker skin

Code, saved predictions and results for the manuscript

> Azam I, Saqib MS. *Calibration and net benefit of a skin cancer triage model in lighter and darker skin: external validation on biopsy-confirmed images.*

## Study in brief

An EfficientNet-B0 classifier was trained on 19,506 histopathology-confirmed dermoscopic images from ISIC 2017, 2019 and 2020 to separate melanoma and keratinocyte carcinoma from benign lesions, and recalibrated by temperature scaling. It was then evaluated on clinical photographs by skin tone:

| Analysis | Data | Comparison |
|---|---|---|
| Primary | DDI (every lesion biopsied) | Fitzpatrick I–II vs III–VI; III–IV and V–VI also reported separately |
| Secondary | PAD-UFES-20 (biopsied when clinically suspicious) | I–III vs IV–VI |
| Sensitivity | Fitzpatrick17k (atlas labels, not biopsy-confirmed) | I–III vs IV–VI |

Outcomes are discrimination (AUC), calibration (observed-to-expected ratio, calibration slope, decile calibration) and net benefit at a risk threshold of 0.10, each with 95% bootstrap intervals. Calibration slopes and decile plots are reported only for strata with at least 50 events. The analysis plan was fixed before the model was trained.

## Repository structure

```
notebooks/
  01_cohort_construction.ipynb   data sources, harmonisation, eligibility, cohorts      (Kaggle, internet)
  02_model_development.ipynb     grouped split, training, temperature scaling, scoring  (Kaggle, GPU)
  03_evaluation.ipynb            Tables 1-4 and Figures 1-4 from the saved predictions  (any computer)
src/dermcal/
  config.py     paths, diagnosis vocabulary, dataset discovery by column signature
  schemas.py    one loader per dataset, all returning the same columns
  dedup.py      identifier overlap between releases, grouping structure
  cohort.py     eligibility criteria, attrition, verification tables
  power.py      approximate precision per evaluation stratum
  metrics.py    AUC, O/E ratio, calibration slope, net benefit, bootstrap intervals
  plots.py      cohort description figures
results/
  cohorts/      cohort files, descriptive tables and figures from notebook 01
  predictions/  predicted risks from notebook 02
  model/        model weights (model_best.pt) and run settings
  logs/         console output of the training run and of the scoring run
  tables/       Tables 1-4 and decision curves from notebook 03
  figures/      Figures 1-4 from notebook 03
```

## Reproducing the results

Notebooks 01 and 02 are the versions that were run on Kaggle and are saved with their outputs; notebook 03 is saved with the outputs of the run reported in the manuscript.

**Tables and figures (a few minutes, no GPU or images needed).** The predictions and cohort files of the reported run are included, so notebook 03 can be run directly:

```
pip install -r requirements.txt
jupyter notebook notebooks/03_evaluation.ipynb
```

**Full pipeline.** Notebooks 01 and 02 were run on Kaggle.

1. Upload this repository to Kaggle as a dataset, or clone it into the notebook's working directory. The notebooks find it by looking for `src/dermcal`.
2. Create a notebook from `01_cohort_construction.ipynb`, attach the seven datasets below, turn internet access on and run all cells (about 20 minutes).
3. Save a version of notebook 01 so that its output is kept.
4. Create a notebook from `02_model_development.ipynb`, attach the same datasets, the repository and the output of notebook 01, turn the GPU on and run all cells. By default (`USE_RELEASED_MODEL = True`) it only scores the images, using the weights and temperature of the reported run. With `USE_RELEASED_MODEL = False` it trains a new model (about 40 minutes on an NVIDIA T4).
5. Copy the files written to `outputs/` by both notebooks into `results/cohorts/` and `results/predictions/`, and run notebook 03.

## Data and licences

Every dataset is obtained from its own source under its own terms; none of the images are in this repository.

**DDI rows are not published here.** The Diverse Dermatology Images research use agreement states that no part of the dataset may be distributed, published or reproduced, so the published copy of this repository carries no DDI rows in its cohort and prediction files, and notebook 03 skips the DDI sections when they are absent. Aggregate results for DDI remain in `results/tables`, in the figures and in the manuscript. Nothing else was changed: the DDI rows were dropped from the cohort and prediction files, and every other row and column is as the notebooks wrote it. To restore the rows, register for DDI at <https://ddi-dataset.github.io/>, then run notebooks 01 and 02.

The remaining per-image rows come from datasets whose terms allow redistribution with attribution: Fitzpatrick17k under CC BY-NC-SA 3.0, PAD-UFES-20 under CC BY 4.0, and the ISIC releases under the terms of the ISIC Archive. Those rows carry the diagnosis, confirmation method and skin tone recorded by the source, together with the risk predicted by our model. Non-commercial use and share-alike therefore apply to them, and the datasets must be cited as listed below.

The Kaggle mirrors used in the reported run:

| Dataset | Role | Kaggle dataset | Reference |
|---|---|---|---|
| ISIC 2017 | Development | `johnchfr/isic-2017` | Codella et al., ISBI 2018 |
| ISIC 2018 Task 3 (HAM10000) | Development | `nightfury007/ham10000-isic2018-raw` | Tschandl et al., Sci Data 2018 |
| ISIC 2019 (incl. BCN20000) | Development | `salviohexia/isic-2019-skin-lesion-images-for-classification` | Hernández-Pérez et al., Sci Data 2024 |
| ISIC 2020 | Development | `prashantjeswani/siimisic2020` | Rotemberg et al., Sci Data 2021 |
| DDI | External, primary | `souvikda/ddidiversedermatologyimages-multimodal-dataset` | Daneshjou et al., Sci Adv 2022 |
| PAD-UFES-20 | External, secondary | `maxjen/pad-ufes-20` | Pacheco et al., Data Brief 2020 |
| Fitzpatrick17k | External, sensitivity | `nazmusresan/fitzpatrick17k` | Groh et al., CVPR Workshops 2021 |

Notebook 01 also downloads the official ISIC label files from the ISIC challenge archive and the ISIC Archive metadata with `isic-cli`, which records how each diagnosis was confirmed.

## Model settings

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B0, ImageNet weights, single output unit |
| Input | 224 × 224 pixels, image only (no clinical metadata) |
| Augmentation | horizontal and vertical flips, rotation up to 20°, translation, scaling, colour jitter |
| Optimiser | AdamW, learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, one-cycle schedule, batch size 64, 4 epochs |
| Loss | binary cross-entropy, positive class weighted by the benign-to-malignant ratio |
| Split | 85% / 15% of groups (lesion, then patient, then image), seed 1234 |
| Checkpoint | highest held-out AUC (epoch 3) |
| Recalibration | temperature scaling on the held-out split, T = 1.343 |
| Intervals | 2,000 percentile bootstrap resamples, seed 1234 |

## Notes

- 3,366 of the 22,872 histopathology-confirmed development images are ISIC 2019 test images that are not in the Kaggle ISIC 2019 mirror, so the model was trained and tuned on 19,506 images (notebook 03, section 8).
- The checkpoint was selected and the temperature fitted on the same held-out split, so the internal validation estimates are somewhat optimistic. External estimates are not affected.
- Actinic keratosis is premalignant and is counted as benign. ISIC 2018 combines it with Bowen's disease in one class, but every ISIC 2018 image is also in ISIC 2019, which labels the two separately, and the ISIC 2019 label is used.
- For DDI the outcome is the dataset's own malignant flag, and for Fitzpatrick17k its three-way label (malignant, benign, non-neoplastic).
- DDI reports skin tone in three groups (I–II, III–IV, V–VI), so type IV cannot be separated from type III in DDI.
- In PAD-UFES-20 a lesion was biopsied when it was clinically suspicious, so its calibration describes a biopsied sample and is not a fair test of the model. `results/cohorts/data_provenance.csv` calls it a consistency set, which was the wording used when the cohorts were built.

## Licence

The code is under the MIT licence (see `LICENSE`). The data files under `results/` keep the terms of the datasets they come from, as described above.

## Requirements

Python 3.10 or later. Notebook 03 needs numpy, pandas, scipy and matplotlib; notebook 02 also needs PyTorch and torchvision; notebook 01 installs `isic-cli` itself. See `requirements.txt`.
