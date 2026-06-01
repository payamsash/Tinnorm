# Tinnorm — Project Overview & Pipeline Reference

> Single authoritative document. Replaces the earlier `PIPELINE.md` and `PROJECT_OVERVIEW.md`.  
> **Data root:** `/Volumes/Extreme_SSD/payam_data/Tinnorm/`  
> **Code root:** `/Users/payamsadeghishabestari/Tinnorm/`  
> All scripts are run from the `src/` directory.

---

## 1. What This Project Is

A multi-site resting-state EEG study aiming to **classify chronic tinnitus patients vs. healthy controls** and **characterise how their brain activity deviates from normality**. The dataset comes from the **TIDE consortium** (7 sites, ~544 subjects after exclusions: 276 controls, 268 tinnitus). The central idea is normative modelling: train a Bayesian regression model on controls only, then quantify how far each tinnitus subject falls outside the normal distribution — per brain region, per frequency band, per modality.

---

## 2. Dataset

| Site | Code prefix | Country |
|------|-------------|---------|
| Austin | 1xxxxx | USA |
| Dublin | 2xxxxx | Ireland |
| Ghent | 3xxxxx | Belgium |
| Illinois | 4xxxxx | USA |
| Regensburg | 5xxxxx | Germany |
| Tübingen | 6xxxxx | Germany |
| Zürich | 7xxxxx | Switzerland |

- **~571 raw subjects** in BIDS format; ~544 retained after quality filtering.
- **Group label**: `esit_a17 ≤ 4 → tinnitus (group=1)`, else control `(group=0)`.
- **Key covariates**: age, sex, `PTA4_mean` (pure-tone average hearing threshold), site.
- Two subjects (`70072`, `70079`) excluded for elevated HADS anxiety/depression scores.
- Clinical scores available: **THI** (Tinnitus Handicap Inventory, 0–100), **TFI** (Tinnitus Functional Index, 0–100), **PTA4_HF** (high-frequency hearing threshold).

---

## 3. Pipeline Phases

### Phase 0 — One-time data preparation

| Script | What it does |
|--------|-------------|
| `00_create_zurich_audio_df.py` | Parses Zürich audiometry `.mat` files → tidy CSV |
| `00_create_zurich_quest_df.py` | Parses Zürich Unipark questionnaire files (THI, demographics) → Regensburg-style CSV |
| `00_convert_to_tide_ids.py` | Renames raw EEG files to the TIDE consortium subject-ID convention |
| `00_compute_aparc_adj_matrix.py` | Builds the FreeSurfer `aparc` (Desikan-Killiany) region adjacency matrix used by graph-metric scripts |
| `01_create_tinnorm_df.py` | Merges questionnaire + audiometry across all 7 sites → `material/master.csv` |
| `01b_filter_master.py` | Applies exclusion criteria (HADS, data quality flags) → `material/master_clean.csv` |

---

### Phase 1 — Demographics & data quality

| Script | What it does |
|--------|-------------|
| `02_plot_demographics.py` | Ridge/KDE plots of age per site × group × sex; cross-checks subject counts |
| `02b_plot_audiograms.py` | Site-level audiogram profiles and PTA4 distributions |
| `02c_stats_checks.py` | Formal demographics table, Mann-Whitney / chi-square group comparisons, VIF multicollinearity check, Spearman correlation matrix, site × group balance, THI severity distribution, PTA vs THI scatter, normality tests |
| `02d_eeg_quality.py` | Five resting-state EEG quality metrics per subject (alpha SNR, 1/f exponent, muscle index, line-noise ratio, scree ratio); per-site boxplots |

---

### Phase 2 — Preprocessing

Script: **`04_preprocess.py`**

Three sequential preprocessing levels saved as `.fif` epoch files:

| Level | Operations |
|-------|-----------|
| `preproc_1` | Channel drop → montage → band-pass 1–100 Hz → resample 250 Hz → average re-reference → 10 s fixed-length epochs |
| `preproc_2` | `preproc_1` + **AutoReject** (automated bad-epoch/channel interpolation) — *default for all downstream analyses* |
| `preproc_3` | `preproc_2` + **ICA** artifact removal via ICLabel (drops eye/muscle/heart components) |

Site-specific handling:
- **Zürich**: eyes-open segments extracted from event triggers (5 × 60 s blocks, 3 s guard removed); subject-specific trigger codes.
- **Austin**: montage converted from HydroCel-64 to easycap-M1 for ICA consistency.

An MNE HTML report is generated per subject. BIDS conversion handled by **`03_convert_to_bids.py`**.

---

### Phase 3 — Feature extraction

Script: **`05_extract_features.py`**

Computed at all 3 preprocessing levels, in both **sensor** and **source** space (source reconstruction: dSPM minimum-norm inverse, fsaverage template, `aparc` atlas → **68 ROIs**).

| Feature type | Description | Columns per ROI |
|-------------|-------------|----------------|
| `power` | Band-limited spectral power (trapz over Welch PSD) | 1 per band |
| `aperiodic` | FOOOF 1/f fit: `offset` + `exponent` | 2 (no bands) |
| `conn` (PLI / PLV / COH) | CWT-Morlet time-resolved functional connectivity | 68×68 matrix per band |
| `regional` | Anatomically-weighted node strength (adjacency-masked) | 1 per band |
| `global` | Total unweighted node strength | 1 per band |
| `graph` | Clustering coefficient + local efficiency (from `07b`) | 2 per band |

**Frequency bands**: delta (1–6 Hz), theta (6.5–8.5 Hz), alpha_0/1/2, beta_0/1/2/3, gamma (30–40 Hz) — 10 bands total.

---

### Phase 4 — Harmonization

Script: **`06_harmonize.py`**

Uses **neuroHarmonize** (ComBat empirical Bayes) to remove inter-site effects while preserving age, sex, and PTA4_mean.

- ComBat model learned **on controls only**, then applied to all subjects.
- Two outputs per modality: `*_hm.csv` (harmonized) and `*_residual.csv` (site residuals, used as alternative ML feature set).
- Aperiodic features are not harmonized (just merged and saved).
- Output path: `harmonized/preproc_{1,2,3}/source/{modality}.csv`

Regional and global summary metrics derived from harmonized connectivity via **`07_create_regional_metrics.py`**. Weighted graph-theory metrics (clustering coefficient, global/local efficiency) computed by **`07b_create_graph_metrics.py`**.

---

### Phase 5 — Normative modelling

Script: **`08_create_norm_models.py`**

Fits **PCNtoolkit BLR** (Bayesian Linear Regression with B-spline basis, heteroskedastic noise, SinhArcSinh warping) normative models **on control subjects only** for every combination of modality × space × preprocessing level × connectivity mode.

Two model configurations:

| Config | Training set | Test set | Purpose |
|--------|-------------|---------|---------|
| `for_eval` | 80% controls (stratified split) | 20% controls | Model quality evaluation (EXPV, MSLL, SMSE) |
| `full_model` | **All controls** | **All tinnitus** | Unbiased clinical deviation estimation |

**Unbiased Z-scores** (required for classification and all downstream analyses):
- **Tinnitus**: `full_model/results/Z_test.csv` — never seen during training.
- **Controls**: `loso/loso_controls_Z.csv` — leave-one-site-out, each control's home site excluded during training → avoids circularity.

> ⚠️ `for_eval` Z-scores for controls come from an 80/20 split and are **only** valid for model diagnostics. Do not use them for classification or deviation analyses.

Model outputs saved under: `models/preproc_{1,2,3}/source/{modality}/{full_model,for_eval,loso}/`

---

### Phase 6 — Multi-modal diffusion metric

Script: **`09_multimodal_diffusion.py`**

Combines Z-scores from three normative models (power + regional + global) into a single per-subject, per-region **Mahalanobis deviation score**.

**Algorithm per ROI × frequency-band pair:**
1. Build 3D feature vector `[power_Z, regional_Z, global_Z]`
2. Estimate 3×3 covariance matrix from controls (regularised: +1e-4 · I)
3. Compute Mahalanobis distance from the control distribution for every subject
4. Average across frequency bands → one score per ROI

**Z-score sourcing**: controls from LOSO, tinnitus from full_model (unbiased, as above).

Output: `diffusive_mm/{space}_preproc_{level}_{conn_mode}.csv`
Columns: `subject_ids`, `group`, one column per ROI (68 total, named `{region}-lh` / `{region}-rh`).

Run for all combinations of `preproc_level ∈ {1, 2, 3}` × `conn_mode ∈ {pli, plv, coh}`.

---

## 4. Normative Model Diagnostics (inspect before ML)

| Script | Output | What it shows |
|--------|--------|---------------|
| `10_plot_nm_metrics.py` | `plots/nm_metrics/` | Brain surface maps + topomaps of EXPV, MSLL, SMSE per modality × freq band; cubehelix sequential palette; 2nd–98th percentile colormap clipping |
| `11_plot_deviations.py` | `plots/deviations/` | Group-level mean Z-score and % deviant (|Z| > 1.96) maps on fsaverage (4 views); IQR-based colormap clipping; difference maps (tinnitus − control) with diverging palette; all preproc levels × modalities × freq bands |
| `12_plot_centiles.py` | `plots/centiles/` | Normative centile trajectories (5th/50th/95th) vs covariate (age, PTA4) with individual subjects overlaid; requires `full_model` to be saved |

---

## 5. Classification

### Main pipeline — `13_compare_clfs.py`

Reads pre-computed features (harmonized, Z-score deviations, or diffusive_mm scores) and runs **site-stratified cross-validation** (`StratifiedGroupKFold`, held-out site per fold) with optional Optuna hyperparameter tuning.

**Supported feature modes:**

| `mode` | `data_mode` | Description |
|--------|------------|-------------|
| `diffusive_mm` | — | Multi-modal Mahalanobis distance per ROI (best-performing) |
| `power` / `aperiodic` / `regional` / `global` / `graph` | `deviation` | Z-scores from normative models |
| `power` / `regional` / `global` | `residual` | ComBat site residuals |

**Supported classifiers:** RF (Random Forest), SVM, LGBM (LightGBM + Optuna Bayesian HPO).

**Feature selection options (inside CV, no leakage):** none, `kbest` (SelectKBest f_classif), `rfe` (RFE with RF base), `elasticnet` (SelectFromModel with sparse logistic).

**Scenarios defined in `__main__`:**

| Label | Model | Notes |
|-------|-------|-------|
| `preproc{1,2,3}_lgbm_tuned` | LGBM + Optuna | Preprocessing level sweep |
| `preproc2_{rf,svm}` | RF / SVM | Classifier comparison at preproc_2 |
| `preproc2_lgbm_kbest50` | LGBM + KBest | Feature selection comparison |
| `preproc2_rf_rfe30` | RF + RFE-30 | Feature selection comparison |
| `thi{25,36}_preproc2_lgbm` | LGBM | THI severity threshold sweep |
| `preproc2_lgbm_graph` | LGBM | Graph topology Z-scores |

Results saved under: `clfs/{scenario_label}/permutation/{timestamp}/`

### Plotting results — `14_plot_clf_results.py`

Loads all saved `clfs/` results and produces:
- Permutation-test histograms (real AUC vs null distribution, p-value annotated)
- ROC + PR curve panels per scenario
- Multi-metric bar charts across all scenarios
- Summary CSV: `results/tables/all_scenario_metrics.csv`

---

## 6. Explainability & Interpretability

| Script | What it does |
|--------|-------------|
| `16_explain_clfs.py` | **SHAP** values computed inside LOSO folds (TreeExplainer, interventional); beeswarm summary, feature scatter plots, hierarchical clustering bar, waterfall plots (TP/FN/FP/TN), fold-stability boxplot, subject × feature heatmap, THI/TFI–SHAP correlation scatters, misclassification analysis, THI threshold sweep |
| `20_lime_pdp.py` | **Permutation importance** (pooled over held-out folds); **partial dependence plots** (mean ± SD across LOSO folds + rug plots); **LIME** explanation for the highest-confidence tinnitus subject per site |

---

## 7. Comparative Analyses

| Script | What it does |
|--------|-------------|
| `17_compare_preproc_levels.py` | LOSO-CV of preproc levels 1/2/3 using diffusive_mm; ROC+PR with fold lines, AUC bar chart, per-site probability strip+boxen plots, metrics heatmap |
| `18_compare_modes.py` | **A** — residual vs deviation vs diffusive feature representation; **B** — RF vs SVM vs LGBM classifier comparison; **C** — no selection vs KBest-50 vs RFE-30; **D** — modality ablation: power / aperiodic / regional_coh / global_coh / graph_coh / diffusive_mm |

---

## 8. Advanced Analyses

| Script | What it does |
|--------|-------------|
| `15_clinical_analysis.py` | Spearman correlations (FDR-corrected) between EEG features and THI/TFI; THI vs TFI agreement scatter; UMAP embedding (group + THI severity); **age-stratified deviation analysis** (violin + scatter by age tertile × group) |
| `19_network_lateralization.py` | **Yeo-7 network aggregation**: maps 68 aparc ROIs to 6 functional networks, radar chart + grouped bar chart of network-level deviation (tinnitus vs controls), FDR-corrected Mann-Whitney tests; **Lateralization index** LI = (LH − RH)/(LH + RH) for 4 auditory ROIs (transversetemporal, superiortemporal, supramarginal, insula), half-violin+strip plots, LH vs RH scatter |
| `21_thi_regression.py` | **Severity prediction**: Ridge + LGBMRegressor to predict continuous THI/TFI from diffusive_mm features (tinnitus subjects, GroupKFold site-level CV); actual-vs-predicted scatter, per-fold R² bars, feature importance comparison, residuals + Q-Q plot |
| `22_site_generalization.py` | **Per-site LOSO breakdown**: AUC with 1000-resample bootstrap 95% CI, horizontal bar chart + per-site ROC curves + confusion matrix heatmap; **Cross-site transfer matrix**: train on one site, test on another — 7×7 heatmap showing which site patterns transfer across sites |

---

## 9. Recommended Run Order

```bash
# Phase 0 — once per dataset update
python 00_create_zurich_audio_df.py
python 00_create_zurich_quest_df.py
python 00_convert_to_tide_ids.py
python 00_compute_aparc_adj_matrix.py
python 01_create_tinnorm_df.py
python 01b_filter_master.py

# Phase 1 — demographics & quality
python 02_plot_demographics.py
python 02b_plot_audiograms.py
python 02c_stats_checks.py
python 02d_eeg_quality.py

# Phase 2 — preprocessing (run once, saves .fif files)
python 03_convert_to_bids.py
python 04_preprocess.py

# Phase 3 — feature extraction (run once, saves zipped CSVs)
python 05_extract_features.py
python 06_harmonize.py
python 07_create_regional_metrics.py
python 07b_create_graph_metrics.py

# Phase 4 — normative models (run once, saves model directories)
python 08_create_norm_models.py     # trains BLR for all modalities
python 09_multimodal_diffusion.py   # computes diffusive_mm scores

# Phase 5 — model diagnostics (optional, inspect before ML)
python 10_plot_nm_metrics.py
python 11_plot_deviations.py
python 12_plot_centiles.py          # only if full_model was saved to disk

# Phase 6 — classification
python 13_compare_clfs.py           # runs all pre-defined scenarios → clfs/
python 14_plot_clf_results.py       # reads clfs/, produces figures

# Phase 7 — explainability
python 16_explain_clfs.py
python 20_lime_pdp.py

# Phase 8 — comparative analyses
python 17_compare_preproc_levels.py
python 18_compare_modes.py          # sections A/B/C/D

# Phase 9 — advanced analyses
python 15_clinical_analysis.py
python 19_network_lateralization.py
python 21_thi_regression.py
python 22_site_generalization.py
```

---

## 10. Directory Structure

```
/Users/payamsadeghishabestari/Tinnorm/          ← Code repository
├── src/                                         ← All pipeline scripts (00–22)
└── material/
    ├── master.csv                               ← Raw merged subject metadata
    ├── master_clean.csv                         ← After exclusion criteria (used by all scripts)
    ├── questionnaires/                          ← Per-site questionnaire CSVs
    ├── audiograms/                              ← Per-site audiometry XLSXs
    ├── aparc_adjacency.csv                      ← Anatomical adjacency matrix (68×68)
    └── ant_to_tide.csv                          ← ID mapping (Antinomics → TIDE)

/Volumes/Extreme_SSD/payam_data/Tinnorm/        ← Data + results (external SSD)
├── BIDS/                                        ← Raw EEG in BIDS format (~571 subjects)
├── preprocessed/                                ← .fif epoch files (preproc_1/2/3 per subject)
├── features/                                    ← Per-subject zipped feature CSVs
├── harmonized/
│   └── preproc_{1,2,3}/source/
│       ├── power_hm.csv                         ← ComBat-harmonized spectral power
│       ├── aperiodic_hm.csv
│       ├── regional_{coh,pli,plv}_hm.csv
│       ├── global_{coh,pli,plv}_hm.csv
│       ├── graph_{coh,pli,plv}_hm.csv
│       └── *_residual.csv                       ← Site residuals (alternative ML features)
├── models/
│   └── preproc_{1,2,3}/source/{modality}/
│       ├── for_eval/results/                    ← 80/20 split metrics (EXPV, MSLL, SMSE)
│       ├── full_model/results/
│       │   ├── Z_train.csv                      ← Z-scores for controls (training fold)
│       │   └── Z_test.csv                       ← Z-scores for tinnitus (UNBIASED)
│       └── loso/
│           └── loso_controls_Z.csv              ← Unbiased control Z-scores (LOSO)
├── diffusive_mm/
│   └── source_preproc_{1,2,3}_{coh,pli,plv}.csv  ← Mahalanobis deviation scores (68 ROIs)
├── clfs/
│   └── {scenario_label}/
│       └── permutation/{timestamp}/
│           ├── metrics.csv                      ← AUC, balanced_accuracy, F1, ...
│           ├── y.npy                            ← True labels
│           ├── y_prob.npy                       ← Predicted probabilities
│           └── y_pred.npy
├── plots/
│   ├── nm_metrics/                              ← Script 10 outputs
│   ├── deviations/                              ← Script 11 outputs
│   └── centiles/                                ← Script 12 outputs
└── results/
    ├── figures/                                 ← Scripts 14–22 publication figures (PDF)
    └── tables/                                  ← Scripts 14–22 summary CSVs
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Normative model trained on controls only** | Defining architectural choice — tinnitus subjects are never seen during BLR training, so their Z-scores are fully unbiased |
| **LOSO Z-scores for controls, full_model for tinnitus** | `for_eval` only covers 20% of controls (stratified split); LOSO covers all 276 controls with site-held-out training, giving unbiased Z-scores for every control |
| **preproc_2 as the working default** | AutoReject-cleaned epochs balance data quality and data retention; ICA (preproc_3) removes more artifacts but is more aggressive |
| **Source space only** | Sensor features are extracted but harmonization and normative modelling currently only run on source-reconstructed signals (dSPM, fsaverage, 68 aparc ROIs) |
| **COH as the primary connectivity mode** | PLI and PLV are computed but coherence showed better normative model fit metrics; the pipeline supports all three |
| **diffusive_mm as the primary ML feature** | Multi-modal Mahalanobis distance (power + regional + global Z-scores) outperforms single-modality Z-scores or residuals for classification |
| **IQR-based colormap clipping** | With only 68 ROIs, 2nd–98th percentile clips ≤1 ROI; Tukey IQR fences (Q1 ± 1.5 × IQR) are more robust to single outlier regions |
| **Site-stratified CV throughout** | `StratifiedGroupKFold` with sites as groups ensures the model is never tested on a site it trained on — critical for multi-site generalizability claims |

---

## 12. Known Limitations & Future Directions

- **Modest classification AUC (~0.60–0.70)**: expected given the heterogeneity of tinnitus; the normative deviation approach characterises the population rather than providing a simple biomarker.
- **Sensor space unused downstream**: extracting sensor features adds runtime but they are currently never passed to harmonization or normative modelling.
- **PLI/PLV unexplored in NM**: only COH enters the diffusive_mm pipeline; PLI/PLV may capture different aspects of phase coupling.
- **`_read_the_file` duplicated**: the data-loading function is independently maintained in `13_compare_clfs.py` and `16_explain_clfs.py`; a shared `utils.py` module would reduce drift.
- **Models not saved by default**: `12_plot_centiles.py` requires saved model objects (`NormativeModel.load`), which are large and not written to disk unless explicitly configured.
- **HBR alternative**: PCNtoolkit also offers Hierarchical Bayesian Regression (HBR) designed for multi-site data; worth comparing against BLR for this dataset.
- **Band-resolved diffusive features**: `09_multimodal_diffusion.py` currently averages Mahalanobis distances across frequency bands per ROI; keeping the per-band resolution (68 ROIs × 10 bands = 680 features) may improve classifier performance for LGBM.
