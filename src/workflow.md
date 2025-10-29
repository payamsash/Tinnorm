## Tinnorm workflow

1) Download all resting-state data and convert them into BIDS.
2) Preprocessing and create report per each.
   - preproc_1
   - preproc_2
   - preproc_3
3) Feature extraction
   - compute band power (per epoch -> per subject)
   - connectivity (per epoch -> per subject)
   - aperiodic (offset + exponent) (per subject)

4) Site Diagnosis
   - Compute per-feature percent variance explained by Site
   - PCA/UMAP of feature matrixes colored by site

5) Harmonization
   
   - Implement neuroHarmonize with site as batch effect
   - maybe compute it only for controls and apply to all
   - Save harmonized features
   - Compute per-feature percent variance explained by Site
   - PCA/UMAP of feature matrixes colored by site

6) Split dataset
   
   - Primary: all controls for NM
   - LOSO-CV + (external test)

7) Model Selection and fitting
   
   - Only HBR (becasue of multi site)
   - fit on controls + CV for hyperparameter tuning
   - save the model

8) Cross validation and generalization
   
   - For each left-out site (7 folds), train on 6 sites
   - Predict on held-out site (controls and tinnitus).
   - Compute calibration metrics on held-out controls: mean(z), std(z) per feature (z = (y - yhat)/y_sd).
   - Aggregate per-site metrics across folds and save
   - per-site calibration plots.

9) Compute normative outputs for all subjects
    
   - Use best HBR
   - compute metrics per feature, per subject

1) Identify abnormal features
    
   - Site PCA / UMAP pre/post harmonization.
   - brain maps
   - LOSO-CV calibration plot (per-site mean/SD of z).
   - Centile (growth) curves for key features vs age and PTA.
   - Group mean topography of z-scores (tinnitus vs controls).
   - Per-subject radar + scalp topography report (exemplar patients).
   - Deviation burden histograms / ECDFs.
   - Cluster heatmap (subjects × features)
