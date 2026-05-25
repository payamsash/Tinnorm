import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from neuroHarmonize import (
    harmonizationLearn,
    harmonizationApply
    )


def _select_bio_covars(df_merged: pd.DataFrame,
                       bio_covars: list,
                       r_threshold: float = 0.85) -> list:
    """
    Drop PTA4_mean from bio_covars if it is highly correlated with PTA4_HF
    (|Pearson r| > r_threshold), to avoid multicollinearity in ComBat.

    PTA4_HF is retained because it is more specific to the high-frequency
    hearing loss associated with tinnitus.  Run 02c_stats_checks.py section 12
    for a full collinearity report with scatter plot and VIF table.
    """
    pair = [c for c in ["PTA4_mean", "PTA4_HF"]
            if c in bio_covars and c in df_merged.columns]
    if len(pair) < 2:
        return bio_covars

    valid = df_merged[pair].dropna()
    r = float(np.corrcoef(valid["PTA4_mean"].values, valid["PTA4_HF"].values)[0, 1])
    print(f"  PTA4_mean–PTA4_HF Pearson r = {r:.3f}  (threshold = {r_threshold})")

    if abs(r) > r_threshold:
        print("  → High collinearity: dropping PTA4_mean, retaining PTA4_HF in ComBat.")
        return [c for c in bio_covars if c != "PTA4_mean"]

    print("  → Collinearity OK: using both PTA4_mean and PTA4_HF.")
    return bio_covars

def harmonize(                
                preproc_level,
                space,
                modality,
                conn_mode,
                saving_dir,
                features_dir
                ):
    """
    Harmonize subject-level neurophysiological features across acquisition sites
    using neuroHarmonize (ComBat-style empirical Bayes correction).

    This function:
    1. Loads subject-wise feature files matching the specified preprocessing level,
        space, and modality.
    2. Aggregates features per subject (mean across channels / connections).
    3. Merges features with subject covariates (site, age, sex, PTA4_mean, group).
    4. Learns a harmonization model on control subjects only (group == 0).
    5. Applies the learned model to all subjects.
    6. Saves both harmonized data ("hm") and residualized data ("residual") as CSVs.

    Parameters
    ----------
    preproc_level : int
        Preprocessing level identifier used in feature filenames.
    space : str
        Feature space (e.g., "sensor" or "source").
    modality : str
        Feature modality (e.g., "power", "conn", "aperiodic").
    conn_mode : str
        Connectivity metric (e.g., "pli", "plv", "coh").
        Only used when modality == "conn".
    saving_dir : pathlib.Path
        Base directory where harmonized CSV files will be saved.
    features_dir : pathlib.Path
        Directory containing extracted feature files.

    Notes
    -----
    - Subject IDs are inferred from filenames.
    - Covariates are loaded from '../material/master_clean.csv'.
    - Site effects are harmonized while preserving biological covariates
        (age, sex, PTA4_mean).
    - For 'aperiodic' modality, no harmonization is applied; merged data
        are saved directly for consistency.
    - Output filenames encode preprocessing level, space, modality,
        and (if applicable) connectivity mode.
    """

    ## Build expected file paths from master_clean subject IDs.
    ## This avoids scanning the whole directory (which could pick up corrupted
    ## or unrelated files and cause zipfile.BadZipFile errors).
    df_master = pd.read_csv("../material/master_clean.csv")
    valid_ids = df_master["subject_id"].astype(str).tolist()
    suffix = f"{modality}_{space}_preproc_{preproc_level}.zip"

    fnames, missing = [], []
    for sid in valid_ids:
        fpath = features_dir / f"sub-{sid}_{suffix}"
        if fpath.exists():
            fnames.append((sid, fpath))
        else:
            missing.append(sid)

    if missing:
        print(f"  {len(missing)} subjects have no {suffix} file "
              f"(first 5: {missing[:5]})")
    if not fnames:
        print(f"  No feature files found for {suffix} — skipping.")
        return

    ## read and create data matrix
    print(f"reading the {modality} in {space} features ({len(fnames)} subjects)...")
    dfs_list = []
    for subject_id, fname in tqdm(fnames):
        try:
            df_subject = pd.read_csv(fname, index_col=None)
        except Exception as e:
            print(f"  Warning: could not read {fname.name}: {e} — skipping.")
            continue

        if modality == "conn":
            df_subject = df_subject.set_index(df_subject.columns[0]).T
            df_subject.columns.name = None
            df_subject = df_subject.filter(regex=rf'_{conn_mode}$')

        df_subject.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
        df_mean = df_subject.mean(axis=0).to_frame().T
        df_mean["subject_id"] = subject_id
        dfs_list.append(df_mean)

    if not dfs_list:
        print(f"  All reads failed for {suffix} — skipping.")
        return

    df_data = pd.concat(dfs_list)
    df_data['subject_id'] = df_data['subject_id'].astype(str)

    ## create covariates df and match 2 dfs
    df_q = pd.read_csv("../material/master_clean.csv")
    _covar_candidates = ["site", "age", "sex", "subject_id", "PTA4_mean", "PTA4_HF", "group"]
    cols = [c for c in _covar_candidates if c in df_q.columns]
    # Biological covariates passed to ComBat (site handled separately as batch effect)
    _bio_covars = [c for c in ["age", "sex", "PTA4_mean", "PTA4_HF"] if c in df_q.columns]
    df_covars = df_q[cols]
    df_covars.rename(columns={"site": "SITE"}, inplace=True)
    df_covars['subject_id'] = df_covars['subject_id'].astype(str)
    df_merged = pd.merge(df_data, df_covars, on='subject_id', how='inner')
    df_merged.sort_values(by="subject_id", inplace=True)

    dropped_from_df1 = set(df_data['subject_id']) - set(df_covars['subject_id'])
    print(f"subjects missing in covar: {list(sorted(dropped_from_df1))}")

    dropped_from_df2 = set(df_covars['subject_id']) - set(df_data['subject_id'])
    print(f"subjects missing in data: {list(sorted(dropped_from_df2))}")

    ## Check PTA collinearity — drop PTA4_mean if highly correlated with PTA4_HF
    _bio_covars = _select_bio_covars(df_merged, _bio_covars)

    ## harmonize
    if modality == "aperiodic":
        for hm_mode in ["hm", "residual"]:
            fname_save = saving_dir / f"preproc_{preproc_level}" / space / f"{modality}_{hm_mode}.csv"
            df_merged = df_merged.fillna(df_merged.mean(numeric_only=True)) # fix NaN values
            df_merged.to_csv(fname_save, index=False)

    else:
        # Fill NaN in features with column means before ComBat.
        # NaN from bad channels/epochs cause harmonizationLearn to return
        # NaN residuals and zero harmonized values.
        feature_slice = df_merged.iloc[:, :-len(cols)]
        nan_count = int(feature_slice.isna().sum().sum())
        if nan_count > 0:
            print(f"  Filling {nan_count} NaN values in features with column means.")
            col_means = feature_slice.mean()
            df_merged.iloc[:, :-len(cols)] = feature_slice.fillna(col_means)

        data_full = df_merged.iloc[:, :-len(cols)].to_numpy().astype(np.float64)

        # Sensor-space power values are on the order of 1e-12 V²/Hz — too small
        # for ComBat's numerical routines (rounds to zero). Log10-transform brings
        # them to a normal scale and also makes the distribution approximately
        # Gaussian, satisfying ComBat's assumptions. Source power is already on a
        # large scale so no transform is applied there.
        log_transform = (modality == "power" and space == "sensor")
        if log_transform:
            data_full = np.log10(np.clip(data_full, a_min=1e-30, a_max=None))
            print("  Applied log10 transform to sensor power features.")

        df_cov_full = df_merged[["SITE"] + _bio_covars].reset_index(drop=True)

        print(f"  data_full shape: {data_full.shape}  "
              f"NaN: {np.isnan(data_full).sum()}  "
              f"sites: {df_cov_full['SITE'].unique().tolist()}")

        ## get the residuals (s_data)
        _, _, s_data = harmonizationLearn(
                                            data_full,
                                            df_cov_full,
                                            eb=True,
                                            seed=0,
                                            return_s_data=True
                                            )

        ## learn the model on controls
        df_train = df_merged.query('group == 0')
        data_controls = df_train.iloc[:, :-len(cols)].to_numpy().astype(np.float64)
        if log_transform:
            data_controls = np.log10(np.clip(data_controls, a_min=1e-30, a_max=None))
        df_cov_controls = df_train[["SITE"] + _bio_covars].reset_index(drop=True)
        hm_model, _ = harmonizationLearn(
                                            data_controls,
                                            df_cov_controls,
                                            eb=True,
                                            seed=0,
                                            return_s_data=False
                                            )
        
        ## apply the model on all subjects
        bayes_data = harmonizationApply(
                                        data_full,
                                        df_cov_full,
                                        hm_model
                                        )
        
        ## replace and save
        column_names = df_merged.columns[:-len(cols)]
        for data, title in zip([bayes_data, s_data], ["hm", "residual"]):
            df_hm = pd.DataFrame(data, columns=column_names)
            df_hm = df_hm.round(4)
            df_hm = pd.concat([
                                df_hm,
                                df_merged[df_merged.columns[-len(cols):]].reset_index(drop=True)],
                                axis=1
                                )
            if modality == "conn":
                fname_save = saving_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_{title}.csv"
            else:
                fname_save = saving_dir / f"preproc_{preproc_level}" / space / f"{modality}_{title}.csv"

            df_hm.to_csv(fname_save, index=False)

if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    features_dir = tinnorm_dir / "features"
    hm_dir = tinnorm_dir / "harmonized"
    os.makedirs(hm_dir, exist_ok=True)
    
    preproc_levels = [1, 2, 3]
    spaces = ["sensor", "source"]
    modalities = ["power", "conn", "aperiodic"][2:]
    conn_modes = ["pli", "plv", "coh"]

    for preproc_level in preproc_levels:
        os.makedirs(hm_dir / f"preproc_{preproc_level}", exist_ok=True)

        for space in spaces:
            os.makedirs(hm_dir / f"preproc_{preproc_level}" / space, exist_ok=True)

            for modality in modalities:
                if modality == "conn":
                    for conn_mode in conn_modes:
                        fname_save = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_hm.csv"
                        if fname_save.exists():
                            continue
                        harmonize(preproc_level, space, modality, conn_mode, hm_dir, features_dir)
                else:
                    fname_save = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_hm.csv"
                    if fname_save.exists():
                        continue
                    harmonize(preproc_level, space, modality, None, hm_dir, features_dir)