"""
09_multimodal_diffusion.py — Multi-modal Mahalanobis deviation score.

Stacks Z-scores from three normative models per brain region:

    power    spectral power deviation (10 freq bands per ROI)
    regional anatomically-weighted connectivity node strength deviation
    global   total node strength deviation

For each ROI, the 3D feature vector [power_Z, regional_Z, global_Z]
(averaged across frequency bands that exist in all three modalities) is
embedded in a Mahalanobis distance space estimated from controls.

Z-score sourcing (unbiased):
    controls  — loso/loso_controls_Z.csv  (held out by site during training)
    tinnitus  — full_model/results/Z_test.csv  (never seen during training)

With only 3 dimensions per ROI the covariance is well-conditioned even
with ~150 controls.  Regularization (1e-4 * I) is still applied.

Output: diffusive_mm/{space}_preproc_{level}_{conn_mode}.csv
        read by 13_compare_clfs.py via mode="diffusive_mm".
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _load_z(model_dir: Path) -> pd.DataFrame:
    """Load unbiased Z-scores: controls from LOSO, tinnitus from full model.

    Controls that were part of full-model training get unbiased z-scores from
    the leave-one-site-out (LOSO) model that held out their site.  Tinnitus
    subjects were never seen during training, so the full model is correct.
    """
    loso_path = model_dir / "loso" / "loso_controls_Z.csv"
    test_path  = model_dir / "full_model" / "results" / "Z_test.csv"

    for p, label in [(loso_path, "LOSO controls"), (test_path, "tinnitus Z_test")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    df_ctrl = pd.read_csv(loso_path)
    # LOSO file uses 'subject_id' (no 's'); normalise to match full-model naming
    df_ctrl = df_ctrl.rename(columns={"subject_id": "subject_ids"})
    df_ctrl["group"] = 0

    df_tin = pd.read_csv(test_path)
    df_tin["group"] = 1

    return pd.concat([df_ctrl, df_tin], axis=0, ignore_index=True)


def _feature_cols(df: pd.DataFrame) -> list:
    """Return columns that are EEG features (drop metadata)."""
    drop = {"subject_ids", "observations", "Unnamed: 0", "group"}
    return [c for c in df.columns if c not in drop]


def _align_rois_across_modalities(power_cols, regional_cols, common_rois, freq_band):
    """Return column names for a given ROI × freq_band across modalities."""
    # Power column: <roi>_<freq_band>
    pow_col = f"{common_rois}_{freq_band}"
    # Regional / global: <roi>_<freq_band>_<conn_mode>  (conn_mode appended in regional step)
    # The suffix differs — use endswith matching
    reg_col = next((c for c in regional_cols if c.startswith(f"{common_rois}_{freq_band}")), None)
    return pow_col if pow_col in power_cols else None, reg_col


def compute_multimodal_diffusion(
        space: str,
        preproc_level: int,
        conn_mode: str,
        tinnorm_dir: Path,
) -> None:
    """
    Compute per-ROI multi-modal Mahalanobis distance and save.
    """
    models_dir = tinnorm_dir / "models"

    # ── Load Z-scores ────────────────────────────────────────────────────
    power_dir    = models_dir / f"preproc_{preproc_level}" / space / "power"
    regional_dir = models_dir / f"preproc_{preproc_level}" / space / f"regional_{conn_mode}"
    global_dir   = models_dir / f"preproc_{preproc_level}" / space / f"global_{conn_mode}"

    for d, label in [(power_dir, "power"), (regional_dir, f"regional_{conn_mode}"),
                     (global_dir, f"global_{conn_mode}")]:
        if not (d / "loso" / "loso_controls_Z.csv").exists():
            print(f"Missing LOSO controls for {label} — run 08_create_norm_models.py with LOSO first.")
            return
        if not (d / "full_model" / "results" / "Z_test.csv").exists():
            print(f"Missing tinnitus Z-scores for {label} — run 08_create_norm_models.py first.")
            return

    df_power    = _load_z(power_dir)
    df_regional = _load_z(regional_dir)
    df_global   = _load_z(global_dir)

    # ── Align subjects ───────────────────────────────────────────────────
    subject_col = "subject_ids"
    shared_ids = (
        set(df_power[subject_col])
        & set(df_regional[subject_col])
        & set(df_global[subject_col])
    )
    df_power    = df_power[df_power[subject_col].isin(shared_ids)].sort_values(subject_col).reset_index(drop=True)
    df_regional = df_regional[df_regional[subject_col].isin(shared_ids)].sort_values(subject_col).reset_index(drop=True)
    df_global   = df_global[df_global[subject_col].isin(shared_ids)].sort_values(subject_col).reset_index(drop=True)

    assert (df_power[subject_col].values == df_regional[subject_col].values).all(), \
        "Subject ID mismatch after alignment."

    # ── Identify shared ROI × freq-band pairs ────────────────────────────
    pow_cols = _feature_cols(df_power)
    reg_cols = _feature_cols(df_regional)
    glo_cols = _feature_cols(df_global)
    group_col = df_power["group"].values

    # ROI names: strip the trailing _freqband from power columns
    # Power cols look like:  bankssts-lh_alpha_1
    # Regional cols look like: bankssts-lh_alpha_1_coh
    # Find ROIs that appear in all three
    def col_to_roi_band(col, strip_suffix=None):
        if strip_suffix and col.endswith(strip_suffix):
            col = col[: -len(strip_suffix) - 1]
        parts = col.rsplit("_", 2)
        if len(parts) >= 2 and parts[-1].isdigit():
            band = "_".join(parts[-2:])
            roi = col[: -(len(band) + 1)]
        elif len(parts) >= 1:
            band = parts[-1]
            roi = "_".join(parts[:-1])
        else:
            return None, None
        return roi, band

    # Build mapping: (roi, band) → col for each modality
    pow_map = {}
    for c in pow_cols:
        r, b = col_to_roi_band(c)
        if r and b:
            pow_map[(r, b)] = c

    reg_map = {}
    for c in reg_cols:
        stripped = c[: -(len(f"_{conn_mode}") + 0)] if c.endswith(f"_{conn_mode}") else c
        r, b = col_to_roi_band(stripped)
        if r and b:
            reg_map[(r, b)] = c

    glo_map = {}
    for c in glo_cols:
        stripped = c[: -(len(f"_{conn_mode}") + 0)] if c.endswith(f"_{conn_mode}") else c
        r, b = col_to_roi_band(stripped)
        if r and b:
            glo_map[(r, b)] = c

    shared_keys = sorted(set(pow_map) & set(reg_map) & set(glo_map))
    if not shared_keys:
        print("No shared ROI × freq-band pairs found across modalities. "
              "Check column naming in Z-score files.")
        return

    print(f"  Shared ROI × band pairs: {len(shared_keys)}")

    # Unique ROIs (for output column structure)
    unique_rois = sorted({k[0] for k in shared_keys})

    # ── Build stacked feature tensor ─────────────────────────────────────
    # W[subject, roi×band, modality]
    n_subj = len(df_power)
    n_pairs = len(shared_keys)

    W = np.zeros((n_subj, n_pairs, 3))
    for idx, (roi, band) in enumerate(shared_keys):
        W[:, idx, 0] = df_power[pow_map[(roi, band)]].values
        W[:, idx, 1] = df_regional[reg_map[(roi, band)]].values
        W[:, idx, 2] = df_global[glo_map[(roi, band)]].values

    # ── Estimate covariance on controls only ─────────────────────────────
    control_mask = group_col == 0
    W_ctrl = W[control_mask]

    inv_cov = {}
    for idx, key in enumerate(shared_keys):
        X_ctrl = W_ctrl[:, idx, :]       # (n_controls × 3)
        cov = np.cov(X_ctrl, rowvar=False)
        cov += np.eye(3) * 1e-4          # stronger regularisation (3×3 is small)
        inv_cov[key] = np.linalg.inv(cov)

    # ── Compute Mahalanobis distance ─────────────────────────────────────
    mu = np.zeros(3)   # Z-scores centred at 0
    maha = np.zeros((n_subj, n_pairs))

    for idx, key in enumerate(shared_keys):
        diff = W[:, idx, :] - mu
        maha[:, idx] = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov[key], diff))

    # ── Aggregate per ROI (mean across freq bands) ────────────────────────
    roi_idx = {roi: [i for i, k in enumerate(shared_keys) if k[0] == roi]
               for roi in unique_rois}
    roi_scores = np.stack([maha[:, idxs].mean(axis=1) for idxs in roi_idx.values()], axis=1)

    df_md = pd.DataFrame(roi_scores, columns=list(roi_idx.keys()))
    df_md.insert(0, "group", group_col)
    df_md.insert(0, "subject_ids", df_power[subject_col].values)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = tinnorm_dir / "diffusive_mm"
    os.makedirs(out_dir, exist_ok=True)
    fname = out_dir / f"{space}_preproc_{preproc_level}_{conn_mode}.csv"
    df_md.to_csv(fname, index=False)
    print(f"  Saved → {fname}")


if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    space = "source"
    preproc_levels = [1, 2, 3]
    conn_modes = ["pli", "plv", "coh"]

    for preproc_level in preproc_levels:
        for conn_mode in conn_modes:
            out = tinnorm_dir / "diffusive_mm" / f"{space}_preproc_{preproc_level}_{conn_mode}.csv"
            if out.exists():
                print(f"Already exists, skipping: {out.name}")
                continue
            print(f"\npreproc_{preproc_level} | {conn_mode}")
            compute_multimodal_diffusion(space, preproc_level, conn_mode, tinnorm_dir)
