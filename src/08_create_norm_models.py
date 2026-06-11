import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from pcntoolkit import (
    NormData,
    BLR,
    HBR,
    BsplineBasisFunction,
    NormativeModel,
    NormalLikelihood,
    make_prior
)


def _select_bio_covars(df_train: pd.DataFrame,
                       bio_covars: list,
                       r_threshold: float = 0.85) -> list:
    """
    Drop PTA4_mean if highly correlated with PTA4_HF (mirrors 06_harmonize.py).
    Applied to the training split (controls only) to match the ComBat decision.
    """
    pair = [c for c in ["PTA4_mean", "PTA4_HF"]
            if c in bio_covars and c in df_train.columns]
    if len(pair) < 2:
        return bio_covars
    valid = df_train[pair].dropna()
    r = float(np.corrcoef(valid["PTA4_mean"].values, valid["PTA4_HF"].values)[0, 1])
    print(f"  PTA4_mean–PTA4_HF Pearson r = {r:.3f}  (threshold = {r_threshold})")
    if abs(r) > r_threshold:
        print("  → High collinearity: dropping PTA4_mean, retaining PTA4_HF.")
        return [c for c in bio_covars if c != "PTA4_mean"]
    print("  → Collinearity OK: using both PTA4_mean and PTA4_HF.")
    return bio_covars


def _build_template_model(model_type: str, age_col_idx: int):
    """Return a fresh BLR or HBR template (called once per NormativeModel instance)."""
    if model_type == "hbr":
        mu = make_prior(
            linear=True,
            slope=make_prior(dist_name="Normal", dist_params=(0.0, 10.0)),
            intercept=make_prior(
                random=True,
                mu=make_prior(dist_name="Normal", dist_params=(0.0, 1.0)),
                sigma=make_prior(dist_name="Normal", dist_params=(0.0, 1.0),
                                 mapping="softplus", mapping_params=(0.0, 3.0)),
            ),
            basis_function=BsplineBasisFunction(basis_column=age_col_idx, nknots=5, degree=3),
        )
        sigma = make_prior(
            linear=True,
            slope=make_prior(dist_name="Normal", dist_params=(0.0, 2.0)),
            intercept=make_prior(dist_name="Normal", dist_params=(1.0, 1.0)),
            basis_function=BsplineBasisFunction(basis_column=age_col_idx, nknots=5, degree=3),
            mapping="softplus",
            mapping_params=(0.0, 3.0),
        )
        likelihood = NormalLikelihood(mu, sigma)
        return HBR(
            name="payam_hbr",
            cores=1,           # >1 deadlocks on macOS due to PyMC spawn method
            progressbar=True,
            draws=1000,
            tune=1000,         # tune >= draws for reliable NUTS mass-matrix adaptation
            chains=4,
            nuts_sampler="nutpie",
            likelihood=likelihood,
        )
    return BLR(
        name="payam_blr",
        basis_function_mean=BsplineBasisFunction(degree=3, nknots=5),
        fixed_effect=True,
        heteroskedastic=True,
        warp_name=None # "warpsinharcsinh"
    )


def _load_features(fname_feature):
    """Load harmonized feature CSV and return a clean dataframe."""
    df = pd.read_csv(fname_feature)
    df["subject_id"] = df["subject_id"].astype(str)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df


def _split_response_cols(df):
    """
    Return (response_cols, demographic_cols) preserving original CSV column order.
    Index.difference is avoided because it sorts alphabetically.
    """
    _demo_candidates = ["subject_id", "SITE", "group", "age", "sex",
                        "PTA4_mean", "PTA4_HF", "THI", "TFI"]
    demographic_cols = [c for c in _demo_candidates if c in df.columns]
    demo_set = set(demographic_cols)
    response_cols = [c for c in df.columns if c not in demo_set]
    return response_cols, demographic_cols


def run_nm(
            fname_feature,
            model_dir,
            outscaler="none",
            model_type="blr",   # "blr" or "hbr"
            random_state=42
        ):
    """
    Fit normative models and evaluate.

    Runs two configs:
      for_eval   — 80/20 site-stratified split of controls; sanity-checks calibration.
      full_model — all controls as training; tinnitus as test; saved for inference.

    Z-scores for controls (needed for classifier) come from run_nm_loso, not here,
    because controls cannot get unbiased Z-scores from a model they trained.
    """
    df = _load_features(fname_feature)
    response_cols, demographic_cols = _split_response_cols(df)

    _candidate_covars = [c for c in ["age", "sex", "PTA4_mean", "PTA4_HF"] if c in df.columns]
    df_controls = df.query("group == 0")
    _core_covariates = _select_bio_covars(df_controls, _candidate_covars)
    age_col_idx = _core_covariates.index("age") if "age" in _core_covariates else 0

    kwargs = {
                "covariates": _core_covariates,
                "batch_effects": ["SITE"],
                "response_vars": response_cols,
                "subject_ids": "subject_id"
                }

    norm_train_all = NormData.from_dataframe(
                                            name="train",
                                            dataframe=df_controls,
                                            **kwargs
                                            )
    norm_test_tinnitus = NormData.from_dataframe(
                                            name="test",
                                            dataframe=df.query('group == 1'),
                                            **kwargs
                                            )

    # Stratified split by SITE so every site is represented in both train and test.
    # Random split risks putting an entire small site in only one partition,
    # making the batch-effect parameters unevaluable for that site.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(sss.split(df_controls, df_controls["SITE"]))
    df_train_ctrl = df_controls.iloc[train_idx].reset_index(drop=True)
    df_test_ctrl  = df_controls.iloc[test_idx].reset_index(drop=True)

    print(f"  Control split — train: {len(df_train_ctrl)}, test: {len(df_test_ctrl)}")
    print(f"    train sites: {df_train_ctrl['SITE'].value_counts().to_dict()}")
    print(f"    test  sites: {df_test_ctrl['SITE'].value_counts().to_dict()}")

    norm_train_control = NormData.from_dataframe(name="train", dataframe=df_train_ctrl, **kwargs)
    norm_test_control  = NormData.from_dataframe(name="test",  dataframe=df_test_ctrl,  **kwargs)

    configs = [
                {
                    "save_dir": model_dir / "for_eval",
                    "train": norm_train_control,
                    "test": norm_test_control,
                    "savemodel": False,
                },
                {
                    "save_dir": model_dir / "full_model",
                    "train": norm_train_all,
                    "test": norm_test_tinnitus,
                    "savemodel": False,
                },
                ]

    for cfg in configs:
        template_model = _build_template_model(model_type, age_col_idx)
        model = NormativeModel(
            template_regression_model=template_model,
            savemodel=cfg["savemodel"],
            evaluate_model=True,
            saveresults=True,
            saveplots=False,
            save_dir=str(cfg["save_dir"]),
            inscaler="standardize",
            outscaler=outscaler,
        )
        model.fit_predict(cfg["train"], cfg["test"])
        del model, template_model


def run_nm_loso(
            fname_feature,
            model_dir,
            outscaler="none",
            model_type="blr",
            random_state=42
        ):
    """
    Leave-one-site-out (LOSO) cross-validation on controls.

    For each of the N sites, trains a normative model on all other sites'
    controls and predicts on the held-out site's controls.  Every control
    subject ends up with exactly one Z-score from a model it never trained on,
    making them directly comparable to tinnitus Z-scores from run_nm's full_model.

    Workflow
    --------
    1. Iterate over sites: hold out site s, train on the remaining 6.
    2. Save pcntoolkit outputs per fold to  model_dir/loso/fold_<site>/.
    3. After all folds, aggregate into    model_dir/loso/loso_controls_Z.csv.

    That CSV + full_model's tinnitus Z-scores feed 13_compare_clfs.py.
    """
    df = _load_features(fname_feature)
    response_cols, demographic_cols = _split_response_cols(df)

    _candidate_covars = [c for c in ["age", "sex", "PTA4_mean", "PTA4_HF"] if c in df.columns]
    df_controls = df.query("group == 0").reset_index(drop=True)
    _core_covariates = _select_bio_covars(df_controls, _candidate_covars)
    age_col_idx = _core_covariates.index("age") if "age" in _core_covariates else 0

    # No batch_effects here: input is already ComBat-harmonized so site effects are
    # gone. More importantly, LOSO holds out one site entirely from training, so
    # pcntoolkit would have no batch-effect estimate for it and raise
    # "Data is not compatible with the model" in compute_zscores.
    kwargs = {
                "covariates": _core_covariates,
                "batch_effects": [],
                "response_vars": response_cols,
                "subject_ids": "subject_id",
                }

    loso_dir = model_dir / "loso"
    sites = sorted(df_controls["SITE"].unique())
    print(f"  LOSO: {len(sites)} folds — {sites}")

    # Columns to carry forward into the aggregated output for downstream use
    meta_cols = [c for c in ["subject_id", "SITE", "group"] + _core_covariates
                 if c in df_controls.columns]

    fold_dirs = []   # fold_dir per site — collected for aggregation step

    for site in sites:
        fold_dir = loso_dir / f"fold_{site}"
        z_csv = fold_dir / "results" / "Z_test.csv"

        if z_csv.exists():
            print(f"  Fold {site}: already done — skipping fit.")
            fold_dirs.append(fold_dir)
            continue

        os.makedirs(fold_dir, exist_ok=True)

        df_train = df_controls[df_controls["SITE"] != site].reset_index(drop=True)
        df_test  = df_controls[df_controls["SITE"] == site].reset_index(drop=True)
        print(f"\n  Fold {site}: train n={len(df_train)}, test n={len(df_test)}")

        norm_train = NormData.from_dataframe(name="train", dataframe=df_train, **kwargs)
        norm_test  = NormData.from_dataframe(name="test",  dataframe=df_test,  **kwargs)

        template_model = _build_template_model(model_type, age_col_idx)
        model = NormativeModel(
            template_regression_model=template_model,
            savemodel=False,
            evaluate_model=True,
            saveresults=True,
            saveplots=False,
            save_dir=str(fold_dir),
            inscaler="standardize",
            outscaler=outscaler,
        )
        model.fit_predict(norm_train, norm_test)
        del model, template_model

        fold_dirs.append(fold_dir)

    # ── Aggregate Z-scores across all folds ───────────────────────────────────
    # pcntoolkit saves results/Z_test.csv in each fold directory.
    # That CSV already contains a subject_ids column — no need to re-associate
    # from df_test ordering. Metadata (SITE, group, covariates) is merged in
    # from df_controls on subject_id.
    z_frames = []
    for fold_dir in fold_dirs:
        z_path = fold_dir / "results" / "Z_test.csv"
        if not z_path.exists():
            print(f"  Warning: {z_path} not found — skipping fold.")
            continue
        df_z = pd.read_csv(z_path)
        df_z = df_z.rename(columns={"subject_ids": "subject_id"})
        df_z = df_z.drop(columns=["observations"], errors="ignore")
        df_z["subject_id"] = df_z["subject_id"].astype(str)
        df_z = df_z.merge(df_controls[meta_cols], on="subject_id", how="left")
        z_frames.append(df_z)

    if not z_frames:
        print("\n  LOSO: aggregation failed — no results/Z_test.csv found in fold directories.")
        return

    df_loso = pd.concat(z_frames, ignore_index=True)
    out_path = loso_dir / "loso_controls_Z.csv"
    df_loso.to_csv(out_path, index=False)
    print(f"\n  LOSO complete: {len(df_loso)} control Z-scores → {out_path}")


if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    # tinnorm_dir = Path("/home/ubuntu/volume/Tinnorm")
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    os.makedirs(models_dir, exist_ok=True)

    preproc_levels = [1, 3]   # preproc 2 already done; generate 1 and 3 for script 17
    spaces = ["sensor", "source"][1:]
    modalities = ["power"]    # power only — rerun with outscaler="standardize" to fix
                              # numerical blow-up from large-magnitude preproc_1/3 features
    conn_modes = ["pli"]
    model_type = "blr"

    # outscaler per modality: power features span ~10^9 for preproc_1 (vs ~10^5 for preproc_2)
    # which causes BLR numerical instability on the test set.  "standardize" divides
    # the response by its training-set std before fitting, giving identical Z-score
    # semantics but stable arithmetic. PLI/connectivity models are fine with "none".
    OUTSCALER = {"power": "standardize"}

    import shutil

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:
                for conn_mode in conn_modes:

                    if modality in ["conn", "global", "regional"]:
                        fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_hm.csv"
                        model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}"
                    elif modality in ["aperiodic", "power"]:
                        fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_hm.csv"
                        model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}"
                    elif modality == "graph":
                        fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"graph_{conn_mode}_hm.csv"
                        model_dir = models_dir / f"preproc_{preproc_level}" / space / f"graph_{conn_mode}"
                    else:
                        raise ValueError(f"Unknown modality: {modality!r}")

                    if not fname_feature.is_file():
                        print(f"  Missing: {fname_feature.name} — skipping")
                        continue

                    outscaler = OUTSCALER.get(modality, "none")

                    # Force-remove stale model dir so the fixed outscaler is applied
                    if model_dir.is_dir():
                        print(f"  Removing stale model dir: {model_dir}")
                        shutil.rmtree(model_dir)

                    # ── Step 1: fit full model + for_eval ─────────────────────
                    print(f"\npreproc_{preproc_level} | {space} | {modality} | outscaler={outscaler}")
                    run_nm(
                        fname_feature,
                        model_dir,
                        outscaler=outscaler,
                        model_type=model_type,
                        random_state=42
                    )

                    # ── Step 2: LOSO to get unbiased control Z-scores ─────────
                    run_nm_loso(
                        fname_feature,
                        model_dir,
                        outscaler=outscaler,
                        model_type=model_type,
                        random_state=42
                    )