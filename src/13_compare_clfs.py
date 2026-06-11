"""
Classification pipeline: tinnitus vs. controls.

Reads pre-computed features (harmonized residuals, normative Z-scores, or
multi-modal Mahalanobis diffusion scores) and runs site-stratified LOSO CV
with optional Optuna tuning and permutation tests.

Feature modes
-------------
diffusive_mm        68 ROIs, band-averaged Mahalanobis distance (default)
diffusive_mm_bands  ~680 ROI×band Mahalanobis distances (full resolution)
power / aperiodic   normative Z-score features (deviation mode)
regional / global   connectivity Z-scores (deviation mode)
graph               graph-topology Z-scores (deviation mode)
residual            harmonized HM residuals (residual mode)

Classifiers: RF, SVM, LGBM (requires lightgbm).
Tuning:      Optuna Bayesian HPO (requires optuna).

Run from src/:  python 13_compare_clfs.py
"""

import json
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, SelectFromModel, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    df_metric: pd.DataFrame
    y: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_prob_1: Optional[np.ndarray] = None
    y_prob_2: Optional[np.ndarray] = None
    delta_null: Optional[np.ndarray] = None
    real_delta: Optional[float] = None
    p_value: Optional[float] = None
    y1_folds: Optional[list] = None
    p1_folds: Optional[list] = None
    y2_folds: Optional[list] = None
    p2_folds: Optional[list] = None
    per_fold_auc: Optional[dict] = None   # site → AUC on held-out fold
    sites: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_the_file(
        tinnorm_dir,
        data_mode,
        mode,
        space,
        freq_band,
        preproc_level,
        conn_mode,
        thi_threshold=None,
):
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    # df_master = pd.read_csv("../material/master_clean.csv")
    df_master = pd.read_csv(tinnorm_dir / "master_clean.csv")

    # Support both old (thi_score) and new (THI) column naming
    _thi_col = "THI" if "THI" in df_master.columns else "thi_score"

    # ── diffusive_mm / diffusive_mm_bands ───────────────────────────────
    if mode in ["diffusive", "diffusive_mm", "diffusive_mm_bands"]:
        suffix = "_bands" if mode == "diffusive_mm_bands" else ""
        folder = "diffusive_mm" if mode != "diffusive" else mode
        fname = tinnorm_dir / folder / f"{space}_preproc_{preproc_level}_{conn_mode}{suffix}.csv"
        df = pd.read_csv(fname)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
        df.rename(columns={"site": "SITE"}, inplace=True)

        if thi_threshold is not None:
            df = df.merge(df_master[["subject_id", _thi_col]], on="subject_id", how="left")
            df = df.query(f"group == 0 or {_thi_col} > {thi_threshold}")

        drop_cols = ["Unnamed: 0", "subject_id", "age", "sex",
                     "PTA4_mean", "PTA4_HF", "thi_score", "THI"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ── residual ──────────────────────────────────────────────────────────
    elif data_mode == "residual":
        mode_prefix = f"_{conn_mode}" if mode in ["conn", "global", "regional"] else ""
        fname = hm_dir / f"preproc_{preproc_level}" / space / f"{mode}{mode_prefix}_residual.csv"
        df = pd.read_csv(fname)

        if thi_threshold is not None:
            df = df.merge(df_master[["subject_id", _thi_col]], on="subject_id", how="left")
            df = df.query(f"group == 0 or {_thi_col} > {thi_threshold}")

        drop_cols = ["Unnamed: 0", "subject_id", "age", "sex",
                     "PTA4_mean", "PTA4_HF", "thi_score", "THI"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ── deviation (Z-scores) ──────────────────────────────────────────────
    elif data_mode == "deviation":
        # "graph" uses graph_{conn_mode} model dir, same prefix logic as regional/conn
        mode_prefix = f"_{conn_mode}" if mode in ["conn", "global", "regional", "graph"] else ""
        dfs_group = []
        for group_name, group_id in zip(["train", "test"], [0, 1]):
            fname = (
                models_dir
                / f"preproc_{preproc_level}"
                / space
                / f"{mode}{mode_prefix}"
                / "full_model"
                / "results"
                / f"Z_{group_name}.csv"
            )
            df_group = pd.read_csv(fname)
            df_group["group"] = group_id
            dfs_group.append(df_group)

        df = pd.concat(dfs_group, axis=0, ignore_index=True)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
        df.rename(columns={"site": "SITE"}, inplace=True)

        if thi_threshold is not None:
            df = df.merge(df_master[["subject_id", _thi_col]], on="subject_id", how="left")
            df = df.query(f"group == 0 or {_thi_col} > {thi_threshold}")

        df.sort_values("subject_id", inplace=True)
        drop_cols = ["observations", "subject_id", "thi_score", "THI", "PTA4_HF"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    else:
        raise ValueError(f"Unknown data_mode: '{data_mode}'")

    y = df["group"].to_numpy(dtype=int)
    sites = df["SITE"].to_numpy()

    print(f"\n  Group counts: {dict(pd.Series(y).value_counts().sort_index())}")

    if freq_band is not None:
        X = df.filter(regex=rf"{freq_band}")
    else:
        X = df.drop(columns=["group", "SITE"], errors="ignore")

    print(f"  Features selected: {X.shape[1]}")

    if data_mode == "deviation":
        X = X.clip(-5, 5)
        print("  Z-scores clipped to ±5")

    return X, y, sites


# ---------------------------------------------------------------------------
# Pipeline construction  (P2: scaler + selector + clf all inside CV)
# ---------------------------------------------------------------------------

def _build_pipeline(
        ml_model,
        n_jobs,
        random_state,
        feature_selection=None,
        n_features=50,
        **model_params,
):
    """Return a sklearn Pipeline.  Every label-touching step lives inside."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if feature_selection == "kbest":
        steps.append(("selector", SelectKBest(f_classif, k=n_features)))
    elif feature_selection == "rfe":
        rfe_base = RandomForestClassifier(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        )
        steps.append(("rfe", RFE(estimator=rfe_base, n_features_to_select=n_features, step=10)))
    elif feature_selection == "elasticnet":
        lr_sel = LogisticRegression(
            penalty="elasticnet", l1_ratio=0.5, C=0.1,
            solver="saga", class_weight="balanced",
            max_iter=1000, random_state=random_state,
        )
        steps.append(("selector", SelectFromModel(lr_sel, max_features=n_features)))

    if ml_model == "RF":
        clf = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 300),
            max_depth=model_params.get("max_depth", 10),
            max_features=model_params.get("max_features", "sqrt"),
            min_samples_leaf=model_params.get("min_samples_leaf", 3),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif ml_model == "SVM":
        clf = SVC(
            kernel="rbf",
            C=model_params.get("C", 1.0),
            gamma=model_params.get("gamma", 0.001),
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
    elif ml_model == "LGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed: pip install lightgbm")
        clf = lgb.LGBMClassifier(
            n_estimators=model_params.get("n_estimators", 500),
            learning_rate=model_params.get("learning_rate", 0.05),
            num_leaves=model_params.get("num_leaves", 31),
            max_depth=model_params.get("max_depth", -1),
            min_child_samples=model_params.get("min_child_samples", 20),
            subsample=model_params.get("subsample", 0.8),
            colsample_bytree=model_params.get("colsample_bytree", 0.8),
            reg_alpha=model_params.get("reg_alpha", 0.0),
            reg_lambda=model_params.get("reg_lambda", 1.0),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown ml_model: '{ml_model}'. Choose 'RF', 'SVM', or 'LGBM'.")

    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning  (P4)
# ---------------------------------------------------------------------------

def _tune_with_optuna(ml_model, X_train, y_train, n_trials=30, random_state=42):
    """Bayesian HPO on the training fold via 3-fold inner CV.  Returns best_params dict."""
    if not HAS_OPTUNA:
        raise ImportError("optuna not installed: pip install optuna")

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        if ml_model == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            }
        elif ml_model == "SVM":
            params = {
                "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 1e-1, log=True),
            }
        elif ml_model == "LGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            }
        else:
            return 0.0

        pipe = _build_pipeline(ml_model, n_jobs=1, random_state=random_state, **params)
        scores = cross_val_score(pipe, X_train, y_train, cv=inner_cv,
                                 scoring="roc_auc", n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics_to_dataframe(model_name, y, y_pred, y_prob=None):
    bal_acc = balanced_accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovo", average="weighted") if y_prob is not None else None
    except ValueError:
        auc = None

    return pd.DataFrame({
        "model": [model_name],
        "balanced_accuracy": [bal_acc],
        "precision": [prec],
        "recall": [rec],
        "f1-score": [f1],
        "roc_auc": [auc],
    })


def _per_fold_auc(y, y_prob, sites):
    """AUC per held-out site, computed from the assembled LOSO predictions."""
    result = {}
    for site in np.unique(sites):
        mask = sites == site
        if len(np.unique(y[mask])) > 1:
            result[str(site)] = float(roc_auc_score(y[mask], y_prob[mask]))
        else:
            result[str(site)] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
        X, y, sites, ml_model, n_jobs, random_state,
        feature_selection=None, n_features=50,
        tune=False, n_trials=30,
):
    """
    Site-stratified CV. All preprocessing is fitted inside each fold (no leakage).
    """
    splitter = StratifiedGroupKFold(
        n_splits=len(np.unique(sites)),
        shuffle=True,
        random_state=random_state,
    )

    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y))

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=sites)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]

        print(f"  Fold {fold_i+1}: held-out={np.unique(sites[test_idx])}, "
              f"test balance={np.bincount(y[test_idx])}")

        model_params = {}
        if tune:
            print(f"    Optuna tuning ({n_trials} trials)…")
            model_params = _tune_with_optuna(ml_model, X_train, y_train, n_trials, random_state)
            print(f"    Best params: {model_params}")

        n_feat = min(n_features, X_train.shape[1])
        pipe = _build_pipeline(ml_model, n_jobs, random_state, feature_selection, n_feat, **model_params)
        pipe.fit(X_train, y_train)

        y_pred[test_idx] = pipe.predict(X_test)
        y_prob[test_idx] = pipe.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


def run_loso_cv_with_folds(
        X, y, sites, ml_model, n_jobs, random_state,
        feature_selection=None, n_features=50,
):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_probs, fold_true = [], []
    for train_idx, test_idx in splitter.split(X, y, groups=sites):
        pipe = _build_pipeline(
            ml_model, n_jobs, random_state, feature_selection,
            min(n_features, X.shape[1])
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        fold_probs.append(pipe.predict_proba(X.iloc[test_idx])[:, 1])
        fold_true.append(y[test_idx])
    return fold_true, fold_probs


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _run_permutation(
        X, y, sites, ml_model, n_jobs, random_state,
        feature_selection, n_features, tune, n_trials, n_permutations,
):
    y_pred, y_prob = run_cv(X, y, sites, ml_model, n_jobs, random_state,
                             feature_selection, n_features, tune, n_trials)
    df_metric = metrics_to_dataframe(f"{ml_model}_real", y, y_pred, y_prob)
    fold_auc = _per_fold_auc(y, y_prob, sites)

    rng = np.random.default_rng(random_state)
    print("Running permutation tests…")
    for i in tqdm(range(n_permutations)):
        y_perm = rng.permutation(y)
        y_pred_p, y_prob_p = run_cv(
            X, y_perm, sites, ml_model, n_jobs, random_state,
            feature_selection, n_features, tune=False, n_trials=0,
        )
        df_p = metrics_to_dataframe(f"{ml_model}_perm_{i+1}", y_perm, y_pred_p, y_prob_p)
        df_metric = pd.concat([df_metric, df_p], ignore_index=True)

    return df_metric, y_pred, y_prob, fold_auc


# ---------------------------------------------------------------------------
# Comparison test (residual vs. deviation)
# ---------------------------------------------------------------------------

def _run_comparison(
        X_1, X_2, y, y_2, sites, sites_2,
        ml_model, n_jobs, random_state,
        feature_selection, n_features, tune, n_trials, n_permutations,
):
    if not np.array_equal(y, y_2):
        raise ValueError("y mismatch between residual and deviation data.")
    if not np.array_equal(sites, sites_2):
        raise ValueError("sites mismatch between residual and deviation data.")

    y_pred_1, y_prob_1 = run_cv(X_1, y, sites, ml_model, n_jobs, random_state,
                                  feature_selection, n_features, tune, n_trials)
    y_pred_2, y_prob_2 = run_cv(X_2, y, sites, ml_model, n_jobs, random_state,
                                  feature_selection, n_features, tune, n_trials)

    df_metric = pd.concat([
        metrics_to_dataframe("X1", y, y_pred_1, y_prob_1),
        metrics_to_dataframe("X2", y, y_pred_2, y_prob_2),
    ], ignore_index=True)

    auc_1 = roc_auc_score(y, y_prob_1)
    auc_2 = roc_auc_score(y, y_prob_2)
    real_delta = auc_1 - auc_2
    print(f"AUC residual={auc_1:.3f}  deviation={auc_2:.3f}  Δ={real_delta:.3f}")

    rng = np.random.default_rng(random_state)
    delta_null = np.zeros(n_permutations)
    for i in range(n_permutations):
        swap = rng.random(len(y)) < 0.5
        y1_p, y2_p = y_prob_1.copy(), y_prob_2.copy()
        y1_p[swap], y2_p[swap] = y2_p[swap], y1_p[swap]
        delta_null[i] = roc_auc_score(y, y1_p) - roc_auc_score(y, y2_p)

    p_value = float(np.mean(np.abs(delta_null) >= np.abs(real_delta)))
    print(f"Permutation p-value: {p_value:.4f}")

    y1_folds, p1_folds = run_loso_cv_with_folds(
        X_1, y, sites, ml_model, n_jobs, random_state, feature_selection, n_features
    )
    y2_folds, p2_folds = run_loso_cv_with_folds(
        X_2, y, sites, ml_model, n_jobs, random_state, feature_selection, n_features
    )

    fold_auc = {
        "residual":  _per_fold_auc(y, y_prob_1, sites),
        "deviation": _per_fold_auc(y, y_prob_2, sites),
    }

    return (df_metric, y, y_prob_1, y_prob_2, delta_null, real_delta, p_value,
            y1_folds, p1_folds, y2_folds, p2_folds, fold_auc)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_clf_result(res, save_dir: Path, run_mode: str, params: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    res.df_metric.to_csv(save_dir / "metrics.csv", index=False)

    with open(save_dir / "params.json", "w") as f:
        json.dump({k: str(v) for k, v in params.items()}, f, indent=2)

    if res.per_fold_auc is not None:
        with open(save_dir / "per_fold_auc.json", "w") as f:
            json.dump(res.per_fold_auc, f, indent=2)

    if res.sites is not None:
        np.save(save_dir / "sites.npy", res.sites)

    if run_mode in ("permutation", "cv_only"):
        arrays = {"y": res.y, "y_pred": res.y_pred, "y_prob": res.y_prob_1}
    else:
        arrays = {
            "y": res.y, "y_prob_1": res.y_prob_1, "y_prob_2": res.y_prob_2,
            "delta_null": res.delta_null, "real_delta": res.real_delta,
            "y1_folds": res.y1_folds, "p1_folds": res.p1_folds,
            "y2_folds": res.y2_folds, "p2_folds": res.p2_folds,
            "p_value": res.p_value,
        }

    for name, arr in arrays.items():
        if arr is None:
            continue
        if name.endswith("folds"):
            with open(save_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(arr, f)
        else:
            np.save(save_dir / f"{name}.npy", arr, allow_pickle=True)

    print(f"  Saved → {save_dir}")


# ---------------------------------------------------------------------------
# Main classify function
# ---------------------------------------------------------------------------

def classify(
        tinnorm_dir,
        data_mode,
        preproc_level,
        space,
        mode,
        conn_mode,
        freq_band,
        high_corr_drop,
        corr_thr,
        ml_model,
        n_jobs,
        feature_selection=None,
        n_features=50,
        run_permutation=True,
        n_permutations=100,
        run_comparison=False,
        tune_hyperparams=False,
        n_trials=30,
        thi_threshold=None,
        random_state=42,
):
    if run_permutation and run_comparison:
        raise ValueError("run_permutation and run_comparison cannot both be True.")

    params = {k: v for k, v in locals().items() if k != "tinnorm_dir"}

    X, y, sites = _read_the_file(
        tinnorm_dir, data_mode, mode, space, freq_band,
        preproc_level, conn_mode, thi_threshold,
    )

    if high_corr_drop:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thr)]
        X.drop(columns=to_drop, inplace=True)
        print(f"  Correlation filter: dropped {len(to_drop)}, remaining {X.shape[1]}")

    n_feat = min(n_features, X.shape[1]) if feature_selection is not None else X.shape[1]
    print(f"\n  Feature matrix: {X.shape[0]} subjects × {X.shape[1]} features"
          + (f"  (selection → {n_feat})" if feature_selection else ""))

    def _stamp_params(df):
        for k, v in params.items():
            df[k] = [v] * len(df)
        df["actual_n_features"] = X.shape[1]

    # ── CV only ───────────────────────────────────────────────────────────
    if not run_permutation and not run_comparison:
        y_pred, y_prob = run_cv(X, y, sites, ml_model, n_jobs, random_state,
                                 feature_selection, n_feat, tune_hyperparams, n_trials)
        df_metric = metrics_to_dataframe(f"{ml_model}_real", y, y_pred, y_prob)
        fold_auc = _per_fold_auc(y, y_prob, sites)
        _stamp_params(df_metric)
        return ClassificationResult(df_metric=df_metric, y=y, y_pred=y_pred,
                                    y_prob_1=y_prob, per_fold_auc=fold_auc, sites=sites)

    # ── Permutation test ──────────────────────────────────────────────────
    if run_permutation:
        df_metric, y_pred, y_prob, fold_auc = _run_permutation(
            X, y, sites, ml_model, n_jobs, random_state,
            feature_selection, n_feat, tune_hyperparams, n_trials, n_permutations,
        )
        _stamp_params(df_metric)
        return ClassificationResult(df_metric=df_metric, y=y, y_pred=y_pred,
                                    y_prob_1=y_prob, per_fold_auc=fold_auc, sites=sites)

    # ── Comparison (residual vs. deviation) ───────────────────────────────
    X_2, y_2, sites_2 = _read_the_file(
        tinnorm_dir, "deviation", mode, space, freq_band,
        preproc_level, conn_mode, thi_threshold,
    )
    (df_metric, y, y_prob_1, y_prob_2, delta_null, real_delta, p_value,
     y1_folds, p1_folds, y2_folds, p2_folds, fold_auc) = _run_comparison(
        X, X_2, y, y_2, sites, sites_2,
        ml_model, n_jobs, random_state,
        feature_selection, n_feat, tune_hyperparams, n_trials, n_permutations,
    )
    _stamp_params(df_metric)
    df_metric.at[1, "data_mode"] = "deviation"
    return ClassificationResult(
        df_metric=df_metric, y=y,
        y_prob_1=y_prob_1, y_prob_2=y_prob_2,
        delta_null=delta_null, real_delta=real_delta, p_value=p_value,
        y1_folds=y1_folds, p1_folds=p1_folds,
        y2_folds=y2_folds, p2_folds=p2_folds,
        per_fold_auc=fold_auc, sites=sites,
    )


# ---------------------------------------------------------------------------
# Entry point  (P1 preproc sweep · P4 LGBM · P7 THI sweep)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION — edit before running on VM
    # ═══════════════════════════════════════════════════════════════════════
    # TINNORM_DIR     = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    TINNORM_DIR   = Path("/home/ubuntu/volume/Tinnorm")

    RUN_PERMUTATION = True   # False = quick dry-run (no null distribution)
    N_PERMUTATIONS  = 100
    N_TRIALS_OPTUNA = 30     # Bayesian HPO trials per outer fold
    N_JOBS          = -1     # -1 = all cores; set e.g. 8 on shared VM nodes

    # Leave empty to run every scenario; or list labels to run a subset:
    # SCENARIOS_TO_RUN = ["A_preproc2_lgbm", "B_preproc2_rf"]
    SCENARIOS_TO_RUN: list = []

    clfs_dir = TINNORM_DIR / "clfs"

    # ── Base configuration (overridden per scenario) ──────────────────────
    base_kwargs = dict(
        space="source",
        data_mode="deviation",      # ignored when mode=diffusive_mm*
        mode="diffusive_mm",        # 68 ROI band-averaged Mahalanobis features
        conn_mode="coh",
        freq_band=None,
        high_corr_drop=False,
        corr_thr=0.95,
        n_jobs=N_JOBS,
        feature_selection=None,     # None = use ALL features (no selection)
        n_features=50,              # only used when feature_selection is set
        run_permutation=RUN_PERMUTATION,
        n_permutations=N_PERMUTATIONS,
        run_comparison=False,
        tune_hyperparams=False,
        n_trials=N_TRIALS_OPTUNA,
        thi_threshold=None,
        random_state=42,
    )

    # ── Scenarios ─────────────────────────────────────────────────────────
    scenarios = [

        # ── A: Preprocessing levels (LGBM, diffusive_mm, coh) ─────────────
        ("A_preproc1_lgbm",           dict(preproc_level=1, ml_model="LGBM")),
        ("A_preproc2_lgbm",           dict(preproc_level=2, ml_model="LGBM")),  # baseline
        ("A_preproc3_lgbm",           dict(preproc_level=3, ml_model="LGBM")),

        # ── B: Classifier comparison (preproc=2, diffusive_mm, coh) ──────
        ("B_preproc2_rf",             dict(preproc_level=2, ml_model="RF")),
        ("B_preproc2_svm",            dict(preproc_level=2, ml_model="SVM")),

        # ── C: Connectivity measure (preproc=2, LGBM, diffusive_mm) ──────
        ("C_preproc2_lgbm_pli",       dict(preproc_level=2, ml_model="LGBM", conn_mode="pli")),
        ("C_preproc2_lgbm_plv",       dict(preproc_level=2, ml_model="LGBM", conn_mode="plv")),

        # ── D: Feature dimensionality — 68 avg vs ~680 band-resolved ─────
        #    Requires 09_multimodal_diffusion.py to have been re-run (saves _bands.csv)
        ("D_preproc2_lgbm_bands",     dict(preproc_level=2, ml_model="LGBM",
                                           mode="diffusive_mm_bands")),
        ("D_preproc2_lgbm_bands_k50", dict(preproc_level=2, ml_model="LGBM",
                                           mode="diffusive_mm_bands",
                                           feature_selection="kbest", n_features=50)),
        ("D_preproc2_rf_bands",       dict(preproc_level=2, ml_model="RF",
                                           mode="diffusive_mm_bands")),

        # ── E: Feature selection (preproc=2, diffusive_mm) ────────────────
        ("E_preproc2_lgbm_kbest50",   dict(preproc_level=2, ml_model="LGBM",
                                           feature_selection="kbest", n_features=50)),
        ("E_preproc2_rf_rfe30",       dict(preproc_level=2, ml_model="RF",
                                           feature_selection="rfe",   n_features=30)),
        ("E_preproc2_lgbm_enet50",    dict(preproc_level=2, ml_model="LGBM",
                                           feature_selection="elasticnet", n_features=50)),

        # ── F: Optuna hyperparameter tuning (preproc=2, diffusive_mm) ────
        ("F_preproc2_lgbm_tuned",     dict(preproc_level=2, ml_model="LGBM",
                                           tune_hyperparams=True)),
        ("F_preproc2_rf_tuned",       dict(preproc_level=2, ml_model="RF",
                                           tune_hyperparams=True)),
        ("F_preproc2_svm_tuned",      dict(preproc_level=2, ml_model="SVM",
                                           tune_hyperparams=True)),

        # ── G: THI severity threshold (preproc=2, LGBM, diffusive_mm) ───
        ("G_thi25_preproc2_lgbm",     dict(preproc_level=2, ml_model="LGBM",
                                           thi_threshold=25)),
        ("G_thi36_preproc2_lgbm",     dict(preproc_level=2, ml_model="LGBM",
                                           thi_threshold=36)),
        ("G_thi56_preproc2_lgbm",     dict(preproc_level=2, ml_model="LGBM",
                                           thi_threshold=56)),

        # ── H: Individual modality Z-scores (preproc=2, LGBM, deviation) ─
        ("H_preproc2_lgbm_power",     dict(preproc_level=2, ml_model="LGBM",
                                           mode="power",    data_mode="deviation")),
        ("H_preproc2_lgbm_aperiodic", dict(preproc_level=2, ml_model="LGBM",
                                           mode="aperiodic",data_mode="deviation")),
        ("H_preproc2_lgbm_regional",  dict(preproc_level=2, ml_model="LGBM",
                                           mode="regional", data_mode="deviation")),
        ("H_preproc2_lgbm_global",    dict(preproc_level=2, ml_model="LGBM",
                                           mode="global",   data_mode="deviation")),
        ("H_preproc2_lgbm_graph",     dict(preproc_level=2, ml_model="LGBM",
                                           mode="graph",    data_mode="deviation")),

        # ── I: Residual vs Deviation (comparison mode) ────────────────────
        ("I_preproc2_lgbm_power_cmp", dict(preproc_level=2, ml_model="LGBM",
                                           mode="power", data_mode="residual",
                                           run_permutation=False, run_comparison=True)),

        # ── J: Band-resolved diffusive × connectivity + tuning on best features ─
        # PLI/PLV bands: connectivity measure × best feature representation
        ("J_bands_pli_lgbm",       dict(preproc_level=2, ml_model="LGBM",
                                        mode="diffusive_mm_bands", conn_mode="pli")),
        ("J_bands_plv_lgbm",       dict(preproc_level=2, ml_model="LGBM",
                                        mode="diffusive_mm_bands", conn_mode="plv")),
        ("J_bands_pli_rf",         dict(preproc_level=2, ml_model="RF",
                                        mode="diffusive_mm_bands", conn_mode="pli")),
        # Optuna tuning on band-resolved features (F group only tuned averaged 68-feat)
        ("J_bands_coh_lgbm_tuned", dict(preproc_level=2, ml_model="LGBM",
                                        mode="diffusive_mm_bands",
                                        tune_hyperparams=True)),
        ("J_bands_pli_lgbm_tuned", dict(preproc_level=2, ml_model="LGBM",
                                        mode="diffusive_mm_bands", conn_mode="pli",
                                        tune_hyperparams=True)),
    ][-5:]

    if SCENARIOS_TO_RUN:
        scenarios = [(l, o) for l, o in scenarios if l in SCENARIOS_TO_RUN]

    print(f"Running {len(scenarios)} scenario(s).\n")

    summary_rows = []

    for label, overrides in scenarios:
        kwargs = {**base_kwargs, **overrides}
        run_mode = ("comparison" if kwargs["run_comparison"]
                    else "permutation" if kwargs["run_permutation"]
                    else "cv_only")

        print(f"\n{'='*60}")
        print(f"Scenario: {label}  [{run_mode}]")
        print(f"{'='*60}")

        try:
            res = classify(TINNORM_DIR, **kwargs)
            save_clf_result(res, clfs_dir / label, run_mode, kwargs)

            real_row = res.df_metric.iloc[0]
            row = {"scenario": label, "run_mode": run_mode,
                   "roc_auc": real_row.get("roc_auc"),
                   "balanced_accuracy": real_row.get("balanced_accuracy"),
                   "f1-score": real_row.get("f1-score")}
            if res.per_fold_auc and isinstance(res.per_fold_auc, dict):
                for site, auc in res.per_fold_auc.items():
                    row[f"auc_{site}"] = auc
            summary_rows.append(row)

        except FileNotFoundError as e:
            print(f"  SKIP — data not found: {e}")
            summary_rows.append({"scenario": label, "run_mode": run_mode,
                                  "roc_auc": "SKIP"})
        except Exception as e:
            print(f"  ERROR — {e}")
            summary_rows.append({"scenario": label, "run_mode": run_mode,
                                  "roc_auc": f"ERROR: {e}"})

    # ── Ensemble: average predictions from top-N scenarios ───────────────
    print(f"\n{'='*60}")
    print("Ensemble of top scenarios …")
    try:
        ens_candidates = []
        for label, _ in scenarios:
            folder = clfs_dir / label
            if not (folder / "y_prob.npy").exists() or not (folder / "y.npy").exists():
                continue
            mp = folder / "metrics.csv"
            if not mp.exists():
                continue
            auc_val = pd.read_csv(mp).iloc[0].get("roc_auc")
            if auc_val is not None:
                try:
                    ens_candidates.append((float(auc_val), label))
                except (ValueError, TypeError):
                    pass
        ens_candidates.sort(reverse=True)

        y_ref = None
        for n_top in [3, 5, len(ens_candidates)]:
            top_n = ens_candidates[:n_top]
            if len(top_n) < 2:
                continue
            if y_ref is None:
                y_ref = np.load(clfs_dir / top_n[0][1] / "y.npy",
                                allow_pickle=True).astype(int)
            probs = [np.load(clfs_dir / lbl / "y_prob.npy", allow_pickle=True)
                     for _, lbl in top_n]
            ens_prob = np.mean(probs, axis=0)
            ens_auc  = roc_auc_score(y_ref, ens_prob)
            y_pred_e = (ens_prob >= 0.5).astype(int)
            from sklearn.metrics import balanced_accuracy_score as _ba
            bal_acc  = _ba(y_ref, y_pred_e)

            ens_label = f"ensemble_top{n_top}"
            print(f"  {ens_label}: AUC={ens_auc:.4f}  BalAcc={bal_acc:.4f}")

            ens_dir = clfs_dir / ens_label
            ens_dir.mkdir(exist_ok=True)
            np.save(ens_dir / "y.npy",      y_ref)
            np.save(ens_dir / "y_prob.npy", ens_prob)
            pd.DataFrame({
                "model":             [ens_label],
                "roc_auc":           [ens_auc],
                "balanced_accuracy": [bal_acc],
                "members":           [str([l for _, l in top_n])],
            }).to_csv(ens_dir / "metrics.csv", index=False)

            summary_rows.append({
                "scenario": ens_label, "run_mode": "ensemble",
                "roc_auc": ens_auc, "balanced_accuracy": bal_acc,
            })
    except Exception as e:
        print(f"  Ensemble failed: {e}")

    df_summary = pd.DataFrame(summary_rows)
    summary_path = clfs_dir / "summary_all_scenarios.csv"
    clfs_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"All done. Summary → {summary_path}")
    cols = ["scenario", "roc_auc", "balanced_accuracy"]
    print(df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))
