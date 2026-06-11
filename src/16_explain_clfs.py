"""
Priority 8 — Model explainability with SHAP computed on held-out folds.

SHAP values are computed inside the LOSO CV loop on each held-out test fold,
then reassembled in original subject order. This avoids the optimistic bias
from fitting the explainer on training data the model has already memorized.

All figures are saved automatically to results/figures/ (or to save_dir if
passed to the plot functions).

Usage: run as a script or import compute_shap_loso().
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from scipy.stats import spearmanr, mannwhitneyu
import shap

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ── Parameters ────────────────────────────────────────────────────────────────

tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = tinnorm_dir / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "16_shap"
read_kwargs = {
    "data_mode": "deviation",
    "mode": "diffusive_mm_bands",  # best AUC (0.82 with PLI); use "diffusive_mm" for 68-ROI avg
    "space": "source",
    "freq_band": None,
    "preproc_level": 2,
    "conn_mode": "pli",            # J_bands_pli_lgbm is the top-performing scenario
    "thi_threshold": None,
}

ml_model      = "LGBM"   # "RF" or "LGBM"
folding_mode  = "loso" # "loso" or "sgkf5"
n_jobs        = -1
random_state  = 42
top_k         = 10     # features shown in summary plot
scatter_k     = 3      # features shown in scatter plots


# ── Data loading ──────────────────────────────────────────────────────────────

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
    hm_dir     = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    df_master  = pd.read_csv("../material/master_clean.csv")

    # Support both old (thi_score) and new (THI) column naming
    _thi_col  = "THI" if "THI" in df_master.columns else "thi_score"
    # Metadata to preserve in returned df for downstream correlation analysis
    _meta_cols = [c for c in ["THI", "TFI", "PTA4_HF"] if c in df_master.columns]

    subject_ids = None   # populated inside each branch before subject_id is dropped

    if mode in ("diffusive", "diffusive_mm", "diffusive_mm_bands"):
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

        if _meta_cols:
            df = df.merge(df_master[["subject_id"] + _meta_cols], on="subject_id", how="left")

        subject_ids = df["subject_id"].to_numpy() if "subject_id" in df.columns else None
        drop_cols = ["Unnamed: 0", "subject_id", "thi_score", "THI",
                     "PTA4_mean", "PTA4_HF", "age", "sex"]
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    elif data_mode == "residual":
        mode_prefix = f"_{conn_mode}" if mode in ("conn", "global", "regional") else ""
        fname = hm_dir / f"preproc_{preproc_level}" / space / f"{mode}{mode_prefix}_residual.csv"
        df = pd.read_csv(fname)

        if thi_threshold is not None:
            df = df.merge(df_master[["subject_id", _thi_col]], on="subject_id", how="left")
            df = df.query(f"group == 0 or {_thi_col} > {thi_threshold}")

        if _meta_cols and "subject_id" in df.columns:
            df = df.merge(df_master[["subject_id"] + _meta_cols], on="subject_id", how="left")

        subject_ids = df["subject_id"].to_numpy() if "subject_id" in df.columns else None
        df.drop(columns=["Unnamed: 0", "subject_id", "thi_score"], inplace=True, errors="ignore")

    else:  # data_mode == "deviation"
        mode_prefix = f"_{conn_mode}" if mode in ("conn", "global", "regional", "graph") else ""
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

        if _meta_cols:
            df = df.merge(df_master[["subject_id"] + _meta_cols], on="subject_id", how="left")

        df.sort_values("subject_id", inplace=True)
        subject_ids = df["subject_id"].to_numpy() if "subject_id" in df.columns else None
        df.drop(columns=["observations", "subject_id", "thi_score"], inplace=True, errors="ignore")

    if subject_ids is None:
        subject_ids = np.arange(len(df))

    y     = df["group"].to_numpy(dtype=int)
    sites = df["SITE"].to_numpy()

    print(f"\nSubject counts:\n{df['group'].value_counts()}")

    if freq_band is not None:
        df = df.filter(regex=rf"{freq_band}")

    df = df.drop(columns=["SITE"])
    print(f"Features: {df.shape[1] - 1}")
    # THI, TFI, PTA4_HF remain in df; caller pops them before building X

    return df, y, sites, subject_ids


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(ml_model, random_state, n_jobs):
    if ml_model == "LGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm is not installed.")
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ── SHAP inside LOSO CV ───────────────────────────────────────────────────────

def compute_shap_loso(
    X: pd.DataFrame,
    y: np.ndarray,
    sites: np.ndarray,
    ml_model: str = "RF",
    folding_mode: str = "loso",
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Fit model and compute SHAP values fold-by-fold.

    Returns
    -------
    shap_vals : np.ndarray  shape (n_subjects, n_features, 2)
    y_pred    : np.ndarray  shape (n_subjects,)
    y_prob    : np.ndarray  shape (n_subjects,)
    metrics   : dict
    """
    if folding_mode == "loso":
        splitter = LeaveOneGroupOut()
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    n_subj    = len(y)
    n_feat    = X.shape[1]
    shap_vals      = np.full((n_subj, n_feat, 2), np.nan)
    y_pred         = np.zeros(n_subj, dtype=int)
    y_prob         = np.zeros(n_subj)
    base_vals      = np.zeros(n_subj)   # per-subject expected-value from fold explainer
    fold_membership = np.full(n_subj, -1, dtype=int)

    for fold_i, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=sites)
    ):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx],  y[test_idx]

        pipe = _make_model(ml_model, random_state, n_jobs)
        pipe.fit(X_tr, y_tr)

        y_pred[test_idx] = pipe.predict(X_te)
        y_prob[test_idx] = pipe.predict_proba(X_te)[:, 1]
        fold_membership[test_idx] = fold_i

        # SHAP on held-out test fold, using the imputed+scaled features the clf sees
        clf     = pipe.named_steps["clf"]
        imputer = pipe.named_steps["imputer"]
        scaler  = pipe.named_steps["scaler"]
        X_te_imp = imputer.transform(X_te)
        X_tr_imp = imputer.transform(X_tr)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=X.columns)
        X_tr_scaled = pd.DataFrame(scaler.transform(X_tr_imp), columns=X.columns)

        explainer = shap.TreeExplainer(
            clf,
            data=X_tr_scaled,          # background for expected value
            feature_perturbation="interventional",
        )
        sv = explainer(X_te_scaled)    # Explanation object, shape (n_test, n_feat, n_classes)

        if sv.values.ndim == 2:
            # binary RF sometimes returns (n, f) — duplicate for both classes
            sv_arr = np.stack([-sv.values, sv.values], axis=2)
            ev = float(explainer.expected_value)
            base_vals[test_idx] = ev
        else:
            sv_arr = sv.values         # (n_test, n_feat, 2)
            ev = explainer.expected_value
            base_vals[test_idx] = ev[1] if hasattr(ev, "__len__") else float(ev)

        shap_vals[test_idx] = sv_arr
        site_label = np.unique(sites[test_idx])[0] if folding_mode == "loso" else fold_i
        print(f"  Fold {fold_i + 1} ({site_label}): n_test={len(test_idx)}, "
              f"AUC={roc_auc_score(y_te, y_prob[test_idx]):.3f}")

    metrics = {
        "accuracy":          accuracy_score(y, y_pred),
        "precision":         precision_score(y, y_pred, zero_division=0),
        "recall":            recall_score(y, y_pred, zero_division=0),
        "f1":                f1_score(y, y_pred, zero_division=0),
        "roc_auc":           roc_auc_score(y, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
    }
    return shap_vals, y_pred, y_prob, metrics, fold_membership, base_vals


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_pr(y, y_prob, save_dir: Path = FIGURES_DIR):
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    RocCurveDisplay.from_predictions(y, y_prob, ax=ax_roc, plot_chance_level=True, despine=True)
    PrecisionRecallDisplay.from_predictions(y, y_prob, ax=ax_pr, plot_chance_level=True, despine=True)
    for ax in [ax_roc, ax_pr]:
        ax.set_title(ax.get_title(), style="italic", fontsize="small")
        ax.legend(frameon=False, loc="lower right", fontsize="small")
    fpath = save_dir / "shap_roc_pr.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_shap_summary(shap_vals, X, y, top_k=10, scatter_k=3, save_dir: Path = FIGURES_DIR):
    """
    shap_vals : (n_subjects, n_features, 2)  — tinnitus class = index 1
    """
    shap_tinnitus = shap_vals[:, :, 1]   # (n_subj, n_feat)

    # ── Beeswarm summary ──────────────────────────────────────────────────
    expl = shap.Explanation(
        values=shap_tinnitus,
        base_values=np.zeros(len(y)),
        data=X.values,
        feature_names=list(X.columns),
    )
    # Blue (low feature value) → white → red (high): clean, perceptually balanced
    cmap1 = LinearSegmentedColormap.from_list(
        "shap_feat", ["#2471A3", "#D6EAF8", "#F9F9F9", "#FADBD8", "#C0392B"]
    )
    # Do NOT use constrained_layout — shap.summary_plot calls tight_layout internally
    fig_sum, _ = plt.subplots(figsize=(8, 6.5))
    shap.summary_plot(expl, X, max_display=top_k, cmap=cmap1, alpha=0.65,
                      show=False, plot_size=None)
    # Thicken axes after SHAP renders (shap takes over plt.gca())
    ax = plt.gca()
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=4)
    fpath = save_dir / "shap_summary.pdf"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {fpath}")

    # ── Scatter plots for top features ────────────────────────────────────
    mean_abs  = np.abs(shap_tinnitus).mean(axis=0)
    top_idxs  = np.argsort(mean_abs)[::-1][:scatter_k]
    col_names = X.columns[top_idxs]
    colors    = ["#1f77b4", "#C99700"]
    cmap2     = LinearSegmentedColormap.from_list("ctrl_tin", colors)

    fig, axs = plt.subplots(1, scatter_k, figsize=(4 * scatter_k, 3),
                             constrained_layout=True)
    for i, (ax, f_idx, col_name) in enumerate(zip(axs, top_idxs, col_names)):
        feat_expl = shap.Explanation(
            values=shap_tinnitus[:, f_idx],
            base_values=np.zeros(len(y)),
            data=X.iloc[:, f_idx].values,
            feature_names=col_name,
        )
        shap.plots.scatter(feat_expl, color=y, dot_size=15, hist=False,
                           alpha=0.8, cmap=cmap2, ax=ax, show=False)
        for coll in ax.collections:
            if hasattr(coll, "colorbar") and coll.colorbar:
                coll.colorbar.remove()
        ax.set_title(col_name, style="italic", fontsize=10)
        ax.set_xlabel("Feature value", style="italic", fontsize=10)
        ax.set_ylabel("SHAP value" if i == 0 else "", fontsize=10)

    fpath = save_dir / "shap_scatter.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_thi_correlation(y_prob, thi_scores, y, save_dir: Path = FIGURES_DIR):
    """Scatter of P(Tinnitus) vs THI score for tinnitus subjects, with Spearman r."""
    tin_mask = y == 1
    prob_tin = y_prob[tin_mask]
    thi_tin  = thi_scores[tin_mask] if thi_scores is not None else np.array([])
    valid    = ~np.isnan(thi_tin)

    if valid.sum() < 5:
        print("  Insufficient THI data for probability-correlation plot — skipping.")
        return

    prob_tin = prob_tin[valid]
    thi_tin  = thi_tin[valid]
    r, p = spearmanr(thi_tin, prob_tin)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(thi_tin, prob_tin, alpha=0.65, s=35, color="#C99700")
    m, b = np.polyfit(thi_tin, prob_tin, 1)
    x_line = np.linspace(thi_tin.min(), thi_tin.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")
    for cutoff in [16, 36, 56, 76]:
        if cutoff < thi_tin.max():
            ax.axvline(cutoff, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("THI score", fontsize=11)
    ax.set_ylabel("P(Tinnitus)", fontsize=11)
    ax.set_title(f"Classifier probability vs THI  (tinnitus group)\n"
                 f"Spearman r = {r:.3f}, p = {p:.3f}", style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "thi_prob_correlation.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_shap_thi_correlation(shap_vals, X, thi_scores, y, top_k=5,
                               save_dir: Path = FIGURES_DIR):
    """Scatter of top-k feature SHAP values vs THI for tinnitus subjects."""
    if thi_scores is None:
        print("  No THI data — skipping SHAP-THI correlation.")
        return

    tin_mask = y == 1
    thi_tin  = thi_scores[tin_mask]
    valid    = ~np.isnan(thi_tin)

    if valid.sum() < 5:
        print("  Insufficient THI data for SHAP-THI correlation — skipping.")
        return

    shap_tin = shap_vals[tin_mask, :, 1]          # (n_tin, n_feat)
    mean_abs = np.abs(shap_tin).mean(axis=0)
    top_idxs = np.argsort(mean_abs)[::-1][:top_k]

    fig, axs = plt.subplots(1, top_k, figsize=(4 * top_k, 3.5), constrained_layout=True)
    if top_k == 1:
        axs = [axs]

    for ax, f_idx in zip(axs, top_idxs):
        col = X.columns[f_idx]
        sv  = shap_tin[valid, f_idx]
        thi = thi_tin[valid]
        r, p_val = spearmanr(thi, sv)
        ax.scatter(thi, sv, alpha=0.6, s=20, color="#C99700")
        m, b = np.polyfit(thi, sv, 1)
        x_ = np.linspace(thi.min(), thi.max(), 100)
        ax.plot(x_, m * x_ + b, color="black", lw=1.2, linestyle="--")
        ax.set_xlabel("THI", fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{col}\nr={r:.2f}, p={p_val:.3f}", style="italic", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "shap_thi_correlation.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_shap_clustering(shap_vals, X, y, top_k=10, save_dir: Path = FIGURES_DIR):
    """Hierarchical-cluster bar using cosine similarity of SHAP vectors."""
    shap_tinnitus = shap_vals[:, :, 1]
    clustering    = shap.utils.hclust(X, y, metric="cosine")
    expl = shap.Explanation(
        values=shap_tinnitus,
        base_values=np.zeros(len(y)),
        data=X.values,
        feature_names=list(X.columns),
    )
    shap.plots.bar(expl, max_display=top_k, clustering=clustering,
                   clustering_cutoff=0.5, show=False)
    fpath = save_dir / "shap_clustering.pdf"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {fpath}")


# ── TFI clinical correlations ─────────────────────────────────────────────────

def plot_tfi_correlation(y_prob, tfi_scores, y, save_dir: Path = FIGURES_DIR):
    """Scatter of P(Tinnitus) vs TFI score for tinnitus subjects."""
    tin_mask  = y == 1
    prob_tin  = y_prob[tin_mask]
    tfi_tin   = tfi_scores[tin_mask] if tfi_scores is not None else np.array([])
    valid     = ~np.isnan(tfi_tin)
    if valid.sum() < 5:
        print("  Insufficient TFI data for probability-correlation plot — skipping.")
        return
    prob_tin = prob_tin[valid]
    tfi_tin  = tfi_tin[valid]
    r, p = spearmanr(tfi_tin, prob_tin)

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(tfi_tin, prob_tin, alpha=0.65, s=35, color="#9B59B6")
    m, b = np.polyfit(tfi_tin, prob_tin, 1)
    x_ = np.linspace(tfi_tin.min(), tfi_tin.max(), 100)
    ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("TFI score", fontsize=11)
    ax.set_ylabel("P(Tinnitus)", fontsize=11)
    ax.set_title(f"Classifier probability vs TFI  (tinnitus group)\n"
                 f"Spearman r = {r:.3f}, p = {p:.3f}", style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)
    fpath = save_dir / "tfi_prob_correlation.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_shap_tfi_correlation(shap_vals, X, tfi_scores, y, top_k=5,
                               save_dir: Path = FIGURES_DIR):
    """Scatter of top-k feature SHAP values vs TFI for tinnitus subjects."""
    if tfi_scores is None:
        print("  No TFI data — skipping SHAP-TFI correlation.")
        return
    tin_mask = y == 1
    tfi_tin  = tfi_scores[tin_mask]
    valid    = ~np.isnan(tfi_tin)
    if valid.sum() < 5:
        print("  Insufficient TFI data for SHAP-TFI correlation — skipping.")
        return

    shap_tin = shap_vals[tin_mask, :, 1]
    mean_abs = np.abs(shap_tin).mean(axis=0)
    top_idxs = np.argsort(mean_abs)[::-1][:top_k]

    fig, axs = plt.subplots(1, top_k, figsize=(4 * top_k, 3.5), constrained_layout=True)
    if top_k == 1:
        axs = [axs]
    for ax, f_idx in zip(axs, top_idxs):
        col = X.columns[f_idx]
        sv  = shap_tin[valid, f_idx]
        tfi = tfi_tin[valid]
        r, p_val = spearmanr(tfi, sv)
        ax.scatter(tfi, sv, alpha=0.6, s=20, color="#9B59B6")
        m, b = np.polyfit(tfi, sv, 1)
        x_ = np.linspace(tfi.min(), tfi.max(), 100)
        ax.plot(x_, m * x_ + b, color="black", lw=1.2, linestyle="--")
        ax.set_xlabel("TFI", fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{col}\nr={r:.2f}, p={p_val:.3f}", style="italic", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
    fpath = save_dir / "shap_tfi_correlation.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── SHAP waterfall for representative subjects ────────────────────────────────

def plot_shap_waterfall_subjects(shap_vals, base_vals, X, y, y_pred, y_prob,
                                  top_k: int = 12, save_dir: Path = FIGURES_DIR):
    """
    Save one waterfall PDF per outcome category (TP, FN, FP, TN).
    TP  → highest P(Tinnitus) among correct tinnitus
    FN  → highest P(Control)  among missed tinnitus  (most confidently missed)
    FP  → highest P(Tinnitus) among misclassified controls
    TN  → lowest  P(Tinnitus) among correct controls
    """
    shap_tinnitus = shap_vals[:, :, 1]
    cases = [
        ((y == 1) & (y_pred == 1), "TP — best classified tinnitus",  "tp",  True),
        ((y == 1) & (y_pred == 0), "FN — missed tinnitus",           "fn",  False),
        ((y == 0) & (y_pred == 1), "FP — misclassified control",     "fp",  True),
        ((y == 0) & (y_pred == 0), "TN — best classified control",   "tn",  False),
    ]
    for mask, label, tag, select_high in cases:
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            print(f"  {tag.upper()}: no subjects — skipping waterfall.")
            continue
        probs  = y_prob[idxs]
        chosen = idxs[np.argmax(probs) if select_high else np.argmin(probs)]
        expl = shap.Explanation(
            values=shap_tinnitus[chosen],
            base_values=float(base_vals[chosen]),
            data=X.iloc[chosen].values,
            feature_names=list(X.columns),
        )
        plt.figure(figsize=(7, 5))
        shap.plots.waterfall(expl, max_display=top_k, show=False)
        plt.title(f"{label}  |  P(Tinnitus) = {y_prob[chosen]:.3f}",
                  style="italic", fontsize=9, pad=6)
        fpath = save_dir / f"shap_waterfall_{tag}.pdf"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved → {fpath}")


# ── SHAP fold-level stability ─────────────────────────────────────────────────

def plot_shap_fold_stability(shap_vals, X, sites, fold_membership,
                              top_k: int = 15, save_dir: Path = FIGURES_DIR):
    """Boxplot of mean |SHAP| per site-fold for top-k features."""
    shap_tinnitus = shap_vals[:, :, 1]
    global_mean   = np.abs(shap_tinnitus).mean(axis=0)
    top_idxs      = np.argsort(global_mean)[::-1][:top_k]
    top_names     = list(X.columns[top_idxs])

    rows = []
    for f in np.unique(fold_membership):
        mask       = fold_membership == f
        fold_shap  = np.abs(shap_tinnitus[mask]).mean(axis=0)
        site_label = np.unique(sites[mask])[0]
        for feat, idx in zip(top_names, top_idxs):
            rows.append({"feature": feat, "mean_abs_shap": fold_shap[idx], "site": site_label})
    df_stab = pd.DataFrame(rows)

    unique_sites  = sorted(df_stab["site"].unique())
    # Perceptually uniform, saturated palette — one rich hue per site
    site_palette  = dict(zip(unique_sites,
                             sns.color_palette("husl", len(unique_sites))))
    box_edge = "#2C3E50"   # dark slate — clean neutral against coloured dots

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    # Hollow box: fill=False (seaborn ≥ 0.12); whiskers and median in dark slate
    sns.boxplot(data=df_stab, x="mean_abs_shap", y="feature",
                order=top_names,
                fill=False,
                color=box_edge,
                linewidth=1.6,
                fliersize=0,
                width=0.55,
                medianprops={"color": box_edge, "linewidth": 2.5,
                             "solid_capstyle": "round"},
                ax=ax)

    # Coloured dots per site — slightly larger, white halo for readability
    sns.stripplot(data=df_stab, x="mean_abs_shap", y="feature",
                  order=top_names, hue="site",
                  palette=site_palette,
                  alpha=0.88, size=7.5, jitter=0.25,
                  linewidth=0.5, edgecolor="white",
                  ax=ax)

    # Subtle vertical grid behind everything
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, zorder=0)

    # Thicker axes
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=4)

    ax.set_xlabel("Mean |SHAP| (tinnitus class)", fontsize=11)
    ax.set_ylabel("", fontsize=11)
    ax.set_title(f"Feature importance stability across LOSO folds\n"
                 f"(top {top_k} features — each dot = one held-out site)",
                 style="italic", fontsize=10)
    ax.legend(title="Held-out site", frameon=False, fontsize="x-small",
              title_fontsize="x-small",
              bbox_to_anchor=(1.01, 1), loc="upper left")

    fpath = save_dir / "shap_fold_stability.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── SHAP subject × feature heatmap ───────────────────────────────────────────

def plot_shap_heatmap(shap_vals, X, y, y_prob, top_k: int = 20,
                      save_dir: Path = FIGURES_DIR):
    """Heatmap of SHAP values (tinnitus class) for all subjects × top-k features."""
    shap_tinnitus = shap_vals[:, :, 1]
    global_mean   = np.abs(shap_tinnitus).mean(axis=0)
    top_idxs      = np.argsort(global_mean)[::-1][:top_k]
    top_names     = [str(c)[:30] for c in X.columns[top_idxs]]   # truncate long names

    # Sort: controls ascending by prob, tinnitus descending by prob
    ctrl_ord = np.where(y == 0)[0][np.argsort(y_prob[y == 0])]
    tin_ord  = np.where(y == 1)[0][np.argsort(y_prob[y == 1])[::-1]]
    order    = np.concatenate([ctrl_ord, tin_ord])

    data = shap_tinnitus[np.ix_(order, top_idxs)]
    lim  = float(np.percentile(np.abs(data), 95))

    n_subj = len(order)
    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.55), max(5, n_subj * 0.045)),
                            constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="SHAP (tinnitus class)", shrink=0.6, pad=0.01)

    n_ctrl = (y == 0).sum()
    ax.axhline(n_ctrl - 0.5, color="black", lw=1.5)
    ax.text(top_k, n_ctrl / 2,         "Control",  va="center", fontsize=8, color="#1f77b4")
    ax.text(top_k, n_ctrl + len(tin_ord) / 2, "Tinnitus", va="center", fontsize=8, color="#C99700")

    ax.set_xticks(range(top_k))
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks([])
    ax.set_ylabel("Subjects (sorted by group & P(Tinnitus))", fontsize=9)
    ax.set_title(f"SHAP values — top {top_k} features × all subjects",
                 style="italic", fontsize=10)
    fpath = save_dir / "shap_heatmap.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Misclassification analysis ────────────────────────────────────────────────

def analyze_misclassifications(
    y, y_pred, y_prob, thi_scores, tfi_scores, sites, subject_ids,
    save_dir: Path = FIGURES_DIR, tables_dir: Path = None,
):
    """
    Comprehensive breakdown of FP, FN, TP, TN:
    - Per-site confusion counts
    - P(Tinnitus) violin by outcome category
    - THI and TFI distributions for TP vs FN (are missed cases milder?)
    - Saved CSV of misclassified subjects
    """
    labels = np.where(
        (y == 1) & (y_pred == 1), "TP",
        np.where(
            (y == 1) & (y_pred == 0), "FN",
            np.where((y == 0) & (y_pred == 1), "FP", "TN"),
        ),
    )

    df_meta = pd.DataFrame({
        "subject_id":    subject_ids if subject_ids is not None else np.arange(len(y)),
        "y_true":        y,
        "y_pred":        y_pred,
        "prob_tinnitus": y_prob,
        "outcome":       labels,
        "site":          sites,
    })
    if thi_scores is not None:
        df_meta["THI"] = thi_scores
    if tfi_scores is not None:
        df_meta["TFI"] = tfi_scores

    if tables_dir is not None:
        misc = df_meta[df_meta["outcome"].isin(["FP", "FN"])].copy()
        misc.to_csv(tables_dir / "misclassified_subjects.csv", index=False)
        print(f"  Saved → {tables_dir / 'misclassified_subjects.csv'}")

    # ── Console summary ──────────────────────────────────────────────────
    print("\n  Outcome totals:")
    for out in ["TP", "TN", "FP", "FN"]:
        print(f"    {out}: {(labels == out).sum()}")
    print("\n  Per-site breakdown:")
    for site in sorted(np.unique(sites)):
        m = sites == site
        tp = ((labels == "TP") & m).sum()
        tn = ((labels == "TN") & m).sum()
        fp = ((labels == "FP") & m).sum()
        fn = ((labels == "FN") & m).sum()
        print(f"    {site:<15}: TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    # ── Figure 1: P(Tinnitus) by outcome + per-site FP/FN bar ────────────
    cat_order  = ["TN", "FP", "FN", "TP"]
    cat_colors = {"TN": "#1f77b4", "FP": "#9B59B6", "FN": "#e67e22", "TP": "#C99700"}
    fig, axes  = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    df_plot = df_meta[df_meta["outcome"].isin(cat_order)].copy()
    sns.violinplot(data=df_plot, x="outcome", y="prob_tinnitus",
                   order=cat_order, palette=cat_colors,
                   inner="quartile", alpha=0.8, ax=axes[0])
    axes[0].axhline(0.5, color="black", lw=1, linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Outcome")
    axes[0].set_ylabel("P(Tinnitus)")
    axes[0].set_title("Classifier probability by outcome category", style="italic")
    axes[0].spines[["right", "top"]].set_visible(False)

    site_ct = df_meta.groupby(["site", "outcome"]).size().unstack(fill_value=0)
    for col in ["FP", "FN"]:
        if col not in site_ct.columns:
            site_ct[col] = 0
    site_ct[["FP", "FN"]].plot(kind="bar", ax=axes[1],
                                color=["#9B59B6", "#e67e22"],
                                edgecolor="white", width=0.6)
    axes[1].set_title("Misclassifications per site", style="italic")
    axes[1].set_xlabel("Site")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(frameon=False)
    axes[1].spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "misclassification_summary.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── Figure 2: THI and TFI — TP vs FN (are missed cases milder?) ──────
    tin_df = df_meta[df_meta["y_true"] == 1].copy()
    tin_df["classified"] = tin_df["outcome"].map({"TP": "Correct (TP)", "FN": "Missed (FN)"})
    clinical_cols = [(c, col) for c, col in [("THI", "#C99700"), ("TFI", "#9B59B6")]
                     if c in tin_df.columns and tin_df[c].notna().any()]

    if clinical_cols:
        n = len(clinical_cols)
        fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True)
        if n == 1:
            axs = [axs]
        pal = {"Correct (TP)": "#C99700", "Missed (FN)": "#e67e22"}
        for ax, (col, _) in zip(axs, clinical_cols):
            sub = tin_df[["classified", col]].dropna()
            if len(sub) < 5:
                continue
            sns.violinplot(data=sub, x="classified", y=col,
                           palette=pal, inner="quartile", alpha=0.8, ax=ax)
            tp_vals = sub[sub["classified"] == "Correct (TP)"][col].dropna()
            fn_vals = sub[sub["classified"] == "Missed (FN)"][col].dropna()
            if len(tp_vals) >= 3 and len(fn_vals) >= 3:
                _, p = mannwhitneyu(tp_vals, fn_vals, alternative="two-sided")
                sig = "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.set_title(f"{col}: TP vs FN\nMann-Whitney {sig} (p={p:.3f})",
                             style="italic", fontsize=10)
            else:
                ax.set_title(f"{col}: TP vs FN", style="italic", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(f"{col} score")
            ax.spines[["right", "top"]].set_visible(False)

        fpath = save_dir / "misclassification_clinical.pdf"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fpath}")

    return df_meta


# ── THI severity threshold sweep ──────────────────────────────────────────────

def _loso_metrics_only(X, y, sites, ml_model="RF", random_state=42, n_jobs=-1):
    """LOSO without SHAP — fast metrics-only run."""
    splitter = LeaveOneGroupOut()
    n        = len(y)
    y_pred   = np.zeros(n, dtype=int)
    y_prob   = np.zeros(n)
    for train_idx, test_idx in splitter.split(X, y, groups=sites):
        pipe = _make_model(ml_model, random_state, n_jobs)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X.iloc[test_idx])
        y_prob[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    return {
        "roc_auc":           roc_auc_score(y, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1":                f1_score(y, y_pred, zero_division=0),
        "recall":            recall_score(y, y_pred, zero_division=0),
    }


def thi_threshold_sweep(
    X, y, sites, thi_scores,
    thresholds=None,
    ml_model="RF", random_state=42, n_jobs=-1,
    save_dir: Path = FIGURES_DIR, tables_dir: Path = None,
):
    """
    Re-run LOSO restricting tinnitus subjects to those with THI > threshold.
    thresholds: list where None means 'all tinnitus regardless of THI'.
    Plots AUC, balanced accuracy, recall vs threshold and sample-size panel.
    """
    if thi_scores is None:
        print("  No THI scores — skipping threshold sweep.")
        return

    if thresholds is None:
        thresholds = [None, 16, 36, 56]

    results = []
    for thresh in thresholds:
        if thresh is None:
            mask  = np.ones(len(y), dtype=bool)
            label = "All"
        else:
            valid = ~np.isnan(thi_scores)
            mask  = (y == 0) | ((y == 1) & valid & (thi_scores > thresh))
            label = f">{thresh}"

        X_sub     = X[mask].reset_index(drop=True)
        y_sub     = y[mask]
        sites_sub = sites[mask]
        n_tin     = int((y_sub == 1).sum())
        n_ctrl    = int((y_sub == 0).sum())

        if n_tin < 5 or len(np.unique(sites_sub)) < 2:
            print(f"  Threshold {label}: too few subjects — skipping.")
            continue

        print(f"  Threshold {label}: n_ctrl={n_ctrl}, n_tin={n_tin} … ", end="", flush=True)
        m = _loso_metrics_only(X_sub, y_sub, sites_sub, ml_model, random_state, n_jobs)
        print(f"AUC={m['roc_auc']:.3f}  bal_acc={m['balanced_accuracy']:.3f}")
        m["threshold_label"] = label
        m["n_tinnitus"]      = n_tin
        m["n_total"]         = int(len(y_sub))
        results.append(m)

    if not results:
        return

    df_res = pd.DataFrame(results)
    if tables_dir is not None:
        df_res.to_csv(tables_dir / "thi_threshold_sweep.csv", index=False)
        print(f"  Saved → {tables_dir / 'thi_threshold_sweep.csv'}")

    x_pos  = np.arange(len(df_res))
    labels = df_res["threshold_label"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for col, style, color, lbl in [
        ("roc_auc",           "o-",  "#C99700", "ROC-AUC"),
        ("balanced_accuracy", "s--", "#9B59B6", "Balanced Accuracy"),
        ("recall",            "^:",  "#e67e22", "Recall (sensitivity)"),
    ]:
        ax1.plot(x_pos, df_res[col], style, color=color, lw=2, ms=8, label=lbl)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_xlabel("Tinnitus subset  (THI threshold)", fontsize=10)
    ax1.set_ylabel("Score", fontsize=10)
    ax1.set_ylim(0.3, 1.02)
    ax1.set_title("Performance vs THI severity threshold\n(LOSO-CV, all sites)",
                  style="italic", fontsize=10)
    ax1.legend(frameon=False, fontsize="small")
    ax1.spines[["right", "top"]].set_visible(False)

    ax2.bar(x_pos, df_res["n_tinnitus"], color="#C99700", alpha=0.8,
            edgecolor="white", width=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_xlabel("THI threshold", fontsize=10)
    ax2.set_ylabel("N tinnitus retained", fontsize=10)
    ax2.set_title("Sample size per threshold", style="italic", fontsize=10)
    ax2.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "thi_threshold_sweep.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")
    return df_res


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR = RESULTS_DIR / "tables"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df, y, sites, subject_ids = _read_the_file(tinnorm_dir, **read_kwargs)

    # Extract clinical metadata before building feature matrix
    thi_scores = df.pop("THI").to_numpy()    if "THI"    in df.columns else None
    tfi_scores = df.pop("TFI").to_numpy()    if "TFI"    in df.columns else None
    pta_hf     = df.pop("PTA4_HF").to_numpy() if "PTA4_HF" in df.columns else None
    X = df.drop(columns=["group"])

    print(f"\nRunning LOSO SHAP with {ml_model} …")
    shap_vals, y_pred, y_prob, metrics, fold_membership, base_vals = compute_shap_loso(
        X, y, sites,
        ml_model=ml_model,
        folding_mode=folding_mode,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    print("\n── Metrics (aggregated across folds) ──")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(TABLES_DIR / "shap_loso_metrics.csv", index=False)

    # Save SHAP arrays for downstream analysis (scripts 23 & 24 load these)
    np.save(TABLES_DIR / "shap_values.npy", shap_vals)
    np.save(TABLES_DIR / "shap_feature_names.npy", np.array(list(X.columns), dtype=object))
    np.save(TABLES_DIR / "shap_fold_membership.npy", fold_membership)
    np.save(TABLES_DIR / "shap_sites.npy", np.array(sites, dtype=object))
    mean_abs_shap = np.abs(shap_vals[:, :, 1]).mean(axis=0)
    pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True).to_csv(TABLES_DIR / "shap_mean_abs.csv", index=False)
    print(f"  Saved SHAP arrays + fold metadata → {TABLES_DIR}")

    # ── Core SHAP plots ────────────────────────────────────────────────────
    print("\n── Core SHAP plots ──")
    plot_roc_pr(y, y_prob, save_dir=FIGURES_DIR)
    plot_shap_summary(shap_vals, X, y, top_k=top_k, scatter_k=scatter_k,
                      save_dir=FIGURES_DIR)
    plot_shap_clustering(shap_vals, X, y, top_k=top_k, save_dir=FIGURES_DIR)

    # ── Enhanced explainability ────────────────────────────────────────────
    print("\n── Waterfall plots (TP / FN / FP / TN) ──")
    plot_shap_waterfall_subjects(shap_vals, base_vals, X, y, y_pred, y_prob,
                                  top_k=top_k, save_dir=FIGURES_DIR)

    print("\n── SHAP fold stability ──")
    plot_shap_fold_stability(shap_vals, X, sites, fold_membership,
                              top_k=top_k, save_dir=FIGURES_DIR)

    print("\n── SHAP subject heatmap ──")
    plot_shap_heatmap(shap_vals, X, y, y_prob, top_k=20, save_dir=FIGURES_DIR)

    # ── Clinical score correlations ────────────────────────────────────────
    if thi_scores is not None:
        print("\n── THI correlation analyses ──")
        plot_thi_correlation(y_prob, thi_scores, y, save_dir=FIGURES_DIR)
        plot_shap_thi_correlation(shap_vals, X, thi_scores, y,
                                  top_k=scatter_k, save_dir=FIGURES_DIR)

    if tfi_scores is not None:
        print("\n── TFI correlation analyses ──")
        plot_tfi_correlation(y_prob, tfi_scores, y, save_dir=FIGURES_DIR)
        plot_shap_tfi_correlation(shap_vals, X, tfi_scores, y,
                                  top_k=scatter_k, save_dir=FIGURES_DIR)

    # ── Misclassification analysis ─────────────────────────────────────────
    print("\n── Misclassification analysis ──")
    analyze_misclassifications(
        y, y_pred, y_prob,
        thi_scores=thi_scores,
        tfi_scores=tfi_scores,
        sites=sites,
        subject_ids=subject_ids,
        save_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
    )

    # ── THI severity threshold sweep ───────────────────────────────────────
    print("\n── THI threshold sweep ──")
    thi_threshold_sweep(
        X, y, sites, thi_scores,
        thresholds=[None, 16, 36, 56],
        ml_model=ml_model,
        random_state=random_state,
        n_jobs=n_jobs,
        save_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
    )

    print(f"\nAll figures → {FIGURES_DIR}")
    print("Done.")
