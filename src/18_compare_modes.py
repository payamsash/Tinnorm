"""
Compare feature modes, classifiers, feature-selection strategies, and modality ablation.

Four analysis sections (all LOSO, no data leakage):

  A. Mode comparison
       residual conn  vs  deviation conn  vs  diffusive (Mahalanobis)
       → results/figures/mode_roc_pr.pdf
       → results/figures/mode_auc_bar.pdf

  B. Classifier comparison  (on diffusive features, preproc 2)
       RF  vs  SVM  vs  LGBM
       → results/figures/clf_roc_pr.pdf
       → results/figures/clf_metrics_bar.pdf

  C. Feature-selection comparison  (RF, diffusive, preproc 2)
       none  vs  kbest-50  vs  rfe-30
       → results/figures/featsel_roc_pr.pdf

  D. Modality ablation  (RF, preproc 2, source)
       power  vs  aperiodic  vs  regional_coh  vs  global_coh
       vs  graph_coh  vs  diffusive_mm
       → results/figures/modality_ablation_roc_pr.pdf
       → results/figures/modality_ablation_metrics.pdf

  results/tables/mode_comparison.csv
  results/tables/clf_comparison.csv
  results/tables/featsel_comparison.csv
  results/tables/modality_ablation.csv

Run from src/:  python 18_compare_modes.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR  = TINNORM_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"

SPACE        = "source"
PREPROC      = 2
CONN_MODE    = "coh"
RANDOM_STATE = 42

COLORS = {
    "residual":      "#1f77b4",
    "deviation":     "#245C43",
    "diffusive":     "#C99700",
    "RF":            "#D55E00",
    "SVM":           "#0072B2",
    "LGBM":          "#009E73",
    "none":          "#9B59B6",
    "kbest":         "#E67E22",
    "rfe":           "#16A085",
    # Modality ablation colours
    "power":         "#E8A838",
    "aperiodic":     "#5B7FA6",
    "regional_coh":  "#6DB86B",
    "global_coh":    "#D6655A",
    "graph_coh":     "#9B59B6",
    "diffusive_mm":  "#C99700",
}
CHANCE_COLOR = "#7f8c8d"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data(data_mode: str, mode: str, preproc: int = PREPROC,
               conn_mode: str = CONN_MODE):
    """Return (X, y, sites) for the requested data_mode / mode combo."""
    df_master   = pd.read_csv("../material/master_clean.csv")
    hm_dir      = TINNORM_DIR / "harmonized"
    models_dir  = TINNORM_DIR / "models"

    if mode in ("diffusive", "diffusive_mm"):
        folder = mode
        fname = TINNORM_DIR / folder / f"{SPACE}_preproc_{preproc}_{conn_mode}.csv"
        df = pd.read_csv(fname)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
        df.rename(columns={"site": "SITE"}, inplace=True)
        df.drop(columns=["Unnamed: 0", "subject_id", "age", "sex",
                          "PTA4_mean", "PTA4_HF", "thi_score", "THI"],
                inplace=True, errors="ignore")

    elif data_mode == "residual":
        pfx  = f"_{conn_mode}" if mode in ("conn", "global", "regional") else ""
        fname = hm_dir / f"preproc_{preproc}" / SPACE / f"{mode}{pfx}_residual.csv"
        df = pd.read_csv(fname)
        df.drop(columns=["Unnamed: 0", "subject_id", "age", "sex",
                          "PTA4_mean", "PTA4_HF", "thi_score", "THI"],
                inplace=True, errors="ignore")

    elif data_mode == "deviation":
        pfx = f"_{conn_mode}" if mode in ("conn", "global", "regional", "graph") else ""
        dfs_g = []
        for gname, gid in zip(["train", "test"], [0, 1]):
            f = (models_dir / f"preproc_{preproc}" / SPACE
                 / f"{mode}{pfx}" / "full_model" / "results" / f"Z_{gname}.csv")
            df_g = pd.read_csv(f)
            df_g["group"] = gid
            dfs_g.append(df_g)
        df = pd.concat(dfs_g, axis=0, ignore_index=True)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
        df.rename(columns={"site": "SITE"}, inplace=True)
        df.sort_values("subject_id", inplace=True)
        df.drop(columns=["observations", "subject_id"], inplace=True, errors="ignore")

    else:
        raise ValueError(f"Unknown combo: data_mode={data_mode!r}, mode={mode!r}")

    y     = df["group"].to_numpy()
    sites = df["SITE"].to_numpy()
    X     = df.drop(columns=["group", "SITE"], errors="ignore")
    print(f"    Loaded: {len(y)} subjects, {X.shape[1]} features")
    return X, y, sites


# ── Pipeline factory ──────────────────────────────────────────────────────────

def _make_pipe(clf_name: str, feat_sel: str = "none", n_features: int = 50):
    steps = [("scaler", StandardScaler())]

    if feat_sel == "kbest":
        steps.append(("sel", SelectKBest(f_classif, k=n_features)))
    elif feat_sel == "rfe":
        base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        steps.append(("sel", RFE(base, n_features_to_select=n_features, step=10)))

    if clf_name == "RF":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    elif clf_name == "SVM":
        clf = SVC(kernel="rbf", C=1.0, gamma=0.001, probability=True,
                  class_weight="balanced", random_state=RANDOM_STATE)
    elif clf_name == "LGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed.")
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  class_weight="balanced", random_state=RANDOM_STATE,
                                  n_jobs=-1, verbose=-1)
    else:
        raise ValueError(clf_name)

    steps.append(("clf", clf))
    return Pipeline(steps)


# ── LOSO CV ───────────────────────────────────────────────────────────────────

def run_loso(X, y, sites, clf_name: str, feat_sel: str = "none", n_features: int = 50):
    sgkf = StratifiedGroupKFold(n_splits=len(np.unique(sites)),
                                 shuffle=True, random_state=RANDOM_STATE)
    n_feat = min(n_features, X.shape[1])
    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y))
    fold_true, fold_prob = [], []

    for train_idx, test_idx in sgkf.split(X, y, groups=sites):
        pipe = _make_pipe(clf_name, feat_sel, n_feat)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X.iloc[test_idx])
        y_prob[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        fold_true.append(y[test_idx])
        fold_prob.append(y_prob[test_idx])

    return y_pred, y_prob, fold_true, fold_prob


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _roc_pr_panel(series_list: list, title: str, fname_stem: str):
    """
    series_list : list of (label, color, y, y_prob, fold_true, fold_prob)
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5),
                                          constrained_layout=True)
    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)

    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2, label="Chance")
    ax_pr.axhline(0.5, linestyle="--", color=CHANCE_COLOR, lw=1.2, label="Chance")

    for label, color, y, y_prob, fold_true, fold_prob in series_list:
        tprs, precs = [], []
        for yt, yp in zip(fold_true, fold_prob):
            if len(np.unique(yt)) < 2:
                continue
            fpr_, tpr_, _ = roc_curve(yt, yp)
            tprs.append(np.interp(mean_fpr, fpr_, tpr_))
            ax_roc.plot(fpr_, tpr_, color=color, alpha=0.15, lw=1)
            prec_, rec_, _ = precision_recall_curve(yt, yp)
            precs.append(np.interp(mean_rec, rec_[::-1], prec_[::-1]))
            ax_pr.plot(rec_, prec_, color=color, alpha=0.15, lw=1)

        auc_val = roc_auc_score(y, y_prob)
        ax_roc.plot(mean_fpr, np.mean(tprs, axis=0), color=color, lw=3,
                    label=f"{label}  (AUC={auc_val:.3f})")
        ap_val = auc(mean_rec, np.mean(precs, axis=0))
        ax_pr.plot(mean_rec, np.mean(precs, axis=0), color=color, lw=3,
                   label=f"{label}  (AP={ap_val:.3f})")

    for ax, xl, yl in [(ax_roc, "FPR", "TPR"), (ax_pr, "Recall", "Precision")]:
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(f"{title}", style="italic", fontsize=10)
        ax.legend(frameon=False, fontsize="small")
        ax.spines[["right", "top"]].set_visible(False)

    fpath = FIGURES_DIR / f"{fname_stem}.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def _metrics_bar(rows: list, group_col: str, fname_stem: str, title: str = ""):
    """Bar chart of AUC + balanced accuracy."""
    df = pd.DataFrame(rows)
    metrics = ["roc_auc", "balanced_accuracy", "f1"]
    metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4),
                              constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    palette = sns.color_palette("viridis", len(df))

    for ax, met in zip(axes, metrics):
        vals = df[met].values
        labels = df[group_col].values
        colors = [COLORS.get(str(l), palette[i]) for i, l in enumerate(labels)]
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white", width=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", lw=1)
        ax.set_ylim(0.35, 1.0)
        ax.set_title(met.replace("_", " ").title(), style="italic")
        ax.spines[["right", "top"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, style="italic")

    if title:
        fig.suptitle(title, style="italic")

    fpath = FIGURES_DIR / f"{fname_stem}.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Section A — Mode comparison ───────────────────────────────────────────────

def compare_modes():
    print("\n── A. Mode comparison ──────────────────────────────")
    specs = [
        ("Residual conn",  "residual",  "conn"),
        ("Deviation conn", "deviation", "conn"),
        ("Diffusive",      "diffusive", "diffusive_mm"),
    ]
    series, rows = [], []
    for label, data_mode, mode in specs:
        try:
            X, y, sites = _load_data(data_mode, mode)
        except FileNotFoundError as e:
            print(f"  Skipping {label}: {e}")
            continue
        print(f"  {label}:")
        y_pred, y_prob, ft, fp = run_loso(X, y, sites, "RF")
        color = COLORS.get(data_mode, "#888888")
        series.append((label, color, y, y_prob, ft, fp))
        rows.append({"mode": label,
                     "roc_auc": roc_auc_score(y, y_prob),
                     "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                     "f1": f1_score(y, y_pred, zero_division=0)})

    if series:
        _roc_pr_panel(series, "Feature mode comparison (RF, LOSO)", "mode_roc_pr")
        _metrics_bar(rows, "mode", "mode_auc_bar", "Mode comparison")
        df = pd.DataFrame(rows)
        df.to_csv(TABLES_DIR / "mode_comparison.csv", index=False)
        print(f"  Table → {TABLES_DIR / 'mode_comparison.csv'}")


# ── Section B — Classifier comparison ────────────────────────────────────────

def compare_classifiers():
    print("\n── B. Classifier comparison ────────────────────────")
    try:
        X, y, sites = _load_data("diffusive", "diffusive_mm")
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return

    clfs = ["RF", "SVM"]
    if HAS_LGBM:
        clfs.append("LGBM")

    series, rows = [], []
    for clf_name in clfs:
        print(f"  {clf_name}:")
        y_pred, y_prob, ft, fp = run_loso(X, y, sites, clf_name)
        series.append((clf_name, COLORS[clf_name], y, y_prob, ft, fp))
        rows.append({"classifier": clf_name,
                     "roc_auc": roc_auc_score(y, y_prob),
                     "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                     "f1": f1_score(y, y_pred, zero_division=0)})

    _roc_pr_panel(series, "Classifier comparison (diffusive, preproc 2, LOSO)", "clf_roc_pr")
    _metrics_bar(rows, "classifier", "clf_metrics_bar", "Classifier comparison")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "clf_comparison.csv", index=False)
    print(f"  Table → {TABLES_DIR / 'clf_comparison.csv'}")


# ── Section C — Feature-selection comparison ──────────────────────────────────

def compare_feature_selection():
    print("\n── C. Feature-selection comparison ────────────────")
    try:
        X, y, sites = _load_data("diffusive", "diffusive_mm")
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return

    configs = [
        ("No selection",     "RF", "none",  X.shape[1]),
        ("KBest-50",         "RF", "kbest",  50),
        ("RFE-30",           "RF", "rfe",    30),
    ]

    series, rows = [], []
    for label, clf_name, feat_sel, n_feat in configs:
        print(f"  {label}:")
        y_pred, y_prob, ft, fp = run_loso(X, y, sites, clf_name, feat_sel, n_feat)
        series.append((label, COLORS.get(feat_sel, "#888888"), y, y_prob, ft, fp))
        rows.append({"strategy": label,
                     "roc_auc": roc_auc_score(y, y_prob),
                     "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                     "f1": f1_score(y, y_pred, zero_division=0)})

    _roc_pr_panel(series, "Feature-selection comparison (RF, diffusive, LOSO)", "featsel_roc_pr")
    _metrics_bar(rows, "strategy", "featsel_metrics_bar", "Feature selection")
    pd.DataFrame(rows).to_csv(TABLES_DIR / "featsel_comparison.csv", index=False)
    print(f"  Table → {TABLES_DIR / 'featsel_comparison.csv'}")


# ── Section D — Modality ablation ─────────────────────────────────────────────

def compare_modality_ablation():
    """
    Load individual modality deviation features for preproc=2, source space.
    Run LOSO-CV with RF on each modality.
    Compare power, aperiodic, regional_coh, global_coh, graph_coh, diffusive_mm.
    """
    print("\n── D. Modality ablation ────────────────────────────")

    # Each entry: (label, data_mode, mode, conn_mode_override)
    # For diffusive_mm, data_mode is "diffusive"; for Z-score modalities, "deviation"
    modality_specs = [
        ("power",        "deviation",  "power",       None),
        ("aperiodic",    "deviation",  "aperiodic",   None),
        ("regional_coh", "deviation",  "regional",    "coh"),
        ("global_coh",   "deviation",  "global",      "coh"),
        ("graph_coh",    "deviation",  "graph",       "coh"),
        ("diffusive_mm", "diffusive",  "diffusive_mm", "coh"),
    ]

    series, rows = [], []
    for label, data_mode, mode, conn_override in modality_specs:
        cm = conn_override if conn_override is not None else CONN_MODE
        try:
            X, y, sites = _load_data(data_mode, mode, preproc=PREPROC, conn_mode=cm)
        except FileNotFoundError as e:
            print(f"  Skipping {label}: {e}")
            continue
        except Exception as e:
            print(f"  Skipping {label} (error): {e}")
            continue

        print(f"  {label}:")
        try:
            y_pred, y_prob, ft, fp = run_loso(X, y, sites, "RF")
        except Exception as e:
            print(f"    LOSO failed: {e}")
            continue

        color = COLORS.get(label, "#888888")
        series.append((label, color, y, y_prob, ft, fp))
        rows.append({
            "modality":           label,
            "roc_auc":            roc_auc_score(y, y_prob),
            "balanced_accuracy":  balanced_accuracy_score(y, y_pred),
            "f1":                 f1_score(y, y_pred, zero_division=0),
        })

    if not series:
        print("  No modality data could be loaded — skipping ablation plots.")
        return

    _roc_pr_panel(series,
                  "Modality ablation (RF, preproc 2, LOSO)",
                  "modality_ablation_roc_pr")
    _metrics_bar(rows, "modality",
                 "modality_ablation_metrics",
                 "Modality ablation")
    df_abl = pd.DataFrame(rows)
    df_abl.to_csv(TABLES_DIR / "modality_ablation.csv", index=False)
    print(f"  Table → {TABLES_DIR / 'modality_ablation.csv'}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    compare_modes()
    compare_classifiers()
    compare_feature_selection()
    compare_modality_ablation()

    print("\nDone.")
