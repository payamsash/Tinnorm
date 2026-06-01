"""
Which preprocessing level works best?

Loads diffusive Mahalanobis features for preproc levels 1, 2, and 3, runs
LOSO cross-validation (Leave-One-Site-Out, stratified by group), and saves:

  results/figures/preproc_roc_pr.pdf          — ROC + PR with per-fold faint lines
  results/figures/preproc_auc_bar.pdf         — AUC bar chart
  results/figures/preproc_prob_sites.pdf      — per-site probability strip plots
  results/tables/preproc_comparison.csv       — AUC, balanced-accuracy, F1 per level

Run from src/:  python 17_compare_preproc_levels.py
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

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR  = TINNORM_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"

SPACE        = "source"
CONN_MODE    = "coh"
RANDOM_STATE = 42

PREPROC_COLORS = {1: "#D6B4F0", 2: "#9B59B6", 3: "#4A235A"}
CTRL_COLOR     = "#1f77b4"
TIN_COLOR      = "#C99700"
CHANCE_COLOR   = "#7f8c8d"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_diffusive(preproc_level: int, conn_mode: str = CONN_MODE):
    df_master = pd.read_csv("../material/master_clean.csv")
    fname = TINNORM_DIR / "diffusive_mm" / f"{SPACE}_preproc_{preproc_level}_{conn_mode}.csv"
    df = pd.read_csv(fname)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
    df.rename(columns={"site": "SITE"}, inplace=True)

    subject_ids = df["subject_id"].to_numpy()
    sites       = df["SITE"].to_numpy()
    y           = df["group"].to_numpy()
    drop_cols   = ["group", "Unnamed: 0", "subject_id", "SITE",
                   "age", "sex", "PTA4_mean", "PTA4_HF", "thi_score", "THI"]
    X = df.drop(columns=drop_cols, errors="ignore")
    print(f"  Preproc {preproc_level}: {len(y)} subjects, {X.shape[1]} features")
    return X, y, sites, subject_ids


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_pipe():
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ── LOSO CV ───────────────────────────────────────────────────────────────────

def run_loso(X, y, sites):
    n_sites = len(np.unique(sites))
    sgkf    = StratifiedGroupKFold(n_splits=n_sites, shuffle=True,
                                   random_state=RANDOM_STATE)
    y_pred  = np.zeros_like(y)
    y_prob  = np.zeros(len(y))
    fold_true, fold_prob, fold_site_labels = [], [], []

    for train_idx, test_idx in sgkf.split(X, y, groups=sites):
        pipe = _make_pipe()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X.iloc[test_idx])
        y_prob[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        fold_true.append(y[test_idx])
        fold_prob.append(y_prob[test_idx])
        fold_site_labels.append(np.unique(sites[test_idx])[0])

    return y_pred, y_prob, fold_true, fold_prob, fold_site_labels


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_pr(results: dict):
    """ROC + PR panel, one curve per preproc level, faint fold lines behind."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5),
                                         constrained_layout=True)
    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)

    for pl, res in results.items():
        color = PREPROC_COLORS[pl]
        tprs, precs = [], []

        for yt, yp in zip(res["fold_true"], res["fold_prob"]):
            if len(np.unique(yt)) < 2:
                continue
            fpr_, tpr_, _ = roc_curve(yt, yp)
            tprs.append(np.interp(mean_fpr, fpr_, tpr_))
            ax_roc.plot(fpr_, tpr_, color=color, alpha=0.18, lw=1)
            prec_, rec_, _ = precision_recall_curve(yt, yp)
            precs.append(np.interp(mean_rec, rec_[::-1], prec_[::-1]))
            ax_pr.plot(rec_, prec_, color=color, alpha=0.18, lw=1)

        auc_val = roc_auc_score(res["y"], res["y_prob"])
        ax_roc.plot(mean_fpr, np.mean(tprs, axis=0), color=color, lw=3,
                    label=f"Preproc {pl}  (AUC={auc_val:.3f})")
        ap_val  = auc(mean_rec, np.mean(precs, axis=0))
        ax_pr.plot(mean_rec, np.mean(precs, axis=0), color=color, lw=3,
                   label=f"Preproc {pl}  (AP={ap_val:.3f})")

    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2, label="Chance")
    ax_pr.axhline(0.5, linestyle="--", color=CHANCE_COLOR, lw=1.2, label="Chance")

    for ax, xl, yl, ttl in [
        (ax_roc, "False Positive Rate", "True Positive Rate", "ROC — preprocessing level"),
        (ax_pr,  "Recall",              "Precision",          "PR  — preprocessing level"),
    ]:
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(ttl, style="italic", fontsize=10)
        ax.legend(frameon=False, fontsize="small")
        ax.spines[["right", "top"]].set_visible(False)

    fpath = FIGURES_DIR / "preproc_roc_pr.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_auc_bar(df_metrics: pd.DataFrame):
    """Simple bar chart: AUC per preprocessing level."""
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    bars = ax.bar(
        [f"Preproc {p}" for p in df_metrics["preproc_level"]],
        df_metrics["roc_auc"],
        color=[PREPROC_COLORS[p] for p in df_metrics["preproc_level"]],
        edgecolor="white", width=0.5,
    )
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", lw=1.2, label="Chance")
    for bar, val in zip(bars, df_metrics["roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, style="italic")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC AUC", fontsize=11)
    ax.set_title("ROC AUC by preprocessing level", style="italic")
    ax.legend(frameon=False)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = FIGURES_DIR / "preproc_auc_bar.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_prob_per_site(results: dict, subject_ids_dict: dict):
    """Per-site probability strip + boxen plots, one column per preproc level."""
    dfs = []
    for pl, res in results.items():
        sids = subject_ids_dict[pl]
        dfs.append(pd.DataFrame({
            "subject_id":    sids,
            "site":          res["sites"],
            "group":         res["y"],
            "prob_tinnitus": res["y_prob"],
            "preproc":       f"Preproc {pl}",
        }))
    df_all = pd.concat(dfs, ignore_index=True)

    site_order = sorted(df_all["site"].unique())
    n_pl = len(results)
    fig, axes = plt.subplots(1, n_pl, figsize=(5 * n_pl, 5), sharey=True,
                              constrained_layout=True)
    if n_pl == 1:
        axes = [axes]

    for ax, (pl, res) in zip(axes, results.items()):
        df_p = df_all[df_all["preproc"] == f"Preproc {pl}"]
        sns.stripplot(data=df_p, x="site", y="prob_tinnitus", order=site_order,
                      hue="group", palette={0: CTRL_COLOR, 1: TIN_COLOR},
                      dodge=True, alpha=0.55, size=5, jitter=0.15,
                      legend=(ax is axes[0]), ax=ax)
        sns.boxenplot(data=df_p, x="site", y="prob_tinnitus", order=site_order,
                      hue="group", palette={0: CTRL_COLOR, 1: TIN_COLOR},
                      dodge=True, alpha=0.22, showfliers=False,
                      legend=False, ax=ax)
        ax.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.5)
        ax.set_title(f"Preproc {pl}", style="italic")
        ax.set_xlabel("Site")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("P(Tinnitus)" if ax is axes[0] else "")
        if ax is axes[0] and ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Control", "Tinnitus"], frameon=False,
                      loc="upper right", fontsize="small")

    fpath = FIGURES_DIR / "preproc_prob_sites.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_metrics_heatmap(df_metrics: pd.DataFrame):
    """Heatmap of all metrics × preproc level."""
    metric_cols = ["roc_auc", "balanced_accuracy", "f1"]
    metric_cols = [c for c in metric_cols if c in df_metrics.columns]
    df_hm = df_metrics.set_index("preproc_level")[metric_cols].T

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    sns.heatmap(df_hm, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0.4, vmax=1.0, linewidths=0.5,
                annot_kws={"size": 11}, ax=ax)
    ax.set_title("Metrics × preprocessing level", style="italic")
    ax.set_xlabel("Preprocessing level")
    ax.set_ylabel("")
    ax.set_xticklabels([f"Preproc {v}" for v in df_metrics["preproc_level"]],
                       rotation=0)

    fpath = FIGURES_DIR / "preproc_metrics_heatmap.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results        = {}
    subject_ids_dict = {}
    metrics_rows   = []

    for pl in [1, 2, 3]:
        print(f"\nPreproc {pl}:")
        try:
            X, y, sites, sids = _load_diffusive(pl)
        except FileNotFoundError as e:
            print(f"  Skipping — file not found: {e}")
            continue

        y_pred, y_prob, fold_true, fold_prob, fold_sites = run_loso(X, y, sites)

        results[pl] = {
            "y": y, "y_pred": y_pred, "y_prob": y_prob,
            "sites": sites,
            "fold_true": fold_true, "fold_prob": fold_prob,
        }
        subject_ids_dict[pl] = sids

        row = {
            "preproc_level":      pl,
            "roc_auc":            roc_auc_score(y, y_prob),
            "balanced_accuracy":  balanced_accuracy_score(y, y_pred),
            "f1":                 f1_score(y, y_pred, zero_division=0),
            "n_subjects":         len(y),
            "n_features":         X.shape[1],
        }
        metrics_rows.append(row)
        print(f"  AUC={row['roc_auc']:.3f}  BAcc={row['balanced_accuracy']:.3f}  F1={row['f1']:.3f}")

    if not results:
        print("No diffusive_mm data found — run 09_multimodal_diffusion.py first.")
    else:
        df_metrics = pd.DataFrame(metrics_rows)
        csv_path = TABLES_DIR / "preproc_comparison.csv"
        df_metrics.to_csv(csv_path, index=False)
        print(f"\nMetrics → {csv_path}")

        plot_roc_pr(results)
        plot_auc_bar(df_metrics)
        plot_prob_per_site(results, subject_ids_dict)
        plot_metrics_heatmap(df_metrics)

    print("\nDone.")
