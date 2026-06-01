"""
Detailed site-by-site generalization analysis.

Two analyses:

A. LOSO per-site breakdown
   - Load diffusive_mm features (preproc=2, source, coh)
   - StratifiedGroupKFold(n_splits=n_sites) LOSO CV
   - Per-site: AUC, balanced_accuracy, sensitivity, specificity, bootstrap CI for AUC
   Plots:
   (a) site_auc_bar.pdf     — horizontal bar chart, sorted by AUC, bootstrap CI
   (b) site_roc_curves.pdf  — all per-site ROC curves on one axes
   (c) site_confusion_heatmap.pdf — 2×n_sites normalised confusion matrices

B. Cross-site transfer matrix
   - For each pair (train_site, test_site): train on train_site, test on test_site
   - RF classifier, balanced class_weight
   - Sites with < 10 subjects of either class → NaN
   Plot:
   (d) cross_site_transfer.pdf — square heatmap, AUC values

Saves:
  results/tables/site_generalization.csv
  results/tables/cross_site_transfer.csv
  results/figures/site_auc_bar.pdf
  results/figures/site_roc_curves.pdf
  results/figures/site_confusion_heatmap.pdf
  results/figures/cross_site_transfer.pdf

Run from src/:  python 22_site_generalization.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
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
    confusion_matrix,
    roc_curve,
)

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"

CSV_PATH    = TINNORM_DIR / "diffusive_mm" / "source_preproc_2_coh.csv"
MASTER_PATH = Path("../material/master_clean.csv")

RANDOM_STATE   = 42
N_BOOTSTRAP    = 1000
MIN_PER_CLASS  = 10


# ── Classifier factory ────────────────────────────────────────────────────────

def _make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data():
    """Load diffusive_mm CSV + master, return (X, y, sites)."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not MASTER_PATH.exists():
        print(f"ERROR: master file not found: {MASTER_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df["subject_id"] = df["subject_id"].astype(str)

    df_master = pd.read_csv(MASTER_PATH)
    df_master["subject_id"] = df_master["subject_id"].astype(str)

    df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
    df.rename(columns={"site": "SITE"}, inplace=True)
    df.drop(columns=["Unnamed: 0", "subject_id", "age", "sex",
                      "PTA4_mean", "PTA4_HF", "thi_score", "THI"],
            inplace=True, errors="ignore")

    y     = df["group"].to_numpy()
    sites = df["SITE"].to_numpy()
    X     = df.drop(columns=["group", "SITE"], errors="ignore")

    print(f"  Loaded: {len(y)} subjects, {X.shape[1]} features")
    print(f"  Sites: {sorted(np.unique(sites))}")
    print(f"  Groups: {dict(pd.Series(y).value_counts().sort_index())}")
    return X, y, sites


# ── Bootstrap AUC CI ──────────────────────────────────────────────────────────

def _bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, alpha=0.05, rng_seed=0):
    """Return (lower, upper) CI for AUC via bootstrap resampling."""
    rng  = np.random.default_rng(rng_seed)
    n    = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        yp  = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


# ── Part A: LOSO per-site breakdown ──────────────────────────────────────────

def run_loso_per_site(X, y, sites):
    """
    StratifiedGroupKFold LOSO. For each held-out site, compute:
    AUC, balanced_accuracy, sensitivity, specificity, bootstrap CI.
    Returns per-site dict and full prediction arrays.
    """
    unique_sites = sorted(np.unique(sites))
    n_sites = len(unique_sites)
    sgkf    = StratifiedGroupKFold(n_splits=n_sites, shuffle=True,
                                    random_state=RANDOM_STATE)

    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y))
    site_results = {}

    for train_idx, test_idx in sgkf.split(X, y, groups=sites):
        held_site = np.unique(sites[test_idx])[0]
        print(f"  Held-out site: {held_site}  (N={len(test_idx)})", end="  ")

        pipe = _make_rf()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_pred[test_idx] = pipe.predict(X.iloc[test_idx])
        y_prob[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

        yt = y[test_idx]
        yp = y_prob[test_idx]
        yd = y_pred[test_idx]

        n_ctrl = int((yt == 0).sum())
        n_tin  = int((yt == 1).sum())

        if len(np.unique(yt)) < 2:
            print(f"only one class present — skipping metrics.")
            site_results[held_site] = {
                "site": held_site, "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "balanced_accuracy": np.nan, "sensitivity": np.nan,
                "specificity": np.nan, "n_controls": n_ctrl, "n_tinnitus": n_tin,
                "y_true": yt, "y_prob": yp, "y_pred": yd,
            }
            continue

        auc_val  = roc_auc_score(yt, yp)
        bal_acc  = balanced_accuracy_score(yt, yd)
        cm       = confusion_matrix(yt, yd, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        auc_lo, auc_hi = _bootstrap_auc_ci(yt, yp)

        print(f"AUC={auc_val:.3f}  bal_acc={bal_acc:.3f}")
        site_results[held_site] = {
            "site": held_site, "auc": auc_val, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "balanced_accuracy": bal_acc, "sensitivity": sens, "specificity": spec,
            "n_controls": n_ctrl, "n_tinnitus": n_tin,
            "y_true": yt, "y_prob": yp, "y_pred": yd,
        }

    return site_results, y_pred, y_prob


# ── Plot A(a): Horizontal bar chart sorted by AUC ────────────────────────────

def plot_site_auc_bar(site_results, save_dir=FIGURES_DIR):
    """Horizontal bar chart sorted by AUC with bootstrap CI error bars."""
    rows = [v for v in site_results.values() if not np.isnan(v["auc"])]
    if not rows:
        print("  No site AUC data to plot.")
        return
    df_plot = pd.DataFrame(rows).sort_values("auc")

    palette = sns.color_palette("cubehelix", len(df_plot))
    norm_vals = (df_plot["auc"].values - df_plot["auc"].min()) / \
                max(df_plot["auc"].max() - df_plot["auc"].min(), 1e-6)
    colors = [palette[int(round(v * (len(palette) - 1)))] for v in norm_vals]

    xerr_lo = (df_plot["auc"] - df_plot["auc_lo"]).fillna(0).values
    xerr_hi = (df_plot["auc_hi"] - df_plot["auc"]).fillna(0).values

    fig, ax = plt.subplots(figsize=(7, max(4, len(df_plot) * 0.55)),
                            constrained_layout=True)
    y_pos = np.arange(len(df_plot))

    ax.barh(y_pos, df_plot["auc"], color=colors, alpha=0.85,
            edgecolor="white", height=0.6,
            xerr=np.vstack([xerr_lo, xerr_hi]),
            error_kw=dict(elinewidth=1.5, capsize=4, ecolor="black"))

    ax.axvline(0.5, color="#7f8c8d", linestyle="--", lw=1.2, label="Chance (0.5)")

    for i, row in df_plot.reset_index(drop=True).iterrows():
        label = f"(ctrl={int(row['n_controls'])}/tin={int(row['n_tinnitus'])})"
        ax.text(row["auc"] + 0.01, i, label, va="center", fontsize=7, style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["site"].values, fontsize=9)
    ax.set_xlabel("ROC-AUC  (95% bootstrap CI)", fontsize=10)
    ax.set_title("Per-site AUC — LOSO cross-validation", style="italic", fontsize=11)
    ax.set_xlim(0.2, 1.15)
    ax.legend(frameon=False, fontsize="small")
    sns.despine(ax=ax)

    fpath = save_dir / "site_auc_bar.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Plot A(b): ROC curves per site ───────────────────────────────────────────

def plot_site_roc_curves(site_results, save_dir=FIGURES_DIR):
    """All per-site ROC curves on one axes, tab10 palette."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    palette = sns.color_palette("tab10", len(site_results))
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d", lw=1, label="Chance")

    for i, (site, res) in enumerate(sorted(site_results.items())):
        if np.isnan(res["auc"]) or len(np.unique(res["y_true"])) < 2:
            continue
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=palette[i], lw=2.5,
                label=f"{site}  (AUC={res['auc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Per-site ROC curves  (LOSO-CV)", style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax)

    fpath = save_dir / "site_roc_curves.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Plot A(c): Confusion matrix heatmaps ─────────────────────────────────────

def plot_site_confusion_heatmap(site_results, save_dir=FIGURES_DIR):
    """2×n_sites grid of normalised confusion matrices."""
    valid_sites = sorted([s for s, r in site_results.items()
                          if not np.isnan(r["auc"])])
    if not valid_sites:
        print("  No valid sites for confusion heatmap.")
        return

    n_sites = len(valid_sites)
    fig, axes = plt.subplots(2, n_sites, figsize=(3.5 * n_sites, 7),
                              constrained_layout=True)
    if n_sites == 1:
        axes = np.array(axes).reshape(2, 1)

    for col_idx, site in enumerate(valid_sites):
        res = site_results[site]
        yt  = res["y_true"]
        yd  = res["y_pred"]

        cm_raw  = confusion_matrix(yt, yd, labels=[0, 1])
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

        for row_idx, (cm, title_sfx) in enumerate([(cm_raw,  " (counts)"),
                                                     (cm_norm, " (normalised)")]):
            ax = axes[row_idx, col_idx]
            fmt = "d" if row_idx == 0 else ".2f"
            cmap = "Blues"
            sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                        xticklabels=["Ctrl", "Tin"],
                        yticklabels=["Ctrl", "Tin"],
                        cbar=False, linewidths=0.5)
            ax.set_title(f"{site}{title_sfx}", fontsize=8, style="italic")
            if col_idx == 0:
                ax.set_ylabel("True label", fontsize=8)
            if row_idx == 1:
                ax.set_xlabel("Predicted label", fontsize=8)

    fig.suptitle("Confusion matrices per held-out site  (LOSO-CV)",
                 style="italic", fontsize=12)
    fpath = save_dir / "site_confusion_heatmap.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Part B: Cross-site transfer matrix ───────────────────────────────────────

def run_cross_site_transfer(X, y, sites):
    """
    For each (train_site, test_site) pair: train on train_site, test on test_site.
    Returns DataFrame of AUC values (NaN for diagonal or skipped pairs).
    """
    unique_sites = sorted(np.unique(sites))
    n  = len(unique_sites)
    auc_matrix = np.full((n, n), np.nan)

    print(f"  Cross-site transfer matrix ({n}×{n}) …")
    for i, train_site in enumerate(unique_sites):
        for j, test_site in enumerate(unique_sites):
            if train_site == test_site:
                continue  # diagonal stays NaN

            train_mask = sites == train_site
            test_mask  = sites == test_site

            y_tr  = y[train_mask]
            y_te  = y[test_mask]
            n_ctrl_tr = (y_tr == 0).sum()
            n_tin_tr  = (y_tr == 1).sum()
            n_ctrl_te = (y_te == 0).sum()
            n_tin_te  = (y_te == 1).sum()

            if (min(n_ctrl_tr, n_tin_tr) < MIN_PER_CLASS or
                    min(n_ctrl_te, n_tin_te) < MIN_PER_CLASS):
                print(f"    {train_site} → {test_site}: skipped (class count < {MIN_PER_CLASS})")
                continue

            if len(np.unique(y_te)) < 2:
                print(f"    {train_site} → {test_site}: skipped (single class in test set)")
                continue

            X_tr = X[train_mask]
            X_te = X[test_mask]

            pipe = _make_rf()
            pipe.fit(X_tr, y_tr)
            y_prob_te = pipe.predict_proba(X_te)[:, 1]
            auc_val = roc_auc_score(y_te, y_prob_te)
            auc_matrix[i, j] = auc_val
            print(f"    {train_site:<12} → {test_site:<12}  AUC={auc_val:.3f}")

    df_transfer = pd.DataFrame(auc_matrix, index=unique_sites, columns=unique_sites)
    return df_transfer


# ── Plot B(d): Cross-site transfer heatmap ────────────────────────────────────

def plot_cross_site_transfer(df_transfer, save_dir=FIGURES_DIR):
    """Square heatmap of cross-site AUC. Diagonal = NaN (grey)."""
    fig, ax = plt.subplots(
        figsize=(max(6, len(df_transfer) * 1.1), max(5, len(df_transfer) * 1.0)),
        constrained_layout=True,
    )

    # Build mask for diagonal (NaN) → display grey
    mask_diag = np.eye(len(df_transfer), dtype=bool)

    # Fill diagonal with grey background
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    cmap.set_bad(color="#aaaaaa")

    # Annotate non-NaN cells with AUC value, NaN with ""
    annot_arr = df_transfer.copy().applymap(lambda v: f"{v:.2f}" if not np.isnan(v) else "")

    sns.heatmap(df_transfer, annot=annot_arr, fmt="", cmap=cmap,
                vmin=0.4, vmax=1.0, ax=ax,
                mask=None, linewidths=0.5, linecolor="#cccccc",
                cbar_kws={"label": "ROC-AUC", "shrink": 0.8},
                annot_kws={"fontsize": 9})

    # Overlay diagonal in grey manually
    n = len(df_transfer)
    for k in range(n):
        ax.add_patch(plt.Rectangle((k, k), 1, 1, fill=True, color="#aaaaaa", lw=0))

    ax.set_xlabel("Test site", fontsize=10)
    ax.set_ylabel("Train site", fontsize=10)
    ax.set_title("Cross-site transfer matrix  (train→test AUC, RF)",
                 style="italic", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    fpath = save_dir / "cross_site_transfer.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    X, y, sites = _load_data()

    # ── Part A: LOSO per-site ─────────────────────────────────────────────
    print("\n── A. LOSO per-site breakdown ──────────────────────────")
    site_results, y_pred_loso, y_prob_loso = run_loso_per_site(X, y, sites)

    # Save site generalization table
    df_site_gen = pd.DataFrame([
        {k: v for k, v in res.items() if k not in ("y_true", "y_prob", "y_pred")}
        for res in site_results.values()
    ])
    df_site_gen.to_csv(TABLES_DIR / "site_generalization.csv", index=False)
    print(f"  Table → {TABLES_DIR / 'site_generalization.csv'}")

    print("\n  Plotting site AUC bar chart …")
    plot_site_auc_bar(site_results, save_dir=FIGURES_DIR)

    print("  Plotting per-site ROC curves …")
    plot_site_roc_curves(site_results, save_dir=FIGURES_DIR)

    print("  Plotting site confusion heatmaps …")
    plot_site_confusion_heatmap(site_results, save_dir=FIGURES_DIR)

    # ── Part B: Cross-site transfer matrix ───────────────────────────────
    print("\n── B. Cross-site transfer matrix ───────────────────────")
    df_transfer = run_cross_site_transfer(X, y, sites)
    df_transfer.to_csv(TABLES_DIR / "cross_site_transfer.csv")
    print(f"  Table → {TABLES_DIR / 'cross_site_transfer.csv'}")

    print("  Plotting cross-site transfer heatmap …")
    plot_cross_site_transfer(df_transfer, save_dir=FIGURES_DIR)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
