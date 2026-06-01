"""
Plot and save all classification results stored under clfs/.

Reads metrics.csv and .npy arrays from timestamped sub-folders and produces:
  • Permutation-test histograms with real-model AUC and p-value annotation
  • ROC / PR curve panels for model comparisons
  • Multi-metric bar charts across scenarios

Run from src/ or set TINNORM_DIR below.  All figures saved to results/figures/.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

TINNORM_DIR  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
CLFS_DIR     = TINNORM_DIR / "clfs"
RESULTS_DIR  = TINNORM_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"

# ── Colour palette ────────────────────────────────────────────────────────────

COLORS = {
    "residual":  "#1f77b4",
    "deviation": "#245C43",
    "diffusive": "#C99700",
    "RF":        "#D55E00",
    "SVM":       "#0072B2",
    "LGBM":      "#009E73",
}
CHANCE_COLOR = "#7f8c8d"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _setup_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load_all_metrics(run_dir: Path) -> pd.DataFrame:
    """Collect metrics.csv from every timestamped sub-folder in run_dir."""
    dfs = []
    if not run_dir.exists():
        return pd.DataFrame()
    for folder in sorted(run_dir.iterdir()):
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        f = folder / "metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["run_id"] = folder.name
        df["run_dir"] = str(folder)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame()


def _load_arrays(folder: Path, *names):
    """Load one or more .npy arrays from folder."""
    out = []
    for name in names:
        p = folder / f"{name}.npy"
        out.append(np.load(p, allow_pickle=True) if p.exists() else None)
    return out[0] if len(out) == 1 else tuple(out)


def _style_axes(*axes):
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(frameon=False, fontsize="small")


# ── 1. Permutation-test plots ─────────────────────────────────────────────────

def plot_permutation_panel(perm_dir: Path, scenario_label: str, metric: str = "roc_auc"):
    """Histogram of permuted AUC + real-model AUC line + ROC / PR curves."""
    df = _load_all_metrics(perm_dir)
    if df.empty:
        print(f"  No permutation results in {perm_dir}")
        return

    # Separate real from permuted
    df_real = df[df["model"].str.endswith("_real", na=False)]
    df_perm = df[df["model"].str.contains("_perm_", na=False)]
    if df_real.empty:
        print("  No real-model row found.")
        return

    real_val = df_real[metric].iloc[0]
    perm_vals = df_perm[metric].values
    p_val = (perm_vals >= real_val).mean() if len(perm_vals) else float("nan")

    # Get run folder for arrays
    run_folder = Path(df_real["run_dir"].iloc[0])
    y, y_prob = _load_arrays(run_folder, "y", "y_prob")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    # (a) Histogram
    ax = axes[0]
    ax.hist(perm_vals, bins=25, color="lightgray", edgecolor="white", alpha=0.85)
    ax.axvline(real_val, color="#1f77b4", lw=2.5, linestyle="--")
    ylim = ax.get_ylim()
    ax.text(real_val + 0.002, ylim[1] * 0.92,
            f"real = {real_val:.3f}\np = {p_val:.3f}",
            fontsize=9, style="italic", color="#1f77b4", va="top")
    ax.set_xlabel(metric.upper().replace("_", " "))
    ax.set_ylabel("Count")
    ax.set_title("Permutation distribution", style="italic")

    # (b) ROC
    if y is not None and y_prob is not None:
        RocCurveDisplay.from_predictions(y, y_prob, ax=axes[1],
                                         plot_chance_level=True, despine=True)
        PrecisionRecallDisplay.from_predictions(y, y_prob, ax=axes[2],
                                                plot_chance_level=True, despine=True)

    _style_axes(*axes)
    fname = FIGURES_DIR / f"permutation_{scenario_label}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── 2. Multi-scenario ROC / PR comparison ─────────────────────────────────────

def plot_scenario_roc_pr(
    scenario_specs: list,
    title: str,
    fname_stem: str,
):
    """
    scenario_specs : list of (label, color, folder_path, prob_key)
       label      – legend label
       color      – line color
       folder     – Path to the run folder containing y.npy + y_prob*.npy
       prob_key   – "y_prob" | "y_prob_1" | "y_prob_2"
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1)
    ax_pr.axhline(0.5, linestyle="--", color=CHANCE_COLOR, lw=1)

    mean_fpr    = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    for label, color, folder, prob_key in scenario_specs:
        y      = _load_arrays(folder, "y")
        y_prob = _load_arrays(folder, prob_key)
        if y is None or y_prob is None:
            continue
        auc_val = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, label=f"{label}  (AUC={auc_val:.3f})")
        prec, rec, _ = precision_recall_curve(y, y_prob)
        ap = auc(rec, prec)
        ax_pr.plot(rec, prec, color=color, lw=2.5, label=f"{label}  (AP={ap:.3f})")

    ax_roc.set(xlabel="FPR", ylabel="TPR", title=f"{title} — ROC")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title=f"{title} — PR")
    for ax in [ax_roc, ax_pr]:
        ax.set_title(ax.get_title(), style="italic")
    _style_axes(ax_roc, ax_pr)

    fname = FIGURES_DIR / f"{fname_stem}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── 3. Multi-metric summary bar chart ────────────────────────────────────────

def plot_metrics_bar(df_metrics: pd.DataFrame, group_col: str, fname_stem: str, title: str = ""):
    """Grouped bar chart of AUC + balanced_accuracy per group."""
    metrics = ["roc_auc", "balanced_accuracy", "f1-score"]
    metrics = [m for m in metrics if m in df_metrics.columns]
    n_metrics = len(metrics)
    groups = df_metrics[group_col].unique()
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), constrained_layout=True)
    if n_metrics == 1:
        axes = [axes]

    palette = sns.color_palette("viridis", n_groups)

    for ax, metric in zip(axes, metrics):
        vals = [df_metrics.loc[df_metrics[group_col] == g, metric].values[0]
                for g in groups]
        bars = ax.bar(range(n_groups), vals, color=palette, edgecolor="white", width=0.5)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title(), style="italic")
        ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", lw=1)
        ax.set_ylim(0.3, 1.05)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, style="italic")
        ax.spines[["right", "top"]].set_visible(False)

    if title:
        fig.suptitle(title, style="italic", fontsize=11)

    fname = FIGURES_DIR / f"{fname_stem}.pdf"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fname}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    _setup_dirs()

    # ── A. Permutation tests — one panel per scenario folder ──────────────────
    perm_root = CLFS_DIR
    for scenario_dir in sorted(perm_root.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
            continue
        # Check for permutation sub-directories
        perm_subdir = scenario_dir / "permutation"
        if perm_subdir.exists():
            print(f"\nPermutation panel: {scenario_dir.name}")
            plot_permutation_panel(perm_subdir, scenario_dir.name)
        # Also handle flat (old-style) permutation folders
        metrics_f = scenario_dir / "metrics.csv"
        if metrics_f.exists():
            df = pd.read_csv(metrics_f)
            if "model" in df.columns and df["model"].str.contains("_perm_", na=False).any():
                print(f"\nPermutation panel (flat): {scenario_dir.name}")
                plot_permutation_panel(scenario_dir, scenario_dir.name)

    # ── B. Scenario comparison bar chart — aggregate real-model rows ───────────
    all_real = []
    for scenario_dir in sorted(perm_root.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
            continue
        for sub in [scenario_dir, scenario_dir / "permutation"]:
            mf = sub / "metrics.csv"
            if mf.exists():
                df = pd.read_csv(mf)
                df_real = df[df.get("model", pd.Series(dtype=str)).str.endswith("_real", na=False)]
                if not df_real.empty:
                    row = df_real.iloc[0].to_dict()
                    row["scenario"] = scenario_dir.name
                    all_real.append(row)
                break

    if all_real:
        df_summary = pd.DataFrame(all_real)
        df_summary.to_csv(TABLES_DIR / "all_scenario_metrics.csv", index=False)
        print(f"\nScenario summary → {TABLES_DIR / 'all_scenario_metrics.csv'}")
        plot_metrics_bar(df_summary, group_col="scenario",
                         fname_stem="scenario_comparison_bar",
                         title="AUC across scenarios")

    print("\nAll result plots saved to:", FIGURES_DIR)
