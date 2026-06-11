"""
Network-level and lateralization analyses of spatial EEG deviation patterns.

Two analyses:

A. Yeo-7 network aggregation
   - Load diffusive_mm features for preproc=2, space=source, conn_mode=coh
   - Map each aparc ROI to its Yeo-7 network using a hardcoded name→network dict
   - For each network: mean deviation per subject, then group-level mean ± SE
   - Plots: (a) radar/spider chart, (b) grouped bar chart with individual dots,
            (c) Mann-Whitney U with FDR correction

B. Lateralization index (LI)
   - Auditory ROIs: transversetemporal, superiortemporal, supramarginal, insula
   - LI = (LH_score - RH_score) / (LH_score + RH_score + 1e-8)
   - Plots: (a) half-violin + strip by group, (b) Mann-Whitney with FDR,
            (c) scatter LH vs RH per subject colored by group

Saves tables to TINNORM_DIR/results/tables/
Saves figures to TINNORM_DIR/results/figures/

Run from src/:  python 19_network_lateralization.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
from math import pi

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, sem
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "19_network_lateralization"
TABLES_DIR  = RESULTS_DIR / "tables"

PREPROC   = 2
SPACE     = "source"
CONN_MODE = "pli"   # best-performing connectivity measure

CTRL_COLOR   = "#1f77b4"
TIN_COLOR    = "#C99700"
CHANCE_COLOR = "#7f8c8d"

# ── Yeo-7 ROI → network mapping ───────────────────────────────────────────────

ROI_TO_NETWORK = {
    "bankssts":               "VAN",
    "caudalanteriorcingulate": "DMN",
    "caudalmiddlefrontal":    "DAN",
    "cuneus":                 "VIS",
    "entorhinal":             "DMN",
    "frontalpole":            "DMN",
    "fusiform":               "VAN",
    "inferiorparietal":       "VAN",
    "inferiortemporal":       "VIS",
    "insula":                 "VAN",
    "isthmuscingulate":       "DMN",
    "lateraloccipital":       "VIS",
    "lateralorbitofrontal":   "FPN",
    "lingual":                "VIS",
    "medialorbitofrontal":    "DMN",
    "middletemporal":         "VAN",
    "paracentral":            "SMN",
    "parahippocampal":        "DMN",
    "parsopercularis":        "FPN",
    "parsorbitalis":          "FPN",
    "parstriangularis":       "FPN",
    "pericalcarine":          "VIS",
    "postcentral":            "SMN",
    "posteriorcingulate":     "DMN",
    "precentral":             "SMN",
    "precuneus":              "DMN",
    "rostralanteriorcingulate": "DMN",
    "rostralmiddlefrontal":   "FPN",
    "superiorfrontal":        "FPN",
    "superiorparietal":       "DAN",
    "superiortemporal":       "SMN",
    "supramarginal":          "VAN",
    "temporalpole":           "DMN",
    "transversetemporal":     "SMN",
}

NETWORKS = ["VIS", "SMN", "DAN", "VAN", "FPN", "DMN"]

AUDITORY_ROIS = ["transversetemporal", "superiortemporal", "supramarginal", "insula"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data():
    """Load diffusive_mm CSV and master, return merged DataFrame."""
    csv_path = TINNORM_DIR / "diffusive_mm" / f"{SPACE}_preproc_{PREPROC}_{CONN_MODE}.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    master_path = Path("../material/master_clean.csv")
    if not master_path.exists():
        print(f"ERROR: master file not found: {master_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df_master = pd.read_csv(master_path)
    df_master["subject_id"] = df_master["subject_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)

    df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
    df.rename(columns={"site": "SITE"}, inplace=True)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    print(f"  Loaded: {len(df)} subjects  |  groups: {dict(df['group'].value_counts())}")
    return df


# ── Identify ROI columns ──────────────────────────────────────────────────────

def _get_roi_columns(df):
    """Return list of columns that are ROI scores (e.g. 'bankssts-lh')."""
    skip = {"group", "SITE", "subject_id", "age", "sex",
            "PTA4_mean", "PTA4_HF", "thi_score", "THI", "TFI"}
    roi_cols = [c for c in df.columns if c not in skip
                and ("-lh" in c or "-rh" in c)]
    return roi_cols


# ── Network aggregation ───────────────────────────────────────────────────────

def _compute_network_scores(df, roi_cols):
    """
    For each subject: compute mean deviation per Yeo-7 network.
    Returns DataFrame indexed by subjects with columns = network names.
    """
    net_scores = {}
    for net in NETWORKS:
        # collect all ROI columns belonging to this network (both hemispheres)
        net_cols = []
        for col in roi_cols:
            # col is like "bankssts-lh" or "bankssts-rh"
            roi_name = col.rsplit("-", 1)[0]
            if ROI_TO_NETWORK.get(roi_name) == net:
                net_cols.append(col)
        if net_cols:
            net_scores[net] = df[net_cols].mean(axis=1).values
        else:
            net_scores[net] = np.zeros(len(df))

    df_net = pd.DataFrame(net_scores, index=df.index)
    df_net["group"] = df["group"].values
    df_net["SITE"]  = df["SITE"].values
    return df_net


# ── Plot A(a): Radar / spider chart ──────────────────────────────────────────

def plot_network_radar(df_net, save_dir=FIGURES_DIR):
    """Radar plot: 6 spokes, two filled polygons (ctrl=blue, tin=gold)."""
    nets = NETWORKS
    n    = len(nets)
    # Simple evenly-spaced angles; set_theta_offset / set_theta_direction
    # handle the rotation so angles must NOT have the offset pre-baked in.
    angles        = np.linspace(0, 2 * pi, n, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    ctrl_mean = df_net[df_net["group"] == 0][nets].mean().values
    tin_mean  = df_net[df_net["group"] == 1][nets].mean().values

    ctrl_closed = np.append(ctrl_mean, ctrl_mean[0])
    tin_closed  = np.append(tin_mean,  tin_mean[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"),
                            facecolor="white")
    ax.set_facecolor("white")
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, ctrl_closed, color=CTRL_COLOR, lw=2.5, label="Control")
    ax.fill(angles_closed, ctrl_closed, color=CTRL_COLOR, alpha=0.2)
    ax.plot(angles_closed, tin_closed, color=TIN_COLOR, lw=2.5, label="Tinnitus")
    ax.fill(angles_closed, tin_closed, color=TIN_COLOR, alpha=0.2)

    ax.set_xticks(angles)
    ax.set_xticklabels(nets, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    ax.spines["polar"].set_color("#cccccc")
    ax.grid(color="#dddddd", linestyle="--", linewidth=0.8)

    ax.set_title("Yeo-7 network deviation\n(mean diffusive score)", style="italic",
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), frameon=False, fontsize=10)

    fpath = save_dir / "network_radar.pdf"
    fig.savefig(fpath, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Plot A(b/c): Grouped bar chart with individual dots + stats ───────────────

def plot_network_bar(df_net, save_dir=FIGURES_DIR, tables_dir=TABLES_DIR):
    """Grouped bar chart: x=network, two bars (ctrl/tin), dots, MWU FDR-corrected."""
    nets = NETWORKS
    df_ctrl = df_net[df_net["group"] == 0][nets]
    df_tin  = df_net[df_net["group"] == 1][nets]

    ctrl_means = df_ctrl.mean()
    ctrl_se    = df_ctrl.sem()
    tin_means  = df_tin.mean()
    tin_se     = df_tin.sem()

    # Mann-Whitney U test per network, then FDR
    p_vals = []
    for net in nets:
        _, p = mannwhitneyu(df_ctrl[net].dropna(), df_tin[net].dropna(),
                            alternative="two-sided")
        p_vals.append(p)
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    # Save stats table
    df_stats = pd.DataFrame({
        "network": nets,
        "ctrl_mean": ctrl_means.values,
        "ctrl_se":   ctrl_se.values,
        "tin_mean":  tin_means.values,
        "tin_se":    tin_se.values,
        "p_mwu":     p_vals,
        "p_fdr":     p_fdr,
    })
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(tables_dir / "network_stats.csv", index=False)
    print(f"  Saved → {tables_dir / 'network_stats.csv'}")

    # Plot
    sns.set_style("white")
    x = np.arange(len(nets))
    width = 0.35
    jitter = 0.06

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    bars_ctrl = ax.bar(x - width / 2, ctrl_means, width, color=CTRL_COLOR,
                       alpha=0.85, edgecolor="white", label="Control", yerr=ctrl_se,
                       capsize=4, error_kw=dict(lw=1.5, capthick=1.5))
    bars_tin  = ax.bar(x + width / 2, tin_means, width, color=TIN_COLOR,
                       alpha=0.85, edgecolor="white", label="Tinnitus", yerr=tin_se,
                       capsize=4, error_kw=dict(lw=1.5, capthick=1.5))

    # Individual subject dots
    rng = np.random.default_rng(42)
    for i, net in enumerate(nets):
        ctrl_vals = df_ctrl[net].dropna().values
        tin_vals  = df_tin[net].dropna().values
        ax.scatter(rng.uniform(i - width / 2 - jitter, i - width / 2 + jitter, len(ctrl_vals)),
                   ctrl_vals, color=CTRL_COLOR, alpha=0.3, s=3, zorder=3)
        ax.scatter(rng.uniform(i + width / 2 - jitter, i + width / 2 + jitter, len(tin_vals)),
                   tin_vals, color=TIN_COLOR, alpha=0.3, s=3, zorder=3)

    # Chance line
    ax.axhline(0, color=CHANCE_COLOR, linestyle="--", lw=1.2, label="Chance (0)")

    # Significance annotations
    y_max = max(tin_means.values.max(), ctrl_means.values.max())
    y_range = y_max - min(df_net[nets].min().min(), 0)
    offset = y_range * 0.05

    for i, (net, p) in enumerate(zip(nets, p_fdr)):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        y_ann = max(ctrl_means[net] + ctrl_se[net], tin_means[net] + tin_se[net]) + offset
        ax.text(x[i], y_ann, stars, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nets, fontsize=11)
    ax.set_xlabel("Yeo-7 Network", fontsize=11)
    ax.set_ylabel("Mean deviation score", fontsize=11)
    ax.set_title("Network-level EEG deviation: Controls vs Tinnitus\n(FDR-corrected Mann-Whitney U)",
                 style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    sns.despine(ax=ax)

    fpath = save_dir / "network_bar.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    return df_stats


# ── Compute lateralization index ──────────────────────────────────────────────

def _compute_lateralization(df, roi_cols):
    """
    For each auditory ROI, compute LI = (LH - RH) / (LH + RH + 1e-8).
    Returns DataFrame with columns = LI per ROI, plus group and SITE.
    """
    li_data = {}
    for roi in AUDITORY_ROIS:
        lh_col = f"{roi}-lh"
        rh_col = f"{roi}-rh"
        if lh_col in df.columns and rh_col in df.columns:
            lh = df[lh_col].values
            rh = df[rh_col].values
            li_data[roi] = (lh - rh) / (np.abs(lh) + np.abs(rh) + 1e-8)
        else:
            li_data[roi] = np.zeros(len(df))
            print(f"  WARNING: {lh_col} or {rh_col} not found in features.")

    df_li = pd.DataFrame(li_data, index=df.index)
    df_li["group"] = df["group"].values
    df_li["SITE"]  = df["SITE"].values
    return df_li


# ── Plot B(a/b): Half-violin + strip of LI per auditory ROI ──────────────────

def plot_lateralization_violin(df_li, save_dir=FIGURES_DIR, tables_dir=TABLES_DIR):
    """2×2 grid of half-violin + strip plots of LI, split by group. FDR-corrected MWU."""
    auditory_rois = [r for r in AUDITORY_ROIS if r in df_li.columns]
    if not auditory_rois:
        print("  No auditory ROI LI columns found — skipping violin plot.")
        return

    # Mann-Whitney U + FDR across ROIs
    p_vals = []
    for roi in auditory_rois:
        ctrl_vals = df_li[df_li["group"] == 0][roi].dropna()
        tin_vals  = df_li[df_li["group"] == 1][roi].dropna()
        if len(ctrl_vals) >= 5 and len(tin_vals) >= 5:
            _, p = mannwhitneyu(ctrl_vals, tin_vals, alternative="two-sided")
        else:
            p = 1.0
        p_vals.append(p)
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    # Stats table
    rows_stats = []
    for roi, p_raw, p_adj in zip(auditory_rois, p_vals, p_fdr):
        ctrl_vals = df_li[df_li["group"] == 0][roi].dropna()
        tin_vals  = df_li[df_li["group"] == 1][roi].dropna()
        rows_stats.append({
            "roi": roi,
            "ctrl_mean_li": ctrl_vals.mean(),
            "tin_mean_li":  tin_vals.mean(),
            "p_mwu": p_raw,
            "p_fdr": p_adj,
        })
    df_stats = pd.DataFrame(rows_stats)
    tables_dir.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(tables_dir / "lateralization_stats.csv", index=False)
    print(f"  Saved → {tables_dir / 'lateralization_stats.csv'}")

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, (roi, p_adj) in enumerate(zip(auditory_rois, p_fdr)):
        ax = axes_flat[ax_idx]
        df_plot = df_li[["group", roi]].copy()
        df_plot["Group"] = df_plot["group"].map({0: "Control", 1: "Tinnitus"})

        sns.violinplot(data=df_plot, x="Group", y=roi,
                       palette={"Control": CTRL_COLOR, "Tinnitus": TIN_COLOR},
                       inner=None, alpha=0.5, ax=ax, cut=0)
        sns.stripplot(data=df_plot, x="Group", y=roi,
                      palette={"Control": CTRL_COLOR, "Tinnitus": TIN_COLOR},
                      alpha=0.5, size=3, jitter=True, ax=ax)

        ax.axhline(0, color=CHANCE_COLOR, linestyle="--", lw=1)

        stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
        y_top = df_plot[roi].quantile(0.99)
        ax.text(0.5, 1.02, f"FDR p={p_adj:.3f} {stars}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, style="italic")

        ax.set_title(roi, fontsize=11, style="italic")
        ax.set_xlabel("")
        ax.set_ylabel("Lateralization Index (LI)", fontsize=9)
        sns.despine(ax=ax)

    for ax in axes_flat[len(auditory_rois):]:
        ax.set_visible(False)

    fig.suptitle("Auditory ROI Lateralization Index by Group  (FDR-corrected)",
                 style="italic", fontsize=12)

    fpath = save_dir / "lateralization_violin.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Plot B(c): Scatter LH vs RH deviation per ROI ────────────────────────────

def plot_lateralization_scatter(df, roi_cols, save_dir=FIGURES_DIR):
    """LH deviation vs RH deviation per subject, colored by group, one panel per ROI."""
    auditory_rois = [r for r in AUDITORY_ROIS
                     if f"{r}-lh" in df.columns and f"{r}-rh" in df.columns]
    if not auditory_rois:
        print("  No auditory ROI columns for scatter — skipping.")
        return

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 9), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax_idx, roi in enumerate(auditory_rois):
        ax = axes_flat[ax_idx]
        lh_col = f"{roi}-lh"
        rh_col = f"{roi}-rh"

        df_ctrl = df[df["group"] == 0][[lh_col, rh_col]].dropna()
        df_tin  = df[df["group"] == 1][[lh_col, rh_col]].dropna()

        ax.scatter(df_ctrl[lh_col], df_ctrl[rh_col],
                   color=CTRL_COLOR, alpha=0.55, s=18, label="Control")
        ax.scatter(df_tin[lh_col], df_tin[rh_col],
                   color=TIN_COLOR, alpha=0.55, s=18, label="Tinnitus")

        # Identity line
        all_vals = pd.concat([df[[lh_col, rh_col]]]).values.flatten()
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], "--", color=CHANCE_COLOR, lw=1.2,
                label="Identity")

        ax.set_xlabel(f"{roi} LH deviation", fontsize=9)
        ax.set_ylabel(f"{roi} RH deviation", fontsize=9)
        ax.set_title(roi, fontsize=11, style="italic")
        ax.legend(frameon=False, fontsize="small")
        sns.despine(ax=ax)

    for ax in axes_flat[len(auditory_rois):]:
        ax.set_visible(False)

    fig.suptitle("LH vs RH deviation per auditory ROI", style="italic", fontsize=12)

    fpath = save_dir / "lateralization_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df = _load_data()
    roi_cols = _get_roi_columns(df)
    print(f"  ROI columns detected: {len(roi_cols)}")

    # ── Part A: Yeo-7 network analysis ────────────────────────────────────
    print("\n── A. Yeo-7 network aggregation ──")
    df_net = _compute_network_scores(df, roi_cols)
    print("  Plotting radar …")
    plot_network_radar(df_net, save_dir=FIGURES_DIR)
    print("  Plotting network bar chart …")
    plot_network_bar(df_net, save_dir=FIGURES_DIR, tables_dir=TABLES_DIR)

    # ── Part B: Lateralization index ──────────────────────────────────────
    print("\n── B. Lateralization index ──")
    df_li = _compute_lateralization(df, roi_cols)
    print("  Plotting lateralization violin …")
    plot_lateralization_violin(df_li, save_dir=FIGURES_DIR, tables_dir=TABLES_DIR)
    print("  Plotting lateralization scatter …")
    plot_lateralization_scatter(df, roi_cols, save_dir=FIGURES_DIR)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
