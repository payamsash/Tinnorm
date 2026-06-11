"""
Deviation overlap maps: percentage of subjects with extreme Z-score deviations.

For each (ROI, band) feature and each normative modality (power, regional PLI,
global PLI), computes the proportion of tinnitus subjects and controls with
Z > +1.96 (hyperdeviation) or Z < -1.96 (hypodeviation).

Produces:
  1. Network × band prevalence heatmaps (tinnitus vs control baseline)
     — separate for positive and negative deviations, per modality
  2. Top-25 ROI×band bar chart coloured by network
  3. Aggregated bar chart: % subjects with at least one extreme deviation per
     modality and direction (parallels Tabbal 2025 Fig. 3b / Xie 2025 Fig. 2)

Saves:
  results/figures/26_overlap_maps/overlap_heatmap_{modality}_{direction}.pdf
  results/figures/26_overlap_maps/top_rois_prevalence.pdf
  results/figures/26_overlap_maps/overlap_summary_bar.pdf
  results/tables/overlap_prevalence.csv

Run from src/:  python 26_overlap_maps.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
MODELS_DIR  = TINNORM_DIR / "models" / "preproc_2" / "source"
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "26_overlap_maps"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD   = 1.96   # |Z| threshold for "extreme" deviation
PREPROC     = 2
SPACE       = "source"

CTRL_COLOR  = "#4A90D9"
TIN_COLOR   = "#C99700"
NEG_COLOR   = "#E05C5C"   # hypodeviation
POS_COLOR   = "#2ECC71"   # hyperdeviation

FREQ_ORDER  = ["delta", "theta", "alpha_0", "alpha_1", "alpha_2",
               "beta_0", "beta_1", "beta_2", "beta_3", "gamma"]

MODALITIES  = {
    "power":        ("power",       ""),
    "regional_pli": ("regional_pli", "_pli"),
    "global_pli":   ("global_pli",  "_pli"),
}

# Yeo-7 network mapping (from script 19)
ROI_TO_NETWORK = {
    "bankssts": "VAN", "caudalanteriorcingulate": "DMN",
    "caudalmiddlefrontal": "DAN", "cuneus": "VIS", "entorhinal": "DMN",
    "frontalpole": "DMN", "fusiform": "VAN", "inferiorparietal": "VAN",
    "inferiortemporal": "VIS", "insula": "VAN", "isthmuscingulate": "DMN",
    "lateraloccipital": "VIS", "lateralorbitofrontal": "FPN",
    "lingual": "VIS", "medialorbitofrontal": "DMN", "middletemporal": "VAN",
    "paracentral": "SMN", "parahippocampal": "DMN", "parsopercularis": "FPN",
    "parsorbitalis": "FPN", "parstriangularis": "FPN", "pericalcarine": "VIS",
    "postcentral": "SMN", "posteriorcingulate": "DMN", "precentral": "SMN",
    "precuneus": "DMN", "rostralanteriorcingulate": "DMN",
    "rostralmiddlefrontal": "FPN", "superiorfrontal": "FPN",
    "superiorparietal": "DAN", "superiortemporal": "SMN",
    "supramarginal": "VAN", "temporalpole": "DMN",
    "transversetemporal": "SMN",
}

NET_ORDER   = ["FPN", "DAN", "DMN", "VAN", "SMN", "VIS"]
NET_COLORS  = {
    "FPN": "#9B59B6", "DAN": "#3498DB", "DMN": "#E74C3C",
    "VAN": "#1ABC9C", "SMN": "#F39C12", "VIS": "#95A5A6",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_feat(col: str, suffix: str = "") -> tuple:
    """Return (roi_base, hemi, band) from a feature column name."""
    s = col[: -len(suffix)] if suffix and col.endswith(suffix) else col
    parts = s.split("_")
    band  = parts[-1]
    roi_hemi = "_".join(parts[:-1])
    if "-" in roi_hemi:
        roi, hemi = roi_hemi.rsplit("-", 1)
    else:
        roi, hemi = roi_hemi, "lh"
    return roi, hemi, band


def load_z_scores(modality: str, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LOSO control Z-scores and full-model tinnitus Z-scores."""
    base = MODELS_DIR / modality

    df_ctrl = pd.read_csv(base / "loso" / "loso_controls_Z.csv")
    df_ctrl.rename(columns={"subject_id": "subject_id"}, inplace=True)

    df_tin  = pd.read_csv(base / "full_model" / "results" / "Z_test.csv")
    df_tin.rename(columns={"subject_ids": "subject_id", "observations": "obs"},
                  inplace=True)

    meta_cols = {"subject_id", "obs", "SITE", "age", "sex",
                 "PTA4_mean", "PTA4_HF", "group"}
    feat_cols = [c for c in df_ctrl.columns if c not in meta_cols]

    return df_ctrl[feat_cols], df_tin[feat_cols]


def compute_prevalence(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Return a DataFrame with columns [feature, pct_pos, pct_neg, n]."""
    n = len(df)
    rows = []
    for col in df.columns:
        vals = df[col].dropna()
        rows.append({
            "feature": col,
            "pct_pos": (vals > threshold).mean() * 100,
            "pct_neg": (vals < -threshold).mean() * 100,
            "pct_any": (vals.abs() > threshold).mean() * 100,
            "n":       len(vals),
        })
    return pd.DataFrame(rows)


def network_band_matrix(prev_df: pd.DataFrame, suffix: str,
                        col: str = "pct_pos") -> pd.DataFrame:
    """Aggregate prevalence to network × band matrix."""
    rows = []
    for _, row in prev_df.iterrows():
        roi, hemi, band = parse_feat(row["feature"], suffix)
        net = ROI_TO_NETWORK.get(roi, "UNK")
        rows.append({"network": net, "band": band, col: row[col]})

    df = pd.DataFrame(rows)
    df = df[df["network"].isin(NET_ORDER) & df["band"].isin(FREQ_ORDER)]
    mat = df.groupby(["network", "band"])[col].mean().unstack("band")
    mat = mat.reindex(index=NET_ORDER, columns=FREQ_ORDER, fill_value=0.0)
    return mat


# ── Plotting functions ────────────────────────────────────────────────────────

def plot_heatmap_pair(mat_tin, mat_ctrl, modality, direction, vmax_tin=None):
    """Side-by-side heatmaps: tinnitus vs control prevalence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    vmax_t = mat_tin.values.max() if vmax_tin is None else vmax_tin
    vmax_c = 12  # controls expected ~5%, allow up to 12%

    cmap_t = "YlOrRd" if direction == "positive" else "YlGnBu_r"
    cmap_c = "Blues"

    for ax, mat, title, vmax, cmap in [
        (axes[0], mat_tin,  f"Tinnitus — {direction} deviation (Z {'>' if direction=='positive' else '<'} {'+' if direction=='positive' else '-'}{THRESHOLD})", vmax_t, cmap_t),
        (axes[1], mat_ctrl, f"Controls — expected ≈5%", vmax_c, cmap_c),
    ]:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(FREQ_ORDER)))
        ax.set_xticklabels(FREQ_ORDER, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(NET_ORDER)))
        ax.set_yticklabels(NET_ORDER, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("% subjects", fontsize=9)

        for i in range(len(NET_ORDER)):
            for j in range(len(FREQ_ORDER)):
                val = mat.values[i, j]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.suptitle(f"Overlap map — {modality.replace('_', ' ')} | {direction} deviations",
                 fontsize=12, fontweight="bold")
    out = FIGURES_DIR / f"overlap_heatmap_{modality}_{direction}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_top_rois(all_records: list, top_n: int = 25):
    """Horizontal bar chart of top ROI×band features by tinnitus prevalence."""
    df = pd.DataFrame(all_records)
    df_tin = df[df["group"] == "tinnitus"].copy()
    df_tin["network"] = df_tin["roi"].map(ROI_TO_NETWORK).fillna("UNK")

    # Top by positive deviation
    top_pos = (df_tin.sort_values("pct_pos", ascending=False)
                     .drop_duplicates("feature").head(top_n))
    top_neg = (df_tin.sort_values("pct_neg", ascending=False)
                     .drop_duplicates("feature").head(top_n))

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)

    for ax, df_sub, col, title, color_prefix in [
        (axes[0], top_pos, "pct_pos", "Hyperdeviation (Z > +1.96)", "Reds"),
        (axes[1], top_neg, "pct_neg", "Hypodeviation (Z < −1.96)",  "Blues"),
    ]:
        df_sub = df_sub.sort_values(col, ascending=True)
        net_colors = [NET_COLORS.get(r, "#aaaaaa") for r in df_sub["network"]]
        bars = ax.barh(range(len(df_sub)), df_sub[col], color=net_colors,
                       edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_yticks(range(len(df_sub)))
        ax.set_yticklabels(df_sub["feature"], fontsize=8)
        ax.axvline(5, color="gray", linestyle="--", linewidth=1,
                   label="Control baseline (~5%)")
        ax.set_xlabel("% tinnitus subjects", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

        # Network colour legend
        handles = [plt.Rectangle((0,0), 1, 1, color=NET_COLORS[n], label=n)
                   for n in NET_ORDER if n in df_sub["network"].values]
        ax.legend(handles=handles, loc="lower right", fontsize=8,
                  title="Network", framealpha=0.85)

    fig.suptitle("Top ROI × band features by deviation prevalence (tinnitus)",
                 fontsize=12, fontweight="bold")
    out = FIGURES_DIR / "top_rois_prevalence.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_summary_bar(summary_rows: list):
    """% subjects with ≥1 extreme deviation per modality × direction × group."""
    df = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True,
                             constrained_layout=True)

    for ax, mod in zip(axes, ["power", "regional_pli", "global_pli"]):
        sub = df[df["modality"] == mod]
        x = np.arange(2)
        width = 0.25

        for i, (direction, col, col_color) in enumerate([
            ("Hyper (Z>+1.96)", "any_pos", POS_COLOR),
            ("Hypo  (Z<−1.96)", "any_neg", NEG_COLOR),
            ("Any |Z|>1.96",    "any_ext", "#555555"),
        ]):
            vals = [sub[sub["group"] == g][col].values[0] if len(sub[sub["group"]==g])>0 else 0
                    for g in ["controls", "tinnitus"]]
            ax.bar(x + i * width, vals, width, label=direction,
                   color=col_color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Controls", "Tinnitus"], fontsize=10)
        ax.set_title(mod.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylabel("% subjects with ≥1 extreme deviation", fontsize=9)
        ax.set_ylim(0, 105)
        ax.axhline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                   label="Expected baseline")
        if mod == "power":
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Prevalence of extreme normative deviations per modality\n"
                 "(|Z| > 1.96 threshold; controls: LOSO; tinnitus: full model)",
                 fontsize=11, fontweight="bold")
    out = FIGURES_DIR / "overlap_summary_bar.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_records  = []
    summary_rows = []
    all_prev_dfs = []

    for mod_key, (mod_dir, suffix) in MODALITIES.items():
        print(f"\n── {mod_key} ──")
        df_ctrl, df_tin = load_z_scores(mod_dir, suffix)

        prev_ctrl = compute_prevalence(df_ctrl)
        prev_tin  = compute_prevalence(df_tin)

        # Annotate with parsed feature info
        for df_prev, grp in [(prev_ctrl, "controls"), (prev_tin, "tinnitus")]:
            for _, row in df_prev.iterrows():
                roi, hemi, band = parse_feat(row["feature"], suffix)
                all_records.append({
                    "modality": mod_key, "feature": row["feature"],
                    "roi": roi, "hemi": hemi, "band": band,
                    "group": grp,
                    "pct_pos": row["pct_pos"],
                    "pct_neg": row["pct_neg"],
                    "pct_any": row["pct_any"],
                })
            all_prev_dfs.append((mod_key, grp, df_prev))

        # % subjects with ≥1 extreme deviation
        for grp_label, df_z in [("controls", df_ctrl), ("tinnitus", df_tin)]:
            any_pos = (df_z > THRESHOLD).any(axis=1).mean() * 100
            any_neg = (df_z < -THRESHOLD).any(axis=1).mean() * 100
            any_ext = (df_z.abs() > THRESHOLD).any(axis=1).mean() * 100
            summary_rows.append({
                "modality": mod_key, "group": grp_label,
                "any_pos": any_pos, "any_neg": any_neg, "any_ext": any_ext,
            })
            print(f"  {grp_label}: {any_pos:.1f}% hyper | {any_neg:.1f}% hypo | "
                  f"{any_ext:.1f}% any extreme")

        # Network × band matrices
        mat_tin_pos  = network_band_matrix(prev_tin,  suffix, "pct_pos")
        mat_ctrl_pos = network_band_matrix(prev_ctrl, suffix, "pct_pos")
        mat_tin_neg  = network_band_matrix(prev_tin,  suffix, "pct_neg")
        mat_ctrl_neg = network_band_matrix(prev_ctrl, suffix, "pct_neg")

        # Heatmap plots
        plot_heatmap_pair(mat_tin_pos, mat_ctrl_pos, mod_key, "positive")
        plot_heatmap_pair(mat_tin_neg, mat_ctrl_neg, mod_key, "negative")

    # Save prevalence table
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(TABLES_DIR / "overlap_prevalence.csv", index=False)
    print(f"\n  Saved: overlap_prevalence.csv ({len(df_all)} rows)")

    # Top ROI bar chart
    plot_top_rois(all_records)

    # Summary bar chart
    plot_summary_bar(summary_rows)

    print("\nDone — overlap maps saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
