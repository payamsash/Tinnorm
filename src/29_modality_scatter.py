"""
Three-modality Z-score scatter for top ROI × band combinations.

For each of the top SHAP features (by ROI × band), scatter Z_power (x-axis)
vs Z_regional_pli (y-axis).  Point size encodes |Z_global_pli|; points are
coloured by group (controls: blue, tinnitus: gold).

Three projection panels per feature (all three 2-D projections of the 3-D
deviation space):
  1. Z_power          vs  Z_regional_pli    (size ~ |Z_global_pli|)
  2. Z_power          vs  Z_global_pli      (size ~ |Z_regional_pli|)
  3. Z_regional_pli   vs  Z_global_pli      (size ~ |Z_power|)

Diagonal lines at ±1.96 mark the normative boundary per axis.

Reference: Xie 2025 NatComm Fig. 2 (bivariate deviation space).

Saves:
  results/figures/29_modality_scatter/scatter_{roi}_{band}.pdf   (per feature)
  results/figures/29_modality_scatter/scatter_grid_top3.pdf       (combined)

Run from src/:  python 29_modality_scatter.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
MODELS_DIR  = TINNORM_DIR / "models" / "preproc_2" / "source"
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "29_modality_scatter"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD   = 1.96
CTRL_COLOR  = "#4A90D9"
TIN_COLOR   = "#C99700"

# Top SHAP features by (roi, band): we visualise the 3-D deviation space
TOP_FEATURES = [
    # (roi,             hemi, band,      display_label)
    ("parsorbitalis",  "rh", "alpha_1", "Parsorbitalis RH × α₁"),
    ("parsorbitalis",  "rh", "theta",   "Parsorbitalis RH × θ"),
    ("parsorbitalis",  "rh", "alpha_0", "Parsorbitalis RH × α₀"),
]

META_COLS = {"subject_id", "subject_ids", "observations", "obs",
             "SITE", "age", "sex", "PTA4_mean", "PTA4_HF", "group"}


# ── Data loading ──────────────────────────────────────────────────────────────

def feat_col_name(roi: str, hemi: str, band: str, suffix: str = "") -> str:
    return f"{roi}-{hemi}_{band}{suffix}"


def load_z(modality: str) -> pd.DataFrame:
    """
    Load Z-scores for both groups from a given modality.
    Returns DataFrame with columns [subject_id, group, <feature cols>].
    """
    base = MODELS_DIR / modality

    df_ctrl = pd.read_csv(base / "loso" / "loso_controls_Z.csv")
    df_tin  = pd.read_csv(base / "full_model" / "results" / "Z_test.csv")

    # Normalise column names
    for df in [df_ctrl, df_tin]:
        if "subject_ids" in df.columns:
            df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        if "observations" in df.columns:
            df.drop(columns=["observations"], inplace=True)

    df_ctrl["group"] = "Controls"
    df_tin["group"]  = "Tinnitus"

    # Keep only subject_id + group + feature cols
    for df in [df_ctrl, df_tin]:
        drop = [c for c in df.columns if c in META_COLS - {"subject_id", "group"}]
        df.drop(columns=drop, inplace=True, errors="ignore")

    return pd.concat([df_ctrl, df_tin], ignore_index=True, sort=False)


# ── Plotting ──────────────────────────────────────────────────────────────────

def scatter_3modal(ax, x_vals, y_vals, z_vals, groups, title,
                   xlabel, ylabel, size_label):
    """
    Scatter of x vs y, size ∝ |z|, coloured by group.
    Normative ±1.96 grid lines drawn.
    """
    ctrl_mask = groups == "Controls"
    tin_mask  = groups == "Tinnitus"

    # Scale point sizes: min 15, max 120, proportional to |z|
    abs_z  = np.abs(z_vals)
    max_z  = np.nanpercentile(abs_z, 97) or 1.0
    sizes  = 15 + 105 * np.clip(abs_z / max_z, 0, 1)

    for mask, color, label in [
        (ctrl_mask, CTRL_COLOR, "Controls"),
        (tin_mask,  TIN_COLOR,  "Tinnitus"),
    ]:
        ax.scatter(x_vals[mask], y_vals[mask],
                   s=sizes[mask], color=color, alpha=0.45,
                   label=label, edgecolors="none", zorder=4)

    # Normative boundary lines
    lim_l, lim_h = -6, 6
    for v in [-THRESHOLD, THRESHOLD]:
        ax.axvline(v, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(v, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)

    # Shade "both extreme" quadrant
    for xm, ym in [
        ([THRESHOLD, lim_h], [THRESHOLD, lim_h]),   # top-right
        ([lim_l, -THRESHOLD], [lim_l, -THRESHOLD]),  # bottom-left
    ]:
        ax.fill_between(xm, ym[0], ym[1], alpha=0.05, color="red")

    ax.set_xlim(lim_l, lim_h)
    ax.set_ylim(lim_l, lim_h)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\n(size ∝ |{size_label}|)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, markerscale=1.2, loc="upper left", framealpha=0.85)

    # Annotate counts in hyperdeviation quadrant (both axes > 1.96)
    both_tin  = (x_vals[tin_mask]  > THRESHOLD) & (y_vals[tin_mask]  > THRESHOLD)
    both_ctrl = (x_vals[ctrl_mask] > THRESHOLD) & (y_vals[ctrl_mask] > THRESHOLD)
    ax.text(lim_h - 0.15, lim_h - 0.3,
            f"Tin: {both_tin.sum()}/{tin_mask.sum()} ({100*both_tin.mean():.0f}%)\n"
            f"Ctrl: {both_ctrl.sum()}/{ctrl_mask.sum()} ({100*both_ctrl.mean():.0f}%)",
            ha="right", va="top", fontsize=7.5, color="darkred",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))


def plot_feature_scatter(roi: str, hemi: str, band: str, display_label: str,
                         df_pow: pd.DataFrame, df_reg: pd.DataFrame,
                         df_glo: pd.DataFrame, save_path: Path):
    """Three 2-D projection panels for one ROI×band triple."""

    col_pow = feat_col_name(roi, hemi, band, suffix="")
    col_reg = feat_col_name(roi, hemi, band, suffix="_pli")
    col_glo = feat_col_name(roi, hemi, band, suffix="_pli")

    # Align subjects
    df = df_pow[["subject_id", "group", col_pow]].copy()
    df = df.rename(columns={col_pow: "Z_power"})

    if col_reg in df_reg.columns:
        df = df.merge(df_reg[["subject_id", col_reg]].rename(
            columns={col_reg: "Z_regional"}), on="subject_id", how="inner")
    else:
        df["Z_regional"] = np.nan

    if col_glo in df_glo.columns:
        df = df.merge(df_glo[["subject_id", col_glo]].rename(
            columns={col_glo: "Z_global"}), on="subject_id", how="inner")
    else:
        df["Z_global"] = np.nan

    # Replace inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Z_power", "Z_regional", "Z_global"], inplace=True)

    groups  = df["group"].values
    z_pow   = df["Z_power"].values.astype(float)
    z_reg   = df["Z_regional"].values.astype(float)
    z_glo   = df["Z_global"].values.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    scatter_3modal(axes[0], z_pow, z_reg, z_glo, groups,
                   f"{display_label}\nPower vs Regional PLI",
                   "Z-score power", "Z-score regional PLI", "Z_global")
    scatter_3modal(axes[1], z_pow, z_glo, z_reg, groups,
                   f"{display_label}\nPower vs Global PLI",
                   "Z-score power", "Z-score global PLI", "Z_regional")
    scatter_3modal(axes[2], z_reg, z_glo, z_pow, groups,
                   f"{display_label}\nRegional PLI vs Global PLI",
                   "Z-score regional PLI", "Z-score global PLI", "Z_power")

    # Per-axis MWU statistics for marginal comparison
    for ax, (x_ctrl, x_tin), (y_ctrl, y_tin), xlab, ylab in zip(
        axes,
        [(z_pow[groups=="Controls"], z_pow[groups=="Tinnitus"]),
         (z_pow[groups=="Controls"], z_pow[groups=="Tinnitus"]),
         (z_reg[groups=="Controls"], z_reg[groups=="Tinnitus"])],
        [(z_reg[groups=="Controls"], z_reg[groups=="Tinnitus"]),
         (z_glo[groups=="Controls"], z_glo[groups=="Tinnitus"]),
         (z_glo[groups=="Controls"], z_glo[groups=="Tinnitus"])],
        ["power", "power", "regional"],
        ["regional", "global", "global"],
    ):
        for vals_ctrl, vals_tin, axis_name, ax_letter in [
            (x_ctrl, x_tin, xlab, "x"),
            (y_ctrl, y_tin, ylab, "y"),
        ]:
            if len(vals_ctrl) > 1 and len(vals_tin) > 1:
                _, p = mannwhitneyu(vals_ctrl, vals_tin, alternative="two-sided")
                sig  = "***" if p < 0.001 else ("**" if p < 0.01 else
                                                 ("*" if p < 0.05 else "ns"))
                xloc = ax.get_xlim()[1]
                yloc = ax.get_ylim()[1]
                if ax_letter == "x":
                    ax.annotate(f"{axis_name}: {sig}", xy=(0.02, 0.02),
                                xycoords="axes fraction", fontsize=7.5,
                                color="navy", ha="left")
                else:
                    ax.annotate(f"{axis_name}: {sig}", xy=(0.02, 0.06),
                                xycoords="axes fraction", fontsize=7.5,
                                color="darkred", ha="left")

    fig.suptitle(f"Trimodal deviation space — {display_label}\n"
                 f"(dashed lines: ±1.96 normative threshold; "
                 f"shaded region: double-extreme deviation)",
                 fontsize=11, fontweight="bold")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading Z-score data…")
    df_pow = load_z("power")
    df_reg = load_z("regional_pli")
    df_glo = load_z("global_pli")

    # Individual scatter per feature
    for roi, hemi, band, display_label in TOP_FEATURES:
        safe = f"{roi}_{hemi}_{band}"
        out  = FIGURES_DIR / f"scatter_{safe}.pdf"
        plot_feature_scatter(roi, hemi, band, display_label,
                             df_pow, df_reg, df_glo, out)

    # ── Combined 3×3 grid (top-3 features × top-projection only) ─────────────
    # Show the most informative projection (power vs regional) for all 3 features
    print("\nBuilding combined top-3 grid…")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, (roi, hemi, band, display_label) in zip(axes, TOP_FEATURES):
        col_pow = feat_col_name(roi, hemi, band, suffix="")
        col_reg = feat_col_name(roi, hemi, band, suffix="_pli")
        col_glo = feat_col_name(roi, hemi, band, suffix="_pli")

        df = df_pow[["subject_id", "group", col_pow]].copy()
        df = df.rename(columns={col_pow: "Z_power"})

        for col_name, new_name, src in [
            (col_reg, "Z_regional", df_reg),
            (col_glo, "Z_global",   df_glo),
        ]:
            if col_name in src.columns:
                df = df.merge(src[["subject_id", col_name]].rename(
                    columns={col_name: new_name}), on="subject_id", how="inner")
            else:
                df[new_name] = np.nan

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["Z_power", "Z_regional", "Z_global"], inplace=True)

        scatter_3modal(ax,
                       df["Z_power"].values, df["Z_regional"].values,
                       df["Z_global"].values, df["group"].values,
                       display_label,
                       "Z-score power", "Z-score regional PLI", "Z_global")

    fig.suptitle("Trimodal deviation space — top 3 SHAP features\n"
                 "(Z_power vs Z_regional_PLI; point size ∝ |Z_global_PLI|)",
                 fontsize=11, fontweight="bold")

    out_grid = FIGURES_DIR / "scatter_grid_top3.pdf"
    fig.savefig(out_grid, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_grid.name}")

    print("\nDone — modality scatter plots saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
