"""
Normative aging trajectories for top SHAP features.

For each of the top-6 SHAP-ranked features, overlays raw harmonised feature
values (controls in blue, tinnitus in gold) on top of the normative percentile
bands (5th / 25th / 50th / 75th / 95th) derived from the BLR centile predictions.

The centile predictions (centiles_train.csv) give the model-predicted centile
boundary in the original feature space at each control subject's age.  Sorting
controls by age and LOWESS-smoothing these predictions gives a continuous
normative trajectory band.

Reference: Tabbal et al., npj Parkinson's Disease 2025, Fig. 2.

Saves:
  results/figures/28_normative_trajectories/trajectories_top6.pdf   (6-panel)
  results/figures/28_normative_trajectories/trajectory_{feature}.pdf (per-feature)

Run from src/:  python 28_normative_trajectories.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
MODELS_DIR  = TINNORM_DIR / "models" / "preproc_2" / "source"
HARM_DIR    = TINNORM_DIR / "harmonized" / "preproc_2" / "source"
MASTER_CSV  = Path("../material/master_clean.csv")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "28_normative_trajectories"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CTRL_COLOR   = "#4A90D9"
TIN_COLOR    = "#C99700"
BAND_COLORS  = {
    "p95_05": "#DDDDDD",  # outermost shaded region
    "p75_25": "#BBBBBB",  # inner shaded region
    "p50":    "#555555",  # normative median
}
LOWESS_FRAC  = 0.4   # smoothing fraction for normative centile lines

# Top 6 SHAP features (from shap_mean_abs.csv); first 3 are Mahalanobis features
# that correspond to modality-level aggregation; we use the underlying Z-scores:
# parsorbitalis-rh × alpha_1/theta/alpha_0 → regional_pli features
# The harmonized raw feature column and centile column use the same name.
TOP_FEATURES = [
    # (modality_dir,   harmonized_prefix, feature_col,                   display_label)
    ("regional_pli", "regional_pli", "parsorbitalis-rh_alpha_1_pli", "Parsorbitalis RH α₁ PLI"),
    ("regional_pli", "regional_pli", "parsorbitalis-rh_theta_pli",   "Parsorbitalis RH θ PLI"),
    ("regional_pli", "regional_pli", "parsorbitalis-rh_alpha_0_pli", "Parsorbitalis RH α₀ PLI"),
    ("power",        "power",        "parsorbitalis-rh_alpha_1",     "Parsorbitalis RH α₁ power"),
    ("power",        "power",        "parsorbitalis-rh_theta",       "Parsorbitalis RH θ power"),
    ("regional_pli", "regional_pli", "superiorfrontal-rh_theta_pli", "Superior frontal RH θ PLI"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_centiles(modality: str) -> pd.DataFrame:
    """
    Load centiles_train.csv.  Returns a DataFrame with columns:
        ['centile', 'subject_id', <feature_cols>]
    where centile is one of [0.05, 0.25, 0.50, 0.75, 0.95].
    """
    path = MODELS_DIR / modality / "full_model" / "results" / "centiles_train.csv"
    df   = pd.read_csv(path)

    # Normalise column names produced by PCNtoolkit
    if "observations" in df.columns:
        df.drop(columns=["observations"], inplace=True)
    if "subject_ids" in df.columns:
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    if "centile" not in df.columns:
        # centile column may be named differently — inspect
        cand = [c for c in df.columns if "centile" in c.lower()]
        if cand:
            df.rename(columns={cand[0]: "centile"}, inplace=True)

    return df


def load_harmonized(modality_prefix: str) -> pd.DataFrame:
    """Load harmonized feature CSV with all 544 subjects and metadata."""
    path = HARM_DIR / f"{modality_prefix}_hm.csv"
    return pd.read_csv(path)


def get_centile_bands(df_centiles: pd.DataFrame, df_master: pd.DataFrame,
                      feat_col: str) -> pd.DataFrame:
    """
    For a given feature, extract the 5 centile boundaries per subject and merge
    with age.  Returns a sorted DataFrame with columns [age, p05, p25, p50, p75, p95].
    """
    if feat_col not in df_centiles.columns:
        return None

    centile_map = {0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"}

    df_sub  = df_centiles[["subject_id", "centile", feat_col]].copy()
    df_wide = (df_sub
               .pivot_table(index="subject_id", columns="centile", values=feat_col)
               .rename(columns=centile_map)
               .reset_index())

    # Merge age
    if df_master is not None and "subject_id" in df_master.columns:
        df_wide = df_wide.merge(df_master[["subject_id", "age"]], on="subject_id", how="left")
    else:
        df_wide["age"] = np.nan

    df_wide.sort_values("age", inplace=True)
    return df_wide


def smooth_band(ages: np.ndarray, vals: np.ndarray, frac: float = LOWESS_FRAC
                ) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS-smooth a centile band.  Returns (sorted_ages, smoothed_vals)."""
    mask = ~np.isnan(ages) & ~np.isnan(vals)
    if mask.sum() < 5:
        return ages[mask], vals[mask]
    sm = lowess(vals[mask], ages[mask], frac=frac, return_sorted=True)
    return sm[:, 0], sm[:, 1]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(ax, feat_col: str, display_label: str,
                    centile_df: pd.DataFrame, harm_df: pd.DataFrame,
                    master: pd.DataFrame):
    """Draw one trajectory panel on the given Axes."""
    if centile_df is None or feat_col not in harm_df.columns:
        ax.text(0.5, 0.5, f"Data not found:\n{feat_col}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_title(display_label, fontsize=9, fontweight="bold")
        return

    ages  = centile_df["age"].values
    pcols = ["p05", "p25", "p50", "p75", "p95"]

    # Check which centile columns exist
    available = [c for c in pcols if c in centile_df.columns]

    if len(available) >= 2:
        # Outer band: p05–p95
        if "p05" in available and "p95" in available:
            a95, s95 = smooth_band(ages, centile_df["p95"].values)
            a05, s05 = smooth_band(ages, centile_df["p05"].values)
            x_lo = np.intersect1d(a95, a05)
            if len(x_lo) == 0:
                x_lo = a95
                v05 = np.interp(x_lo, a05, s05)
                v95 = np.interp(x_lo, a95, s95)
            else:
                v05 = np.interp(x_lo, a05, s05)
                v95 = np.interp(x_lo, a95, s95)
            ax.fill_between(x_lo, v05, v95, color="#DDDDDD", alpha=0.9,
                            label="5th–95th %ile")

        # Inner band: p25–p75
        if "p25" in available and "p75" in available:
            a75, s75 = smooth_band(ages, centile_df["p75"].values)
            a25, s25 = smooth_band(ages, centile_df["p25"].values)
            x_in  = a75
            v25_i = np.interp(x_in, a25, s25)
            v75_i = np.interp(x_in, a75, s75)
            ax.fill_between(x_in, v25_i, v75_i, color="#AAAAAA", alpha=0.75,
                            label="25th–75th %ile")

        # Median
        if "p50" in available:
            a50, s50 = smooth_band(ages, centile_df["p50"].values)
            ax.plot(a50, s50, color="#333333", linewidth=2, label="Normative median")

    # Raw feature values
    if "age" in harm_df.columns and "group" in harm_df.columns:
        for grp_val, grp_label, color in [(0, "Controls", CTRL_COLOR),
                                           (1, "Tinnitus", TIN_COLOR)]:
            df_g = harm_df[harm_df["group"] == grp_val]
            ax.scatter(df_g["age"], df_g[feat_col],
                       color=color, alpha=0.45, s=18, label=grp_label,
                       zorder=5, edgecolors="none")

    ax.set_xlabel("Age (years)", fontsize=9)
    ax.set_ylabel("Feature value", fontsize=9)
    ax.set_title(display_label, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)


def main():
    master = None
    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)
        if "subject_id" not in master.columns and "Subject" in master.columns:
            master.rename(columns={"Subject": "subject_id"}, inplace=True)

    # Pre-load data per unique modality
    centile_cache = {}
    harm_cache    = {}

    for mod_dir, harm_prefix, feat_col, _ in TOP_FEATURES:
        if mod_dir not in centile_cache:
            print(f"  Loading centiles for {mod_dir}…")
            centile_cache[mod_dir] = load_centiles(mod_dir)
        if harm_prefix not in harm_cache:
            print(f"  Loading harmonized for {harm_prefix}…")
            harm_cache[harm_prefix] = load_harmonized(harm_prefix)

    # ── 6-panel combined figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (mod_dir, harm_prefix, feat_col, display_label) in \
            zip(axes_flat, TOP_FEATURES):

        df_centiles = centile_cache[mod_dir]
        harm_df     = harm_cache[harm_prefix]

        band_df = get_centile_bands(df_centiles, master, feat_col)
        # Merge age into harmonized df
        if master is not None and "age" not in harm_df.columns:
            harm_df = harm_df.merge(master[["subject_id", "age"]], on="subject_id",
                                    how="left")

        plot_trajectory(ax, feat_col, display_label, band_df, harm_df, master)

    fig.suptitle("Normative aging trajectories — top SHAP features\n"
                 "(shaded bands: 5th–95th / 25th–75th percentiles; "
                 "median line = normative trajectory)",
                 fontsize=12, fontweight="bold")

    out_combined = FIGURES_DIR / "trajectories_top6.pdf"
    fig.savefig(out_combined, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_combined.name}")

    # ── Individual per-feature figures ────────────────────────────────────────
    for mod_dir, harm_prefix, feat_col, display_label in TOP_FEATURES:
        fig_s, ax_s = plt.subplots(figsize=(7, 5), constrained_layout=True)
        df_centiles = centile_cache[mod_dir]
        harm_df     = harm_cache[harm_prefix].copy()

        band_df = get_centile_bands(df_centiles, master, feat_col)
        if master is not None and "age" not in harm_df.columns:
            harm_df = harm_df.merge(master[["subject_id", "age"]], on="subject_id",
                                    how="left")

        plot_trajectory(ax_s, feat_col, display_label, band_df, harm_df, master)

        safe_name = feat_col.replace("/", "_").replace(" ", "_")
        out_s = FIGURES_DIR / f"trajectory_{safe_name}.pdf"
        fig_s.savefig(out_s, bbox_inches="tight", dpi=150)
        plt.close(fig_s)
        print(f"  Saved: {out_s.name}")

    print("\nDone — normative trajectories saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
