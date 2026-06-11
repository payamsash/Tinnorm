"""
Per-subject deviation counts and Extreme Deviation Index (EDI).

For each subject, counts the number of (ROI, band) features with |Z| > 1.96,
split by direction (hyperdeviation: Z>+1.96; hypodeviation: Z<-1.96).

EDI (Extreme Deviation Index) is computed as the mean Z-score over all features
where |Z| > 1.96 per subject (signed; captures net direction of deviations).
Reference: Tabbal et al., npj Parkinson's Disease, 2025.

Produces:
  1. Violin + strip plots — deviation count per group × direction × modality
  2. % subjects bar chart — fraction with ≥1 extreme deviation per group
  3. EDI distribution — tinnitus vs controls per modality
  4. Scatter — EDI vs THI/TFI for tinnitus subjects

Saves:
  results/figures/27_deviation_counts/deviation_counts_{modality}.pdf
  results/figures/27_deviation_counts/edi_vs_symptoms_{modality}.pdf
  results/figures/27_deviation_counts/summary_grid.pdf
  results/tables/deviation_counts.csv
  results/tables/edi_scores.csv

Run from src/:  python 27_deviation_counts.py
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
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
MODELS_DIR  = TINNORM_DIR / "models" / "preproc_2" / "source"
MASTER_CSV  = Path("../material/master_clean.csv")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "27_deviation_counts"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD  = 1.96
CTRL_COLOR = "#4A90D9"
TIN_COLOR  = "#C99700"
NEG_COLOR  = "#E05C5C"
POS_COLOR  = "#2ECC71"

MODALITIES = {
    "power":        ("power",        ""),
    "regional_pli": ("regional_pli", "_pli"),
    "global_pli":   ("global_pli",   "_pli"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_z_with_meta(modality: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_ctrl, df_tin) DataFrames with Z-scores + subject_id."""
    base = MODELS_DIR / modality

    df_ctrl = pd.read_csv(base / "loso" / "loso_controls_Z.csv")
    if "subject_ids" in df_ctrl.columns:
        df_ctrl.rename(columns={"subject_ids": "subject_id"}, inplace=True)

    df_tin = pd.read_csv(base / "full_model" / "results" / "Z_test.csv")
    if "subject_ids" in df_tin.columns:
        df_tin.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    if "observations" in df_tin.columns:
        df_tin.rename(columns={"observations": "obs"}, inplace=True)

    meta_base = {"subject_id", "obs", "SITE", "age", "sex",
                 "PTA4_mean", "PTA4_HF", "group"}
    feat_ctrl = [c for c in df_ctrl.columns if c not in meta_base]
    feat_tin  = [c for c in df_tin.columns  if c not in meta_base]

    sub_ctrl = df_ctrl["subject_id"].values if "subject_id" in df_ctrl.columns else None
    sub_tin  = df_tin["subject_id"].values  if "subject_id" in df_tin.columns  else None

    z_ctrl = df_ctrl[feat_ctrl].reset_index(drop=True)
    z_tin  = df_tin[feat_tin].reset_index(drop=True)

    if sub_ctrl is not None:
        z_ctrl.insert(0, "subject_id", sub_ctrl)
    if sub_tin is not None:
        z_tin.insert(0, "subject_id", sub_tin)

    return z_ctrl, z_tin


def compute_edi(df_z: pd.DataFrame, threshold: float = THRESHOLD) -> pd.Series:
    """EDI = mean Z of features where |Z| > threshold (per subject, signed)."""
    feat_cols = [c for c in df_z.columns if c != "subject_id"]
    vals = df_z[feat_cols].values.astype(float)
    edi = np.full(len(vals), np.nan)
    for i, row in enumerate(vals):
        mask = np.abs(row) > threshold
        edi[i] = row[mask].mean() if mask.any() else 0.0
    return pd.Series(edi, name="EDI")


def build_counts_df(z_ctrl: pd.DataFrame, z_tin: pd.DataFrame,
                    modality: str, master: pd.DataFrame) -> pd.DataFrame:
    """Build per-subject deviation count + EDI DataFrame."""
    feat_ctrl = [c for c in z_ctrl.columns if c != "subject_id"]
    feat_tin  = [c for c in z_tin.columns  if c != "subject_id"]

    rows = []
    for df_z, feat_cols, grp_label in [
        (z_ctrl, feat_ctrl, "Controls"),
        (z_tin,  feat_tin,  "Tinnitus"),
    ]:
        vals = df_z[feat_cols].values.astype(float)
        n_pos = (vals > THRESHOLD).sum(axis=1)
        n_neg = (vals < -THRESHOLD).sum(axis=1)
        n_any = n_pos + n_neg
        edi   = compute_edi(df_z[feat_cols])

        ids = df_z["subject_id"].values if "subject_id" in df_z.columns else np.arange(len(df_z))
        for i, sid in enumerate(ids):
            rows.append({
                "subject_id": sid, "group": grp_label, "modality": modality,
                "n_pos": n_pos[i], "n_neg": n_neg[i], "n_any": n_any[i],
                "EDI": float(edi.iloc[i]),
            })

    df = pd.DataFrame(rows)

    # Merge clinical scores for tinnitus
    if master is not None:
        df = df.merge(master[["subject_id", "THI", "TFI"]], on="subject_id", how="left")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def mwu_label(a, b):
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_counts_violin(df: pd.DataFrame, modality: str):
    """Violin + strip: n_pos, n_neg, n_any per group."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    for ax, (col, label, color) in zip(axes, [
        ("n_pos", "Hyperdeviation count\n(Z > +1.96)", POS_COLOR),
        ("n_neg", "Hypodeviation count\n(Z < −1.96)",  NEG_COLOR),
        ("n_any", "Total extreme count\n(|Z| > 1.96)",  "#555555"),
    ]):
        ctrl_vals = df[df["group"] == "Controls"][col].values
        tin_vals  = df[df["group"] == "Tinnitus"][col].values

        parts = ax.violinplot([ctrl_vals, tin_vals], positions=[0, 1],
                              showmedians=True, showextrema=False, widths=0.6)
        for pc, c in zip(parts["bodies"], [CTRL_COLOR, TIN_COLOR]):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Strip plot
        jitter = 0.08
        for x, vals, c in [(0, ctrl_vals, CTRL_COLOR), (1, tin_vals, TIN_COLOR)]:
            jit = np.random.default_rng(42).uniform(-jitter, jitter, len(vals))
            ax.scatter(x + jit, vals, color=c, alpha=0.35, s=10, zorder=3)

        # Significance
        sig = mwu_label(ctrl_vals, tin_vals)
        y_top = max(ctrl_vals.max(), tin_vals.max()) * 1.05
        ax.plot([0, 1], [y_top, y_top], color="black", linewidth=1)
        ax.text(0.5, y_top * 1.02, sig, ha="center", va="bottom", fontsize=12)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Controls", "Tinnitus"], fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label.split(chr(10))[0]}", fontsize=10, fontweight="bold")

        # Add median annotation
        for x, vals in [(0, ctrl_vals), (1, tin_vals)]:
            ax.text(x, np.median(vals), f"  {np.median(vals):.0f}",
                    va="center", fontsize=8, color="black")

    fig.suptitle(f"Per-subject deviation counts — {modality.replace('_', ' ')}",
                 fontsize=12, fontweight="bold")
    out = FIGURES_DIR / f"deviation_counts_{modality}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_edi_distribution(df: pd.DataFrame, modality: str):
    """EDI distribution: histogram + violin, then scatter vs THI/TFI."""
    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Panel 1 — EDI histogram per group
    ax1 = fig.add_subplot(gs[0])
    for grp, color in [("Controls", CTRL_COLOR), ("Tinnitus", TIN_COLOR)]:
        vals = df[df["group"] == grp]["EDI"].dropna()
        ax1.hist(vals, bins=25, color=color, alpha=0.6, density=True, label=grp,
                 edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("EDI (mean Z of extreme features)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("EDI distribution", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9)

    # Add medians
    for grp, color in [("Controls", CTRL_COLOR), ("Tinnitus", TIN_COLOR)]:
        m = df[df["group"] == grp]["EDI"].median()
        ax1.axvline(m, color=color, linewidth=2, linestyle=":")

    # Panel 2 & 3 — EDI vs THI and TFI (tinnitus only)
    df_tin = df[df["group"] == "Tinnitus"]

    for ax, symptom in zip([fig.add_subplot(gs[1]), fig.add_subplot(gs[2])],
                           ["THI", "TFI"]):
        sub = df_tin[["EDI", symptom]].dropna()
        if len(sub) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        rho, p = spearmanr(sub["EDI"], sub[symptom])
        ax.scatter(sub["EDI"], sub[symptom], color=TIN_COLOR, alpha=0.6,
                   s=25, edgecolors="none")

        # Trend line
        z = np.polyfit(sub["EDI"].values, sub[symptom].values, 1)
        x_line = np.linspace(sub["EDI"].min(), sub["EDI"].max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), color="black", linewidth=1.5,
                linestyle="--")

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.set_title(f"EDI vs {symptom}\nρ={rho:.2f} {sig}", fontsize=10,
                     fontweight="bold")
        ax.set_xlabel("EDI", fontsize=10)
        ax.set_ylabel(symptom, fontsize=10)

    fig.suptitle(f"Extreme Deviation Index (EDI) — {modality.replace('_', ' ')}",
                 fontsize=12, fontweight="bold")
    out = FIGURES_DIR / f"edi_vs_symptoms_{modality}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_summary_grid(all_df: pd.DataFrame):
    """Combined figure: median deviation counts per modality × group."""
    mods = ["power", "regional_pli", "global_pli"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    for ax, mod in zip(axes, mods):
        df_m = all_df[all_df["modality"] == mod]
        grps   = ["Controls", "Tinnitus"]
        colors = [CTRL_COLOR, TIN_COLOR]
        x      = np.arange(3)
        width  = 0.3

        for i, (grp, color) in enumerate(zip(grps, colors)):
            df_g = df_m[df_m["group"] == grp]
            means = [df_g["n_pos"].mean(), df_g["n_neg"].mean(), df_g["n_any"].mean()]
            sems  = [df_g["n_pos"].sem(), df_g["n_neg"].sem(), df_g["n_any"].sem()]
            bars  = ax.bar(x + i * width, means, width, label=grp, color=color,
                           alpha=0.85, edgecolor="white")
            ax.errorbar(x + i * width, means, yerr=sems, fmt="none",
                        color="black", linewidth=1.2, capsize=3)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(["Hyper", "Hypo", "Any"], fontsize=10)
        ax.set_title(mod.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean # deviated features", fontsize=9)
        ax.legend(fontsize=9)

    fig.suptitle("Mean per-subject deviated feature counts (mean ± SEM)",
                 fontsize=12, fontweight="bold")
    out = FIGURES_DIR / "summary_grid.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

import matplotlib.gridspec as gridspec

def main():
    master = None
    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)
        # Normalise subject_id column
        if "subject_id" not in master.columns and "Subject" in master.columns:
            master.rename(columns={"Subject": "subject_id"}, inplace=True)

    all_dfs = []

    for mod_key, (mod_dir, suffix) in MODALITIES.items():
        print(f"\n── {mod_key} ──")
        z_ctrl, z_tin = load_z_with_meta(mod_dir)

        df_counts = build_counts_df(z_ctrl, z_tin, mod_key, master)
        all_dfs.append(df_counts)

        for grp in ["Controls", "Tinnitus"]:
            sub = df_counts[df_counts["group"] == grp]
            print(f"  {grp}: median hyper={sub['n_pos'].median():.0f}, "
                  f"hypo={sub['n_neg'].median():.0f}, any={sub['n_any'].median():.0f}, "
                  f"EDI={sub['EDI'].mean():.3f}±{sub['EDI'].std():.3f}")

        plot_counts_violin(df_counts, mod_key)
        plot_edi_distribution(df_counts, mod_key)

    all_df = pd.concat(all_dfs, ignore_index=True)

    # Save tables
    counts_path = TABLES_DIR / "deviation_counts.csv"
    edi_path    = TABLES_DIR / "edi_scores.csv"

    all_df.to_csv(counts_path, index=False)
    all_df[["subject_id", "group", "modality", "EDI",
            "THI", "TFI"]].to_csv(edi_path, index=False)
    print(f"\n  Saved: {counts_path.name}")
    print(f"  Saved: {edi_path.name}")

    plot_summary_grid(all_df)

    print("\nDone — deviation counts saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
