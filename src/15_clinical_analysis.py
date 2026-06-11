"""
Clinical correlation analysis: EEG deviations vs THI and TFI.

For tinnitus subjects, computes Spearman correlations between each EEG
feature (Z-score deviation) and THI / TFI scores, applies FDR correction,
and produces scatter plots for surviving features.

Also includes:
  - UMAP embedding coloured by group and THI severity
  - THI vs TFI agreement scatter
  - Per-site THI/TFI profiles
  - Age-stratified deviation analysis

Saves:
  results/tables/corr_thi_features.csv
  results/tables/corr_tfi_features.csv
  results/figures/clinical_embedding.pdf
  results/figures/thi_tfi_agreement.pdf
  results/figures/corr_thi_top.pdf
  results/figures/corr_tfi_top.pdf
  results/figures/age_stratified_violin.pdf
  results/figures/age_stratified_scatter.pdf
  results/figures/age_thi_scatter.pdf

Run from src/:  python 15_clinical_analysis.py
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
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "15_clinical"
TABLES_DIR  = RESULTS_DIR / "tables"

MODE      = "diffusive_mm"
SPACE     = "source"
PREPROC   = 2
CONN_MODE = "pli"   # PLI is the top-performing connectivity measure

CTRL_COLOR = "#1f77b4"
TIN_COLOR  = "#C99700"
TFI_COLOR  = "#9B59B6"

# THI severity bins (Newman et al. 1996)
def _thi_sev(v):
    if pd.isna(v):  return "Unknown"
    if v <= 16:     return "Slight"
    if v <= 36:     return "Mild"
    if v <= 56:     return "Moderate"
    if v <= 76:     return "Severe"
    return "Catastrophic"

SEV_ORDER  = ["Slight", "Mild", "Moderate", "Severe", "Catastrophic"]
SEV_COLORS = {"Slight": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22",
              "Severe": "#e74c3c", "Catastrophic": "#8e44ad", "Unknown": "#aaaaaa"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_features():
    """Load EEG features + clinical scores (including age) for all subjects."""
    models_dir = TINNORM_DIR / "models"
    df_master  = pd.read_csv("../material/master_clean.csv")
    df_master["subject_id"] = df_master["subject_id"].astype(str)

    if "THI" not in df_master.columns and "thi_score" in df_master.columns:
        df_master.rename(columns={"thi_score": "THI"}, inplace=True)

    if MODE in ("diffusive_mm", "diffusive"):
        fname  = TINNORM_DIR / MODE / f"{SPACE}_preproc_{PREPROC}_{CONN_MODE}.csv"
        df = pd.read_csv(fname)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    else:
        dfs = []
        pfx = f"_{CONN_MODE}" if MODE in ("conn", "global", "regional") else ""
        for gname, gid in zip(["train", "test"], [0, 1]):
            f = (models_dir / f"preproc_{PREPROC}" / SPACE
                 / f"{MODE}{pfx}" / "full_model" / "results" / f"Z_{gname}.csv")
            dg = pd.read_csv(f)
            dg["group"] = gid
            dfs.append(dg)
        df = pd.concat(dfs, ignore_index=True)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)

    df["subject_id"] = df["subject_id"].astype(str)

    # Include "age" in the merge so it is available for age-stratified analysis
    clinical_cols = [c for c in ["site", "age", "THI", "TFI", "PTA4_mean", "PTA4_HF"]
                     if c in df_master.columns]
    df = df.merge(df_master[["subject_id"] + clinical_cols], on="subject_id", how="left")
    df.rename(columns={"site": "SITE"}, inplace=True)
    df.sort_values("subject_id", inplace=True)

    # Drop metadata; keep age, PTA4_mean, THI, TFI for downstream analyses
    df.drop(columns=["Unnamed: 0", "observations", "subject_id",
                     "sex", "thi_score"],
            inplace=True, errors="ignore")

    print(f"  {len(df)} subjects  |  groups: {dict(df['group'].value_counts())}")
    return df


# ── Feature-clinical correlation ──────────────────────────────────────────────

def _correlate_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Spearman r between each feature and `target`, FDR-corrected, tinnitus only."""
    df_tin = df[df["group"] == 1].dropna(subset=[target])
    skip   = {"group", "SITE", "THI", "TFI", "PTA4_mean", "PTA4_HF", "age"}
    feat_cols = [c for c in df_tin.columns if c not in skip]

    rows = []
    for col in feat_cols:
        vals = df_tin[[col, target]].dropna()
        if len(vals) < 10:
            continue
        r, p = spearmanr(vals[col], vals[target])
        rows.append({"feature": col, "r": r, "p": p})

    if not rows:
        print(f"  No features correlated with {target}.")
        return pd.DataFrame()

    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    res = res.sort_values("r", key=abs, ascending=False).reset_index(drop=True)
    n_sig = (res["p_fdr"] < 0.05).sum()
    print(f"  {target}: {n_sig} / {len(res)} features survive FDR < 0.05")
    return res


def plot_top_correlations(df: pd.DataFrame, res: pd.DataFrame, target: str,
                           top_n: int = 6, color: str = TIN_COLOR,
                           save_path: Path = None):
    survivors = res[res["p_fdr"] < 0.05].head(top_n)
    if survivors.empty:
        print(f"  Nothing to plot for {target} — no FDR survivors.")
        return

    n     = len(survivors)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    df_tin = df[df["group"] == 1].dropna(subset=[target])

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.8 * nrows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (_, row) in zip(axes_flat, survivors.iterrows()):
        col = row["feature"]
        sub = df_tin[[col, target]].dropna()
        ax.scatter(sub[target], sub[col], alpha=0.65, s=25, color=color)
        m, b = np.polyfit(sub[target].values, sub[col].values, 1)
        x_ = np.linspace(sub[target].min(), sub[target].max(), 100)
        ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
        ax.set_xlabel(target, fontsize=9)
        ax.set_ylabel("Feature value", fontsize=9)
        ax.set_title(f"{col}\nr={row['r']:.2f}, FDR-p={row['p_fdr']:.3f}",
                     style="italic", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Top {n} features correlated with {target}  (tinnitus, FDR<0.05)",
                 style="italic", fontsize=10)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {save_path}")


# ── Per-site clinical profiles ────────────────────────────────────────────────

def plot_site_clinical_profiles(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    """
    Per-site violin plots of THI and (if available) TFI scores for tinnitus subjects,
    and a stacked bar chart of group composition per site.
    Saves: site_thi_distribution.pdf, site_composition.pdf
    """
    if "SITE" not in df.columns:
        print("  No SITE column — skipping site profiles.")
        return

    sites = sorted(df["SITE"].dropna().unique())

    # ── (a) THI distribution per site ────────────────────────────────────
    score_cols = [c for c in ("THI", "TFI") if c in df.columns]
    if score_cols:
        df_tin = df[df["group"] == 1].copy()
        n_panels = len(score_cols)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(max(7, len(sites) * 0.9) * n_panels, 4.5),
                                 constrained_layout=True, squeeze=False)
        axes = axes[0]

        for ax, score in zip(axes, score_cols):
            data_plot = [df_tin[df_tin["SITE"] == s][score].dropna().values for s in sites]
            vp = ax.violinplot(
                [d for d in data_plot if len(d) >= 3],
                positions=[i for i, d in enumerate(data_plot) if len(d) >= 3],
                showmedians=True, showextrema=False, widths=0.6,
            )
            for body in vp["bodies"]:
                body.set_facecolor(TIN_COLOR)
                body.set_alpha(0.55)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.8)

            for i, d in enumerate(data_plot):
                if len(d) < 3:
                    continue
                rng = np.random.default_rng(i)
                jitter = rng.uniform(-0.08, 0.08, len(d))
                ax.scatter(i + jitter, d, color=TIN_COLOR, alpha=0.35, s=8, zorder=3)

            ax.set_xticks(range(len(sites)))
            ax.set_xticklabels(sites, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel(score, fontsize=11)
            ax.set_title(f"{score} distribution per site (tinnitus subjects)",
                         style="italic", fontsize=10)
            ax.spines[["right", "top"]].set_visible(False)

        fpath = save_dir / "site_thi_distribution.pdf"
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fpath}")

    # ── (b) Group composition per site ───────────────────────────────────
    comp = df.groupby(["SITE", "group"]).size().unstack(fill_value=0)
    comp.columns = [("Control" if c == 0 else "Tinnitus") for c in comp.columns]
    comp = comp.reindex(sites)
    total = comp.sum(axis=1)

    fig, ax = plt.subplots(figsize=(max(6, len(sites) * 1.0), 4.5), constrained_layout=True)
    bottom = np.zeros(len(sites))
    palette = {"Control": CTRL_COLOR, "Tinnitus": TIN_COLOR}
    for col in ("Control", "Tinnitus"):
        if col not in comp.columns:
            continue
        vals = comp[col].values.astype(float)
        bars = ax.bar(sites, vals, bottom=bottom, color=palette[col],
                      alpha=0.85, label=col, edgecolor="white", linewidth=0.5)
        bottom += vals

    for i, (site, tot) in enumerate(zip(sites, total)):
        ax.text(i, tot + 1, str(int(tot)), ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("N subjects", fontsize=11)
    ax.set_title("Group composition per site", style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "site_composition.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── PTA4 correlation ───────────────────────────────────────────────────────────

def plot_pta_thi_scatter(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    """
    Scatter of PTA4_mean vs THI for tinnitus subjects, coloured by age group,
    with Spearman r. Saves: pta_thi_scatter.pdf
    """
    if "PTA4_mean" not in df.columns or "THI" not in df.columns:
        print("  PTA4_mean or THI not available — skipping PTA scatter.")
        return

    df_tin = df[(df["group"] == 1)][["PTA4_mean", "THI"]].dropna()
    if len(df_tin) < 10:
        return

    r, p = spearmanr(df_tin["PTA4_mean"], df_tin["THI"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    sc = ax.scatter(df_tin["PTA4_mean"], df_tin["THI"],
                    alpha=0.6, s=30, c=df_tin["THI"], cmap="YlOrRd", zorder=3)
    plt.colorbar(sc, ax=ax, label="THI score", shrink=0.85)

    m, b = np.polyfit(df_tin["PTA4_mean"].values, df_tin["THI"].values, 1)
    x_ = np.linspace(df_tin["PTA4_mean"].min(), df_tin["PTA4_mean"].max(), 100)
    ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--", zorder=4)

    ax.set_xlabel("PTA4 mean (dB HL)", fontsize=11)
    ax.set_ylabel("THI score", fontsize=11)
    ax.set_title(f"Audiometric severity vs tinnitus distress\n"
                 f"Spearman r = {r:.3f},  p = {p:.3f}  (N={len(df_tin)})",
                 style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "pta_thi_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── THI vs TFI agreement ──────────────────────────────────────────────────────

def plot_thi_tfi_agreement(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if "THI" not in df.columns or "TFI" not in df.columns:
        print("  THI or TFI not available — skipping agreement plot.")
        return

    df_tin = df[(df["group"] == 1) & df[["THI", "TFI"]].notna().all(axis=1)]
    if len(df_tin) < 10:
        print("  Not enough data for THI-TFI agreement plot.")
        return

    r, p = spearmanr(df_tin["THI"], df_tin["TFI"])
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)

    sc = ax.scatter(df_tin["THI"], df_tin["TFI"],
                    alpha=0.65, s=35, c=df_tin["THI"], cmap="YlOrRd")
    plt.colorbar(sc, ax=ax, label="THI score", shrink=0.85)
    m, b = np.polyfit(df_tin["THI"].values, df_tin["TFI"].values, 1)
    x_ = np.linspace(df_tin["THI"].min(), df_tin["THI"].max(), 100)
    ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
    for cutoff in [16, 36, 56, 76]:
        ax.axvline(cutoff, color="gray", lw=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("THI score", fontsize=11)
    ax.set_ylabel("TFI score", fontsize=11)
    ax.set_title(f"THI vs TFI agreement  (tinnitus, N={len(df_tin)})\n"
                 f"Spearman r = {r:.3f},  p = {p:.3f}", style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "thi_tfi_agreement.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── UMAP embedding ────────────────────────────────────────────────────────────

def plot_embedding(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if not HAS_UMAP:
        print("  umap-learn not installed — skipping embedding.  pip install umap-learn")
        return

    skip      = {"group", "SITE", "THI", "TFI", "PTA4_mean", "PTA4_HF", "age"}
    feat_cols = [c for c in df.columns if c not in skip]
    X = df[feat_cols].fillna(0).values

    print("  Fitting UMAP …")
    Z = UMAP(n_components=2, random_state=42).fit_transform(X)

    df_p = pd.DataFrame({
        "u1":    Z[:, 0],
        "u2":    Z[:, 1],
        "group": df["group"].values,
        "SITE":  df["SITE"].values if "SITE" in df.columns else "?",
        "THI":   df["THI"].values if "THI" in df.columns else np.nan,
        "TFI":   df["TFI"].values if "TFI" in df.columns else np.nan,
    })
    df_p["sev"] = df_p["THI"].apply(_thi_sev)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Panel 1: group
    for g, gname, color in [(0, "Control", CTRL_COLOR), (1, "Tinnitus", TIN_COLOR)]:
        sub = df_p[df_p["group"] == g]
        axes[0].scatter(sub["u1"], sub["u2"], label=gname, color=color, alpha=0.6, s=25)
    axes[0].set_title("UMAP — group", style="italic")
    axes[0].legend(frameon=False, fontsize="small")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    axes[0].spines[["right", "top"]].set_visible(False)

    # Panel 2: THI severity
    ctrl = df_p[df_p["group"] == 0]
    axes[1].scatter(ctrl["u1"], ctrl["u2"], color=CTRL_COLOR, alpha=0.3, s=18, label="Control")
    for sev in SEV_ORDER:
        sub = df_p[(df_p["group"] == 1) & (df_p["sev"] == sev)]
        if len(sub):
            axes[1].scatter(sub["u1"], sub["u2"],
                            color=SEV_COLORS[sev], alpha=0.75, s=35, label=sev)
    axes[1].set_title("UMAP — THI severity (tinnitus)", style="italic")
    axes[1].legend(frameon=False, fontsize="small", title="THI severity")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    axes[1].spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "clinical_embedding.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Age-stratified analysis ───────────────────────────────────────────────────

def plot_age_stratified(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    """
    Age-stratified EEG deviation analysis.

    (a) Violin + strip of mean diffusion score by age_group × group
    (b) Scatter: age vs mean diffusion score, coloured by group, trend lines
    (c) If THI available: scatter age vs THI for tinnitus subjects

    Saves: age_stratified_violin.pdf, age_stratified_scatter.pdf, age_thi_scatter.pdf
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Ensure age is present; re-merge from master if needed
    if "age" not in df.columns:
        try:
            df_master = pd.read_csv("../material/master_clean.csv")
            df_master["subject_id"] = df_master["subject_id"].astype(str)
            if "age" in df_master.columns:
                # We can only re-merge if subject_id survived; since it was dropped,
                # fall back to positional merge by sort order (last resort)
                print("  WARNING: 'age' not in df and subject_id already dropped — "
                      "age-stratified analysis will be skipped.")
                return
        except Exception as e:
            print(f"  Could not load age from master: {e}  — skipping age analysis.")
            return

    if df["age"].isna().all():
        print("  'age' column is all NaN — skipping age-stratified analysis.")
        return

    # Build feature score columns (all non-metadata columns)
    skip = {"group", "SITE", "THI", "TFI", "PTA4_mean", "PTA4_HF", "age"}
    feat_cols = [c for c in df.columns if c not in skip]

    if not feat_cols:
        print("  No feature columns found for mean diffusion score — skipping.")
        return

    df_work = df.copy()
    df_work["mean_score"] = df_work[feat_cols].mean(axis=1)
    df_work["group_label"] = df_work["group"].map({0: "Control", 1: "Tinnitus"})

    # Age tertile bins
    bins   = [0, 45, 60, 120]
    labels = ["<45", "45–60", ">60"]
    df_work["age_group"] = pd.cut(df_work["age"], bins=bins, labels=labels, right=True)

    # ── (a) Violin + strip ────────────────────────────────────────────────
    ctrl_palette = {
        "<45":   "#9ecae1",   # light blue
        "45–60": "#3182bd",   # medium blue
        ">60":   "#084594",   # dark blue
    }
    tin_palette = {
        "<45":   "#fed976",   # light gold
        "45–60": "#C99700",   # medium gold
        ">60":   "#8b6400",   # dark gold
    }

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    age_group_order = labels
    x_positions = {"<45": 0, "45–60": 1, ">60": 2}
    group_offsets = {0: -0.2, 1: 0.2}

    for grp, grp_label, pal in [(0, "Control", ctrl_palette), (1, "Tinnitus", tin_palette)]:
        for ag in age_group_order:
            sub = df_work[(df_work["group"] == grp) & (df_work["age_group"] == ag)]["mean_score"].dropna()
            if len(sub) < 3:
                continue
            xpos = x_positions[ag] + group_offsets[grp]
            vp = ax.violinplot(sub.values, positions=[xpos], widths=0.35,
                               showmeans=True, showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(pal[ag])
                body.set_alpha(0.55)
            vp["cmeans"].set_color(pal[ag])
            vp["cmeans"].set_linewidth(2)

            rng = np.random.default_rng(hash(ag + grp_label) % (2**32))
            jitter = rng.uniform(-0.08, 0.08, len(sub))
            ax.scatter(xpos + jitter, sub.values, color=pal[ag], alpha=0.4, s=8)

    # Legend patches
    import matplotlib.patches as mpatches
    handles = []
    for ag in labels:
        handles.append(mpatches.Patch(color=ctrl_palette[ag], alpha=0.7, label=f"Ctrl {ag}"))
    for ag in labels:
        handles.append(mpatches.Patch(color=tin_palette[ag], alpha=0.7, label=f"Tin {ag}"))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(age_group_order, fontsize=11)
    ax.set_xlabel("Age group", fontsize=11)
    ax.set_ylabel("Mean deviation score", fontsize=11)
    ax.set_title("Mean EEG deviation by age group and group",
                 style="italic", fontsize=11)
    ax.legend(handles=handles, frameon=False, fontsize="x-small",
              ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    sns.despine(ax=ax)

    fpath = save_dir / "age_stratified_violin.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── (b) Scatter: age vs mean diffusion score ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    for grp, grp_label, color in [(0, "Control", CTRL_COLOR), (1, "Tinnitus", TIN_COLOR)]:
        sub = df_work[df_work["group"] == grp][["age", "mean_score"]].dropna()
        if len(sub) < 3:
            continue
        ax.scatter(sub["age"], sub["mean_score"], color=color, alpha=0.4, s=20,
                   label=grp_label)
        # Trend line
        m, b = np.polyfit(sub["age"].values, sub["mean_score"].values, 1)
        x_ = np.linspace(sub["age"].min(), sub["age"].max(), 100)
        ax.plot(x_, m * x_ + b, color=color, lw=2, linestyle="--")

        r, p = spearmanr(sub["age"].values, sub["mean_score"].values)
        ax.text(0.02, 0.97 - grp * 0.08,
                f"{grp_label}: r={r:.2f}, p={p:.3f}",
                transform=ax.transAxes, fontsize=9, color=color,
                va="top", style="italic")

    ax.set_xlabel("Age (years)", fontsize=11)
    ax.set_ylabel("Mean deviation score", fontsize=11)
    ax.set_title("Age vs mean EEG deviation", style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    sns.despine(ax=ax)

    fpath = save_dir / "age_stratified_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── (c) Age vs THI for tinnitus subjects ──────────────────────────────
    if "THI" in df_work.columns:
        df_tin = df_work[(df_work["group"] == 1)][["age", "THI", "age_group"]].dropna()
        if len(df_tin) >= 5:
            fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
            tertile_colors = {"<45": "#fed976", "45–60": "#C99700", ">60": "#8b6400"}
            for ag, color in tertile_colors.items():
                sub = df_tin[df_tin["age_group"] == ag]
                if len(sub) > 0:
                    ax.scatter(sub["age"], sub["THI"], color=color, alpha=0.65, s=30,
                               label=ag, edgecolors="white", linewidths=0.3)

            # Global trend line
            m, b = np.polyfit(df_tin["age"].values, df_tin["THI"].values, 1)
            x_ = np.linspace(df_tin["age"].min(), df_tin["age"].max(), 100)
            ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
            r, p = spearmanr(df_tin["age"].values, df_tin["THI"].values)

            ax.set_xlabel("Age (years)", fontsize=11)
            ax.set_ylabel("THI score", fontsize=11)
            ax.set_title(f"Age vs THI  (tinnitus only)\nSpearman r={r:.2f}, p={p:.3f}",
                         style="italic", fontsize=10)
            ax.legend(title="Age tertile", frameon=False, fontsize=9)
            sns.despine(ax=ax)

            fpath = save_dir / "age_thi_scatter.pdf"
            fig.savefig(fpath, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {fpath}")
        else:
            print("  Not enough tinnitus data with THI + age for age_thi_scatter.")
    else:
        print("  THI not available — skipping age vs THI scatter.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features …")
    df = _load_features()

    # ── THI ────────────────────────────────────────────────────────────────
    if "THI" in df.columns:
        print("\n── THI feature correlations ──")
        res_thi = _correlate_features(df, "THI")
        if not res_thi.empty:
            res_thi.to_csv(TABLES_DIR / "corr_thi_features.csv", index=False)
            plot_top_correlations(df, res_thi, "THI", top_n=6,
                                  color=TIN_COLOR,
                                  save_path=FIGURES_DIR / "corr_thi_top.pdf")

    # ── TFI ────────────────────────────────────────────────────────────────
    if "TFI" in df.columns:
        print("\n── TFI feature correlations ──")
        res_tfi = _correlate_features(df, "TFI")
        if not res_tfi.empty:
            res_tfi.to_csv(TABLES_DIR / "corr_tfi_features.csv", index=False)
            plot_top_correlations(df, res_tfi, "TFI", top_n=6,
                                  color=TFI_COLOR,
                                  save_path=FIGURES_DIR / "corr_tfi_top.pdf")

    # ── THI vs TFI agreement ───────────────────────────────────────────────
    print("\n── THI vs TFI agreement ──")
    plot_thi_tfi_agreement(df)

    # ── UMAP embedding ─────────────────────────────────────────────────────
    print("\n── UMAP embedding ──")
    plot_embedding(df)

    # ── Age-stratified analysis ────────────────────────────────────────────
    print("\n── Age-stratified analysis ──")
    plot_age_stratified(df, FIGURES_DIR)

    # ── Per-site clinical profiles ─────────────────────────────────────────
    print("\n── Per-site clinical profiles ──")
    plot_site_clinical_profiles(df, FIGURES_DIR)

    # ── PTA4 vs THI correlation ────────────────────────────────────────────
    print("\n── PTA4 vs THI scatter ──")
    plot_pta_thi_scatter(df, FIGURES_DIR)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
