"""
Statistical checks on demographics, clinical variables, and data quality.

Sections:
  1.  Demographics summary table by group × site
  2.  Group-level statistical tests (Mann-Whitney U, chi-square, Kruskal-Wallis)
  3.  VIF: multicollinearity among continuous demographics (with site as dummy)
  4.  Spearman correlation matrix (age, PTA4_mean, PTA4_HF, THI, TFI — tinnitus-only for THI/TFI)
  5.  Site × group balance (chi-square + visualization)
  6.  THI clinical severity distribution (tinnitus group, Newman et al. 1996 cutoffs)
  7.  PTA4_mean and PTA4_HF vs THI scatter (tinnitus group)
  8.  Normality checks (Shapiro-Wilk per variable per group)
  9.  Demographics violin plots by group
  10. TFI distribution and per-site profile (tinnitus group)
  11. THI vs TFI agreement scatter (tinnitus group)
  12. PTA4_mean vs PTA4_HF collinearity check (Pearson r, Spearman r, VIF + recommendation)

Saves:
  results/tables/demographics_summary.csv
  results/tables/group_comparisons.csv
  results/tables/vif_demographics.csv
  results/tables/correlations.csv
  results/tables/normality_tests.csv
  results/tables/pta_collinearity.csv
  results/figures/demographics_corr_heatmap.pdf
  results/figures/demographics_by_group.pdf
  results/figures/thi_distribution.pdf
  results/figures/tfi_distribution.pdf
  results/figures/thi_tfi_agreement_stats.pdf
  results/figures/pta_thi_scatter.pdf
  results/figures/pta_collinearity.pdf
  results/figures/site_group_balance.pdf

Run from src/:  python 21_stats_checks.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency, shapiro, spearmanr, kruskal

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "plots" / "diagnosis"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"

CTRL_COLOR = "#1f77b4"
TIN_COLOR  = "#C99700"

# THI clinical severity cutoffs (Newman et al. 1996)
THI_CUTOFFS = [(0, 16, "Slight"), (18, 36, "Mild"), (38, 56, "Moderate"),
               (58, 76, "Severe"), (78, 100, "Catastrophic")]
THI_SEV_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


# ── Load master ───────────────────────────────────────────────────────────────

def load_master():
    df = pd.read_csv("../material/master_clean.csv")
    df["subject_id"] = df["subject_id"].astype(str)
    df["group"] = df["group"].astype(int)

    # Normalize THI column name (support old 'thi_score' naming)
    if "THI" not in df.columns and "thi_score" in df.columns:
        df.rename(columns={"thi_score": "THI"}, inplace=True)

    continuous_cols = [c for c in ["age", "PTA4_mean", "PTA4_HF", "THI", "TFI"] if c in df.columns]
    return df, continuous_cols


# ── 1. Demographics summary ───────────────────────────────────────────────────

def demographics_summary(df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
    rows = []
    for site in sorted(df["site"].unique()):
        for group, gname in [(0, "Control"), (1, "Tinnitus")]:
            sub = df[(df["site"] == site) & (df["group"] == group)]
            row = {"site": site, "group": gname, "N": len(sub)}
            if "sex" in df.columns:
                n_male = int((sub["sex"] == 1).sum())
                row["N_male"] = n_male
                row["pct_male"] = round(100 * n_male / len(sub), 1) if len(sub) > 0 else np.nan
            for c in continuous_cols:
                if c not in df.columns:
                    continue
                if c in ("THI", "TFI") and group == 0:
                    row[f"{c}_mean"] = np.nan
                    row[f"{c}_sd"]   = np.nan
                else:
                    vals = sub[c].dropna()
                    row[f"{c}_mean"] = round(vals.mean(), 2) if len(vals) else np.nan
                    row[f"{c}_sd"]   = round(vals.std(),  2) if len(vals) else np.nan
            rows.append(row)

    # Append totals row
    for group, gname in [(0, "Control"), (1, "Tinnitus")]:
        sub = df[df["group"] == group]
        row = {"site": "TOTAL", "group": gname, "N": len(sub)}
        if "sex" in df.columns:
            n_male = int((sub["sex"] == 1).sum())
            row["N_male"] = n_male
            row["pct_male"] = round(100 * n_male / len(sub), 1)
        for c in continuous_cols:
            if c not in df.columns:
                continue
            if c in ("THI", "TFI") and group == 0:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_sd"]   = np.nan
            else:
                vals = sub[c].dropna()
                row[f"{c}_mean"] = round(vals.mean(), 2) if len(vals) else np.nan
                row[f"{c}_sd"]   = round(vals.std(),  2) if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ── 2. Group comparison tests ─────────────────────────────────────────────────

def group_comparisons(df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
    ctrl = df[df["group"] == 0]
    tin  = df[df["group"] == 1]
    rows = []

    for col in continuous_cols:
        if col not in df.columns or col in ("THI", "TFI"):
            continue  # THI/TFI are tinnitus-only — no between-group comparison
        a = ctrl[col].dropna().values
        b = tin[col].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "variable":    col,
            "test":        "Mann-Whitney U",
            "ctrl":        f"{np.mean(a):.2f} ± {np.std(a):.2f}",
            "tinnitus":    f"{np.mean(b):.2f} ± {np.std(b):.2f}",
            "statistic":   round(stat, 1),
            "p_value":     round(p, 4),
            "significant": p < 0.05,
        })

    # Sex: chi-square
    if "sex" in df.columns:
        ct = pd.crosstab(df["group"], df["sex"])
        chi2, p, _, _ = chi2_contingency(ct)
        rows.append({
            "variable":    "sex",
            "test":        "chi-square",
            "ctrl":        f"N={len(ctrl)}",
            "tinnitus":    f"N={len(tin)}",
            "statistic":   round(chi2, 3),
            "p_value":     round(p, 4),
            "significant": p < 0.05,
        })

    # Kruskal-Wallis across sites (site effect on each continuous variable)
    for col in [c for c in continuous_cols if c not in ("THI", "TFI") and c in df.columns]:
        groups_per_site = [df[df["site"] == s][col].dropna().values
                           for s in df["site"].unique()]
        groups_per_site = [g for g in groups_per_site if len(g) >= 3]
        if len(groups_per_site) < 2:
            continue
        stat, p = kruskal(*groups_per_site)
        rows.append({
            "variable":    f"{col}  (site effect, K-W)",
            "test":        "Kruskal-Wallis",
            "ctrl":        "",
            "tinnitus":    "",
            "statistic":   round(stat, 3),
            "p_value":     round(p, 4),
            "significant": p < 0.05,
        })

    return pd.DataFrame(rows)


# ── 3. VIF ────────────────────────────────────────────────────────────────────

def compute_vif(df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
    """VIF for continuous demographics with site encoded as one-hot dummies."""
    cols = [c for c in continuous_cols if c in df.columns and c not in ("THI", "TFI")]
    if not cols:
        return pd.DataFrame()

    df_vif = df[cols + ["site"]].dropna()
    site_dummies = pd.get_dummies(df_vif["site"], prefix="site", drop_first=True)
    design = pd.concat(
        [df_vif[cols].reset_index(drop=True), site_dummies.reset_index(drop=True)],
        axis=1,
    ).astype(float)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_vals = [variance_inflation_factor(design.values, i)
                    for i in range(design.shape[1])]
    except ImportError:
        # Fallback: manual R² computation
        X = design.values
        vif_vals = []
        for i in range(X.shape[1]):
            y_i   = X[:, i]
            X_rest = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
            coef, *_ = np.linalg.lstsq(X_rest, y_i, rcond=None)
            y_hat = X_rest @ coef
            ss_res = np.sum((y_i - y_hat) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif_vals.append(1.0 / (1.0 - r2) if r2 < 1.0 else np.inf)

    df_result = pd.DataFrame({
        "variable": design.columns,
        "VIF": np.round(vif_vals, 3),
    })
    df_result["flag"] = df_result["VIF"].apply(
        lambda v: "HIGH (>10)" if v > 10 else ("moderate (5–10)" if v > 5 else "ok (<5)")
    )
    return df_result


# ── 4. Spearman correlation matrix ────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame, continuous_cols: list,
                        save_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    cols = [c for c in continuous_cols if c in df.columns]
    if len(cols) < 2:
        print("  Not enough continuous columns for correlation matrix.")
        return pd.DataFrame()

    n = len(cols)
    rho  = np.full((n, n), np.nan)
    pmat = np.full((n, n), np.nan)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                rho[i, j] = 1.0
                pmat[i, j] = 0.0
                continue
            valid = df[[ci, cj]].dropna()
            if len(valid) < 5:
                continue
            r, p = spearmanr(valid[ci], valid[cj])
            rho[i, j]  = r
            pmat[i, j] = p

    df_rho = pd.DataFrame(rho, index=cols, columns=cols).round(3)

    # ── Heatmap ──────────────────────────────────────────────────────────
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(max(5, n + 1), max(4, n)), constrained_layout=True)
    sns.heatmap(df_rho, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, mask=mask, square=True,
                linewidths=0.5, annot_kws={"size": 10}, ax=ax)

    for i in range(n):
        for j in range(i):
            p = pmat[i, j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if stars:
                ax.text(j + 0.5, i + 0.82, stars, ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")

    ax.set_title("Spearman correlations — demographics & clinical\n"
                 "(* p<.05  ** p<.01  *** p<.001)", style="italic", fontsize=10)
    fpath = save_dir / "demographics_corr_heatmap.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    return df_rho


# ── 5. Site × group balance ───────────────────────────────────────────────────

def site_group_balance(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    ct = pd.crosstab(df["site"], df["group"])
    ct.columns = ["Control", "Tinnitus"]
    ct.index.name = "Site"
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"  Site × Group chi-square: χ²={chi2:.3f}, df={dof}, p={p:.4f}")

    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ct.plot(kind="bar", ax=ax1, color=[CTRL_COLOR, TIN_COLOR], edgecolor="white", width=0.6)
    ax1.set_title(f"N per site  (χ²={chi2:.2f}, p={p:.3f})", style="italic")
    ax1.set_xlabel("Site")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=35)
    ax1.legend(frameon=False)
    ax1.spines[["right", "top"]].set_visible(False)

    ct_norm.plot(kind="bar", stacked=True, ax=ax2,
                 color=[CTRL_COLOR, TIN_COLOR], edgecolor="white", width=0.6)
    ax2.set_title("Group proportion per site", style="italic")
    ax2.set_xlabel("Site")
    ax2.set_ylabel("Proportion")
    ax2.axhline(0.5, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=35)
    ax2.legend(frameon=False)
    ax2.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "site_group_balance.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    return ct


# ── 6. THI clinical severity distribution ────────────────────────────────────

def thi_distribution(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if "THI" not in df.columns:
        print("  THI column not found — skipping.")
        return

    df_tin = df[(df["group"] == 1) & df["THI"].notna()].copy()
    if len(df_tin) < 5:
        print("  Not enough THI data — skipping.")
        return

    def _thi_cat(v):
        for lo, hi, label in THI_CUTOFFS:
            if lo <= v <= hi:
                return label
        return "Catastrophic"

    df_tin["THI_cat"] = df_tin["THI"].apply(_thi_cat)
    cat_order = [c for _, _, c in THI_CUTOFFS if c in df_tin["THI_cat"].values]

    thi_vals = df_tin["THI"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax1.hist(thi_vals.values, bins=20, color=TIN_COLOR, alpha=0.75, edgecolor="white")
    for lo, _, label in THI_CUTOFFS[:-1]:
        ax1.axvline(lo, color="gray", lw=0.9, linestyle="--", alpha=0.65)
    ax1.set_xlabel("THI score")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"THI distribution  (tinnitus group, N={len(df_tin)})\n"
        f"Mean = {thi_vals.mean():.1f}  SD = {thi_vals.std():.1f}  "
        f"Range = [{thi_vals.min():.0f}, {thi_vals.max():.0f}]",
        style="italic", fontsize=9,
    )
    ax1.spines[["right", "top"]].set_visible(False)

    ct_cat = pd.crosstab(df_tin["site"], df_tin["THI_cat"])[
        [c for c in cat_order if c in df_tin["THI_cat"].values]
    ]
    ct_cat.plot(kind="bar", stacked=True, ax=ax2,
                color=THI_SEV_COLORS[:len(ct_cat.columns)], edgecolor="white", width=0.6)
    ax2.set_title("THI severity per site", style="italic")
    ax2.set_xlabel("Site")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=35)
    ax2.legend(title="Severity", frameon=False, fontsize="small")
    ax2.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "thi_distribution.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    print("\n  THI severity breakdown:")
    counts = df_tin["THI_cat"].value_counts()
    for _, _, cat in THI_CUTOFFS:
        n = counts.get(cat, 0)
        print(f"    {cat:<20}: {n:>3}  ({100*n/len(df_tin):.1f}%)")


# ── 7. PTA vs THI scatter ─────────────────────────────────────────────────────

def pta_thi_scatter(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if "THI" not in df.columns:
        print("  THI not in master_clean.csv — skipping PTA-THI scatter.")
        return

    pta_cols = [c for c in ["PTA4_mean", "PTA4_HF"] if c in df.columns]
    if not pta_cols:
        print("  No PTA columns — skipping PTA-THI scatter.")
        return

    n = len(pta_cols)
    colors = [TIN_COLOR, "#9B59B6"]
    fig, axs = plt.subplots(1, n, figsize=(5.5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, col, color in zip(axs, pta_cols, colors):
        sub = df[(df["group"] == 1) & df[[col, "THI"]].notna().all(axis=1)]
        if len(sub) < 5:
            ax.set_visible(False)
            continue
        r, p = spearmanr(sub[col], sub["THI"])
        ax.scatter(sub[col], sub["THI"], alpha=0.65, s=30, color=color)
        m, b = np.polyfit(sub[col].values, sub["THI"].values, 1)
        x_ = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
        ax.set_xlabel(f"{col} (dB HL)", fontsize=11)
        ax.set_ylabel("THI score", fontsize=11)
        ax.set_title(f"{col} vs THI  (tinnitus group)\n"
                     f"Spearman r = {r:.3f},  p = {p:.3f}", style="italic", fontsize=10)
        ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "pta_thi_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── 10. TFI distribution ──────────────────────────────────────────────────────

def tfi_distribution(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if "TFI" not in df.columns:
        print("  TFI column not found — skipping.")
        return

    df_tin = df[(df["group"] == 1) & df["TFI"].notna()].copy()
    if len(df_tin) < 5:
        print("  Not enough TFI data — skipping.")
        return

    tfi_vals = df_tin["TFI"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax1.hist(tfi_vals.values, bins=20, color="#9B59B6", alpha=0.75, edgecolor="white")
    ax1.set_xlabel("TFI score")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"TFI distribution  (tinnitus group, N={len(df_tin)})\n"
        f"Mean = {tfi_vals.mean():.1f}  SD = {tfi_vals.std():.1f}  "
        f"Range = [{tfi_vals.min():.0f}, {tfi_vals.max():.0f}]",
        style="italic", fontsize=9,
    )
    ax1.spines[["right", "top"]].set_visible(False)

    # Per-site box plot
    site_order = sorted(df_tin["site"].unique())
    site_data  = [df_tin[df_tin["site"] == s]["TFI"].dropna().values for s in site_order]
    ax2.boxplot(site_data, labels=site_order, patch_artist=True,
                boxprops=dict(facecolor="#9B59B6", alpha=0.65),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color="gray"),
                capprops=dict(color="gray"),
                flierprops=dict(marker="o", color="gray", alpha=0.4, markersize=4))
    ax2.set_title("TFI per site (tinnitus group)", style="italic")
    ax2.set_xlabel("Site")
    ax2.set_ylabel("TFI score")
    ax2.tick_params(axis="x", rotation=35)
    ax2.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "tfi_distribution.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    print(f"\n  TFI summary (tinnitus, N={len(df_tin)}):")
    print(f"    Mean={tfi_vals.mean():.1f}  Median={tfi_vals.median():.1f}  "
          f"SD={tfi_vals.std():.1f}  Range=[{tfi_vals.min():.0f},{tfi_vals.max():.0f}]")
    n_missing = (df["group"] == 1).sum() - len(df_tin)
    if n_missing:
        print(f"    Missing TFI in tinnitus group: {n_missing}")


# ── 11. THI vs TFI agreement ──────────────────────────────────────────────────

def thi_tfi_agreement(df: pd.DataFrame, save_dir: Path = FIGURES_DIR):
    if "THI" not in df.columns or "TFI" not in df.columns:
        print("  THI or TFI not available — skipping agreement plot.")
        return

    df_tin = df[(df["group"] == 1) & df[["THI", "TFI"]].notna().all(axis=1)]
    if len(df_tin) < 10:
        print("  Not enough paired THI/TFI data — skipping.")
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
                 f"Spearman r = {r:.3f},  p = {p:.4f}", style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "thi_tfi_agreement_stats.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")
    print(f"  THI-TFI Spearman r = {r:.3f},  p = {p:.4f}")


# ── 8. Normality checks ───────────────────────────────────────────────────────

def normality_checks(df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
    rows = []
    for col in continuous_cols:
        if col not in df.columns:
            continue
        for group, gname in [(0, "Control"), (1, "Tinnitus")]:
            vals = df[(df["group"] == group)][col].dropna().values
            if len(vals) < 8:
                continue
            # Shapiro-Wilk is most powerful for small-ish samples (<2000)
            sample = vals if len(vals) <= 2000 else vals[:2000]
            stat, p = shapiro(sample)
            rows.append({
                "variable":    col,
                "group":       gname,
                "N":           len(vals),
                "W":           round(stat, 4),
                "p_value":     round(p, 4),
                "normal":      p >= 0.05,
            })
    return pd.DataFrame(rows)


# ── 9. Demographics violin by group ──────────────────────────────────────────

def demographics_by_group(df: pd.DataFrame, continuous_cols: list,
                           save_dir: Path = FIGURES_DIR):
    plot_cols = [c for c in continuous_cols if c in df.columns and c not in ("THI", "TFI")]
    if not plot_cols:
        return

    n = len(plot_cols)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4.5), constrained_layout=True)
    if n == 1:
        axs = [axs]

    for ax, col in zip(axs, plot_cols):
        sns.violinplot(data=df, x="group", y=col,
                       palette={"0": CTRL_COLOR, "1": TIN_COLOR},
                       order=[0, 1],
                       inner="quartile", alpha=0.8, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Tinnitus"])
        ax.set_xlabel("")
        ax.set_ylabel(col)
        ax.set_title(col, style="italic")
        ax.spines[["right", "top"]].set_visible(False)

    fig.suptitle("Continuous demographics by group", style="italic")
    fpath = save_dir / "demographics_by_group.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── 12. PTA4_mean vs PTA4_HF collinearity ────────────────────────────────────

def pta_collinearity_check(df: pd.DataFrame,
                           save_dir: Path = FIGURES_DIR) -> pd.DataFrame:
    """
    Check collinearity between PTA4_mean and PTA4_HF.

    Reports Pearson r, Spearman r, and VIF (computed with site dummies to
    mimic the harmonization context). Issues a recommendation on whether both
    can safely be used as ComBat covariates.

    VIF > 5 → moderate collinearity; > 10 → severe collinearity.
    Rule of thumb: if |Pearson r| > 0.85 or VIF > 5, prefer using PTA4_HF
    only (more specific to the tinnitus-related high-frequency hearing loss).
    """
    pair = [c for c in ["PTA4_mean", "PTA4_HF"] if c in df.columns]
    if len(pair) < 2:
        print("  PTA4_mean or PTA4_HF not available — skipping collinearity check.")
        return pd.DataFrame()

    from scipy.stats import pearsonr
    # Include age alongside site dummies so this VIF mirrors the compute_vif() design (section 3).
    extra_cols = [c for c in ["age"] if c in df.columns]
    valid = df[pair + extra_cols + ["site"]].dropna()

    r_pearson,  p_pearson  = pearsonr(valid["PTA4_mean"], valid["PTA4_HF"])
    r_spearman, p_spearman = spearmanr(valid["PTA4_mean"], valid["PTA4_HF"])

    # VIF including age and site dummies (mirrors the harmonization/normative-model design)
    site_dummies = pd.get_dummies(valid["site"], prefix="site", drop_first=True)
    design = pd.concat(
        [valid[pair + extra_cols].reset_index(drop=True),
         site_dummies.reset_index(drop=True)],
        axis=1,
    ).astype(float)
    X = design.values
    vif_vals = {}
    for i, col in enumerate(pair):
        y_i   = X[:, i]
        X_rest = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
        coef, *_ = np.linalg.lstsq(X_rest, y_i, rcond=None)
        y_hat = X_rest @ coef
        ss_res = np.sum((y_i - y_hat) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif_vals[col] = round(1.0 / (1.0 - r2) if r2 < 1.0 else np.inf, 2)

    high_collinear = abs(r_pearson) > 0.85 or any(v > 5.0 for v in vif_vals.values())
    rec = (
        "HIGH COLLINEARITY — drop PTA4_mean, use PTA4_HF only in ComBat covariates.\n"
        "    (PTA4_HF is more specific to tinnitus-associated high-frequency hearing loss.)"
        if high_collinear else
        "Collinearity acceptable — both PTA4_mean and PTA4_HF can be used as covariates."
    )

    print(f"  Pearson  r = {r_pearson:.3f}  (p = {p_pearson:.4f})")
    print(f"  Spearman r = {r_spearman:.3f}  (p = {p_spearman:.4f})")
    print(f"  VIF  →  PTA4_mean = {vif_vals.get('PTA4_mean','?')}   "
          f"PTA4_HF = {vif_vals.get('PTA4_HF','?')}")
    print(f"  → {rec}")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    sc = ax.scatter(valid["PTA4_mean"], valid["PTA4_HF"],
                    alpha=0.45, s=22, color="#475569")
    m, b = np.polyfit(valid["PTA4_mean"].values, valid["PTA4_HF"].values, 1)
    x_ = np.linspace(valid["PTA4_mean"].min(), valid["PTA4_mean"].max(), 200)
    ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")

    flag_color = "#dc2626" if high_collinear else "#16a34a"
    ax.set_xlabel("PTA4_mean (dB HL)", fontsize=11)
    ax.set_ylabel("PTA4_HF (dB HL)", fontsize=11)
    ax.set_title(
        f"PTA4_mean vs PTA4_HF — collinearity check  (N = {len(valid)})\n"
        f"Pearson r = {r_pearson:.3f}   Spearman r = {r_spearman:.3f}\n"
        f"VIF (design: age + site): PTA4_mean = {vif_vals.get('PTA4_mean','?')}   "
        f"PTA4_HF = {vif_vals.get('PTA4_HF','?')}",
        style="italic", fontsize=9,
    )
    # Recommendation box
    ax.text(0.02, 0.02, rec, transform=ax.transAxes,
            fontsize=7.5, color=flag_color, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=flag_color, alpha=0.85))
    ax.spines[["right", "top"]].set_visible(False)

    fpath = save_dir / "pta_collinearity.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── Save table ────────────────────────────────────────────────────────────
    df_result = pd.DataFrame({
        "metric": ["Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p",
                   "VIF_PTA4_mean", "VIF_PTA4_HF", "high_collinear"],
        "value":  [round(r_pearson, 3), round(p_pearson, 4),
                   round(r_spearman, 3), round(p_spearman, 4),
                   vif_vals.get("PTA4_mean", np.nan),
                   vif_vals.get("PTA4_HF",   np.nan),
                   int(high_collinear)],
    })
    df_result.to_csv(TABLES_DIR / "pta_collinearity.csv", index=False)
    return df_result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading master_clean.csv …")
    df, continuous_cols = load_master()
    print(f"  {len(df)} subjects | columns: {list(df.columns)}")
    print(f"  Groups: {dict(df['group'].value_counts().sort_index())}")
    print(f"  Sites:  {sorted(df['site'].unique())}")
    print(f"  Continuous demographics: {continuous_cols}")

    # ── 1. Demographics summary ────────────────────────────────────────────
    print("\n── 1. Demographics summary ──")
    df_sum = demographics_summary(df, continuous_cols)
    df_sum.to_csv(TABLES_DIR / "demographics_summary.csv", index=False)
    print(df_sum.to_string(index=False))

    # ── 2. Group comparison tests ──────────────────────────────────────────
    print("\n── 2. Group comparison tests ──")
    df_comp = group_comparisons(df, continuous_cols)
    df_comp.to_csv(TABLES_DIR / "group_comparisons.csv", index=False)
    print(df_comp.to_string(index=False))

    # ── 3. VIF (multicollinearity) ─────────────────────────────────────────
    print("\n── 3. VIF (multicollinearity) ──")
    df_vif = compute_vif(df, continuous_cols)
    if not df_vif.empty:
        df_vif.to_csv(TABLES_DIR / "vif_demographics.csv", index=False)
        print(df_vif.to_string(index=False))

    # ── 4. Spearman correlation matrix ─────────────────────────────────────
    print("\n── 4. Spearman correlations ──")
    df_corr = correlation_matrix(df, continuous_cols)
    if not df_corr.empty:
        df_corr.to_csv(TABLES_DIR / "correlations.csv")

    # ── 5. Site × group balance ────────────────────────────────────────────
    print("\n── 5. Site × group balance ──")
    ct = site_group_balance(df)
    print(ct.to_string())

    # ── 6. THI distribution ────────────────────────────────────────────────
    print("\n── 6. THI clinical severity ──")
    thi_distribution(df)

    # ── 7. PTA vs THI ──────────────────────────────────────────────────────
    print("\n── 7. PTA vs THI scatter ──")
    pta_thi_scatter(df)

    # ── 8. Normality tests ─────────────────────────────────────────────────
    print("\n── 8. Normality tests (Shapiro-Wilk) ──")
    df_norm = normality_checks(df, continuous_cols)
    if not df_norm.empty:
        df_norm.to_csv(TABLES_DIR / "normality_tests.csv", index=False)
        print(df_norm.to_string(index=False))
        non_normal = df_norm[~df_norm["normal"]]
        if len(non_normal):
            print(f"\n  Non-normal distributions (p<.05): "
                  f"{list(zip(non_normal['variable'], non_normal['group']))}")
        else:
            print("\n  All distributions pass normality (p≥.05).")

    # ── 9. Demographics violins ────────────────────────────────────────────
    print("\n── 9. Demographics by group ──")
    demographics_by_group(df, continuous_cols)

    # ── 10. TFI distribution ───────────────────────────────────────────────
    print("\n── 10. TFI distribution ──")
    tfi_distribution(df)

    # ── 11. THI vs TFI agreement ───────────────────────────────────────────
    print("\n── 11. THI vs TFI agreement ──")
    thi_tfi_agreement(df)

    # ── 12. PTA collinearity check ─────────────────────────────────────────────
    print("\n── 12. PTA4_mean vs PTA4_HF collinearity ──")
    pta_collinearity_check(df)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
