"""
23_deep_feature_analysis.py — Deep analysis of the best PLI-band diffusive features.

Three clinically-oriented analyses on diffusive_mm_bands (PLI, preproc=2, source space):

  A. Network × Band deviation heatmap
       Aggregate 680 ROI×band features into Yeo-7 networks × frequency bands.
       Three-panel figure: control mean | tinnitus mean | difference with FDR stars.
       → figures/23_deep_feature/network_band_heatmap.pdf
       → tables/network_band_stats.csv

  B. Tinnitus subgroup discovery
       UMAP 2-D embedding of tinnitus subjects; k-means clustering (k chosen by
       silhouette score); clinical profiles per cluster (THI, PTA4_HF, site).
       → figures/23_deep_feature/subgroup_umap.pdf
       → figures/23_deep_feature/subgroup_profiles.pdf
       → tables/subgroup_stats.csv

  C. Band-specific lateralization — SHAP-guided ROI selection
       Loads mean |SHAP| from script 16 output (tables/shap_mean_abs.csv) and ranks
       all 34 ROIs by total SHAP importance. The top-N ROIs (default 15) are used for
       the lateralization index analysis instead of the four hard-coded auditory ROIs.
       Falls back to auditory ROIs if the SHAP file is absent.
       LI = (LH−RH) / (|LH|+|RH|+ε) per ROI × band, group-difference heatmap +
       violin plots for the most significant cells.
       → figures/23_deep_feature/band_lateralization_heatmap.pdf
       → figures/23_deep_feature/band_lateralization_violin.pdf
       → tables/band_lateralization_stats.csv

Run from src/:  python 23_deep_feature_analysis.py
"""

import re
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed — PCA used instead.  pip install umap-learn")


# ── Paths ──────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "23_deep_feature"
TABLES_DIR  = RESULTS_DIR / "tables"

CONN_MODE = "pli"
PREPROC   = 2
SPACE     = "source"


# ── Visual constants ───────────────────────────────────────────────────────────

CTRL_COLOR   = "#1f77b4"
TIN_COLOR    = "#C99700"
CHANCE_COLOR = "#7f8c8d"

# Canonical slow-to-fast band order and display labels
BAND_ORDER = ["delta", "theta",
              "alpha_0", "alpha_1", "alpha_2",
              "beta_0",  "beta_1",  "beta_2",  "beta_3",
              "gamma"]
BAND_LABELS = {
    "delta":   "δ",
    "theta":   "θ",
    "alpha_0": "α₀", "alpha_1": "α₁", "alpha_2": "α₂",
    "beta_0":  "β₀", "beta_1":  "β₁", "beta_2":  "β₂", "beta_3": "β₃",
    "gamma":   "γ",
}

# Yeo-7 network assignment (Desikan–Killiany ROI names, both hemispheres)
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
NETWORKS      = ["VIS", "SMN", "DAN", "VAN", "FPN", "DMN"]
AUDITORY_ROIS = ["transversetemporal", "superiortemporal", "supramarginal", "insula"]

THI_SEV_ORDER  = ["Slight", "Mild", "Moderate", "Severe", "Catastrophic", "Unknown"]
THI_SEV_COLORS = {
    "Slight": "#2ecc71", "Mild": "#f1c40f", "Moderate": "#e67e22",
    "Severe": "#e74c3c", "Catastrophic": "#8e44ad", "Unknown": "#aaaaaa",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stars(p):
    if pd.isna(p):    return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return ""


def _thi_sev(v):
    if pd.isna(v):  return "Unknown"
    if v <= 16:     return "Slight"
    if v <= 36:     return "Mild"
    if v <= 56:     return "Moderate"
    if v <= 76:     return "Severe"
    return "Catastrophic"


def _despine(ax):
    ax.spines[["right", "top"]].set_visible(False)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data():
    """Load PLI-band diffusive CSV merged with clinical data from master_clean."""
    feat_path   = (TINNORM_DIR / "diffusive_mm"
                   / f"{SPACE}_preproc_{PREPROC}_{CONN_MODE}_bands.csv")
    master_path = Path("../material/master_clean.csv")

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    df = pd.read_csv(feat_path)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df["subject_id"] = df["subject_id"].astype(str)

    df_master = pd.read_csv(master_path)
    df_master["subject_id"] = df_master["subject_id"].astype(str)
    if "THI" not in df_master.columns and "thi_score" in df_master.columns:
        df_master.rename(columns={"thi_score": "THI"}, inplace=True)

    clinical = [c for c in ["site", "age", "THI", "TFI", "PTA4_HF", "PTA4_mean"]
                if c in df_master.columns]
    df = df.merge(df_master[["subject_id"] + clinical], on="subject_id", how="left")

    print(f"  {len(df)} subjects | controls={int((df['group']==0).sum())}  "
          f"tinnitus={int((df['group']==1).sum())}")
    return df


def _parse_columns(df):
    """
    Parse feature columns into (roi, hemi, band) and return a lookup DataFrame.
    Column format: {roi_base}-{lh|rh}_{band}
    """
    skip    = {"subject_id", "group", "site", "age", "THI", "TFI",
               "PTA4_HF", "PTA4_mean", "Unnamed: 0"}
    pattern = re.compile(r'^(.+)-(lh|rh)_(.+)$')
    records = []
    for col in df.columns:
        if col in skip:
            continue
        m = pattern.match(col)
        if m:
            records.append({"col": col, "roi": m.group(1),
                            "hemi": m.group(2), "band": m.group(3)})
    return pd.DataFrame(records)


def _feat_cols(df, df_parsed):
    """Return list of pure feature column names (no metadata)."""
    return df_parsed["col"].tolist()


def _load_shap_rois(n_top: int = 15):
    """
    Load mean |SHAP| per feature from script 16 output and return the top-n ROIs
    ranked by aggregate SHAP importance (summed across all bands and hemispheres).
    Falls back to AUDITORY_ROIS when the SHAP file is missing (e.g. script 16 not yet run).

    Returns
    -------
    rois     : list[str]   — ROI names in SHAP-importance order
    roi_rank : dict | None — {roi: rank} for heatmap annotation (None on fallback)
    """
    shap_path = TABLES_DIR / "shap_mean_abs.csv"
    if not shap_path.exists():
        print(f"  SHAP file not found ({shap_path.name}) — falling back to anatomical auditory ROIs.")
        return AUDITORY_ROIS, None

    df_shap = pd.read_csv(shap_path)
    pattern = re.compile(r'^(.+)-(lh|rh)_(.+)$')
    records = []
    for _, row in df_shap.iterrows():
        m = pattern.match(str(row["feature"]))
        if m:
            records.append({"roi": m.group(1), "mean_abs_shap": float(row["mean_abs_shap"])})

    if not records:
        print("  Could not parse SHAP features — falling back to auditory ROIs.")
        return AUDITORY_ROIS, None

    roi_importance = (pd.DataFrame(records)
                      .groupby("roi")["mean_abs_shap"]
                      .sum()
                      .sort_values(ascending=False))
    top_rois = list(roi_importance.head(n_top).index)
    roi_rank = {roi: i + 1 for i, roi in enumerate(top_rois)}

    net_labels = [f"{r} [{ROI_TO_NETWORK.get(r, '?')}]" for r in top_rois[:5]]
    print(f"  SHAP top-{n_top} ROIs (by summed mean|SHAP|): {', '.join(net_labels)} …")
    return top_rois, roi_rank


# ── Analysis A: Network × Band heatmap ────────────────────────────────────────

def analysis_a_network_band_heatmap(df, df_parsed):
    print("\n── A. Network × Band heatmap ──────────────────────────────────────")

    df_parsed = df_parsed.copy()
    df_parsed["network"] = df_parsed["roi"].map(ROI_TO_NETWORK)
    df_parsed = df_parsed.dropna(subset=["network"])

    avail_bands = [b for b in BAND_ORDER if b in df_parsed["band"].unique()]
    y           = df["group"].values
    ctrl_mask   = y == 0
    tin_mask    = y == 1

    # Build (network × band) matrices
    ctrl_mat = pd.DataFrame(np.nan, index=NETWORKS, columns=avail_bands)
    tin_mat  = pd.DataFrame(np.nan, index=NETWORKS, columns=avail_bands)
    diff_mat = pd.DataFrame(np.nan, index=NETWORKS, columns=avail_bands)
    pval_mat = pd.DataFrame(np.nan, index=NETWORKS, columns=avail_bands)

    stat_rows = []
    for net in NETWORKS:
        for band in avail_bands:
            cols = df_parsed[(df_parsed["network"] == net) &
                             (df_parsed["band"]    == band)]["col"].tolist()
            if not cols:
                continue
            vals      = df[cols].mean(axis=1).values
            ctrl_vals = vals[ctrl_mask]
            tin_vals  = vals[tin_mask]

            ctrl_mat.loc[net, band] = ctrl_vals.mean()
            tin_mat.loc[net, band]  = tin_vals.mean()
            diff_mat.loc[net, band] = tin_vals.mean() - ctrl_vals.mean()

            _, p = mannwhitneyu(ctrl_vals, tin_vals, alternative="two-sided")
            pval_mat.loc[net, band] = p

            n_rois = len(cols) // 2   # each ROI has lh + rh column
            stat_rows.append({"network": net, "band": band,
                               "ctrl_mean": ctrl_vals.mean(),
                               "tin_mean":  tin_vals.mean(),
                               "diff":      tin_vals.mean() - ctrl_vals.mean(),
                               "p_mwu":     p, "n_rois": n_rois})

    # FDR correction across all non-NaN cells
    df_stats  = pd.DataFrame(stat_rows)
    _, p_fdr, _, _ = multipletests(df_stats["p_mwu"].values, method="fdr_bh")
    df_stats["p_fdr"] = p_fdr

    # Rebuild FDR matrix
    pval_fdr_mat = pd.DataFrame(np.nan, index=NETWORKS, columns=avail_bands)
    for _, row in df_stats.iterrows():
        pval_fdr_mat.loc[row["network"], row["band"]] = row["p_fdr"]

    # Annotation matrix (stars for difference panel)
    annot_diff = pval_fdr_mat.applymap(_stars)

    # ── Three-panel figure ─────────────────────────────────────────────────
    band_tick_labels = [BAND_LABELS.get(b, b) for b in avail_bands]
    vmax = max(np.nanmax(ctrl_mat.values), np.nanmax(tin_mat.values))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)

    for ax, mat, title in zip(axes[:2], [ctrl_mat, tin_mat],
                               ["Control", "Tinnitus"]):
        sns.heatmap(mat.astype(float), ax=ax, cmap="YlOrRd",
                    vmin=0, vmax=vmax, annot=False,
                    linewidths=0.4, linecolor="#dddddd",
                    xticklabels=band_tick_labels,
                    yticklabels=NETWORKS,
                    cbar_kws={"label": "Mean deviation", "shrink": 0.8})
        ax.set_title(title, style="italic", fontsize=12)
        ax.set_xlabel("Frequency band", fontsize=10)
        ax.set_ylabel("Yeo-7 Network" if ax is axes[0] else "", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    # Difference panel
    abs_max = np.nanmax(np.abs(diff_mat.values))
    sns.heatmap(diff_mat.astype(float), ax=axes[2],
                cmap="RdBu_r", center=0, vmin=-abs_max, vmax=abs_max,
                annot=annot_diff, fmt="", annot_kws={"size": 10, "weight": "bold"},
                linewidths=0.4, linecolor="#dddddd",
                xticklabels=band_tick_labels, yticklabels=NETWORKS,
                cbar_kws={"label": "Δ deviation (Tin − Ctrl)", "shrink": 0.8})
    axes[2].set_title("Tinnitus − Control  (FDR-corrected stars)",
                      style="italic", fontsize=12)
    axes[2].set_xlabel("Frequency band", fontsize=10)
    axes[2].set_ylabel("", fontsize=10)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].tick_params(axis="y", rotation=0)

    fpath = FIGURES_DIR / "network_band_heatmap.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    df_stats.to_csv(TABLES_DIR / "network_band_stats.csv", index=False)
    sig = (df_stats["p_fdr"] < 0.05).sum()
    print(f"  Significant network×band cells (FDR<0.05): {sig}/{len(df_stats)}")


# ── Analysis B: Tinnitus subgroup discovery ───────────────────────────────────

def analysis_b_subgroup_discovery(df, df_parsed):
    print("\n── B. Tinnitus subgroup discovery ─────────────────────────────────")

    feat_cols  = _feat_cols(df, df_parsed)
    tin_mask   = df["group"] == 1
    df_tin     = df[tin_mask].copy().reset_index(drop=True)
    X          = df_tin[feat_cols].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Dimensionality reduction ───────────────────────────────────────────
    if HAS_UMAP:
        print("  Running UMAP …")
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       random_state=42, n_jobs=-1)
        embed   = reducer.fit_transform(X_scaled)
        embed_label = "UMAP"
    else:
        print("  Running PCA (UMAP not installed) …")
        reducer = PCA(n_components=2, random_state=42)
        embed   = reducer.fit_transform(X_scaled)
        embed_label = "PCA"

    df_tin["embed_1"] = embed[:, 0]
    df_tin["embed_2"] = embed[:, 1]

    # ── K-means: pick k by silhouette ─────────────────────────────────────
    print("  Silhouette scores:")
    best_k, best_score, best_labels = 2, -1, None
    sil_scores = {}
    for k in [2, 3, 4]:
        km     = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels,
                                   sample_size=min(500, len(X_scaled)),
                                   random_state=42)
        sil_scores[k] = score
        print(f"    k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"  → Best k={best_k}  (silhouette={best_score:.3f})")
    df_tin["cluster"] = best_labels

    if "THI" in df_tin.columns:
        df_tin["THI_severity"] = df_tin["THI"].apply(_thi_sev)

    cluster_palette = sns.color_palette("Set2", best_k)
    site_col = "site" if "site" in df_tin.columns else None
    sites_uniq = sorted(df_tin[site_col].dropna().unique()) if site_col else []
    site_palette = {s: sns.color_palette("tab10", len(sites_uniq))[i]
                    for i, s in enumerate(sites_uniq)}

    # ── Figure 1: 2×2 embedding panels ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.flatten()

    # Panel A: cluster assignment
    ax = axes[0]
    for k_idx in range(best_k):
        m = df_tin["cluster"] == k_idx
        ax.scatter(df_tin.loc[m, "embed_1"], df_tin.loc[m, "embed_2"],
                   c=[cluster_palette[k_idx]], s=22, alpha=0.75, zorder=3,
                   label=f"Cluster {k_idx+1}  (n={int(m.sum())})")
    ax.set_title(f"Cluster assignment  (k={best_k}, sil={best_score:.3f})",
                 style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize="small")
    ax.set_xlabel(f"{embed_label} 1"); ax.set_ylabel(f"{embed_label} 2")
    _despine(ax)

    # Panel B: THI severity
    ax = axes[1]
    if "THI_severity" in df_tin.columns:
        for sev in THI_SEV_ORDER:
            m = df_tin["THI_severity"] == sev
            if not m.any():
                continue
            ax.scatter(df_tin.loc[m, "embed_1"], df_tin.loc[m, "embed_2"],
                       c=[THI_SEV_COLORS[sev]], s=22, alpha=0.75, zorder=3,
                       label=f"{sev}  (n={int(m.sum())})")
        ax.legend(frameon=False, fontsize="small", title="THI severity",
                  title_fontsize="small")
    ax.set_title("THI severity", style="italic", fontsize=11)
    ax.set_xlabel(f"{embed_label} 1"); ax.set_ylabel("")
    _despine(ax)

    # Panel C: PTA4_HF (continuous audiometry)
    ax = axes[2]
    if "PTA4_HF" in df_tin.columns:
        pta  = df_tin["PTA4_HF"].values.astype(float)
        vlo  = np.nanpercentile(pta, 5)
        vhi  = np.nanpercentile(pta, 95)
        sc   = ax.scatter(df_tin["embed_1"], df_tin["embed_2"],
                          c=pta, cmap="plasma", s=22, alpha=0.75, zorder=3,
                          vmin=vlo, vmax=vhi)
        plt.colorbar(sc, ax=ax, shrink=0.75, label="PTA4_HF (dB HL)")
    ax.set_title("High-frequency hearing threshold (PTA4_HF)",
                 style="italic", fontsize=11)
    ax.set_xlabel(f"{embed_label} 1"); ax.set_ylabel(f"{embed_label} 2")
    _despine(ax)

    # Panel D: recording site
    ax = axes[3]
    for site in sites_uniq:
        m = df_tin[site_col] == site
        ax.scatter(df_tin.loc[m, "embed_1"], df_tin.loc[m, "embed_2"],
                   c=[site_palette[site]], s=22, alpha=0.75, zorder=3,
                   label=f"{site}  (n={int(m.sum())})")
    ax.legend(frameon=False, fontsize="small", title="Site",
              title_fontsize="small")
    ax.set_title("Recording site", style="italic", fontsize=11)
    ax.set_xlabel(f"{embed_label} 1"); ax.set_ylabel("")
    _despine(ax)

    fig.suptitle(
        f"Tinnitus subgroup discovery — {embed_label} + k-means (k={best_k})",
        style="italic", fontsize=13)
    fpath = FIGURES_DIR / "subgroup_umap.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── Figure 2: Clinical profiles per cluster ────────────────────────────
    targets = [c for c in ["THI", "TFI", "PTA4_HF"] if c in df_tin.columns]
    if targets:
        n_t   = len(targets)
        fig, axes = plt.subplots(1, n_t, figsize=(5.5 * n_t, 5.5),
                                  constrained_layout=True)
        if n_t == 1:
            axes = [axes]

        for ax, target in zip(axes, targets):
            df_plot = (df_tin[["cluster", target]]
                       .dropna()
                       .assign(Cluster=lambda d: (d["cluster"] + 1).astype(str)))
            order = [str(k + 1) for k in range(best_k)]

            sns.boxenplot(data=df_plot, x="Cluster", y=target,
                          palette=cluster_palette[:best_k],
                          order=order, linewidth=1.2, ax=ax)
            sns.stripplot(data=df_plot, x="Cluster", y=target,
                          order=order, color="black", alpha=0.22,
                          size=3, jitter=True, ax=ax)

            groups_kw = [df_plot.loc[df_plot["cluster"] == k, target].dropna().values
                         for k in range(best_k)]
            groups_kw = [g for g in groups_kw if len(g) >= 3]
            if len(groups_kw) >= 2:
                try:
                    _, p_kw = kruskal(*groups_kw)
                    sig = _stars(p_kw)
                    ax.set_title(f"{target}  (KW p={p_kw:.3f} {sig})",
                                 style="italic", fontsize=11)
                except Exception:
                    ax.set_title(target, style="italic", fontsize=11)
            else:
                ax.set_title(target, style="italic", fontsize=11)

            ax.set_xlabel("Cluster", fontsize=10)
            ax.set_ylabel(target, fontsize=10)
            _despine(ax)

        fig.suptitle(f"Clinical profiles per cluster  (k={best_k})",
                     style="italic", fontsize=12)
        fpath = FIGURES_DIR / "subgroup_profiles.pdf"
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fpath}")

    # ── Stats table ────────────────────────────────────────────────────────
    rows_stats = []
    for k_idx in range(best_k):
        m_k = df_tin["cluster"] == k_idx
        row = {"cluster": k_idx + 1, "n": int(m_k.sum())}
        for t in targets:
            vals = df_tin.loc[m_k, t].dropna()
            row[f"{t}_mean"]   = round(vals.mean(), 2) if len(vals) else np.nan
            row[f"{t}_sd"]     = round(vals.std(), 2)  if len(vals) else np.nan
            row[f"{t}_median"] = round(vals.median(), 2) if len(vals) else np.nan
        if site_col:
            vc = df_tin.loc[m_k, site_col].value_counts()
            row["dominant_site"] = vc.index[0] if len(vc) else ""
        rows_stats.append(row)
    pd.DataFrame(rows_stats).to_csv(TABLES_DIR / "subgroup_stats.csv", index=False)
    print(f"  Saved → {TABLES_DIR / 'subgroup_stats.csv'}")


# ── Analysis C: Band-specific lateralization (SHAP-guided ROI selection) ──────

def analysis_c_band_lateralization(df, df_parsed, n_top_rois: int = 15):
    print("\n── C. Band-specific lateralization (SHAP-guided) ───────────────────")

    # Load SHAP-ranked ROIs; fall back to auditory ROIs if script 16 hasn't run yet
    rois, roi_rank = _load_shap_rois(n_top=n_top_rois)
    avail_rois  = [r for r in rois if r in df_parsed["roi"].unique()]
    avail_bands = [b for b in BAND_ORDER if b in df_parsed["band"].unique()]

    if not avail_rois:
        print("  No matching ROIs found in feature set — skipping.")
        return

    y         = df["group"].values
    ctrl_mask = y == 0
    tin_mask  = y == 1

    stat_rows = []
    li_long   = []

    for roi in avail_rois:
        for band in avail_bands:
            lh_col = f"{roi}-lh_{band}"
            rh_col = f"{roi}-rh_{band}"
            if lh_col not in df.columns or rh_col not in df.columns:
                continue

            lh = df[lh_col].values.astype(float)
            rh = df[rh_col].values.astype(float)
            li = (lh - rh) / (np.abs(lh) + np.abs(rh) + 1e-8)

            ctrl_li = li[ctrl_mask]
            tin_li  = li[tin_mask]

            _, p = mannwhitneyu(ctrl_li, tin_li, alternative="two-sided")

            network = ROI_TO_NETWORK.get(roi, "?")
            stat_rows.append({
                "roi":          roi,
                "network":      network,
                "shap_rank":    roi_rank[roi] if roi_rank else None,
                "band":         band,
                "ctrl_mean_li": float(ctrl_li.mean()),
                "tin_mean_li":  float(tin_li.mean()),
                "diff_li":      float(tin_li.mean() - ctrl_li.mean()),
                "p_mwu":        p,
            })

            for v in ctrl_li:
                li_long.append({"roi": roi, "band": band, "LI": v, "group": "Control"})
            for v in tin_li:
                li_long.append({"roi": roi, "band": band, "LI": v, "group": "Tinnitus"})

    if not stat_rows:
        print("  No ROI × band pairs found — skipping.")
        return

    df_stats = pd.DataFrame(stat_rows)
    _, p_fdr, _, _ = multipletests(df_stats["p_mwu"].values, method="fdr_bh")
    df_stats["p_fdr"] = p_fdr
    df_stats.to_csv(TABLES_DIR / "band_lateralization_stats.csv", index=False)

    sig_n = (df_stats["p_fdr"] < 0.05).sum()
    source = "SHAP top-" + str(len(avail_rois)) if roi_rank else "anatomical auditory"
    print(f"  ROIs used ({source}): {', '.join(avail_rois)}")
    print(f"  Significant ROI×band pairs (FDR<0.05): {sig_n}/{len(df_stats)}")

    # ── Figure 1: LI difference heatmap ───────────────────────────────────
    diff_pivot = df_stats.pivot(index="roi", columns="band", values="diff_li")
    fdr_pivot  = df_stats.pivot(index="roi", columns="band", values="p_fdr")

    col_order = [b for b in avail_bands if b in diff_pivot.columns]
    diff_pivot = diff_pivot.reindex(index=avail_rois, columns=col_order)
    fdr_pivot  = fdr_pivot.reindex(index=avail_rois, columns=col_order)
    diff_pivot.dropna(how="all", inplace=True)
    fdr_pivot  = fdr_pivot.loc[diff_pivot.index]

    # Y-axis labels: "roi [NETWORK] (#shap_rank)"
    def _row_label(roi):
        net  = ROI_TO_NETWORK.get(roi, "?")
        rank = f" (#{roi_rank[roi]})" if roi_rank and roi in roi_rank else ""
        return f"{roi} [{net}]{rank}"

    row_labels       = [_row_label(r) for r in diff_pivot.index]
    band_tick_labels = [BAND_LABELS.get(b, b) for b in col_order]
    annot_mat        = fdr_pivot.applymap(_stars)
    abs_max          = max(np.nanmax(np.abs(diff_pivot.values)), 1e-6)

    n_rows_heat = len(diff_pivot)
    fig, ax = plt.subplots(figsize=(12, max(4.0, n_rows_heat * 0.55)),
                            constrained_layout=True)
    sns.heatmap(diff_pivot.astype(float), ax=ax,
                cmap="RdBu_r", center=0, vmin=-abs_max, vmax=abs_max,
                annot=annot_mat, fmt="", annot_kws={"size": 9, "weight": "bold"},
                linewidths=0.4, linecolor="#dddddd",
                xticklabels=band_tick_labels, yticklabels=row_labels,
                cbar_kws={"label": "ΔLI  (Tinnitus − Control)", "shrink": 0.7})
    ax.set_xlabel("Frequency band", fontsize=11)
    ax.set_ylabel("ROI  [network]  (SHAP rank)", fontsize=10)
    subtitle = "SHAP-ranked ROIs" if roi_rank else "anatomical auditory ROIs"
    ax.set_title(
        f"Band-specific lateralization index — {subtitle}\n"
        "(Tinnitus − Control, FDR-corrected stars)",
        style="italic", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    fpath = FIGURES_DIR / "band_lateralization_heatmap.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # ── Figure 2: Violin plots for top significant cells ───────────────────
    sig_cells = df_stats[df_stats["p_fdr"] < 0.05].sort_values("p_fdr")
    fallback  = False
    if len(sig_cells) == 0:
        sig_cells = df_stats.nsmallest(6, "p_mwu")
        fallback  = True
        print("  No FDR-significant cells — showing top 6 by raw p-value.")

    n_viol = min(9, len(sig_cells))
    if n_viol == 0:
        return

    df_li_long = pd.DataFrame(li_long)
    ncols      = min(n_viol, 3)
    nrows      = int(np.ceil(n_viol / ncols))
    fig, axes  = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 4.5 * nrows),
                               constrained_layout=True)
    axes_flat  = np.array(axes).flatten() if n_viol > 1 else [axes]

    palette = {"Control": CTRL_COLOR, "Tinnitus": TIN_COLOR}

    for ax, (_, row) in zip(axes_flat, sig_cells.head(n_viol).iterrows()):
        roi  = row["roi"]
        band = row["band"]
        net  = ROI_TO_NETWORK.get(roi, "?")
        rank = f" #{roi_rank[roi]}" if roi_rank and roi in roi_rank else ""
        df_sub = df_li_long[
            (df_li_long["roi"] == roi) & (df_li_long["band"] == band)
        ]

        sns.violinplot(data=df_sub, x="group", y="LI",
                       order=["Control", "Tinnitus"],
                       palette=palette, inner="box",
                       alpha=0.72, cut=0, ax=ax, linewidth=1.2)
        sns.stripplot(data=df_sub, x="group", y="LI",
                      order=["Control", "Tinnitus"],
                      palette=palette, alpha=0.28, size=3, jitter=True, ax=ax)
        ax.axhline(0, color=CHANCE_COLOR, linestyle="--", lw=1.1)

        p_label = _stars(row["p_fdr"]) if not fallback else f"p={row['p_mwu']:.3f}"
        ax.set_title(f"{roi} [{net}]{rank}\n{BAND_LABELS.get(band, band)}  {p_label}",
                     style="italic", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Lateralization Index (LI)", fontsize=9)
        _despine(ax)

    for ax in axes_flat[n_viol:]:
        ax.set_visible(False)

    suptitle_sfx = "FDR-significant" if not fallback else "top by raw p"
    fig.suptitle(
        f"Band-specific LI — {suptitle_sfx} ROI × band pairs  (SHAP-guided)",
        style="italic", fontsize=12)
    fpath = FIGURES_DIR / "band_lateralization_violin.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    try:
        df = _load_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nRun 09_multimodal_diffusion.py first.")
        raise SystemExit(1)

    df_parsed = _parse_columns(df)
    print(f"  Feature columns parsed: {len(df_parsed)}  "
          f"({df_parsed['roi'].nunique()} ROIs × "
          f"{df_parsed['band'].nunique()} bands × 2 hemispheres)")

    analysis_a_network_band_heatmap(df, df_parsed)
    analysis_b_subgroup_discovery(df, df_parsed)
    analysis_c_band_lateralization(df, df_parsed, n_top_rois=15)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
