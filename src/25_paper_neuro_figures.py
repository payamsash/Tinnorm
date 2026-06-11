"""
25_paper_neuro_figures.py — Four supplementary paper figures (Discussion §9 A–D).

  A. Normative model quality
       KDE distributions of Z-scores (control LOSO vs tinnitus full-model) for
       the top SHAP features.  Validates that normative modelling correctly captures
       the control distribution and reveals tinnitus deviation.
       → figures/25_paper_neuro/A_normative_quality.pdf

  B. Per-site classification performance
       Heatmap: sensitivity, specificity, balanced-accuracy per site (J_bands_pli_lgbm).
       → figures/25_paper_neuro/B_site_performance.pdf

  C. SHAP × Yeo-7 network bar chart
       Aggregate mean |SHAP| per feature, summed per network.
       Shows FPN ≫ DAN ≫ DMN as the dominant networks.
       → figures/25_paper_neuro/C_shap_network_bar.pdf

  D. Brain parcellation heatmap
       Project mean |SHAP| per ROI onto a flat parcellation grid, grouped by
       Yeo-7 network and sorted by SHAP within network.
       Optionally also renders MNE brain surface views (lateral + medial,
       both hemispheres) if pyvistaqt is available.
       → figures/25_paper_neuro/D_brain_parcellation.pdf
       → figures/25_paper_neuro/D_brain_surface.png  (if MNE rendering works)

Run from src/:  python 25_paper_neuro_figures.py
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
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score, confusion_matrix


# ── Paths ──────────────────────────────────────────────────────────────────────

TINNORM_DIR  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR  = TINNORM_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures" / "25_paper_neuro"
TABLES_DIR   = RESULTS_DIR / "tables"
MATERIAL_DIR = Path("../material")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR   = TINNORM_DIR / "models" / "preproc_2" / "source" / "regional_pli"
BEST_CLF_DIR = TINNORM_DIR / "clfs" / "J_bands_pli_lgbm"
FEAT_CSV     = TINNORM_DIR / "diffusive_mm" / "source_preproc_2_pli_bands.csv"
MASTER_CSV   = MATERIAL_DIR / "master_clean.csv"


# ── Yeo-7 mapping (same as script 23) ─────────────────────────────────────────

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
NETWORKS     = ["VIS", "SMN", "DAN", "VAN", "FPN", "DMN"]
NETWORK_COLORS = {
    "VIS": "#7b68ee", "SMN": "#4caf50", "DAN": "#ff9800",
    "VAN": "#f06292", "FPN": "#1976d2", "DMN": "#e53935",
}

BAND_ORDER = ["delta", "theta",
              "alpha_0", "alpha_1", "alpha_2",
              "beta_0",  "beta_1",  "beta_2",  "beta_3",
              "gamma"]
BAND_LABELS = {
    "delta":   "δ",  "theta":   "θ",
    "alpha_0": "α₀", "alpha_1": "α₁", "alpha_2": "α₂",
    "beta_0":  "β₀", "beta_1":  "β₁", "beta_2":  "β₂", "beta_3": "β₃",
    "gamma":   "γ",
}

CTRL_COLOR = "#2471A3"
TIN_COLOR  = "#C0392B"

SITE_ORDER = ["austin", "dublin", "ghent", "illinois", "regensburg", "tuebingen", "zuerich"]
SITE_COLORS = dict(zip(SITE_ORDER, sns.color_palette("husl", len(SITE_ORDER))))


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _despine(ax, lw: float = 1.5):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)
    ax.tick_params(width=lw, length=4)


def _save(fig, name: str, dpi: int = 200):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path.name}")


def _load_shap_mean_abs():
    df = pd.read_csv(TABLES_DIR / "shap_mean_abs.csv")
    pattern = re.compile(r'^(.+)-(lh|rh)_(.+)$')
    rows = []
    for _, r in df.iterrows():
        m = pattern.match(str(r["feature"]))
        if m:
            rows.append({
                "feature": r["feature"],
                "roi": m.group(1),
                "hemi": m.group(2),
                "band": m.group(3),
                "mean_abs_shap": float(r["mean_abs_shap"]),
                "network": ROI_TO_NETWORK.get(m.group(1), "?"),
            })
    return pd.DataFrame(rows)


# ── Figure A: Normative model quality ─────────────────────────────────────────

def fig_a_normative_quality():
    print("\n── A. Normative model quality ─────────────────────────────────────")

    ctrl_z_path = MODELS_DIR / "loso" / "loso_controls_Z.csv"
    tin_z_path  = MODELS_DIR / "full_model" / "results" / "Z_test.csv"
    if not ctrl_z_path.exists() or not tin_z_path.exists():
        print(f"  Missing Z-score files — skipping Figure A.")
        return

    ctrl_z = pd.read_csv(ctrl_z_path)
    tin_z  = pd.read_csv(tin_z_path)

    # Top SHAP features (ROI_theta_pli naming in Z-score files)
    df_shap = _load_shap_mean_abs()
    top_feats = df_shap.sort_values("mean_abs_shap", ascending=False).head(8)

    # Map shap feature name → Z-score column name  (add _pli suffix)
    feat_map = {}
    for _, row in top_feats.iterrows():
        z_col = f"{row['roi']}-{row['hemi']}_{row['band']}_pli"
        if z_col in ctrl_z.columns and z_col in tin_z.columns:
            feat_map[row["feature"]] = z_col
        if len(feat_map) == 6:
            break

    if not feat_map:
        print("  Could not map SHAP features to Z-score columns — skipping.")
        return

    n_feats = len(feat_map)
    fig, axes = plt.subplots(1, n_feats, figsize=(2.4 * n_feats, 3.8), sharey=False)
    if n_feats == 1:
        axes = [axes]

    x_range = np.linspace(-6, 10, 300)
    # Reference N(0,1) curve
    from scipy.stats import norm as sp_norm
    ref_y = sp_norm.pdf(x_range)

    # Compute KDEs first so we can set a shared ylim
    kde_data = {}
    for feat_name, z_col in feat_map.items():
        ctrl_vals = ctrl_z[z_col].dropna().values
        tin_vals  = tin_z[z_col].dropna().values
        kde_ctrl  = gaussian_kde(ctrl_vals, bw_method=0.4)
        kde_tin   = gaussian_kde(tin_vals,  bw_method=0.4)
        kde_data[feat_name] = (z_col, ctrl_vals, tin_vals, kde_ctrl, kde_tin)

    from scipy.stats import norm as sp_norm
    ref_y = sp_norm.pdf(x_range)
    global_ymax = max(
        max(kde_ctrl(x_range).max(), kde_tin(x_range).max(), ref_y.max())
        for _, _, _, kde_ctrl, kde_tin in kde_data.values()
    ) * 1.12

    for ax, (feat_name, (z_col, ctrl_vals, tin_vals, kde_ctrl, kde_tin)) in \
            zip(axes, kde_data.items()):
        ax.plot(x_range, ref_y,    color="#AAAAAA", lw=1.2, ls="--", label="N(0,1)", zorder=1)
        ax.fill_between(x_range, kde_ctrl(x_range), alpha=0.30, color=CTRL_COLOR)
        ax.plot(x_range, kde_ctrl(x_range), color=CTRL_COLOR, lw=1.8, label="Control")
        ax.fill_between(x_range, kde_tin(x_range), alpha=0.30, color=TIN_COLOR)
        ax.plot(x_range, kde_tin(x_range),  color=TIN_COLOR,  lw=1.8, label="Tinnitus")
        ax.axvline(0, color="#BBBBBB", lw=0.8, ls=":")

        roi_base, hemi_band = feat_name.rsplit("-", 1)
        hemi, band = hemi_band.split("_", 1)
        band_nice = BAND_LABELS.get(band, band)
        net = ROI_TO_NETWORK.get(roi_base, "?")
        label = f"{roi_base}\n{hemi} · {band_nice} [{net}]"
        ax.set_title(label, fontsize=7.5, pad=4)
        ax.set_xlabel("Z-score", fontsize=8)
        ax.set_xlim(-6, 10)
        ax.set_ylim(0, global_ymax)
        _despine(ax)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Density", fontsize=8)
    # Remove yticks from non-first panels (same scale is shown by axis[0])
    for ax in axes[1:]:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    handles = [
        mpatches.Patch(color="#AAAAAA", label="N(0,1) ref."),
        mpatches.Patch(color=CTRL_COLOR, label="Control (LOSO)"),
        mpatches.Patch(color=TIN_COLOR,  label="Tinnitus (full model)"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=7.5,
               bbox_to_anchor=(1.0, 1.02), frameon=False)
    fig.suptitle("Normative model Z-score distributions — top SHAP features",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, "A_normative_quality.pdf")


# ── Figure B: Per-site classification performance ─────────────────────────────

def fig_b_site_performance():
    print("\n── B. Per-site classification performance ────────────────────────")

    y     = np.load(BEST_CLF_DIR / "y.npy")
    y_prob = np.load(BEST_CLF_DIR / "y_prob.npy")
    y_pred = (y_prob >= 0.5).astype(int)

    df_feat   = pd.read_csv(FEAT_CSV, usecols=["subject_ids", "group"])
    df_master = pd.read_csv(MASTER_CSV)
    df_feat   = df_feat.rename(columns={"subject_ids": "subject_id"})
    df_feat["subject_id"] = df_feat["subject_id"].astype(str)
    df_master["subject_id"] = df_master["subject_id"].astype(str)
    df_merged = df_feat.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
    sites_arr = df_merged["site"].values

    rows = []
    for site in SITE_ORDER:
        idx = np.where(sites_arr == site)[0]
        if len(idx) == 0:
            continue
        yt   = y[idx]
        yp   = y_prob[idx]
        ypred = y_pred[idx]

        if len(np.unique(yt)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(yt, yp)

        n_ctrl = (yt == 0).sum()
        n_tin  = (yt == 1).sum()

        tn, fp, fn, tp = (0, 0, 0, 0)
        if n_ctrl > 0 and n_tin > 0:
            tn, fp, fn, tp = confusion_matrix(yt, ypred).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        balacc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else np.nan

        rows.append({
            "site": site.capitalize(),
            "n_ctrl": n_ctrl,
            "n_tin": n_tin,
            "AUC": auc,
            "Sensitivity": sens,
            "Specificity": spec,
            "PPV": ppv,
            "NPV": npv,
            "Bal.Acc": balacc,
        })

    df_perf = pd.DataFrame(rows)
    print(df_perf[["site", "n_ctrl", "n_tin", "AUC", "Sensitivity", "Specificity", "Bal.Acc"]].to_string(index=False))

    # Save table
    df_perf.to_csv(TABLES_DIR / "site_clf_performance.csv", index=False)

    # ── Heatmap: sites × metrics ──────────────────────────────────────────────
    metrics   = ["AUC", "Sensitivity", "Specificity", "PPV", "NPV", "Bal.Acc"]
    hm_data   = df_perf.set_index("site")[metrics]
    site_labels = [
        f"{r['site']}\n(ctrl={r['n_ctrl']}, tin={r['n_tin']})"
        for _, r in df_perf.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cmap = LinearSegmentedColormap.from_list("perf", ["#FDEBD0", "#E74C3C", "#1A5276"])
    im = ax.imshow(hm_data.values, cmap=cmap, vmin=0.35, vmax=0.85, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_yticks(range(len(site_labels)))
    ax.set_yticklabels(site_labels, fontsize=8.5)
    ax.tick_params(left=False, bottom=False)

    # Annotate cells
    for i, row in enumerate(hm_data.values):
        for j, val in enumerate(row):
            txt = f"{val:.2f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color="white" if val < 0.45 or val > 0.72 else "#1a1a1a",
                    style="italic")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Chance-level annotation
    ax.axvline(-0.5, color="none")
    ax.set_title("Per-site LOSO performance — J_bands_pli_lgbm\n"
                 "(each subject predicted by model trained without their site)",
                 fontsize=8.5, pad=8)

    # Highlight best / worst AUC rows
    auc_vals = hm_data["AUC"].values
    best_idx = np.nanargmax(auc_vals)
    worst_idx = np.nanargmin(auc_vals)
    for idx, color in [(best_idx, "#27AE60"), (worst_idx, "#E74C3C")]:
        rect = plt.Rectangle((-0.5, idx - 0.5), len(metrics), 1,
                              linewidth=2, edgecolor=color, facecolor="none", zorder=3)
        ax.add_patch(rect)

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "B_site_performance.pdf")


# ── Figure C: SHAP × Network bar chart ────────────────────────────────────────

def fig_c_shap_network_bar():
    print("\n── C. SHAP × Network bar chart ────────────────────────────────────")

    df_shap = _load_shap_mean_abs()
    df_shap = df_shap[df_shap["network"] != "?"]

    # Sum mean |SHAP| per network
    net_sum = (df_shap.groupby("network")["mean_abs_shap"]
               .sum()
               .reindex(NETWORKS)
               .fillna(0)
               .sort_values(ascending=True))  # ascending for horizontal barh

    # Also compute per-band breakdown per network
    net_band = (df_shap.groupby(["network", "band"])["mean_abs_shap"]
                .sum()
                .unstack(fill_value=0)
                .reindex(index=net_sum.index, columns=BAND_ORDER, fill_value=0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.7]})

    # ── Panel 1: aggregated bar ───────────────────────────────────────────────
    ax = axes[0]
    colors = [NETWORK_COLORS.get(n, "#888888") for n in net_sum.index]
    bars = ax.barh(net_sum.index, net_sum.values, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.65)
    ax.set_xlabel("Σ mean |SHAP|", fontsize=9)
    ax.set_title("Total SHAP importance\nper Yeo-7 network", fontsize=9)
    for bar, val in zip(bars, net_sum.values):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, net_sum.max() * 1.22)
    _despine(ax)
    ax.tick_params(labelsize=8.5)

    # ── Panel 2: stacked bar by frequency band ───────────────────────────────
    # Frequency-ordered custom palette: slow (indigo) → fast (deep red)
    BAND_PALETTE = {
        "delta":   "#37306B",  # deep indigo
        "theta":   "#1565C0",  # royal blue
        "alpha_0": "#0277BD",  # medium blue
        "alpha_1": "#00838F",  # teal
        "alpha_2": "#00695C",  # dark teal
        "beta_0":  "#2E7D32",  # forest green
        "beta_1":  "#558B2F",  # olive green
        "beta_2":  "#F9A825",  # amber
        "beta_3":  "#E65100",  # deep orange
        "gamma":   "#B71C1C",  # deep red
    }
    ax2 = axes[1]
    bottom = np.zeros(len(net_band))
    for i, band in enumerate(BAND_ORDER):
        vals = net_band[band].values
        ax2.barh(net_band.index, vals, left=bottom,
                 color=BAND_PALETTE.get(band, "#888888"),
                 label=BAND_LABELS.get(band, band),
                 edgecolor="white", linewidth=0.3, height=0.65)
        bottom += vals

    ax2.set_xlabel("Σ mean |SHAP|", fontsize=9)
    ax2.set_title("SHAP importance by frequency band\nper Yeo-7 network", fontsize=9)
    ax2.legend(title="Band", fontsize=7, title_fontsize=7.5,
               bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, ncol=1)
    _despine(ax2)
    ax2.tick_params(labelsize=8.5)

    fig.tight_layout()
    _save(fig, "C_shap_network_bar.pdf")


# ── Figure D: Brain surface — script-11 style ─────────────────────────────────

def fig_d_brain_surface():
    print("\n── D. Brain surface — SHAP parcellation map ───────────────────────")
    try:
        import mne
        import matplotlib.colors as mcolors
        from mne.viz import Brain

        df_shap = _load_shap_mean_abs()

        # Per (roi, hemi): mean over bands → separate lh/rh SHAP values
        roi_hemi_mean = (df_shap.groupby(["roi", "hemi"])["mean_abs_shap"].mean())

        fs_dir       = Path(mne.datasets.fetch_fsaverage(verbose=False))
        subjects_dir = str(fs_dir.parent)
        labels = mne.read_labels_from_annot(
            "fsaverage", parc="aparc", subjects_dir=subjects_dir, verbose=False
        )
        labels = labels[:-1]  # drop "unknown" (last entry)

        # Build metric_vals aligned with label list
        metric_vals = np.zeros(len(labels))
        for i, lbl in enumerate(labels):
            m = re.match(r'^(.+)-(lh|rh)$', lbl.name)
            if m:
                roi, hemi = m.group(1), m.group(2)
                try:
                    metric_vals[i] = roi_hemi_mean.loc[(roi, hemi)]
                except KeyError:
                    pass

        # "rocket_r" reversed = dark-to-bright: very dark at low, vivid amber/cream at peak
        # "flare" = soft warm cream → deep rust red (perceptually uniform, seaborn)
        palette = sns.color_palette("flare", n_colors=256)
        colors_hex = np.array([mcolors.to_hex(c) for c in palette])

        # Robust vmax: 98th percentile of non-zero values (outlier parsorbitalis)
        nonzero = metric_vals[metric_vals > 0]
        vmax = float(np.percentile(nonzero, 98)) if len(nonzero) else 1.0
        vmin = 0.0
        clipped = np.clip(metric_vals, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin + 1e-12)
        label_colors = colors_hex[np.round(norm * 255).astype(int)]

        brain_kw = dict(
            subject="fsaverage",
            subjects_dir=subjects_dir,
            surf="inflated",
            background="white",
            cortex=["#c8c4bc", "#c8c4bc"],
        )

        def _render(hemi, view):
            brain = Brain(hemi=hemi, views=view, **brain_kw)
            for lbl, color in zip(labels, label_colors):
                if lbl.hemi == hemi:
                    brain.add_label(lbl, hemi=hemi, color=color,
                                    borders=False, alpha=0.90)
            brain.add_annotation("aparc", borders=True, color="white")
            img = brain.screenshot()
            brain.close()
            nw = (img != 255).any(axis=-1)
            return img[nw.any(axis=1)][:, nw.any(axis=0)]

        view_configs = [
            ("lh", "lateral"),  ("rh", "lateral"),
            ("lh", "medial"),   ("rh", "medial"),
        ]
        view_titles = ["LH lateral", "RH lateral", "LH medial", "RH medial"]
        imgs = [_render(h, v) for h, v in view_configs]

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.subplots_adjust(hspace=0.03, wspace=0.03)
        for ax, img in zip(axes.flat, imgs):
            ax.imshow(img)
            ax.axis("off")

        sm = plt.cm.ScalarMappable(
            cmap=mcolors.ListedColormap(palette),
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        )
        cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.06, shrink=0.45)
        cbar.set_label("Mean |SHAP| (avg over frequency bands, per hemisphere)", fontsize=9)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.3g}", f"{vmax:.3g}"])

        fig.suptitle(
            "SHAP importance projected onto fsaverage cortical surface\n"
            "(Desikan–Killiany parcellation — colour shows mean |SHAP| per ROI × hemisphere)",
            fontsize=9, y=1.005,
        )
        _save(fig, "D_brain_surface.pdf", dpi=150)

    except Exception as e:
        print(f"  Brain surface render failed ({type(e).__name__}: {e})")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("25_paper_neuro_figures.py")
    print("=" * 65)

    fig_a_normative_quality()
    fig_b_site_performance()
    fig_c_shap_network_bar()
    fig_d_brain_surface()

    print("\n✓ Done — all figures in:", FIGURES_DIR)
