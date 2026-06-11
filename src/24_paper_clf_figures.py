"""
24_paper_clf_figures.py — Publication-quality versions of main classification figures.

  Figure 1: all_scenarios_auc_ranking  (cubehelix gradient, readable labels,
                                         ensemble divider, tight bar spacing)
  Figure 2: best_scenario_panel        (permutation null + ROC with dashed per-site
                                         lines + PR with dashed per-site lines;
                                         bar colour matched to Figure 1)
  Figure 3: shap_summary               (beeswarm, top-k features, paper-quality)
  Figure 4: shap_fold_stability        (hollow boxplot + HUSL site dots)

Figures 1–2 are auto-detected from clfs/.
Figures 3–4 load pre-saved SHAP arrays from results/tables/ (run script 16 first).
All figures saved to results/figures/24_paper_clf/.

Run from src/:  python 24_paper_clf_figures.py
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    average_precision_score,
)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    print("Warning: shap not installed — Figures 3 & 4 will be skipped.")

# ── Paths ──────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
# TINNORM_DIR = Path("/data/Tinnorm")   # ← uncomment on VM
CLFS_DIR    = TINNORM_DIR / "clfs"
FIGURES_DIR = TINNORM_DIR / "results" / "figures" / "24_paper_clf"

# ── Colour / group maps (shared with script 14) ────────────────────────────────

GROUP_COLORS = {
    "A": "#2196F3", "B": "#E91E63", "C": "#009688", "D": "#FF9800",
    "E": "#9C27B0", "F": "#F44336", "G": "#4CAF50", "H": "#795548",
    "I": "#607D8B", "J": "#FF5722", "e": "#212121",
}
GROUP_NAMES = {
    "A": "Preprocessing levels",  "B": "Classifiers",
    "C": "Connectivity measures", "D": "Feature dimensionality",
    "E": "Feature selection",     "F": "Hyperparameter tuning",
    "G": "THI severity threshold","H": "Individual modalities",
    "I": "Residual vs Deviation", "J": "Bands × connectivity + tuning",
    "e": "Ensemble",
}
CHANCE_COLOR = "#9E9E9E"

_MODE_SHORT = {
    "diffusive_mm_bands": "dev-bands",
    "diffusive_mm":       "deviation",
    "residuals":          "residuals",
    "deviation":          "deviation",
}

# ── Shared helpers ─────────────────────────────────────────────────────────────

def _group(label: str) -> str:
    if label.startswith("ensemble"):
        return "e"
    return label[0].upper() if label and label[0].isalpha() else "?"


def _readable_label(label: str, params: dict) -> str:
    if label.startswith("ensemble"):
        tail = label[len("ensemble"):].lstrip("_").replace("_", " ")
        return f"Ensemble ({tail})" if tail else "Ensemble"
    g = _group(label)
    parts = [f"[{g}]"]
    ml = params.get("ml_model", "")
    if ml:
        parts.append(ml.upper())
    mode = _MODE_SHORT.get(params.get("mode", ""), params.get("mode", ""))
    if mode:
        parts.append(mode)
    conn = params.get("conn_mode", "")
    if conn:
        parts.append(conn.upper())
    preproc = params.get("preproc_level")
    if preproc is not None:
        parts.append(f"p{preproc}")
    if params.get("tune_hyperparams"):
        parts.append("tuned")
    fs = params.get("feature_selection")
    if fs and fs not in (None, False, "none", "None"):
        parts.append(f"fs={fs}")
    thi = params.get("thi_threshold")
    if thi is not None:
        parts.append(f"THI≥{thi}")
    return " · ".join(parts)


def _despine(ax, lw: float = 1.5):
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)
    ax.tick_params(width=lw, length=4)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_scenarios(clfs_dir: Path) -> dict:
    scenarios = {}
    if not clfs_dir.exists():
        return scenarios
    for folder in sorted(clfs_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        target = folder if (folder / "metrics.csv").exists() else next(
            (s for s in sorted(folder.iterdir())
             if s.is_dir() and (s / "metrics.csv").exists()), None)
        if target is None:
            continue
        sc = {"folder": target}
        mf = target / "metrics.csv"
        sc["df_metric"] = pd.read_csv(mf) if mf.exists() else pd.DataFrame()
        pf = target / "params.json"
        sc["params"] = json.loads(pf.read_text()) if pf.exists() else {}
        af = target / "per_fold_auc.json"
        sc["per_fold_auc"] = json.loads(af.read_text()) if af.exists() else {}
        for name in ("y", "y_prob", "y_pred", "y_prob_1", "sites"):
            p = target / f"{name}.npy"
            sc[name] = np.load(p, allow_pickle=True) if p.exists() else None
        if sc["y_prob"] is None and sc["y_prob_1"] is not None:
            sc["y_prob"] = sc["y_prob_1"]
        scenarios[folder.name] = sc
    return scenarios


def _real_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    real = df[df.get("model", pd.Series(dtype=str)).str.endswith("_real", na=False)]
    return real.iloc[0] if not real.empty else df.iloc[0]


def _perm_auc(df: pd.DataFrame) -> np.ndarray:
    perm = df[df.get("model", pd.Series(dtype=str)).str.contains("_perm_", na=False)]
    return perm["roc_auc"].dropna().values if "roc_auc" in perm.columns else np.array([])


def _resolve_sites(sc: dict, y: np.ndarray) -> np.ndarray | None:
    """
    Return a per-subject site array aligned with y.
    Tries sites.npy first; if absent, reconstructs from the feature CSV + master file.
    """
    sites = sc.get("sites")
    if sites is not None:
        return sites

    params     = sc.get("params", {})
    conn_mode  = params.get("conn_mode", "pli")
    preproc    = params.get("preproc_level", 2)
    suffix     = "_bands" if params.get("mode") == "diffusive_mm_bands" else ""
    feat_path  = TINNORM_DIR / "diffusive_mm" / f"source_preproc_{preproc}_{conn_mode}{suffix}.csv"
    master_path = Path("../material/master_clean.csv")
    try:
        df_feat = pd.read_csv(feat_path)
        df_feat.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df_master = pd.read_csv(master_path)
        df_feat = df_feat.merge(df_master[["subject_id", "site"]],
                                on="subject_id", how="left")
        sites = df_feat["site"].to_numpy()
        if len(sites) != len(y):
            print(f"  Site array length {len(sites)} ≠ y length {len(y)} — skipping per-site lines.")
            return None
        return sites
    except Exception as e:
        print(f"  Could not reconstruct sites: {e}")
        return None


# ── Figure 1: AUC ranking ─────────────────────────────────────────────────────

def build_ranking_df(scenarios: dict):
    """Return sorted DataFrame and cubehelix colour list for single-model bars."""
    rows = []
    for lbl, sc in scenarios.items():
        rr    = _real_row(sc["df_metric"])
        auc_v = rr.get("roc_auc")
        if auc_v is None:
            continue
        pfa = sc.get("per_fold_auc") or {}
        if any(isinstance(v, dict) for v in pfa.values()):
            pfa = {}
        fold_vals = [float(v) for v in pfa.values()
                     if v is not None and not np.isnan(float(v))]
        rows.append({
            "label":          lbl,
            "readable_label": _readable_label(lbl, sc.get("params", {})),
            "group":          _group(lbl),
            "is_ensemble":    _group(lbl) == "e",
            "auc":            float(auc_v),
            "fold_min":       float(min(fold_vals)) if fold_vals else float(auc_v),
            "fold_max":       float(max(fold_vals)) if fold_vals else float(auc_v),
        })

    df = pd.DataFrame(rows).sort_values("auc", ascending=True).reset_index(drop=True)
    n_single = int((~df["is_ensemble"]).sum())
    palette  = sns.cubehelix_palette(
        n_colors=n_single, start=0.5, rot=-0.6, dark=0.25, light=0.82, reverse=False
    )
    color_iter = iter(palette)
    df["color"] = [GROUP_COLORS["e"] if ens else next(color_iter)
                   for ens in df["is_ensemble"]]
    return df


def plot_auc_ranking(scenarios: dict) -> tuple:
    """
    Draw Figure 1 and return (df, best_single_label, best_single_color)
    so Figure 2 can use the matched bar colour.
    """
    df = build_ranking_df(scenarios)
    if df.empty:
        return df, None, None

    n     = len(df)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.26 + 1.5)), constrained_layout=True)

    for i, row in df.iterrows():
        is_ens = row["is_ensemble"]
        ax.barh(i, row["auc"], height=0.5,
                color=row["color"], alpha=0.88, zorder=2,
                linewidth=1.2 if is_ens else 0,
                edgecolor=row["color"] if is_ens else "none",
                linestyle="--" if is_ens else "-")

    # Error bars: fold min–max
    xerr_lo = np.clip(df["auc"].values - df["fold_min"].values, 0, None)
    xerr_hi = np.clip(df["fold_max"].values - df["auc"].values, 0, None)
    ax.errorbar(df["auc"].values, y_pos,
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="#333333", lw=1.1, capsize=2.5, zorder=3)

    ax.axvline(0.5, color=CHANCE_COLOR, lw=1.2, linestyle="--")

    # Ensemble divider
    single_pos = y_pos[~df["is_ensemble"].values]
    ens_pos    = y_pos[df["is_ensemble"].values]
    if len(single_pos) and len(ens_pos):
        div_y = (single_pos.max() + ens_pos.min()) / 2
        ax.axhline(div_y, color="#555555", lw=1.0, linestyle=":", zorder=4, alpha=0.7)
        ax.text(0.31, div_y + 0.08, "── ensemble (probability average) ──",
                fontsize=7, color="#555555", va="bottom", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["readable_label"].values, fontsize=8)
    ax.set_xlabel("ROC-AUC", fontsize=11)
    ax.set_title("All scenarios — AUC ranking\n(error bars: per-site fold range)",
                 style="italic", fontsize=11)
    ax.set_xlim(0.3, 0.90)
    ax.grid(axis="x", alpha=0.25, zorder=1)
    _despine(ax)

    _save(fig, FIGURES_DIR / "all_scenarios_auc_ranking.pdf")

    # Identify best single-model scenario and its bar colour
    single_df    = df[~df["is_ensemble"]]
    best_row     = single_df.loc[single_df["auc"].idxmax()]
    return df, best_row["label"], best_row["color"]


# ── Figure 2: Best-scenario panel (permutation + ROC per-site + PR per-site) ──

def plot_best_scenario_panel(label: str, sc: dict, bar_color):
    y      = sc.get("y")
    y_prob = sc.get("y_prob")
    df_m   = sc["df_metric"]

    if df_m.empty or y is None or y_prob is None:
        print(f"  Missing data for {label} — skipping.")
        return

    sites = _resolve_sites(sc, y)

    rr       = _real_row(df_m)
    perm_auc = _perm_auc(df_m)
    real_auc = float(rr.get("roc_auc", roc_auc_score(y, y_prob)))
    p_val    = float((perm_auc >= real_auc).mean()) if len(perm_auc) else float("nan")

    pfa = sc.get("per_fold_auc") or {}
    if any(isinstance(v, dict) for v in pfa.values()):
        pfa = {}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)

    # ── ROC — dashed per-site + bold overall ──────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2)

    if sites is not None:
        for site in sorted(np.unique(sites)):
            mask = sites == site
            if len(np.unique(y[mask])) < 2:
                continue
            try:
                fpr_s, tpr_s, _ = roc_curve(y[mask], y_prob[mask])
                ax_roc.plot(fpr_s, tpr_s, color=bar_color, lw=1.1, alpha=0.35,
                            linestyle="--", zorder=2)
            except Exception:
                pass

    fpr, tpr, _ = roc_curve(y, y_prob)
    ax_roc.plot(fpr, tpr, color=bar_color, lw=2.5, zorder=4)
    ax_roc.fill_between(fpr, tpr, alpha=0.10, color=bar_color, zorder=1)
    ax_roc.text(0.97, 0.06, f"AUC = {real_auc:.3f}",
                ha="right", va="bottom", fontsize=9, style="italic", color=bar_color,
                transform=ax_roc.transAxes)
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax_roc.set_title("ROC curve", style="italic")
    _despine(ax_roc)

    # ── PR — dashed per-site + bold overall ───────────────────────────────────
    ax_pr.axhline(y.mean(), linestyle=":", color=CHANCE_COLOR, lw=1.2)

    if sites is not None:
        for site in sorted(np.unique(sites)):
            mask = sites == site
            if len(np.unique(y[mask])) < 2:
                continue
            try:
                prec_s, rec_s, _ = precision_recall_curve(y[mask], y_prob[mask])
                ax_pr.plot(rec_s, prec_s, color=bar_color, lw=1.1, alpha=0.35,
                           linestyle="--", zorder=2)
            except Exception:
                pass

    prec, rec, _ = precision_recall_curve(y, y_prob)
    ap = auc(rec, prec)
    ax_pr.plot(rec, prec, color=bar_color, lw=2.5, zorder=4)
    ax_pr.fill_between(rec, prec, alpha=0.10, color=bar_color, zorder=1)
    ax_pr.text(0.03, 0.06, f"AP = {ap:.3f}",
               ha="left", va="bottom", fontsize=9, style="italic", color=bar_color,
               transform=ax_pr.transAxes)
    ax_pr.set(xlabel="Recall", ylabel="Precision")
    ax_pr.set_title("Precision–Recall", style="italic")
    _despine(ax_pr)

    readable = _readable_label(label, sc.get("params", {}))
    fig.suptitle(f"Best scenario: {readable}", style="italic", fontsize=11)
    _save(fig, FIGURES_DIR / f"best_scenario_panel_{label}.pdf")


# ── Figures 5–7: script-18 ROC/PR panels ──────────────────────────────────────

def _load_series(stem: str) -> list | None:
    """
    Load a *_results.npz saved by script 18 and return a series_list compatible
    with _roc_pr_panel_paper: [(label, color, y, y_prob), ...]
    """
    path = TABLES_DIR / f"{stem}_results.npz"
    if not path.exists():
        print(f"  {path.name} not found — run script 18 first.")
        return None
    npz = np.load(path, allow_pickle=True)

    # Keys are like "Residual_conn__y", "Residual_conn__y_prob", "Residual_conn__color"
    labels = sorted({k.rsplit("__", 1)[0] for k in npz.files})
    series = []
    for lbl in labels:
        y      = npz[f"{lbl}__y"]
        y_prob = npz[f"{lbl}__y_prob"]
        color  = str(npz[f"{lbl}__color"][0])
        series.append((lbl.replace("_", " "), color, y, y_prob))
    return series


def _roc_pr_panel_paper(series_list: list, title: str, fname_stem: str,
                         sites: np.ndarray | None = None):
    """
    Paper-quality ROC + PR panel with optional per-site dashed lines.
    series_list: [(label, color, y, y_prob), ...]
    sites: optional array aligned with y (same subjects for all series).
    """
    CHANCE = "#9E9E9E"
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 4.5),
                                          constrained_layout=True)
    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE, lw=1.2)
    ax_pr.axhline(
        np.mean([y.mean() for _, _, y, _ in series_list]),
        linestyle="--", color=CHANCE, lw=1.2,
    )

    for label, color, y, y_prob in series_list:
        # Per-site dashed lines (same colour as overall, thinner, transparent)
        if sites is not None:
            for site in sorted(np.unique(sites)):
                mask = sites == site
                if len(np.unique(y[mask])) < 2:
                    continue
                try:
                    fpr_s, tpr_s, _ = roc_curve(y[mask], y_prob[mask])
                    ax_roc.plot(fpr_s, tpr_s, color=color, lw=0.9,
                                alpha=0.35, linestyle="--", zorder=2)
                    prec_s, rec_s, _ = precision_recall_curve(y[mask], y_prob[mask])
                    ax_pr.plot(rec_s, prec_s, color=color, lw=0.9,
                               alpha=0.35, linestyle="--", zorder=2)
                except Exception:
                    pass

        auc_val = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2.5, zorder=4,
                    label=f"{label}  (AUC={auc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y, y_prob)
        ap = auc(rec, prec)
        ax_pr.plot(rec, prec, color=color, lw=2.5, zorder=4,
                   label=f"{label}  (AP={ap:.3f})")

    for ax, xl, yl in [(ax_roc, "False Positive Rate", "True Positive Rate"),
                        (ax_pr,  "Recall",              "Precision")]:
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(title, style="italic", fontsize=10)
        _despine(ax)

    _save(fig, FIGURES_DIR / f"{fname_stem}.pdf")


# ── SHAP data loading ──────────────────────────────────────────────────────────

TABLES_DIR  = TINNORM_DIR / "results" / "tables"
MASTER_PATH = Path("../material/master_clean.csv")


def _load_shared_sites() -> np.ndarray | None:
    """
    Reconstruct a per-subject site array aligned with the feature CSV ordering.
    Used for all three comparison panels (mode, clf, modality_ablation) since
    they all share the same subject population.
    """
    try:
        df_feat = pd.read_csv(_FEAT_FILE, usecols=["subject_ids"])
        df_feat.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df_feat["subject_id"] = df_feat["subject_id"].astype(str)
        df_master = pd.read_csv(MASTER_PATH)
        df_master["subject_id"] = df_master["subject_id"].astype(str)
        df = df_feat.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
        return df["site"].to_numpy()
    except Exception as e:
        print(f"  Could not load shared sites: {e}")
        return None

# Feature file matching the best J scenario
_FEAT_FILE = TINNORM_DIR / "diffusive_mm" / "source_preproc_2_pli_bands.csv"

_META_COLS = {"subject_id", "subject_ids", "SITE", "site", "group",
              "age", "sex", "PTA4_mean", "PTA4_HF", "THI", "TFI"}


def _load_shap_data() -> dict | None:
    """
    Load pre-saved SHAP arrays from results/tables/.
    Returns dict with keys: shap_vals, X, y, sites, fold_membership.
    Returns None if any required file is missing.
    """
    required = {
        "shap_values":        TABLES_DIR / "shap_values.npy",
        "shap_feat_names":    TABLES_DIR / "shap_feature_names.npy",
        "fold_membership":    TABLES_DIR / "shap_fold_membership.npy",
        "sites":              TABLES_DIR / "shap_sites.npy",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        print(f"  SHAP files not found — run script 16 first:\n  " +
              "\n  ".join(missing))
        return None

    shap_vals      = np.load(required["shap_values"],     allow_pickle=True)
    feat_names     = np.load(required["shap_feat_names"], allow_pickle=True)
    fold_membership= np.load(required["fold_membership"], allow_pickle=True)
    sites          = np.load(required["sites"],           allow_pickle=True)

    if not _FEAT_FILE.exists():
        print(f"  Feature file not found: {_FEAT_FILE}")
        return None

    df_feat = pd.read_csv(_FEAT_FILE)
    df_feat.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    feat_cols = [c for c in df_feat.columns if c not in _META_COLS]
    X = df_feat[feat_cols].copy()
    X.columns = feat_names[:len(feat_cols)]   # align with saved SHAP names
    y = df_feat["group"].to_numpy()

    return dict(shap_vals=shap_vals, X=X, y=y,
                sites=sites, fold_membership=fold_membership)


# ── Figure 3: SHAP beeswarm summary ───────────────────────────────────────────

def plot_shap_summary_paper(shap_vals, X, y, top_k: int = 12):
    if not _SHAP_AVAILABLE:
        return
    shap_tinnitus = shap_vals[:, :, 1]
    expl = shap.Explanation(
        values=shap_tinnitus,
        base_values=np.zeros(len(y)),
        data=X.values,
        feature_names=list(X.columns),
    )
    # Blue (low) → near-white → red (high feature value)
    cmap = LinearSegmentedColormap.from_list(
        "shap_feat", ["#2471A3", "#D6EAF8", "#F9F9F9", "#FADBD8", "#C0392B"]
    )
    # Do NOT use constrained_layout — shap calls tight_layout internally
    fig, _ = plt.subplots(figsize=(8, 6.5))
    shap.summary_plot(expl, X, max_display=top_k, cmap=cmap, alpha=0.65,
                      show=False, plot_size=None)
    ax = plt.gca()
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=4)
    _save(fig, FIGURES_DIR / "shap_summary.pdf")


# ── Figure 4: SHAP fold stability ─────────────────────────────────────────────

def plot_shap_fold_stability_paper(shap_vals, X, sites, fold_membership,
                                    top_k: int = 15):
    shap_tinnitus = shap_vals[:, :, 1]
    global_mean   = np.abs(shap_tinnitus).mean(axis=0)
    top_idxs      = np.argsort(global_mean)[::-1][:top_k]
    top_names     = list(X.columns[top_idxs])

    rows = []
    for f in np.unique(fold_membership):
        mask       = fold_membership == f
        fold_shap  = np.abs(shap_tinnitus[mask]).mean(axis=0)
        site_label = np.unique(sites[mask])[0]
        for feat, idx in zip(top_names, top_idxs):
            rows.append({"feature": feat, "mean_abs_shap": fold_shap[idx],
                         "site": site_label})
    df_stab = pd.DataFrame(rows)

    unique_sites = sorted(df_stab["site"].unique())
    site_palette = dict(zip(unique_sites,
                            sns.color_palette("husl", len(unique_sites))))
    box_edge = "#2C3E50"

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    sns.boxplot(data=df_stab, x="mean_abs_shap", y="feature",
                order=top_names, fill=False, color=box_edge,
                linewidth=1.6, fliersize=0, width=0.55,
                medianprops={"color": box_edge, "linewidth": 2.5,
                             "solid_capstyle": "round"},
                ax=ax)
    sns.stripplot(data=df_stab, x="mean_abs_shap", y="feature",
                  order=top_names, hue="site", palette=site_palette,
                  alpha=0.88, size=7.5, jitter=0.25,
                  linewidth=0.5, edgecolor="white", ax=ax)

    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=4)
    ax.set_xlabel("Mean |SHAP| (tinnitus class)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title(f"Feature importance stability across LOSO folds\n"
                 f"(top {top_k} features — each dot = one held-out site)",
                 style="italic", fontsize=10)
    ax.legend(title="Held-out site", frameon=False, fontsize="x-small",
              title_fontsize="x-small",
              bbox_to_anchor=(1.01, 1), loc="upper left")

    _save(fig, FIGURES_DIR / "shap_fold_stability.pdf")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scenarios …")
    scenarios = _load_scenarios(CLFS_DIR)
    if not scenarios:
        print("No scenarios found in", CLFS_DIR)
        raise SystemExit(1)
    print(f"  {len(scenarios)} scenarios loaded.")

    print("\nFigure 1: AUC ranking …")
    _, best_label, best_color = plot_auc_ranking(scenarios)
    print(f"  Best single-model scenario: {best_label}  (colour: {best_color})")

    print("\nFigure 2: Best-scenario diagnostic panel …")
    if best_label and best_label in scenarios:
        plot_best_scenario_panel(best_label, scenarios[best_label], best_color)
    else:
        print("  Could not identify best scenario.")

    shared_sites = _load_shared_sites()
    print(f"  Shared site array: {len(shared_sites) if shared_sites is not None else 'unavailable'} subjects")

    print("\nFigure 5: Feature representation comparison (mode_roc_pr) …")
    series = _load_series("mode")
    if series:
        _roc_pr_panel_paper(series, "Feature representation comparison (LOSO)",
                            "mode_roc_pr", sites=shared_sites)

    print("\nFigure 6: Classifier comparison (clf_roc_pr) …")
    series = _load_series("clf")
    if series:
        _roc_pr_panel_paper(series,
                            "Classifier comparison (diffusive bands PLI, LOSO)",
                            "clf_roc_pr", sites=shared_sites)

    print("\nFigure 7: EEG modality ablation …")
    series = _load_series("modality_ablation")
    if series:
        _roc_pr_panel_paper(series, "EEG modality ablation (LOSO)",
                            "modality_ablation_roc_pr", sites=shared_sites)

    print("\nFigure 3 & 4: SHAP summary + fold stability …")
    shap_data = _load_shap_data()
    if shap_data is not None:
        plot_shap_summary_paper(
            shap_data["shap_vals"], shap_data["X"], shap_data["y"], top_k=12)
        plot_shap_fold_stability_paper(
            shap_data["shap_vals"], shap_data["X"],
            shap_data["sites"], shap_data["fold_membership"], top_k=15)
    else:
        print("  Skipping SHAP figures.")

    print(f"\nDone. Figures → {FIGURES_DIR}")
