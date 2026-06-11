"""
Plot and summarize all classification results stored under clfs/.

Reads metrics.csv, per_fold_auc.json, params.json, and .npy arrays from
each scenario folder produced by 13_compare_clfs.py and produces:

  A. Permutation-test histograms + ROC + PR  (one PDF per scenario)
  B. Within-group ROC / PR comparison panels  (one PDF per group A–I)
  C. Overall AUC ranking chart  (all scenarios, sorted)
  D. Per-fold AUC heatmap  (scenarios × sites)
  E. Calibration curves  (top N scenarios by AUC)
  F. Confusion matrix  (best scenario)
  G. Metric scatter  (AUC vs balanced accuracy, all scenarios)
  H. Summary metrics table  (CSV + console)

Run from src/:  python 14_plot_clf_results.py
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    average_precision_score, confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
# TINNORM_DIR = Path("/data/Tinnorm")  # ← uncomment on VM
CLFS_DIR    = TINNORM_DIR / "clfs"
FIGURES_DIR = TINNORM_DIR / "results" / "figures" / "14_clf_results"
TABLES_DIR  = TINNORM_DIR / "results" / "tables"

N_CALIBRATION = 6   # top-N scenarios shown in calibration plot

# ── Colour scheme ──────────────────────────────────────────────────────────────

GROUP_COLORS = {
    "A": "#2196F3",   # blue   — preprocessing
    "B": "#E91E63",   # pink   — classifiers
    "C": "#009688",   # teal   — connectivity
    "D": "#FF9800",   # orange — dimensionality
    "E": "#9C27B0",   # purple — feature selection
    "F": "#F44336",   # red    — tuning
    "G": "#4CAF50",   # green  — THI threshold
    "H": "#795548",   # brown  — modality
    "I": "#607D8B",   # grey   — comparison
    "J": "#FF5722",   # deep orange — bands × connectivity + tuning
    "e": "#212121",   # near-black  — ensemble
}
GROUP_NAMES = {
    "A": "Preprocessing levels",
    "B": "Classifiers",
    "C": "Connectivity measures",
    "D": "Feature dimensionality",
    "E": "Feature selection",
    "F": "Hyperparameter tuning",
    "G": "THI severity threshold",
    "H": "Individual modalities",
    "I": "Residual vs Deviation",
    "J": "Bands × connectivity + tuning",
    "e": "Ensemble",
}
CHANCE_COLOR = "#9E9E9E"


# ── Data loading ───────────────────────────────────────────────────────────────

def _iter_scenarios(clfs_dir: Path):
    """Yield (label, folder) for every scenario with a metrics.csv."""
    if not clfs_dir.exists():
        return
    for folder in sorted(clfs_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if (folder / "metrics.csv").exists():      # flat (new) structure
            yield folder.name, folder
        else:                                       # old timestamped structure
            for sub in sorted(folder.iterdir()):
                if sub.is_dir() and (sub / "metrics.csv").exists():
                    yield folder.name, sub
                    break


def _load_scenario(folder: Path) -> dict:
    out = {"folder": folder}

    mf = folder / "metrics.csv"
    out["df_metric"] = pd.read_csv(mf) if mf.exists() else pd.DataFrame()

    pf = folder / "params.json"
    out["params"] = json.loads(pf.read_text()) if pf.exists() else {}

    af = folder / "per_fold_auc.json"
    out["per_fold_auc"] = json.loads(af.read_text()) if af.exists() else {}

    for name in ("y", "y_prob", "y_pred", "y_prob_1", "y_prob_2", "sites"):
        p = folder / f"{name}.npy"
        out[name] = np.load(p, allow_pickle=True) if p.exists() else None

    # Canonicalise: y_prob → prefer y_prob, then y_prob_1
    if out["y_prob"] is None and out["y_prob_1"] is not None:
        out["y_prob"] = out["y_prob_1"]

    return out


def _real_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    real = df[df.get("model", pd.Series(dtype=str)).str.endswith("_real", na=False)]
    return real.iloc[0] if not real.empty else df.iloc[0]


def _perm_auc(df: pd.DataFrame) -> np.ndarray:
    perm = df[df.get("model", pd.Series(dtype=str)).str.contains("_perm_", na=False)]
    return perm["roc_auc"].dropna().values if "roc_auc" in perm.columns else np.array([])


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _short(label: str) -> str:
    """Remove group prefix for legend labels."""
    parts = label.split("_", 1)
    return parts[1] if len(parts) > 1 else label


def _group(label: str) -> str:
    if label.startswith("ensemble"):
        return "e"
    return label[0].upper() if label and label[0].isalpha() else "?"


_MODE_SHORT = {
    "diffusive_mm_bands": "dev-bands",
    "diffusive_mm":       "deviation",
    "residuals":          "residuals",
    "deviation":          "deviation",
}

def _readable_label(label: str, params: dict) -> str:
    """Build a compact human-readable ytick label from params.json fields."""
    if label.startswith("ensemble"):
        tail = label[len("ensemble"):].lstrip("_").replace("_", " ")
        return f"Ensemble ({tail})" if tail else "Ensemble"

    group = _group(label)
    parts = [f"[{group}]"]

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


# ── A. Per-scenario permutation + ROC + PR panels ─────────────────────────────

def plot_permutation_panel(label: str, sc: dict):
    y      = sc.get("y")
    y_prob = sc.get("y_prob")
    df_m   = sc["df_metric"]

    if df_m.empty or y is None or y_prob is None:
        return

    rr       = _real_row(df_m)
    perm_auc = _perm_auc(df_m)
    real_auc = float(rr.get("roc_auc", roc_auc_score(y, y_prob)))
    p_val    = float((perm_auc >= real_auc).mean()) if len(perm_auc) else float("nan")
    color    = GROUP_COLORS.get(_group(label), "#1f77b4")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    # (a) Permutation histogram
    ax = axes[0]
    if len(perm_auc):
        ax.hist(perm_auc, bins=30, color="lightgray", edgecolor="white",
                linewidth=0.4, alpha=0.9, zorder=1)
    ax.axvline(0.5, color=CHANCE_COLOR, lw=1.2, linestyle=":", alpha=0.8, label="Chance")
    ax.axvline(real_auc, color=color, lw=2.5, linestyle="--", zorder=2)
    ylim = ax.get_ylim()
    ax.text(real_auc + 0.003, ylim[1] * 0.94,
            f"AUC = {real_auc:.3f}\np = {p_val:.3f}",
            fontsize=9, style="italic", color=color, va="top")
    ax.set_xlabel("AUC (permuted labels)")
    ax.set_ylabel("Count")
    ax.set_title("Permutation null distribution", style="italic")
    _despine(ax)

    # (b) ROC
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    ax.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {real_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.10, color=color)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("ROC curve", style="italic")
    ax.legend(frameon=False, fontsize=10)
    _despine(ax)

    # (c) Precision–Recall
    ax = axes[2]
    prec, rec, _ = precision_recall_curve(y, y_prob)
    ap = auc(rec, prec)
    ax.axhline(y.mean(), linestyle=":", color=CHANCE_COLOR, lw=1.2, label="Chance")
    ax.plot(rec, prec, color=color, lw=2.5, label=f"AP = {ap:.3f}")
    ax.fill_between(rec, prec, alpha=0.10, color=color)
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision–Recall", style="italic")
    ax.legend(frameon=False, fontsize=10)
    _despine(ax)

    fig.suptitle(f"Scenario: {label}", style="italic", fontsize=11)
    _save(fig, FIGURES_DIR / f"permutation_{label}.pdf")


# ── B. Within-group ROC + PR comparison panels ────────────────────────────────

def plot_group_roc_pr(group_letter: str, scenarios: dict):
    members = [(lbl, sc) for lbl, sc in scenarios.items()
               if _group(lbl) == group_letter.upper()
               and sc.get("y") is not None and sc.get("y_prob") is not None]
    if not members:
        return

    n = len(members)
    colors = sns.color_palette("husl", n)
    group_name = GROUP_NAMES.get(group_letter.upper(), group_letter)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2)
    chance_line_drawn = False

    for (lbl, sc), color in zip(members, colors):
        y      = sc["y"]
        y_prob = sc["y_prob"]
        sites  = sc.get("sites")
        rr     = _real_row(sc["df_metric"])
        auc_v  = float(rr.get("roc_auc", roc_auc_score(y, y_prob)))

        # Thin per-site lines
        if sites is not None:
            for site in np.unique(sites):
                mask = sites == site
                if len(np.unique(y[mask])) < 2:
                    continue
                try:
                    fpr_s, tpr_s, _ = roc_curve(y[mask], y_prob[mask])
                    ax_roc.plot(fpr_s, tpr_s, color=color, lw=0.9, alpha=0.25)
                    prec_s, rec_s, _ = precision_recall_curve(y[mask], y_prob[mask])
                    ax_pr.plot(rec_s, prec_s, color=color, lw=0.9, alpha=0.25)
                except Exception:
                    pass

        # Bold average line (with legend entry)
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2.2,
                    label=f"{_short(lbl)}  AUC={auc_v:.3f}")

        prec, rec, _ = precision_recall_curve(y, y_prob)
        ap = auc(rec, prec)
        ax_pr.plot(rec, prec, color=color, lw=2.2,
                   label=f"{_short(lbl)}  AP={ap:.3f}")

        if not chance_line_drawn:
            ax_pr.axhline(y.mean(), linestyle=":", color=CHANCE_COLOR, lw=1.2, label="Chance")
            chance_line_drawn = True

    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=f"{group_name} — ROC")
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title=f"{group_name} — Precision–Recall")
    for ax in [ax_roc, ax_pr]:
        ax.set_title(ax.get_title(), style="italic")
        ax.legend(frameon=False, fontsize="small", loc="lower right" if ax is ax_roc else "upper right")
        _despine(ax)

    _save(fig, FIGURES_DIR / f"group_{group_letter.upper()}_roc_pr.pdf")


# ── C. Overall AUC ranking ────────────────────────────────────────────────────

def plot_auc_ranking(scenarios: dict):
    rows = []
    for lbl, sc in scenarios.items():
        rr = _real_row(sc["df_metric"])
        auc_v = rr.get("roc_auc")
        if auc_v is None:
            continue
        raw_fa = sc.get("per_fold_auc") or {}
        # Skip comparison scenarios whose per_fold_auc stores nested dicts
        if any(isinstance(v, dict) for v in raw_fa.values()):
            raw_fa = {}
        fa = {k: v for k, v in raw_fa.items()
              if v is not None and not np.isnan(float(v))}
        fold_vals = [float(v) for v in fa.values()]
        rows.append({
            "label":          lbl,
            "readable_label": _readable_label(lbl, sc.get("params", {})),
            "group": _group(lbl),
            "auc":       float(auc_v),
            "bal_acc":   float(rr["balanced_accuracy"]) if "balanced_accuracy" in rr else None,
            "fold_mean": float(np.mean(fold_vals)) if fold_vals else float(auc_v),
            "fold_std":  float(np.std(fold_vals))  if fold_vals else 0.0,
            "fold_min":  float(min(fold_vals))      if fold_vals else float(auc_v),
            "fold_max":  float(max(fold_vals))      if fold_vals else float(auc_v),
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("auc", ascending=True).reset_index(drop=True)
    df["is_ensemble"] = df["group"] == "e"
    n = len(df)

    # Cubehelix gradient for single-model rows; ensemble gets fixed near-black
    n_single = int((~df["is_ensemble"]).sum())
    single_colors = sns.cubehelix_palette(
        n_colors=n_single, start=0.5, rot=-0.6, dark=0.25, light=0.82, reverse=False
    )
    ens_color  = GROUP_COLORS["e"]
    color_iter = iter(single_colors)
    bar_colors = [ens_color if ens else next(color_iter) for ens in df["is_ensemble"]]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.26 + 1.5)), constrained_layout=True)
    y_pos = np.arange(n)

    for i, (auc_v, is_ens, color) in enumerate(
            zip(df["auc"].values, df["is_ensemble"].values, bar_colors)):
        ax.barh(i, auc_v, height=0.5, color=color, alpha=0.88, zorder=2,
                linewidth=1.2 if is_ens else 0,
                edgecolor=color if is_ens else "none",
                linestyle="--" if is_ens else "-")

    # Error bars: min–max across folds
    xerr_lo = np.clip(df["auc"].values - df["fold_min"].values, 0, None)
    xerr_hi = np.clip(df["fold_max"].values - df["auc"].values, 0, None)
    ax.errorbar(df["auc"].values, y_pos,
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="#333333", lw=1.1, capsize=2.5, zorder=3)

    ax.axvline(0.5, color=CHANCE_COLOR, lw=1.2, linestyle="--")

    # Divider line between highest single-model bar and lowest ensemble bar
    single_positions = y_pos[~df["is_ensemble"].values]
    ens_positions    = y_pos[df["is_ensemble"].values]
    if len(single_positions) and len(ens_positions):
        divider_y = (single_positions.max() + ens_positions.min()) / 2
        ax.axhline(divider_y, color="#555555", lw=1.0, linestyle=":",
                   zorder=4, alpha=0.7)
        ax.text(0.31, divider_y + 0.08, "── ensemble (probability average) ──",
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


# ── D. Per-fold AUC heatmap ───────────────────────────────────────────────────

def plot_fold_auc_heatmap(scenarios: dict):
    rows, all_sites = {}, set()
    for lbl, sc in scenarios.items():
        fa = sc.get("per_fold_auc")
        if not fa or not isinstance(fa, dict):
            continue
        # Skip nested dicts (comparison scenarios store residual/deviation sub-dicts)
        if any(isinstance(v, dict) for v in fa.values()):
            continue
        rows[lbl] = fa
        all_sites.update(fa.keys())

    if not rows:
        print("  No per-fold AUC data — skipping heatmap.")
        return

    sites = sorted(all_sites)
    df = pd.DataFrame(
        {lbl: [float(rows[lbl].get(s, float("nan"))) for s in sites] for lbl in rows},
        index=sites,
    ).T

    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(
        figsize=(max(5, n_cols * 1.0 + 1), max(4, n_rows * 0.42 + 1.5)),
        constrained_layout=True,
    )
    sns.heatmap(
        df, ax=ax, cmap="RdYlGn", vmin=0.3, vmax=1.0,
        annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Held-out AUC", "shrink": 0.7},
    )
    ax.set_xlabel("Site", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Per-fold AUC heatmap — scenarios × sites",
                 style="italic", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", rotation=30, labelsize=9)

    _save(fig, FIGURES_DIR / "per_fold_auc_heatmap.pdf")


# ── E. Calibration curves ─────────────────────────────────────────────────────

def plot_calibration(scenarios: dict, top_n: int = N_CALIBRATION):
    ranked = sorted(
        [(lbl, sc) for lbl, sc in scenarios.items()
         if sc.get("y") is not None and sc.get("y_prob") is not None],
        key=lambda x: float(_real_row(x[1]["df_metric"]).get("roc_auc", 0)),
        reverse=True,
    )[:top_n]

    if not ranked:
        return

    colors = sns.color_palette("husl", len(ranked))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.4, label="Perfect calibration")

    for (lbl, sc), color in zip(ranked, colors):
        try:
            frac, mean_pred = calibration_curve(sc["y"], sc["y_prob"], n_bins=8)
            ax.plot(mean_pred, frac, "o-", markersize=5, color=color,
                    lw=1.8, label=_short(lbl))
        except Exception:
            pass

    ax.set(xlabel="Mean predicted probability",
           ylabel="Fraction of positives (tinnitus)")
    ax.set_title(f"Calibration curves — top {len(ranked)} scenarios by AUC",
                 style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize="small")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    _despine(ax)

    _save(fig, FIGURES_DIR / "calibration_top_scenarios.pdf")


# ── F. Confusion matrix for best scenario ─────────────────────────────────────

def plot_best_confusion(scenarios: dict):
    ranked = sorted(
        [(lbl, sc) for lbl, sc in scenarios.items()
         if sc.get("y") is not None and sc.get("y_pred") is not None],
        key=lambda x: float(_real_row(x[1]["df_metric"]).get("roc_auc", 0)),
        reverse=True,
    )
    if not ranked:
        return

    lbl, sc = ranked[0]
    y      = sc["y"]
    y_pred = sc["y_pred"]
    rr     = _real_row(sc["df_metric"])
    color  = GROUP_COLORS.get(_group(lbl), "#2196F3")

    cm = confusion_matrix(y, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.5, 4.2), constrained_layout=True)
    sns.heatmap(
        cm, ax=ax, annot=True, fmt=".2f",
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Proportion", "shrink": 0.85},
        xticklabels=["Control", "Tinnitus"],
        yticklabels=["Control", "Tinnitus"],
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    auc_str = f"AUC={float(rr['roc_auc']):.3f}  " if rr.get("roc_auc") is not None else ""
    ba_str  = f"BalAcc={float(rr['balanced_accuracy']):.3f}" if rr.get("balanced_accuracy") is not None else ""
    ax.set_title(f"Confusion matrix — {lbl}\n{auc_str}{ba_str}", style="italic", fontsize=10)

    _save(fig, FIGURES_DIR / f"confusion_best_{lbl}.pdf")


# ── F2. Per-site ROC + PR for best scenario ───────────────────────────────────

def plot_best_site_roc_pr(scenarios: dict):
    """
    For the best non-ensemble scenario: 7 thin per-site ROC/PR curves
    (tab10 colours) + 1 bold average line.  Sites are recovered from
    sites.npy if present, otherwise reconstructed from the feature CSV.
    """
    ranked = sorted(
        [(lbl, sc) for lbl, sc in scenarios.items()
         if sc.get("y") is not None and sc.get("y_prob") is not None
         and not lbl.startswith("ensemble")],
        key=lambda x: float(_real_row(x[1]["df_metric"]).get("roc_auc", 0)),
        reverse=True,
    )
    if not ranked:
        return

    lbl, sc = ranked[0]
    y      = sc["y"]
    y_prob = sc["y_prob"]
    sites  = sc.get("sites")

    # Reconstruct sites from feature CSV when sites.npy was not saved
    if sites is None:
        params     = sc.get("params", {})
        conn_mode  = params.get("conn_mode", "pli")
        preproc    = params.get("preproc_level", 2)
        suffix     = "_bands" if params.get("mode") == "diffusive_mm_bands" else ""
        feat_path  = TINNORM_DIR / "diffusive_mm" / f"source_preproc_{preproc}_{conn_mode}{suffix}.csv"
        master_path = Path("../material/master_clean.csv")
        try:
            df_feat   = pd.read_csv(feat_path)
            df_feat.rename(columns={"subject_ids": "subject_id"}, inplace=True)
            df_master = pd.read_csv(master_path)
            df_feat   = df_feat.merge(df_master[["subject_id", "site"]],
                                      on="subject_id", how="left")
            sites = df_feat["site"].to_numpy()
        except Exception as e:
            print(f"  Could not reconstruct sites for {lbl}: {e}")
            return

    unique_sites = sorted(np.unique(sites))
    palette      = sns.color_palette("tab10", len(unique_sites))
    site_color   = {s: palette[i] for i, s in enumerate(unique_sites)}

    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)
    tprs, precs = [], []

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5),
                                          constrained_layout=True)
    ax_roc.plot([0, 1], [0, 1], "--", color=CHANCE_COLOR, lw=1.2, label="Chance")
    ax_pr.axhline(y.mean(), linestyle=":", color=CHANCE_COLOR, lw=1.2, label="Chance")

    pfa = sc.get("per_fold_auc") or {}
    if any(isinstance(v, dict) for v in pfa.values()):
        pfa = {}

    for site in unique_sites:
        mask = sites == site
        if len(np.unique(y[mask])) < 2:
            continue
        try:
            color  = site_color[site]
            auc_s  = float(pfa[site]) if site in pfa else roc_auc_score(y[mask], y_prob[mask])
            ap_s   = average_precision_score(y[mask], y_prob[mask])

            fpr_s, tpr_s, _ = roc_curve(y[mask], y_prob[mask])
            ax_roc.plot(fpr_s, tpr_s, color=color, lw=1.5, alpha=0.55,
                        label=f"{site}  AUC={auc_s:.3f}")
            tprs.append(np.interp(mean_fpr, fpr_s, tpr_s))

            prec_s, rec_s, _ = precision_recall_curve(y[mask], y_prob[mask])
            ax_pr.plot(rec_s, prec_s, color=color, lw=1.5, alpha=0.55,
                       label=f"{site}  AP={ap_s:.3f}")
            precs.append(np.interp(mean_rec, rec_s[::-1], prec_s[::-1]))
        except Exception:
            pass

    # Bold average
    overall_auc = roc_auc_score(y, y_prob)
    overall_ap  = average_precision_score(y, y_prob)
    if tprs:
        ax_roc.plot(mean_fpr, np.mean(tprs, axis=0), color="black", lw=2.8,
                    label=f"Average  AUC={overall_auc:.3f}")
    if precs:
        ax_pr.plot(mean_rec, np.mean(precs, axis=0), color="black", lw=2.8,
                   label=f"Average  AP={overall_ap:.3f}")

    for ax, xl, yl, ttl in [
        (ax_roc, "False Positive Rate", "True Positive Rate",
         f"ROC — {lbl}  (per site)"),
        (ax_pr,  "Recall",              "Precision",
         f"PR  — {lbl}  (per site)"),
    ]:
        ax.set(xlabel=xl, ylabel=yl)
        ax.set_title(ttl, style="italic", fontsize=10)
        ax.legend(frameon=False, fontsize="small",
                  bbox_to_anchor=(1.01, 1), loc="upper left")
        _despine(ax)

    _save(fig, FIGURES_DIR / f"site_roc_pr_{lbl}.pdf")


# ── G. AUC vs Balanced-Accuracy scatter ──────────────────────────────────────

def plot_auc_balacc_scatter(scenarios: dict):
    rows = []
    for lbl, sc in scenarios.items():
        rr = _real_row(sc["df_metric"])
        auc_v = rr.get("roc_auc")
        ba    = rr.get("balanced_accuracy")
        if auc_v is None or ba is None:
            continue
        grp = _group(lbl)
        if grp == "A":          # preprocessing levels — excluded (all near chance)
            continue
        rows.append({"label": lbl, "group": grp,
                     "readable": _readable_label(lbl, sc.get("params", {})),
                     "auc": float(auc_v), "bal_acc": float(ba)})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for g, gdf in df.groupby("group"):
        color = GROUP_COLORS.get(g, "#607D8B")
        ax.scatter(gdf["bal_acc"], gdf["auc"], color=color, s=70,
                   alpha=0.88, zorder=3, label=f"{g}: {GROUP_NAMES.get(g, g)}")
        for _, row in gdf.iterrows():
            ax.text(row["bal_acc"] + 0.003, row["auc"] + 0.001,
                    row["readable"], fontsize=5.5, color=color,
                    style="italic", va="bottom", zorder=4,
                    clip_on=True)

    ax.axhline(0.5, color=CHANCE_COLOR, lw=1, linestyle="--", zorder=1)
    ax.axvline(0.5, color=CHANCE_COLOR, lw=1, linestyle="--", zorder=1)
    ax.set(xlabel="Balanced Accuracy", ylabel="ROC-AUC")
    ax.set_title("AUC vs Balanced Accuracy — all scenarios",
                 style="italic", fontsize=11)
    ax.legend(frameon=False, fontsize="x-small",
              bbox_to_anchor=(1.01, 1), loc="upper left")
    _despine(ax)

    _save(fig, FIGURES_DIR / "auc_vs_balacc_scatter.pdf")


# ── H. Summary table ──────────────────────────────────────────────────────────

def build_summary_table(scenarios: dict) -> pd.DataFrame:
    rows = []
    for lbl, sc in scenarios.items():
        rr = _real_row(sc["df_metric"])
        fa = sc.get("per_fold_auc") or {}
        # Skip nested dicts (comparison mode stores {'residual': {}, 'deviation': {}})
        if any(isinstance(v, dict) for v in fa.values()):
            fa = {}
        fold_vals = [float(v) for v in fa.values()
                     if v is not None and not np.isnan(float(v))]

        row = {
            "scenario":         lbl,
            "group":            _group(lbl),
            "roc_auc":          rr.get("roc_auc"),
            "balanced_accuracy":rr.get("balanced_accuracy"),
            "f1-score":         rr.get("f1-score"),
            "fold_auc_mean":    float(np.mean(fold_vals)) if fold_vals else None,
            "fold_auc_std":     float(np.std(fold_vals))  if fold_vals else None,
            "fold_auc_min":     float(min(fold_vals))     if fold_vals else None,
            "fold_auc_max":     float(max(fold_vals))     if fold_vals else None,
        }
        params = sc.get("params", {})
        for key in ("ml_model", "mode", "conn_mode", "preproc_level",
                    "feature_selection", "tune_hyperparams",
                    "thi_threshold", "actual_n_features"):
            row[key] = params.get(key)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scenarios …")
    scenarios = {}
    for label, folder in _iter_scenarios(CLFS_DIR):
        sc = _load_scenario(folder)
        scenarios[label] = sc
        rr  = _real_row(sc["df_metric"])
        auc_v = rr.get("roc_auc")
        print(f"  {label:50s}  AUC={auc_v:.3f}" if auc_v is not None
              else f"  {label} (no AUC)")

    if not scenarios:
        print("No scenario results found in", CLFS_DIR)
        raise SystemExit(0)

    print(f"\n{len(scenarios)} scenarios loaded.\n")

    # ── A. Per-scenario permutation + ROC + PR ────────────────────────────
    print("A. Permutation panels …")
    for lbl, sc in scenarios.items():
        if len(_perm_auc(sc["df_metric"])) > 0:
            plot_permutation_panel(lbl, sc)

    # ── B. Within-group ROC + PR ──────────────────────────────────────────
    print("\nB. Group ROC + PR comparison …")
    groups = sorted({_group(lbl) for lbl in scenarios if _group(lbl) != "?"})
    for g in groups:
        print(f"  Group {g} …")
        plot_group_roc_pr(g, scenarios)

    # ── C. Overall AUC ranking ────────────────────────────────────────────
    print("\nC. Overall AUC ranking …")
    plot_auc_ranking(scenarios)

    # ── D. Per-fold AUC heatmap ───────────────────────────────────────────
    print("\nD. Per-fold AUC heatmap …")
    plot_fold_auc_heatmap(scenarios)

    # ── E. Calibration curves ─────────────────────────────────────────────
    print("\nE. Calibration curves …")
    plot_calibration(scenarios, top_n=N_CALIBRATION)

    # ── F. Confusion matrix (best) ────────────────────────────────────────
    print("\nF. Confusion matrix for best scenario …")
    plot_best_confusion(scenarios)

    # ── F2. Per-site ROC + PR for best scenario ───────────────────────────
    print("\nF2. Per-site ROC + PR for best scenario …")
    plot_best_site_roc_pr(scenarios)

    # ── G. AUC vs BalAcc scatter ──────────────────────────────────────────
    print("\nG. AUC vs balanced-accuracy scatter …")
    plot_auc_balacc_scatter(scenarios)

    # ── H. Summary table ──────────────────────────────────────────────────
    print("\nH. Summary table …")
    df_sum = build_summary_table(scenarios)
    tbl_path = TABLES_DIR / "all_scenario_metrics.csv"
    df_sum.to_csv(tbl_path, index=False)
    print(f"  Saved → {tbl_path}")

    show_cols = ["scenario", "roc_auc", "balanced_accuracy",
                 "fold_auc_mean", "fold_auc_std"]
    print(df_sum[[c for c in show_cols if c in df_sum.columns]]
          .sort_values("roc_auc", ascending=False)
          .to_string(index=False))

    print(f"\nAll figures → {FIGURES_DIR}")
    print("Done.")
