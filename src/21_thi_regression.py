"""
Predict continuous THI (and TFI if available) from multi-modal EEG deviation features
using Ridge regression and LGBMRegressor, inside site-level group CV.

Data: Load diffusive_mm features (all subjects), restrict to tinnitus group for
      regression. GroupKFold (n_splits=5) for site-level holdout.

Models:
  - Ridge(alpha=10.0)
  - LGBMRegressor (if lightgbm is installed)

Pipeline: StandardScaler → Ridge or LGBM
Metrics per fold: R², MAE, Spearman r

Saves:
  results/figures/thi_regression_scatter.pdf
  results/figures/thi_regression_cv_performance.pdf
  results/figures/thi_regression_feature_importance.pdf
  results/figures/thi_regression_residuals.pdf
  results/tables/thi_regression_metrics.csv

Run from src/:  python 21_thi_regression.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, probplot
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR = TINNORM_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "21_thi_regression"
TABLES_DIR  = RESULTS_DIR / "tables"

CSV_PATH    = TINNORM_DIR / "diffusive_mm" / "source_preproc_2_pli.csv"
MASTER_PATH = Path("../material/master_clean.csv")

CTRL_COLOR   = "#1f77b4"
TIN_COLOR    = "#C99700"
CHANCE_COLOR = "#7f8c8d"

N_SPLITS = 5


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_data():
    """Load diffusive_mm features, merge clinical scores. Return full df."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not MASTER_PATH.exists():
        print(f"ERROR: master file not found: {MASTER_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df["subject_id"] = df["subject_id"].astype(str)

    df_master = pd.read_csv(MASTER_PATH)
    df_master["subject_id"] = df_master["subject_id"].astype(str)

    # Normalise THI column name
    if "THI" not in df_master.columns and "thi_score" in df_master.columns:
        df_master.rename(columns={"thi_score": "THI"}, inplace=True)

    clinical_cols = [c for c in ["site", "THI", "TFI"] if c in df_master.columns]
    df = df.merge(df_master[["subject_id"] + clinical_cols], on="subject_id", how="left")
    if "site" in df.columns:
        df.rename(columns={"site": "SITE"}, inplace=True)

    df.drop(columns=["Unnamed: 0", "age", "sex", "PTA4_mean", "PTA4_HF",
                      "thi_score", "subject_id"],
            inplace=True, errors="ignore")

    print(f"  {len(df)} subjects  |  groups: {dict(df['group'].value_counts())}")
    return df


# ── Feature / target preparation ─────────────────────────────────────────────

def _prepare_regression_data(df, target):
    """
    Restrict to tinnitus group, drop rows with missing target.
    Returns (X, y, sites).
    """
    skip = {"group", "SITE", "THI", "TFI"}
    df_tin = df[df["group"] == 1].dropna(subset=[target]).copy()
    feat_cols = [c for c in df_tin.columns if c not in skip]
    X = df_tin[feat_cols].fillna(0)
    y = df_tin[target].values
    sites = df_tin["SITE"].values if "SITE" in df_tin.columns else np.zeros(len(df_tin))
    print(f"  {target}: N={len(y)}, mean={y.mean():.1f} ± {y.std():.1f}")
    return X, y, sites


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge(alpha=10.0)),
    ])


def _make_lgbm():
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed.")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=42, n_jobs=-1, verbose=-1)),
    ])


# ── Group K-Fold CV ───────────────────────────────────────────────────────────

def _run_cv(X, y, sites, model_factory, model_name):
    """
    GroupKFold CV.  Returns per-fold metrics, fold-level predictions,
    and list of fitted pipelines + feature names for importance.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    y_pred_all = np.full_like(y, np.nan, dtype=float)
    fitted_pipes = []
    feat_names = list(X.columns)

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=sites)):
        pipe = model_factory()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds = pipe.predict(X.iloc[test_idx])
        y_pred_all[test_idx] = preds
        fitted_pipes.append(pipe)

        if len(test_idx) >= 3:
            r2  = r2_score(y[test_idx], preds)
            mae = mean_absolute_error(y[test_idx], preds)
            sp_r, _ = spearmanr(y[test_idx], preds)
        else:
            r2, mae, sp_r = np.nan, np.nan, np.nan

        fold_rows.append({
            "fold": fold_i + 1,
            "r2": r2,
            "mae": mae,
            "spearman_r": sp_r,
        })
        print(f"    Fold {fold_i + 1}: R²={r2:.3f}, MAE={mae:.1f}, "
              f"Spearman r={sp_r:.3f}  (N={len(test_idx)})")

    df_folds = pd.DataFrame(fold_rows)
    print(f"  Mean R²={df_folds['r2'].mean():.3f} ± {df_folds['r2'].std():.3f}  "
          f"MAE={df_folds['mae'].mean():.1f} ± {df_folds['mae'].std():.1f}  "
          f"Spearman r={df_folds['spearman_r'].mean():.3f}")
    return df_folds, y_pred_all, fitted_pipes, feat_names


# ── Extract feature importances ───────────────────────────────────────────────

def _extract_importances(fitted_pipes, feat_names, model_name):
    """Return mean importance array across folds."""
    imps = []
    for pipe in fitted_pipes:
        reg = pipe.named_steps["reg"]
        if model_name == "Ridge":
            imps.append(np.abs(reg.coef_))
        elif model_name == "LGBM":
            imps.append(reg.feature_importances_.astype(float))
    if not imps:
        return np.zeros(len(feat_names))
    return np.mean(imps, axis=0)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(results, df_full, save_dir=FIGURES_DIR):
    """
    2×2 grid: [Ridge, LGBM] × [THI, TFI].
    Each panel: scatter actual vs predicted, regression line, identity line,
    R² and Spearman r annotated, colored by site.
    """
    model_names   = [k for k in results]
    targets_avail = [t for t in ["THI", "TFI"] if t in results.get(model_names[0], {})]

    if not model_names or not targets_avail:
        print("  No results to scatter-plot.")
        return

    nrows = len(model_names)
    ncols = len(targets_avail)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.8 * nrows),
                             constrained_layout=True,
                             squeeze=False)

    site_palette = sns.color_palette("tab10")
    for r_idx, mname in enumerate(model_names):
        for c_idx, target in enumerate(targets_avail):
            ax = axes[r_idx][c_idx]
            if target not in results[mname]:
                ax.set_visible(False)
                continue

            y_true = results[mname][target]["y_true"]
            y_pred = results[mname][target]["y_pred"]
            sites_vec = results[mname][target]["sites"]

            unique_sites = sorted(set(sites_vec))
            site_color_map = {s: site_palette[i % len(site_palette)]
                              for i, s in enumerate(unique_sites)}
            colors_plot = [site_color_map[s] for s in sites_vec]

            ax.scatter(y_true, y_pred, c=colors_plot, alpha=0.65, s=25)

            valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid.sum() >= 3:
                m, b = np.polyfit(y_true[valid], y_pred[valid], 1)
                x_ = np.linspace(y_true[valid].min(), y_true[valid].max(), 100)
                ax.plot(x_, m * x_ + b, color="black", lw=1.5, linestyle="--")
                # Identity line
                vmin = min(y_true[valid].min(), y_pred[valid].min())
                vmax = max(y_true[valid].max(), y_pred[valid].max())
                ax.plot([vmin, vmax], [vmin, vmax], ":", color=CHANCE_COLOR, lw=1.2)

                r2  = r2_score(y_true[valid], y_pred[valid])
                sr, _ = spearmanr(y_true[valid], y_pred[valid])
                ax.text(0.05, 0.92, f"R²={r2:.3f}\nSpearman r={sr:.3f}",
                        transform=ax.transAxes, fontsize=9, style="italic",
                        va="top")

            ax.set_xlabel(f"Actual {target}", fontsize=10)
            ax.set_ylabel(f"Predicted {target}", fontsize=10)
            ax.set_title(f"{mname} — {target}", style="italic", fontsize=10)
            ax.spines[["right", "top"]].set_visible(False)

            # Site legend (first panel only per row)
            if c_idx == 0:
                handles = [plt.Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=site_color_map[s], markersize=7,
                                      label=str(s))
                           for s in unique_sites]
                ax.legend(handles=handles, frameon=False, fontsize="x-small",
                          title="Site", title_fontsize="x-small",
                          bbox_to_anchor=(1.01, 1), loc="upper left")

    fpath = save_dir / "thi_regression_scatter.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_cv_performance(all_fold_df, save_dir=FIGURES_DIR):
    """Bar chart of per-fold R² and Spearman r, grouped by model."""
    if all_fold_df.empty:
        return

    metrics = ["r2", "spearman_r"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5),
                              constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    palette = {"Ridge": CTRL_COLOR, "LGBM": TIN_COLOR}

    for ax, met in zip(axes, metrics):
        sns.barplot(data=all_fold_df, x="fold", y=met, hue="model",
                    palette=palette, ax=ax, errorbar="sd", capsize=0.1,
                    edgecolor="white")
        ax.axhline(0, color=CHANCE_COLOR, linestyle="--", lw=1)
        ax.set_title(met.replace("_", " ").title(), style="italic", fontsize=10)
        ax.set_xlabel("Fold", fontsize=9)
        ax.set_ylabel(met, fontsize=9)
        ax.legend(frameon=False, fontsize="small")
        sns.despine(ax=ax)

    fig.suptitle("CV performance per fold", style="italic", fontsize=11)
    fpath = save_dir / "thi_regression_cv_performance.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_feature_importance(importance_dict, top_n=20, save_dir=FIGURES_DIR):
    """Side-by-side horizontal bar charts of top-20 feature importances."""
    model_names = list(importance_dict.keys())
    if not model_names:
        return

    ncols = len(model_names)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, mname in zip(axes, model_names):
        names = importance_dict[mname]["names"]
        vals  = importance_dict[mname]["values"]
        if len(vals) == 0:
            ax.set_visible(False)
            continue
        idx = np.argsort(vals)[::-1][:top_n]
        top_vals  = vals[idx][::-1]
        top_names = [names[i] for i in idx][::-1]

        y_pos = np.arange(len(top_names))
        color = CTRL_COLOR if mname == "Ridge" else TIN_COLOR
        ax.barh(y_pos, top_vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=7)
        ax.set_xlabel("Mean |coefficient|" if mname == "Ridge" else "Mean feature importance",
                      fontsize=9)
        ax.set_title(f"{mname} — Top {top_n} features", style="italic", fontsize=10)
        sns.despine(ax=ax)

    fpath = save_dir / "thi_regression_feature_importance.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def plot_residuals(y_true, y_pred, model_name, target, save_dir=FIGURES_DIR):
    """Residuals histogram + Q-Q plot side by side for the best model."""
    residuals = y_true - y_pred
    valid = ~np.isnan(residuals)
    residuals = residuals[valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    ax1.hist(residuals, bins=25, color=TIN_COLOR, edgecolor="white", alpha=0.8)
    ax1.axvline(0, color=CHANCE_COLOR, linestyle="--", lw=1.5)
    ax1.set_xlabel("Residual (actual − predicted)", fontsize=10)
    ax1.set_ylabel("Count", fontsize=10)
    ax1.set_title(f"Residuals — {model_name} ({target})", style="italic", fontsize=10)
    sns.despine(ax=ax1)

    (osm, osr), (slope, intercept, r) = probplot(residuals, fit=True)
    ax2.scatter(osm, osr, color=CTRL_COLOR, alpha=0.5, s=15)
    ax2.plot(osm, slope * np.array(osm) + intercept, color="black", lw=1.5)
    ax2.set_xlabel("Theoretical quantiles", fontsize=10)
    ax2.set_ylabel("Sample quantiles", fontsize=10)
    ax2.set_title(f"Q-Q plot — {model_name} ({target})", style="italic", fontsize=10)
    sns.despine(ax=ax2)

    fpath = save_dir / "thi_regression_residuals.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features …")
    df = _load_data()

    targets = [t for t in ["THI", "TFI"] if t in df.columns]
    if not targets:
        print("ERROR: Neither THI nor TFI found in merged DataFrame.")
        sys.exit(1)

    print(f"  Regression targets available: {targets}")

    # Model factories
    model_factories = {"Ridge": _make_ridge}
    if HAS_LGBM:
        model_factories["LGBM"] = _make_lgbm

    results          = {}   # mname → target → dict
    all_fold_rows    = []
    importance_dict  = {}

    for mname, factory in model_factories.items():
        print(f"\n── {mname} ──────────────────────────────────────────────")
        results[mname] = {}
        importance_list = []

        for target in targets:
            print(f"  Target: {target}")
            try:
                X, y, sites = _prepare_regression_data(df, target)
            except Exception as e:
                print(f"    Skipping {target}: {e}")
                continue

            if len(np.unique(sites)) < 2:
                print(f"    Skipping {target}: fewer than 2 unique sites.")
                continue

            df_folds, y_pred_all, fitted_pipes, feat_names = _run_cv(
                X, y, sites, factory, mname)

            df_folds["model"]  = mname
            df_folds["target"] = target
            all_fold_rows.append(df_folds)

            results[mname][target] = {
                "y_true": y,
                "y_pred": y_pred_all,
                "sites":  sites,
            }

            importance_list.append(_extract_importances(fitted_pipes, feat_names, mname))

        if importance_list:
            mean_imp = np.mean(importance_list, axis=0)
            importance_dict[mname] = {
                "names":  feat_names,
                "values": mean_imp,
            }

    # ── Save metrics table ─────────────────────────────────────────────────
    if all_fold_rows:
        df_metrics = pd.concat(all_fold_rows, ignore_index=True)
        df_metrics.to_csv(TABLES_DIR / "thi_regression_metrics.csv", index=False)
        print(f"\n  Table → {TABLES_DIR / 'thi_regression_metrics.csv'}")
    else:
        df_metrics = pd.DataFrame()

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating figures ──")

    print("  Scatter actual vs predicted …")
    plot_scatter(results, df, save_dir=FIGURES_DIR)

    print("  CV performance bar chart …")
    if not df_metrics.empty:
        plot_cv_performance(df_metrics, save_dir=FIGURES_DIR)

    print("  Feature importance …")
    if importance_dict:
        plot_feature_importance(importance_dict, top_n=20, save_dir=FIGURES_DIR)

    # Residuals for the best model (Ridge on THI if available, else first)
    best_model  = None
    best_target = None
    best_r2     = -np.inf
    for mname in results:
        for target in results[mname]:
            y_true = results[mname][target]["y_true"]
            y_pred = results[mname][target]["y_pred"]
            valid  = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if valid.sum() >= 3:
                r2 = r2_score(y_true[valid], y_pred[valid])
                if r2 > best_r2:
                    best_r2     = r2
                    best_model  = mname
                    best_target = target

    if best_model is not None:
        print(f"  Residuals for best model: {best_model} / {best_target} (R²={best_r2:.3f}) …")
        y_t = results[best_model][best_target]["y_true"]
        y_p = results[best_model][best_target]["y_pred"]
        plot_residuals(y_t, y_p, best_model, best_target, save_dir=FIGURES_DIR)

    print(f"\nAll outputs → {RESULTS_DIR}")
    print("Done.")
