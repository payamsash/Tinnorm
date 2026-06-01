"""
Model explainability: permutation importance, PDP, and LIME.

All computed on **held-out** test folds to avoid optimistic in-sample bias.

  A. Permutation importance (LOSO, held-out folds)
       → results/figures/perm_importance.pdf

  B. Partial Dependence Plots (LOSO folds, top features)
       → results/figures/pdp_top_features.pdf

  C. LIME explanation for the highest-confidence tinnitus prediction
     per site (using the fold model for that site)
       → results/figures/lime_<site>.pdf

Run from src/:  python 20_lime_pdp.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import roc_auc_score

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("lime not installed — LIME section will be skipped.  pip install lime")

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
RESULTS_DIR  = TINNORM_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"

SPACE        = "source"
PREPROC      = 2
CONN_MODE    = "coh"
RANDOM_STATE = 42
N_TOP        = 10          # features shown in permutation importance
N_PDP_FEAT   = 4           # features shown in PDP
PDP_GRID_RES = 50
CTRL_COLOR   = "#1f77b4"
TIN_COLOR    = "#C99700"


# ── Data ──────────────────────────────────────────────────────────────────────

def _load_diffusive():
    df_master = pd.read_csv("../material/master_clean.csv")
    fname = TINNORM_DIR / "diffusive_mm" / f"{SPACE}_preproc_{PREPROC}_{CONN_MODE}.csv"
    df = pd.read_csv(fname)
    df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
    df = df.merge(df_master[["subject_id", "site"]], on="subject_id", how="left")
    df.rename(columns={"site": "SITE"}, inplace=True)
    subject_ids = df["subject_id"].to_numpy()
    sites       = df["SITE"].to_numpy()
    y           = df["group"].to_numpy()
    drop_cols   = ["group", "Unnamed: 0", "subject_id", "SITE",
                   "age", "sex", "PTA4_mean", "PTA4_HF", "thi_score", "THI"]
    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y, sites, subject_ids


# ── Model ─────────────────────────────────────────────────────────────────────

def _make_pipe():
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ── A. Permutation importance (LOSO, held-out test folds) ────────────────────

def compute_permutation_importance(X, y, sites, n_repeats: int = 10):
    """
    Run permutation importance on each held-out test fold independently,
    then pool results across folds for a robust global ranking.
    """
    sgkf = StratifiedGroupKFold(n_splits=len(np.unique(sites)),
                                 shuffle=True, random_state=RANDOM_STATE)
    all_rows = []

    for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=sites)):
        site_label = np.unique(sites[test_idx])[0]
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_te, y_te = X.iloc[test_idx],  y[test_idx]

        pipe = _make_pipe()
        pipe.fit(X_tr, y_tr)

        res = permutation_importance(
            pipe, X_te, y_te,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="roc_auc",
        )
        auc_val = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        print(f"  Fold {fold_i+1} ({site_label}): AUC={auc_val:.3f}")

        # Each repeat is one row
        for rep in range(n_repeats):
            row = pd.Series(res.importances[:, rep], index=X.columns)
            row["_fold"]  = fold_i
            row["_site"]  = site_label
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def plot_permutation_importance(df_imp: pd.DataFrame):
    feat_cols = [c for c in df_imp.columns if not c.startswith("_")]
    mean_imp  = df_imp[feat_cols].mean()
    top_feats = mean_imp.nlargest(N_TOP).index.tolist()

    melted = df_imp[top_feats + ["_site"]].melt(
        id_vars=["_site"], var_name="Feature", value_name="Importance"
    )

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.boxplot(data=melted, x="Importance", y="Feature",
                order=top_feats[::-1],
                color="#9B59B6", showfliers=False,
                medianprops={"color": "white", "linewidth": 2},
                width=0.6, linewidth=1.8, ax=ax)
    sns.stripplot(data=melted, x="Importance", y="Feature",
                  order=top_feats[::-1],
                  color=TIN_COLOR, alpha=0.35, size=4, ax=ax)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, lw=1.2)
    ax.set_title(f"Permutation importance — top {N_TOP} features\n"
                 f"(held-out LOSO folds, {(df_imp['_site'].nunique())} sites)",
                 style="italic")
    ax.set_xlabel("ΔHeld-out AUC (mean ± IQR across folds × repeats)")
    ax.spines[["right", "top"]].set_visible(False)

    fpath = FIGURES_DIR / "perm_importance.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")

    # Also save top-features table
    tbl = mean_imp[top_feats].reset_index()
    tbl.columns = ["feature", "mean_importance"]
    tbl.to_csv(TABLES_DIR / "perm_importance_top.csv", index=False)
    return top_feats


# ── B. Partial dependence plots (LOSO) ───────────────────────────────────────

def compute_pdp(X, y, sites, features: list):
    sgkf = StratifiedGroupKFold(n_splits=len(np.unique(sites)),
                                 shuffle=True, random_state=RANDOM_STATE)
    storage = {f: {"folds": [], "grid": None, "raw": X[f].values} for f in features}

    for train_idx, _ in sgkf.split(X, y, groups=sites):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        pipe = _make_pipe()
        pipe.fit(X_tr, y_tr)

        for feat in features:
            res = partial_dependence(
                pipe, X_tr, features=[feat],
                kind="average", grid_resolution=PDP_GRID_RES,
                percentiles=(0.05, 0.95),
            )
            storage[feat]["folds"].append(res["average"][0])
            storage[feat]["grid"] = res["grid_values"][0]

    return storage


def plot_pdp(storage: dict):
    features = list(storage.keys())
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=False,
                              constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        res    = storage[feat]
        folds  = np.array(res["folds"])
        grid   = res["grid"]
        mean_  = folds.mean(axis=0)
        std_   = folds.std(axis=0)

        for fold_vals in folds:
            ax.plot(grid, fold_vals, color="gray", alpha=0.25, lw=1, linestyle="--")
        ax.plot(grid, mean_, color="#9B59B6", lw=3, label="Mean")
        ax.fill_between(grid, mean_ - std_, mean_ + std_,
                         color="#9B59B6", alpha=0.18)

        # Rug plot
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.eventplot(res["raw"], orientation="horizontal", lineoffsets=ax.get_ylim()[0],
                      linelengths=0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                      colors="gray", alpha=0.2)

        short = feat.split("_")[0]  # shorten for title
        ax.set_title(short, style="italic", fontsize=9)
        ax.set_xlabel("Feature value")
        ax.spines[["right", "top"]].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("P(Tinnitus)", fontsize=10)

    fig.suptitle("Partial dependence plots — top features (LOSO mean ± 1 SD)",
                 style="italic", fontsize=10)
    fpath = FIGURES_DIR / "pdp_top_features.pdf"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ── C. LIME per-site explanation ──────────────────────────────────────────────

def compute_lime_explanations(X, y, sites, subject_ids):
    """For each site, explain the highest-confidence tinnitus prediction."""
    if not HAS_LIME:
        return

    sgkf = StratifiedGroupKFold(n_splits=len(np.unique(sites)),
                                 shuffle=True, random_state=RANDOM_STATE)
    y_prob_all = np.zeros(len(y))

    # Collect per-fold predictions and the fold's trained pipe
    fold_models = []
    for train_idx, test_idx in sgkf.split(X, y, groups=sites):
        pipe = _make_pipe()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        y_prob_all[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        fold_models.append((train_idx, test_idx, pipe))

    for train_idx, test_idx, pipe in fold_models:
        site_label = np.unique(sites[test_idx])[0]
        tin_mask   = y[test_idx] == 1
        if not tin_mask.any():
            continue

        # Highest-confidence tinnitus subject in held-out site
        probs_test   = y_prob_all[test_idx]
        local_idx    = np.where(tin_mask)[0][np.argmax(probs_test[tin_mask])]
        global_idx   = test_idx[local_idx]
        sub_id       = subject_ids[global_idx]
        confidence   = y_prob_all[global_idx]

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X.iloc[train_idx].values,
            feature_names=list(X.columns),
            class_names=["Control", "Tinnitus"],
            mode="classification",
            random_state=RANDOM_STATE,
        )
        exp = explainer.explain_instance(
            X.iloc[global_idx].values,
            pipe.predict_proba,
            num_features=10,
        )

        exp_list = exp.as_list()
        lime_df  = pd.DataFrame(exp_list, columns=["Feature", "Contribution"])
        lime_df["Direction"] = lime_df["Contribution"].apply(
            lambda v: "Supports Tinnitus" if v > 0 else "Supports Control"
        )

        pal = {"Supports Tinnitus": TIN_COLOR, "Supports Control": CTRL_COLOR}
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        sns.barplot(data=lime_df, x="Contribution", y="Feature",
                    hue="Direction", palette=pal, dodge=False, ax=ax)
        ax.axvline(0, color="black", lw=1.5, alpha=0.7)
        ax.set_title(f"LIME — {site_label}  sub {sub_id}\n"
                     f"Confidence = {confidence:.3f}",
                     style="italic")
        ax.set_xlabel("Local contribution weight")
        ax.legend(title="", frameon=False, loc="lower right", fontsize="small")
        ax.spines[["left", "top", "right"]].set_visible(False)

        fpath = FIGURES_DIR / f"lime_{site_label}.pdf"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fpath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    try:
        X, y, sites, subject_ids = _load_diffusive()
    except FileNotFoundError as e:
        print(f"Data not found: {e}\nRun 09_multimodal_diffusion.py first.")
        raise SystemExit(1)

    print(f"  {len(y)} subjects, {X.shape[1]} features\n")

    # ── A. Permutation importance ──────────────────────────────────────────
    print("A. Permutation importance …")
    df_imp  = compute_permutation_importance(X, y, sites, n_repeats=10)
    top_feats = plot_permutation_importance(df_imp)

    # ── B. PDP for top N_PDP_FEAT features ────────────────────────────────
    pdp_features = top_feats[:N_PDP_FEAT]
    print(f"\nB. PDP for: {pdp_features}")
    pdp_storage = compute_pdp(X, y, sites, pdp_features)
    plot_pdp(pdp_storage)

    # ── C. LIME per site ───────────────────────────────────────────────────
    print("\nC. LIME explanations …")
    compute_lime_explanations(X, y, sites, subject_ids)

    print("\nDone.")
