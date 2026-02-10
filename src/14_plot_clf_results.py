from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
clfs_dir = tinnorm_dir / "clfs"
plots_dir = tinnorm_dir / "plots"

perm_dir = clfs_dir / "permutation"
comp_dir = clfs_dir / "comparison"

data_type = "comparison"   # "permutation" | "comparison"

DIR_MAP = {
    "permutation": perm_dir,
    "comparison": comp_dir,
}

data_dir = DIR_MAP[data_type]

def filter_df(df, filters):
    """
    filters: dict of column -> value
        value can be list, scalar, or np.nan
    """
    mask = pd.Series(True, index=df.index)

    for col, val in filters.items():
        if isinstance(val, list):
            mask &= df[col].isin(val)
        elif pd.isna(val):
            mask &= df[col].isna()
        else:
            mask &= df[col] == val

    return df[mask]

dfs = []
for folder in data_dir.iterdir():
    if folder.name.startswith("."):
        continue

    metrics_fname = folder / "metrics.csv"
    if not metrics_fname.exists():
        continue

    df_single = pd.read_csv(metrics_fname)
    df_single["id"] = folder.name
    dfs.append(df_single)

df = pd.concat(dfs, axis=0, ignore_index=True)

filters = {
    "data_mode": ["residual", "deviation"],
    "preproc_level": [2],
    "space": ["source"],
    "mode": ["power"],
    "conn_mode": np.nan,
    "freq_band": np.nan,
    "folding_mode": ["stratified"],
    "apply_rfe": [False],
    "thi_threshold": np.nan,
}

df_filtered = filter_df(df, filters)

ys = []
y_probs = []

for id_ in df_filtered["id"].unique():
    ys.append(np.load(data_dir / id_ / "y.npy", allow_pickle=False))

    if data_type == "permutation":
        y_probs.append(
            np.load(data_dir / id_ / "y_prob.npy", allow_pickle=False)
        )

    elif data_type == "comparison":
        y_probs.append([
            np.load(data_dir / id_ / "y_prob_1.npy", allow_pickle=False),
            np.load(data_dir / id_ / "y_prob_2.npy", allow_pickle=False),
        ])

ys = np.array(ys)

if data_type == "permutation":

    for dm, group in df_filtered.groupby("data_mode"):
        real_acc = group.loc[
            group["model"] == "RF_real", "balanced_accuracy"
        ].values[0]

        perm_acc = group.loc[
            group["model"] != "RF_real", "balanced_accuracy"
        ]

        p_val = (perm_acc >= real_acc).mean()
        df_filtered.loc[group.index, "p_value"] = p_val

    metric = "balanced_accuracy"

    g = sns.FacetGrid(
        data=df_filtered,
        col="data_mode",
        xlim=[0.4, 0.7],
        sharex=True,
        sharey=True,
        height=4,
        aspect=1,
    )

    g.map_dataframe(
        sns.histplot,
        x=metric,
        stat="percent",
        color="lightgray",
        alpha=0.7,
    )

    def add_rf_real_line(data, **kwargs):
        ax = plt.gca()

        real_acc = data.loc[
            data["model"] == "RF_real", metric
        ].values[0]

        p_val = data["p_value"].iloc[0]

        ax.axvline(
            real_acc,
            linestyle="--",
            color="#1f77b4",
            linewidth=2,
        )

        ylim = ax.get_ylim()
        ax.text(
            real_acc + 0.01,
            ylim[1] * 0.95,
            f"acc = {real_acc:.3f}\npval = {p_val:.3f}",
            fontsize=10,
            fontstyle="italic",
            color="#1f77b4",
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                alpha=0.5,
                edgecolor="none",
            ),
        )

    g.map_dataframe(add_rf_real_line)
    g.add_legend(title="")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="gray",
    label="Chance (AUC = 0.50)",
)

axs[1].hlines(
    0.5, 0, 1,
    linestyle="--",
    color="gray",
    label="Chance (AUC = 0.50)",
)

if data_type == "permutation":

    for y, y_prob in zip(ys, y_probs):
        RocCurveDisplay.from_predictions(
            y_true=y,
            y_score=y_prob,
            ax=axs[0],
            curve_kwargs=dict(color="#1f77b4"),
            plot_chance_level=False,
            despine=True,
        )

        PrecisionRecallDisplay.from_predictions(
            y_true=y,
            y_score=y_prob,
            ax=axs[1],
            color="#1f77b4",
            plot_chance_level=False,
            despine=True,
        )

elif data_type == "comparison":

    colors = ["#1f77b4", "#245C43"]

    for model_idx, color in enumerate(colors):
        RocCurveDisplay.from_predictions(
            y_true=ys[0],
            y_score=y_probs[0][model_idx],
            ax=axs[0],
            curve_kwargs=dict(color=color),
            plot_chance_level=False,
            despine=True,
        )

        PrecisionRecallDisplay.from_predictions(
            y_true=ys[0],
            y_score=y_probs[0][model_idx],
            ax=axs[1],
            color=color,
            plot_chance_level=False,
            despine=True,
        )

for ax in axs:
    ax.legend(frameon=False, loc="lower right")
