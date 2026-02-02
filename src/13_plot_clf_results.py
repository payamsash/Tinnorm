from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay


## permutation

tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
clfs_dir = tinnorm_dir / "clfs"
plots_dir = tinnorm_dir / "plots"

perm_dir = clfs_dir / "permutation"
comp_dir = clfs_dir / "comparison"

def filter_df(df, filters):
    """
    filters: dict of column -> value
        - value can be a list, single value, or np.nan
    Returns filtered df.
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
for folder in perm_dir.iterdir():
    if folder.name.startswith("."):
        continue
    
    ## read metric dataframes
    id = folder.name
    metrics_fname = perm_dir / folder / "metrics.csv"
    df_single = pd.read_csv(metrics_fname)
    df_single["id"] = [id] * len(df_single)
    dfs.append(df_single)

df = pd.concat(dfs, axis=0)


filters = {
    "data_mode": ["residual", "deviation"],
    "preproc_level": [2],
    "space": ["source"],
    "mode": ["power"],
    "conn_mode": np.nan,
    "freq_band": np.nan,
    "folding_mode": ["stratified"],
    "apply_rfe": [False],
    "thi_threshold": np.nan
}

df_filtered = filter_df(df, filters)

ys, y_probs = [], []
for id in df_filtered["id"].unique():
    ys.append(np.load(perm_dir / id / "y.npy", allow_pickle=False))
    y_probs.append(np.load(perm_dir / id / "y_prob.npy", allow_pickle=False))

ys = np.array(ys)
y_probs = np.array(y_probs)

## permutation histogram

for dm, group in df_filtered.groupby('data_mode'):
    
    real_acc = group[group['model'] == 'RF_real']['balanced_accuracy'].values[0]
    perm_acc = group[group['model'] != 'RF_real']['balanced_accuracy']
    
    p_val = (perm_acc >= real_acc).mean()
    
    
    df_filtered.loc[df_filtered['data_mode'] == dm, 'p_value'] = p_val


metric = "balanced_accuracy"
col_metric = "data_mode"

g = sns.FacetGrid(
                    data=df_filtered,
                    col=col_metric,
                    # col_order=["residual", "deviation"],
                    xlim=[0.4, 0.7],
                    sharex=True,
                    sharey=True,
                    height=4,
                    aspect=1
                )
g.map_dataframe(
                sns.histplot,
                x=metric,
                stat="percent",
                color="lightgray",
                alpha=0.7
                
                )

def add_rf_real_line(data, color, **kwargs):
    ax = plt.gca()
    p_val = data['p_value'].iloc[0]

    ## vertical line
    real_acc = data.loc[data['model'] == 'RF_real', metric].values[0]
    ax.axvline(real_acc, linestyle='--', color="#1f77b4", linewidth=2, label='RF_real')

    textstr = f"acc = {real_acc:.3f}\npval = {p_val:.3f}"
    ylim = ax.get_ylim()
    ax.text(
        real_acc + 0.01,
        ylim[1]*0.95,
        textstr,
        fontsize=10,
        fontstyle='italic',
        color="#1f77b4",
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )


g.map_dataframe(add_rf_real_line)
g.add_legend(title='')
for text in g._legend.texts:
    text.set_fontstyle('italic')



## ROC curves
colors = ["red", "green"]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")
for y, y_prob, color in zip(ys, y_probs, colors):
    RocCurveDisplay.from_predictions(
                                    y_true=y,
                                    y_score=y_prob,
                                    plot_chance_level=False,
                                    despine=True,
                                    ax=ax,
                                    curve_kwargs=dict(color=color),
                                    )
ax.legend(frameon=False, loc="lower right")


######################## still need this part
## plot histogram

def plot_delta_auc_null(delta_null, real_delta, p_value, save_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,4))
    plt.hist(delta_null, bins=40, alpha=0.7)
    plt.axvline(real_delta, color="red", linewidth=3,
                label=f"Real ΔAUC = {real_delta:.3f}\np = {p_value:.4f}")

    plt.xlabel("ΔAUC (X1 - X2)")
    plt.ylabel("Frequency")
    plt.title("Permutation Test: ΔAUC Null Distribution")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


## plot ROC curves

from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

def plot_roc_curves(y, y_prob_1, y_prob_2, save_path=None):

    plt.figure(figsize=(6,6))

    RocCurveDisplay.from_predictions(
        y, y_prob_1, name="X1", linewidth=2
    )
    RocCurveDisplay.from_predictions(
        y, y_prob_2, name="X2", linewidth=2
    )

    plt.plot([0,1], [0,1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: X1 vs X2")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

## 
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metric_boxplot(df_metric, metric="roc_auc", save_path=None):

    plt.figure(figsize=(5,4))

    sns.boxplot(
        x="model",
        y=metric,
        data=df_metric,
        showfliers=False
    )

    plt.title(f"{metric} Across Permutations")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Model")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

def plot_subject_scatter(y_prob_1, y_prob_2, y, save_path=None):

    plt.figure(figsize=(6,6))

    sns.scatterplot(
        x=y_prob_1,
        y=y_prob_2,
        hue=y,
        palette="Set1",
        alpha=0.7
    )

    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("Predicted Probability (X1)")
    plt.ylabel("Predicted Probability (X2)")
    plt.title("Subject-Level Predictions: X1 vs X2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
