from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
import shap

from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
)

## parameters which gave the best result
tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
read_kwargs = {
    "data_mode": "residual",
    "mode": "diffusive",
    "space": "source",
    "freq_band": None,
    "preproc_level": 2,
    "conn_mode": "coh",
    "thi_threshold": None
    }

ml_model = "RF"
apply_rfe = False
n_rfe_features = 100
n_jobs = -1
random_state = 42

## read the file
def _read_the_file(
                tinnorm_dir,
                data_mode,
                mode,
                space,
                freq_band,
                preproc_level,
                conn_mode,
                thi_threshold=None
                ):
    
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    df_master = pd.read_csv("../material/master.csv")

    if mode == "diffusive":
        ## Diffusive data
        fname = tinnorm_dir / mode / f"{space}_preproc_{preproc_level}_{conn_mode}.csv"
        df = pd.read_csv(fname)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)
        df = df.merge(  
                        df_master[['subject_id', 'site']],
                        on='subject_id',
                        how='left'
                    )
        df.rename(columns={"site": "SITE"}, inplace=True)


        if not thi_threshold is None:
            df = df.merge(
                        df_master[['subject_id', 'thi_score']],
                        on='subject_id',
                        how='left'
                    )
            df = df.query(f'group == 0 or thi_score > {thi_threshold}')

        cols_to_drop = ["Unnamed: 0", "subject_id", "age", "sex", "PTA4_mean", "thi_score"]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    else:
        ## Residual data
        if data_mode == "residual":
            mode_prefix = ""
            if mode in ["conn", "global", "regional", "diffusive"]:
                mode_prefix += f"_{conn_mode}"

            fname = hm_dir / f"preproc_{preproc_level}" / space / f"{mode}{mode_prefix}_residual.csv"
            df = pd.read_csv(fname)

            if not thi_threshold is None:
                df = df.merge(
                            df_master[['subject_id', 'thi_score']],
                            on='subject_id',
                            how='left'
                        )
                df = df.query(f'group == 0 or thi_score > {thi_threshold}')

            cols_to_drop = ["Unnamed: 0", "subject_id", "age", "sex", "PTA4_mean", "thi_score"]
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
            
        ## Deviation data
        if data_mode == "deviation":
            mode_prefix = ""
            if mode in ["conn", "global", "regional", "diffusive"]:
                mode_prefix += f"_{conn_mode}"
            
            dfs_group = []
            for group_name, group_id in zip(["train", "test"], [0, 1]):
                fname = (
                            models_dir
                            / f"preproc_{preproc_level}"
                            / space
                            / f"{mode}{mode_prefix}"
                            / "full_model"
                            / "results"
                            / f"Z_{group_name}.csv"
                        )
                df_group = pd.read_csv(fname)
                df_group["group"] = group_id
                dfs_group.append(df_group)

            df = pd.concat(dfs_group, axis=0, ignore_index=True)
            df.rename(columns={"subject_ids": "subject_id"}, inplace=True)

            df = df.merge(df_master[['subject_id', 'site']], on='subject_id', how='left')
            df.rename(columns={"site": "SITE"}, inplace=True)

            if not thi_threshold is None:
                df = df.merge(
                            df_master[['subject_id', 'thi_score']],
                            on='subject_id',
                            how='left'
                        )
                df = df.query(f'group == 0 or thi_score > {thi_threshold}')

            df.sort_values("subject_id", inplace=True)
            df.drop(columns=["observations", "subject_id", "thi_score"], inplace=True, errors="ignore")
    
    y = df["group"].to_numpy()         
    sites = df["SITE"].to_numpy()

    print("\n****************************************************")
    print(f"number of subjects are: {df['group'].value_counts()}")
    print("****************************************************\n")

    if not freq_band is None:
        df = df.filter(regex=rf"{freq_band}")
    
    df = df.drop(columns=["SITE"])
    print("\n****************************************************")
    print(f"number of selected features are: {df.shape[1] -1 }")
    print("****************************************************\n")

    return df, y, sites

df, y, sites = _read_the_file(tinnorm_dir, **read_kwargs)
X = df.drop(columns=["group"])

## split into train-test
sgkf = StratifiedGroupKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=random_state
                        )
train_idx, test_idx = next(
            sgkf.split(X, y, groups=sites)
                        )
X_train = X.iloc[train_idx].reset_index(drop=True)
X_test  = X.iloc[test_idx].reset_index(drop=True)

y_train = y[train_idx]
y_test  = y[test_idx]

sites_train = sites[train_idx]
sites_test  = sites[test_idx]



## train clf
kwargs = {
        "class_weight": "balanced",
        "random_state" : random_state
    }
if ml_model == "RF":
    final_model = RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=5,
                    n_jobs=n_jobs,
                    **kwargs
                )
final_model.fit(X_train, y_train)

## create a small df_metric of test set

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
}

df_metric = pd.DataFrame([metrics])

## plot the evaluation plots
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
RocCurveDisplay.from_predictions(
    y_test, y_prob, 
    ax=ax_roc, 
    plot_chance_level=True, 
    despine=True
)
PrecisionRecallDisplay.from_predictions(
    y_test, y_prob, 
    ax=ax_pr, 
    plot_chance_level=True, 
    despine=True
)

for ax in [ax_roc, ax_pr]:
    # Applying your preferred title style
    ax.set_title(ax.get_title(), style='italic', fontsize='small')
    ax.legend(frameon=False, loc="lower right", fontsize='small')

plt.show()


## run shap and clustering
df["group"] = df["group"].map({0: "Control", 1: "Tinnitus"})
idx_val = df["group"].values[0]
if idx_val == "Control":
    shap_idx = 1
    colors = ['#1f77b4', '#d62728']
if idx_val == "Tinnitus":
    colors = ['#d62728', '#1f77b4']

cmap1 = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
cmap2 = LinearSegmentedColormap.from_list("custom_cmap", colors)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X_train)
clustering = shap.utils.hclust(X_train, y_train, metric="cosine")

## plot first k features 
k = 3
shap_class1 = shap_values[:, :, shap_idx]
mean_abs_shap = np.abs(shap_class1.values).mean(axis=0)
f_idxs = np.argsort(mean_abs_shap)[::-1][:k]
col_names = X_train.columns[f_idxs]

shap.summary_plot(
    shap_values[:, :, shap_idx],
    X_train,
    feature_names=X.columns,
    max_display=10,
    cmap=cmap1,
    alpha=0.2
)

## scatter plot
fig, axs = plt.subplots(1, k, figsize=(4*k, 3), layout="tight")
for i, ax, f_idx, col_name in zip(range(k), axs, f_idxs, col_names):
    shap.plots.scatter(
            shap_values[:, f_idx, shap_idx],
            color=y_train,
            dot_size=25,
            alpha=1,
            cmap=cmap2,
            ax=ax,
            show=False
    )
    if ax.collections:
        for coll in ax.collections:
            if hasattr(coll, "colorbar") and coll.colorbar:
                coll.colorbar.remove()
    if i == 0:
        ax.set_ylabel("Shap values", fontsize=11)
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Feature value", style='italic', fontsize=11)
    ax.set_title(col_name, style='italic', fontsize=11)