
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import color_palette
from umap import UMAP
from sklearn.manifold import MDS

def _read_the_file(
                mode,
                space,
                preproc_level,
                conn_mode,
                thi_threshold=None,
                only_extreme=False
                ):
    
    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    models_dir = tinnorm_dir / "models"
    df_master = pd.read_csv("../material/master.csv")

    if mode in ["conn", "global", "regional", "diffusive"]:
        mode_prefix = f"_{conn_mode}"
    else:
        mode_prefix = ""

    if mode == "diffusive":
        fname = tinnorm_dir / "diffusive" / f"{space}_preproc_{preproc_level}_{conn_mode}.csv"
        df = pd.read_csv(fname)
        cols = [c for c in df.columns if c != "group"] + ["group"]
        df = df[cols]

    else:
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
    df = df.merge(
                df_master[['subject_id', 'thi_score']],
                on='subject_id',
                how='left'
                )

    if not thi_threshold is None:
        df = df.query(f'group == 0 or thi_score > {thi_threshold}')

    df = df.merge(df_master[['subject_id', 'site']], on='subject_id', how='left')
    df.rename(columns={"site": "SITE"}, inplace=True)
    df.sort_values("subject_id", inplace=True)
    df.drop(columns=["Unnamed: 0", "observations", "subject_id"], inplace=True, errors="ignore")

    if only_extreme:
        raise NotImplementedError

    print("\n****************************************************")
    print(f"number of subjects are: {df['group'].value_counts()}")
    print("****************************************************\n")
    return df

def plot_dm_results(
                    mode,
                    space,
                    preproc_level,
                    conn_mode,
                    freq_band,
                    method,
                    thi_threshold=None,
                    ):

    df = _read_the_file(
                        mode,
                        space,
                        preproc_level,
                        conn_mode,
                        thi_threshold,
                        only_extreme
                        )

    if not freq_band is None:
        df_features = df.filter(regex=rf"_{freq_band}")
        X = df_features.values
    else:
        df_features = df.copy()
        feature_cols = df.columns[:-3]
        X = df[feature_cols].values

    print("\n****************************************************")
    print(f"number of features are: {X.shape[1]}")
    print("****************************************************\n")

    kwargs = {"n_components": 2, "random_state": 42}
    if method == "umap":
        reducer = UMAP(**kwargs)
        X_reduced = reducer.fit_transform(X)
    if method == "MDS":
        reducer = MDS(**kwargs)
        X_reduced = reducer.fit_transform(X)

    df_result = pd.DataFrame(
                            X_reduced,
                            columns=["dimension_1", "dimension_2"]
                            )
    df_result = pd.concat([df[df.columns[-3:]], df_result], axis=1)

    palette_colors = ['#1f77b4', '#d62728'] 
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), layout="tight") 

    kwargs = {
                "x": "dimension_1",
                "y": "dimension_2",
                "alpha": 0.8,
                "ax": ax
            }
    sns.scatterplot(
                    data=df_result.query('group == 0'),
                    s=100,
                    color=palette_colors[0],
                    legend=False,
                    **kwargs
                    )
    sns.scatterplot(
                    data=df_result.query('group == 1'),
                    size="thi_score",
                    sizes=(100, 300),
                    color=palette_colors[1],
                    legend=False,
                    **kwargs
                    )
    ax.spines[["right", "top"]].set_visible(False)
    plt.show()

def plot_corr_results(
                    mode,
                    space,
                    preproc_level,
                    conn_mode,
                    freq_band,
                    method,
                    thi_threshold=None,
                    ):

    df = _read_the_file(
                        mode,
                        space,
                        preproc_level,
                        conn_mode,
                        thi_threshold,
                        only_extreme
                        )
    
    if freq_band is not None:
        df_features = df.filter(regex=rf"_{freq_band}")
        df = pd.concat([df_features, df["thi_score"]], axis=1)
    
    df.dropna(inplace=True)
    target_col = "thi_score"
    other_cols = [c for c in df.columns if c not in ["group", "thi_score", "SITE"]]

    results = []
    for col in other_cols:
        r, p = pearsonr(df[col], df[target_col])
        results.append({"feature": col, "r": r, "pval": p})

    results_df = pd.DataFrame(results)
    results_df["pval_fdr"] = multipletests(results_df["pval"], method="fdr_bh")[1]

    results_df = results_df.query('pval_fdr < 0.05')
    if len(results_df) > 0:
        cols = results_df["feature"]
        pal = color_palette("Purples", n_colors=10)
        color = pal[9]

        for col in cols:
            g = sns.jointplot(
                    data=df,
                    x="thi_score",
                    y=col,
                    kind="reg",
                    height=5,
                    scatter_kws = {"s": 20, "color": color},
                    line_kws = {"linestyle": "--", "linewidth": 2, "color": "grey"},
                    marginal_kws={"bins": 25, "fill": False, "color": color}
                    )
            
            r = results_df.loc[results_df['feature'] == col, 'r'].values[0]
            pval = results_df.loc[results_df['feature'] == col, 'pval_fdr'].values[0]
            textstr = f"R = {r:.3f}\npval = {pval:.3f}"
            g.ax_joint.text(
                            0.7, 0.98,                    
                            textstr,
                            transform=g.ax_joint.transAxes,
                            fontsize=10,
                            fontstyle='italic',            
                            verticalalignment='top',       
                            horizontalalignment='left'  
                        )
            plt.show()
    else:
        print("********** nothing survived multiple comparison **********")



if __name__ == "__main__":

    mode = "diffusive"
    space = "source"
    preproc_level = 2
    conn_mode = "coh"
    freq_band = "beta_1"
    method = "umap" 
    thi_threshold = None
    only_extreme = False

    kwargs = {
            "mode": mode,
            "space": space,
            "preproc_level": preproc_level,
            "conn_mode": conn_mode,
            "freq_band": freq_band,
            "method": method,
            "thi_threshold": thi_threshold
            }
    # plot_dm_results(**kwargs)
    plot_corr_results(**kwargs)

## add extreme subjects


