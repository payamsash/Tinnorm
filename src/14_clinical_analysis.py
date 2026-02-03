
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from umap import UMAP
from sklearn.manifold import MDS

def _read_the_file(
                mode,
                space,
                preproc_level,
                conn_mode,
                thi_threshold=None
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

    print("\n****************************************************")
    print(f"number of subjects are: {df['group'].value_counts()}")
    print("****************************************************\n")

    print(df.columns)

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
                        thi_threshold
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

if __name__ == "__main__":

    mode = "diffusive"
    space = "source"
    preproc_level = 2
    conn_mode = "coh"
    method = "umap" 
    freq_band = "alpha_1"

    plot_dm_results(mode,
                    space,
                    preproc_level,
                    conn_mode,
                    freq_band,
                    method,
                    thi_threshold=None
                    )



## map of significant correlations between z-scores and THI