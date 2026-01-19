## Whether subjects organize in deviation space
## Whether THI varies smoothly along that space
## Not just classification, but severity structure

## read the deviation file(s) (add the freq_band option)
## read the deviation file

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

    mode_prefix = f"{mode}_{space}_preproc_{preproc_level}"
    if mode in ["conn", "global", "regional"]:
        mode_prefix += f"_{conn_mode}"
    
    dfs_group = []
    for group_name, group_id in zip(["train", "test"], [0, 1]):
        fname = models_dir / mode_prefix / "full_model" / "results" / f"Z_{group_name}.csv"
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
    df.drop(columns=["observations", "subject_id"], inplace=True, errors="ignore")

    return df

method = "umap"
def plot_dm_results(df, method):

    feature_cols = df.columns[:-3]
    X = df[feature_cols].values

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

