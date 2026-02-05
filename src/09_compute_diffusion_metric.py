import os
from pathlib import Path
import numpy as np
import pandas as pd


def compute_diffusion_scores(
                            space,
                            preproc_level,
                            conn_mode
                            ):

    ## read results of the model
    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    models_dir = tinnorm_dir / "models"

    modes = ["power", "regional", "global"]

    dfs_all = []
    for mode in modes:
        if mode == "power":
            model_dir = models_dir / f"preproc_{preproc_level}" / space / mode
        else:
            model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{mode}_{conn_mode}"

        dfs_group = []
        for group_name, group_id in zip(["train", "test"], [0, 1]):
            fname = model_dir / "full_model" / "results" / f"Z_{group_name}.csv" 
            df = pd.read_csv(fname)
            df["group"] = [group_id] * len(df)
            dfs_group.append(df)

        dfs = pd.concat(dfs_group, axis=0)
        dfs_all.append(dfs)

    ## compute covariance matrx
    group_mask = dfs_all[0].group.values.astype(bool)
    regions = [s for s in dfs_all[1].columns[2:-1]]

    W = np.stack([df[df.columns[2:-1]].values for df in dfs_all], axis=2)
    W_controls = W[~group_mask]

    cov_matrices = {}
    inv_cov_matrices = {}
    for r, region in enumerate(regions):
        X = W_controls[:, r, :]
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6 # regularization
        cov_matrices[region] = cov
        inv_cov_matrices[region] = np.linalg.inv(cov)

    ## compute Mahalanobis distance

    maha_dists = np.zeros((W.shape[0], W.shape[1]))

    for r, region in enumerate(regions):
        inv_cov = inv_cov_matrices[region]
        mu = np.zeros(3)  # W-scores are centered
        
        diff = W[:, r, :] - mu
        maha_dists[:, r] = np.sqrt(
                                    np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
                                    )
        df_md = pd.DataFrame(
                            maha_dists,
                            columns=regions
                            )

    df_md["subject_ids"] = dfs_all[0]["subject_ids"].values
    df_md["group"] = dfs_all[0]["group"].values
    cols = ["subject_ids", "group"] + [c for c in df_md.columns if c not in ["subject_ids", "group"]]
    df_md = df_md[cols]

    fname_save = f"{space}_preproc_{preproc_level}_{conn_mode}.csv"
    os.makedirs(tinnorm_dir / "diffusive", exist_ok=True)
    df_md.to_csv(tinnorm_dir / "diffusive" / fname_save)

    print("********** computed and saved ************")

if __name__ == "__main__":

    space = "source"
    preproc_level = 2
    conn_mode = "coh"

    compute_diffusion_scores(
                            space,
                            preproc_level,
                            conn_mode
                            )
