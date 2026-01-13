from pathlib import Path
import pandas as pd
import numpy as np
from mne import read_labels_from_annot


def compute_regional_metrics(
                            space,
                            conn_mode,
                            preproc_level,
                            suffix,
                            hm_dir
                            ):
    """
    Compute regional and global (node strength) connectivity metrics
    from harmonized connectivity matrices.

    This function:
    1. Loads harmonized connectivity data.
    2. Applies anatomical adjacency constraints.
    3. Computes regional (adjacency-weighted) and global node strengths
        for multiple frequency bands.
    4. Saves the resulting metrics as CSV files.

    Parameters
    ----------
    space : str
        Connectivity space (e.g., "source" or "sensor").
    conn_mode : str
        Connectivity metric (e.g., "pli", "plv", etc.).
    preproc_level : int
        Preprocessing level identifier.
    suffix : str
        harmonized or residual
    hm_dir : pathlib.Path
        Path to the harmonized data directory.

    Outputs
    -------
    Saves two CSV files in hm_dir:
    - Regional (adjacency-weighted) node strengths
    - Global (unweighted) node strengths
    """

    print("Loading connectivity and adjacency data...")
    df_conn = pd.read_csv(hm_dir / f"conn_{space}_preproc_{preproc_level}_{conn_mode}_{suffix}.csv")
    df_adj = pd.read_csv("../material/aparc_adjacency.csv")
    df_adj.drop(columns="Unnamed: 0", inplace=True)

    freq_bands = {
                "delta": [1, 6],
                "theta": [6.5, 8.5],
                "alpha_0": [8.5, 12.5],
                "alpha_1": [8.5, 10.5],
                "alpha_2": [10.5, 12.5],
                "beta_0": [12.5, 30],
                "beta_1": [12.5, 18.5],
                "beta_2": [18.5, 21],
                "beta_3": [21, 30],
                "gamma": [30, 40]
                }

    mne_labels = [lb.name for lb in read_labels_from_annot(subject="fsaverage", verbose=False)[:-1]]

    ## create regional and global node strengths
    n_labels = len(mne_labels)
    rows, cols = np.tril_indices(n_labels, k=-1)
    df_re_list, df_ns_list = [], []
    print("Computing regional and global node strengths...")

    for freq_band in freq_bands.keys():
        print(f"  Processing frequency band: {freq_band}")

        ## filter
        regex = f"_{freq_band}_{conn_mode}"
        df_filtered = df_conn.filter(regex=regex)
        df_adj_sub = df_adj.copy()
        df_adj_sub.columns = [col + regex for col in df_adj_sub.columns]

        ## multiply
        df_product = df_filtered * df_adj_sub.values
        X_re = df_product.to_numpy()
        X_ns = df_filtered.to_numpy()

        sum_per_label_re = np.zeros((X_re.shape[0], n_labels))
        sum_per_label_ns = np.zeros((X_ns.shape[0], n_labels))
        labels = [c + regex for c in mne_labels]

        # vectorized addition
        np.add.at(sum_per_label_re, (np.arange(X_re.shape[0])[:,None], rows[None,:]), X_re)
        np.add.at(sum_per_label_re, (np.arange(X_re.shape[0])[:,None], cols[None,:]), X_re)
        df_re = pd.DataFrame(sum_per_label_re, columns=labels)

        # vectorized addition
        np.add.at(sum_per_label_ns, (np.arange(X_ns.shape[0])[:,None], rows[None,:]), X_ns)
        np.add.at(sum_per_label_ns, (np.arange(X_ns.shape[0])[:,None], cols[None,:]), X_ns)
        df_ns = pd.DataFrame(sum_per_label_ns, columns=labels)

        ## concatenate
        df_re_list.append(df_re)
        df_ns_list.append(df_ns)

    df_res = pd.concat(df_re_list, axis=1)
    df_nss = pd.concat(df_ns_list, axis=1)

    cols = ["subject_id", "SITE", "age", "sex", "PTA4_mean", "group"]
    df_res = pd.concat([df_res, df_conn[cols]], axis=1)
    df_nss = pd.concat([df_nss, df_conn[cols]], axis=1)

    ## save it in harmonized folder path
    for df, title in zip([df_res, df_nss], ["regional", "global"]):
        df.to_csv(hm_dir / f"{title}_{space}_preproc_{preproc_level}_{conn_mode}_{suffix}.csv")
    
    print("All computations completed and saved successfully!")


if __name__ == "__main__":
    
    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    hm_dir = tinnorm_dir / "harmonized"
    
    preproc_levels = [2]
    spaces = ["sensor", "source"][1:]
    conn_modes = ["pli", "plv", "coh"][2:]
    suffixes = ["hm", "residual"]


    for preproc_level in preproc_levels:
        for space in spaces:
            for conn_mode in conn_modes:
                for suffix in suffixes:
                    fname_save = hm_dir / f"conn_{space}_preproc_{preproc_level}_{suffix}.csv"
                
                    if fname_save.exists():
                        continue
                    else:
                        compute_regional_metrics(
                                                space,
                                                conn_mode,
                                                preproc_level,
                                                suffix,
                                                hm_dir
                                                )
                        

## add the 6 columns here