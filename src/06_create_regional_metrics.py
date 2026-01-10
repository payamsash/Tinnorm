from pathlib import Path
import pandas as pd
import numpy as np
from mne import read_labels_from_annot


def compute_regional_metrics(
                            mode,
                            conn_mode,
                            preproc_level,
                            harmonized_dir
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
    mode : str
        Connectivity space (e.g., "source" or "sensor").
    conn_mode : str
        Connectivity metric (e.g., "pli", "plv", etc.).
    preproc_level : int
        Preprocessing level identifier.
    harmonized_dir : pathlib.Path
        Path to the harmonized data directory.

    Outputs
    -------
    Saves two CSV files in harmonized_dir:
    - Regional (adjacency-weighted) node strengths
    - Global (unweighted) node strengths
    """

    print("Loading connectivity and adjacency data...")
    df_conn = pd.read_csv(harmonized_dir / f"conn_{mode}_preproc_{preproc_level}_{conn_mode}_hm.csv")
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

    ## save it in harmonized folder path
    for df, title in zip([df_res, df_nss], ["regional", "global"]):
        df.to_csv(harmonized_dir / f"conn_{mode}_preproc_{preproc_level}_{title}_hm.csv")
    
    print("All computations completed and saved successfully!")


if __name__ == "__main__":
    harmonized_dir = Path("/Volumes/G_USZ_ORL$/Research/ANT/tinnorm/harmonized")
    mode = "source"
    conn_mode = "pli"
    preproc_level = 2

    compute_regional_metrics(
                            mode,
                            conn_mode,
                            preproc_level,
                            harmonized_dir
                            )