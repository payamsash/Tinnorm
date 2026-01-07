import pandas as pd
import numpy as np
from mne import read_labels_from_annot


## fix paths
conn_mode = "pli"

df_conn = pd.read_csv(f"/Users/payamsadeghishabestari/Tinnorm/material/conn_source_preproc_2_{conn_mode}_hm.csv")
df_adj = pd.read_csv("/Users/payamsadeghishabestari/Tinnorm/material/aparc_adjacency.csv")
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

## create regional node strength da
n_labels = len(mne_labels)
rows, cols = np.tril_indices(n_labels, k=-1)
df_re_list, df_ns_list = [], []
for freq_band in freq_bands.keys():
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

## save it
