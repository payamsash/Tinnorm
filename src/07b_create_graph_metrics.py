"""
Priority 5 — Graph-theory metrics from harmonized connectivity matrices.

For each subject, reconstructs the ROI×ROI adjacency matrix per frequency band
and computes three complementary graph metrics:

  clustering   Weighted clustering coefficient (Onnela 2005) per node.
               Captures local clique density.
  efficiency   Global efficiency = mean(1/d_ij) using 1/weight as distance.
               Sensitive to integration across the whole network.
  local_eff    Local efficiency per node = efficiency of each node's
               immediate neighbourhood subgraph.

Output: one CSV per preproc_level × conn_mode in the harmonized folder,
with columns  <roi>_<freq_band>_<metric>  (e.g., bankssts-lh_alpha_1_clustering)
plus subject-level covariates so the file can be fed directly into
08_create_norm_models.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from mne import read_labels_from_annot
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


# ── Graph metric implementations (numpy/scipy, no networkx dependency) ────

def _weighted_clustering(W: np.ndarray) -> np.ndarray:
    """Onnela (2005) weighted clustering coefficient per node.

    W : symmetric adjacency matrix (n×n), diagonal zeroed.
    Returns array of length n.
    """
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    w_max = W.max()
    if w_max == 0:
        return np.zeros(W.shape[0])
    W_norm = W / w_max
    W_cbrt = np.cbrt(W_norm)
    triangles = np.diagonal(W_cbrt @ W_cbrt @ W_cbrt)
    k = (W > 0).sum(axis=1).astype(float)
    denom = k * (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, triangles / denom, 0.0)
    return C


def _global_efficiency(W: np.ndarray) -> float:
    """Global efficiency = mean(1/shortest_path_length) for all node pairs."""
    n = W.shape[0]
    if n < 2:
        return 0.0
    # Convert weights to distances: d_ij = 1/w_ij (higher weight → shorter path)
    with np.errstate(divide="ignore"):
        D = np.where(W > 0, 1.0 / W, 0.0)
    paths = shortest_path(csr_matrix(D), directed=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_paths = np.where((np.isinf(paths)) | (paths == 0), 0.0, 1.0 / paths)
    np.fill_diagonal(inv_paths, 0.0)
    return float(inv_paths.sum() / (n * (n - 1)))


def _local_efficiency(W: np.ndarray) -> np.ndarray:
    """Local efficiency per node = global efficiency of each node's subgraph."""
    n = W.shape[0]
    leff = np.zeros(n)
    for i in range(n):
        nb = np.where(W[i] > 0)[0]
        if len(nb) < 2:
            continue
        W_sub = W[np.ix_(nb, nb)]
        leff[i] = _global_efficiency(W_sub)
    return leff


# ── Core computation ──────────────────────────────────────────────────────

def compute_graph_metrics(
        preproc_level: int,
        conn_mode: str,
        hm_mode: str,
        hm_dir: Path,
        saving_dir: Path,
) -> None:
    """
    Load the harmonized pairwise connectivity matrix and compute per-node
    graph-theoretic metrics for each frequency band.

    Parameters
    ----------
    preproc_level : preprocessing level (1, 2, or 3)
    conn_mode     : "pli", "plv", or "coh"
    hm_mode       : "hm" (harmonized) or "residual"
    hm_dir        : root of harmonized/ directory
    saving_dir    : where to write the output CSV
    """

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
        "gamma": [30, 40],
    }

    conn_file = hm_dir / f"preproc_{preproc_level}" / "source" / f"conn_{conn_mode}_{hm_mode}.csv"
    if not conn_file.is_file():
        print(f"  Missing: {conn_file}  — skipping.")
        return

    print(f"  Loading {conn_file.name} …")
    df_conn = pd.read_csv(conn_file)

    # Covariate columns to carry through
    covariate_cols = ["subject_id", "SITE", "age", "sex", "PTA4_mean", "PTA4_HF", "group"]
    df_covars = df_conn[[c for c in covariate_cols if c in df_conn.columns]]

    # ROI names in the same order as feature extraction
    mne_labels = [lb.name for lb in read_labels_from_annot(subject="fsaverage", verbose=False)[:-1]]
    n_labels = len(mne_labels)
    rows, cols = np.tril_indices(n_labels, k=-1)

    metric_dfs = []

    for freq_band in freq_bands:
        regex = f"_{freq_band}_{conn_mode}"
        df_band = df_conn.filter(regex=regex)

        if df_band.shape[1] == 0:
            continue

        expected_pairs = n_labels * (n_labels - 1) // 2
        if df_band.shape[1] != expected_pairs:
            print(f"  Warning: {freq_band} has {df_band.shape[1]} columns, "
                  f"expected {expected_pairs}. Skipping band.")
            continue

        n_subjects = len(df_band)
        clust_matrix  = np.zeros((n_subjects, n_labels))
        eff_matrix    = np.zeros((n_subjects, n_labels))
        leff_matrix   = np.zeros((n_subjects, n_labels))

        vals = df_band.to_numpy()  # (n_subjects × n_pairs)

        for s in range(n_subjects):
            # Reconstruct symmetric adjacency matrix
            adj = np.zeros((n_labels, n_labels))
            adj[rows, cols] = vals[s]
            adj[cols, rows] = vals[s]

            clust_matrix[s] = _weighted_clustering(adj)
            leff_matrix[s]  = _local_efficiency(adj)
            eff_matrix[s]   = _global_efficiency(adj)   # scalar → broadcast below

        # Global efficiency is per-subject scalar; broadcast to per-node column for consistency
        # (same value for all nodes — added as a separate column)
        glob_eff_col = {f"global_eff_{freq_band}_{conn_mode}": eff_matrix[:, 0]}

        for name, matrix in [("clustering", clust_matrix), ("local_eff", leff_matrix)]:
            cols_names = [f"{roi}_{freq_band}_{name}" for roi in mne_labels]
            metric_dfs.append(pd.DataFrame(matrix, columns=cols_names))

        metric_dfs.append(pd.DataFrame(glob_eff_col))

    df_metrics = pd.concat(metric_dfs, axis=1)
    df_out = pd.concat([df_metrics.reset_index(drop=True), df_covars.reset_index(drop=True)], axis=1)

    os.makedirs(saving_dir, exist_ok=True)
    out_fname = saving_dir / f"graph_{conn_mode}_{hm_mode}.csv"
    df_out.to_csv(out_fname, index=False)
    print(f"  Saved → {out_fname}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    hm_dir      = tinnorm_dir / "harmonized"

    preproc_levels = [1, 2, 3]
    conn_modes     = ["pli", "plv", "coh"]
    hm_modes       = ["hm", "residual"]

    for preproc_level in preproc_levels:
        for conn_mode in conn_modes:
            for hm_mode in hm_modes:
                saving_dir = hm_dir / f"preproc_{preproc_level}" / "source"
                out_fname  = saving_dir / f"graph_{conn_mode}_{hm_mode}.csv"

                if out_fname.exists():
                    print(f"Already exists, skipping: {out_fname.name}")
                    continue

                print(f"\npreproc_{preproc_level} | {conn_mode} | {hm_mode}")
                compute_graph_metrics(preproc_level, conn_mode, hm_mode, hm_dir, saving_dir)
