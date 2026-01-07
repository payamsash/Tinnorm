import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nibabel.freesurfer import read_geometry, read_annot
from mne import read_labels_from_annot

subjects_dir = "/Applications/freesurfer/dev/subjects"
subject = "fsaverage"
EXCLUDE = {"unknown", "corpuscallosum"}

def aparc_adjacency(hemi):
    verts, faces = read_geometry(f"{subjects_dir}/{subject}/surf/{hemi}.white")
    labels, ctab, names = read_annot(f"{subjects_dir}/{subject}/label/{hemi}.aparc.annot")
    names = [n.decode("utf-8") for n in names]

    # Keep only cortical labels
    keep = [i for i, n in enumerate(names) if n not in EXCLUDE]
    names = [names[i] for i in keep]
    remap = {old: new for new, old in enumerate(keep)}  # label ID -> index 0..33

    n = len(names)
    adj = np.zeros((n, n), dtype=int)

    for tri in faces:
        # labels at these vertices
        labs = labels[tri]
        # keep only cortical labels
        tri_cort = [remap[l] for l in labs if l in remap]
        # assign adjacency between all pairs
        for i in range(len(tri_cort)):
            for j in range(i+1, len(tri_cort)):
                adj[tri_cort[i], tri_cort[j]] = 1
                adj[tri_cort[j], tri_cort[i]] = 1

    return adj, names

def interleave_matrices(A, B):
    """
    Interleave two square matrices A (LH) and B (RH) into a larger matrix.

    Ordering:
        lh0, rh0, lh1, rh1, ...

    Between-hemisphere blocks are zero by construction.
    """
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("Both matrices must be square and of the same size.")

    n = A.shape[0]
    out = np.zeros((2*n, 2*n), dtype=A.dtype)
    out[0::2, 0::2] = A
    out[1::2, 1::2] = B

    return out

## save and plot
def lower_triangle_df(mat, names):
    """
    Convert the lower triangle of a square matrix (excluding diagonal)
    into a 1-row DataFrame with named columns.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square.")
    if mat.shape[0] != len(names):
        raise ValueError("names must match matrix size.")

    rows = []
    cols = []

    n = mat.shape[0]
    for i in range(1, n):
        for j in range(i):
            rows.append(mat[i, j])
            cols.append(f"{names[i]}_vs_{names[j]}")

    return pd.DataFrame([rows], columns=cols)

# LH and RH adjacency
adj_lh, names_lh = aparc_adjacency("lh")
adj_rh, names_rh = aparc_adjacency("rh")
mne_labels = [lb.name[:-3] for lb in read_labels_from_annot(subject, hemi="rh")]

idx = [names_lh.index(lbl) for lbl in mne_labels]
adj_lh_reordered = adj_lh[np.ix_(idx, idx)]

idx = [names_rh.index(lbl) for lbl in mne_labels]
adj_rh_reordered = adj_rh[np.ix_(idx, idx)]

adj_mat = interleave_matrices(adj_lh_reordered, adj_rh_reordered)

## plot
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(adj_mat, cmap="Reds")
mne_labels_all = [lb.name for lb in read_labels_from_annot(subject)[:-1]]
n = len(mne_labels_all)
ticks = np.arange(n)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(mne_labels_all, rotation=90, fontsize=6)
ax.set_yticklabels(mne_labels_all, fontsize=6)
ax.set_title("APARC adjacency matrix (RH)")
plt.tight_layout()
plt.show()

## save
df = lower_triangle_df(adj_mat, mne_labels_all)
df.to_csv("../material/aparc_adjacency.csv")