"""
11_plot_deviations.py

Map normative-model Z-score deviations onto the fsaverage cortical surface.

For each (preproc_level × space × modality × conn_mode × freq_band):
  tinnitus → full_model/results/Z_test.csv          (unbiased: never in training)
  controls → loso/loso_controls_Z.csv               (unbiased: held out by site)

Two summaries per ROI per group:
  mean_Z   — mean Z-score across subjects (signed; how far from normal on average)
  pct_dev  — % of subjects with |Z| > 1.96 (proportion with significant deviation)

Where both groups exist a difference map (tinnitus − control mean_Z) is also saved.

Aperiodic modality has no freq bands: exponent and offset are plotted separately.
Graph modality columns carry no conn_mode suffix: regex filtered accordingly.

Run from src/:  python 11_plot_deviations.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne import read_labels_from_annot
from mne.viz import Brain


# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
MODELS_DIR  = TINNORM_DIR / "models"
SAVING_DIR  = TINNORM_DIR / "plots" / "deviations"

THR        = 1.96
PALETTE    = sns.color_palette("ch:", n_colors=256)
PALETTE_DIV = sns.color_palette("RdBu_r", n_colors=256)  # for diff maps
FIG_KW     = {"format": "pdf", "dpi": 300, "bbox_inches": "tight"}

_N_LABELS: int | None = None   # cached label count


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_n_labels(subject: str = "fsaverage", subjects_dir=None) -> int:
    global _N_LABELS
    if _N_LABELS is None:
        _N_LABELS = len(read_labels_from_annot(
            subject=subject, subjects_dir=subjects_dir, verbose=False)) - 1
    return _N_LABELS


def _discover_freq_bands(csv_path: Path) -> list:
    """Return unique EEG freq bands from CSV columns, in physiological order."""
    df = pd.read_csv(csv_path, nrows=1)
    meta = {"subject_id", "subject_ids", "observations", "Unnamed: 0"}
    cols = [c for c in df.columns if c not in meta]
    band_re = re.compile(
        r'_((?:delta|theta|alpha|beta|gamma|broadband)(?:_\d+)?)(?:_\w+)?$')
    bands = set()
    for c in cols:
        m = band_re.search(c)
        if m:
            bands.add(m.group(1))
    order = ["delta", "theta",
             "alpha_0", "alpha_1", "alpha_2",
             "beta_0",  "beta_1",  "beta_2",
             "gamma_0", "gamma_1", "gamma_2", "broadband"]
    return [b for b in order if b in bands] + sorted(bands - set(order))


def _filter_band(df: pd.DataFrame, freq_band: str,
                 conn_mode: str | None, modality: str) -> pd.DataFrame:
    """
    Return columns matching this freq_band.

    graph_* columns have no conn_mode suffix ({roi}_{band}_{metric}), so we
    don't append conn_mode to the regex for that modality.
    """
    if modality == "graph" or conn_mode is None:
        regex = freq_band
    else:
        regex = f"{freq_band}_{conn_mode}"
    result = df.filter(regex=regex).apply(pd.to_numeric, errors="coerce")
    return result.dropna(axis=1, how="all")


def _to_roi_matrix(df_band: pd.DataFrame, n_rois: int) -> np.ndarray:
    """
    Convert a (n_subjects, n_cols) band DataFrame to (n_subjects, n_rois).

    Works when n_cols == n_rois (power, regional, global) and when
    n_cols == n_rois × k (graph: k metrics per ROI — averaged).
    """
    arr   = df_band.values.astype(float)
    n_col = arr.shape[1]
    if n_col == n_rois:
        return arr
    if n_col % n_rois == 0:
        k = n_col // n_rois
        return arr.reshape(arr.shape[0], n_rois, k).mean(axis=2)
    raise ValueError(
        f"Cannot map {n_col} columns to {n_rois} ROIs "
        f"(not divisible). Check band filter.")


# ── Brain surface plot ────────────────────────────────────────────────────────

def plot_brain(
        metric_vals,
        palette=None,
        surf: str = "inflated",
        subject: str = "fsaverage",
        subjects_dir=None,
        alpha: float = 0.8,
        cbar_label: str = "value",
) -> plt.Figure:
    """
    Map per-ROI scalar values onto fsaverage (4 views: LH/RH × lateral/medial).

    Colormap range is the 2nd–98th percentile so outlier ROIs don't compress
    the scale. Values outside the range still receive the nearest palette colour.
    """
    if palette is None:
        palette = PALETTE

    metric_vals = np.asarray(metric_vals, dtype=float)
    if metric_vals.ndim != 1:
        raise ValueError("metric_vals must be 1-D.")

    labels = read_labels_from_annot(
        subject=subject, subjects_dir=subjects_dir, verbose=False)[:-1]
    if len(labels) != len(metric_vals):
        raise ValueError(
            f"Expected {len(labels)} values (n_labels), got {len(metric_vals)}.")

    vmin = float(np.percentile(metric_vals, 2))
    vmax = float(np.percentile(metric_vals, 98))
    if vmin == vmax:
        vmin -= 1e-6; vmax += 1e-6
    clipped = np.clip(metric_vals, vmin, vmax)
    norm    = (clipped - vmin) / (vmax - vmin)

    colors_hex   = np.array([mcolors.to_hex(c) for c in palette])
    label_colors = colors_hex[np.round(norm * (len(colors_hex) - 1)).astype(int)]

    brain_kw = dict(subject=subject, subjects_dir=subjects_dir,
                    surf=surf, background="white",
                    cortex=["#b8b4ac", "#b8b4ac"])

    def _render(hemi, view):
        brain = Brain(hemi=hemi, views=view, **brain_kw)
        for lbl, color in zip(labels, label_colors):
            if lbl.hemi == hemi:
                brain.add_label(lbl, hemi=hemi, color=color,
                                borders=False, alpha=alpha)
        brain.add_annotation("aparc", borders=True, color="white")
        img = brain.screenshot()
        brain.close()
        nw = (img != 255).any(axis=-1)
        return img[nw.any(axis=1)][:, nw.any(axis=0)]

    imgs = [
        _render("lh", "lateral"),
        _render("rh", "lateral"),
        _render("lh", "medial"),
        _render("rh", "medial"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img)
        ax.axis("off")

    sm   = plt.cm.ScalarMappable(
        cmap=mcolors.ListedColormap(palette),
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.06, shrink=0.45)
    cbar.set_label(cbar_label)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.3g}", f"{vmax:.3g}"])
    return fig


# ── Yeo-7 network mapping ─────────────────────────────────────────────────────

def map_roi_to_yeo7() -> dict:
    """Aparc label index → Yeo-7 network abbreviation."""
    return {
        0:  "VAN",  1:  "VAN",   # bankssts
        2:  "DMN",  3:  "DMN",   # caudalanteriorcingulate
        4:  "DAN",  5:  "DAN",   # caudalmiddlefrontal
        6:  "VIS",  7:  "VIS",   # cuneus
        8:  "DMN",  9:  "DMN",   # entorhinal
        10: "DMN",  11: "DMN",   # frontalpole
        12: "VAN",  13: "VAN",   # fusiform
        14: "VAN",  15: "VAN",   # inferiorparietal
        16: "VIS",  17: "VIS",   # inferiortemporal
        18: "VAN",  19: "VAN",   # insula
        20: "DMN",  21: "DMN",   # isthmuscingulate
        22: "VIS",  23: "VIS",   # lateraloccipital
        24: "FPN",  25: "FPN",   # lateralorbitofrontal
        26: "VIS",  27: "VIS",   # lingual
        28: "DMN",  29: "DMN",   # medialorbitofrontal
        30: "VAN",  31: "VAN",   # middletemporal
        32: "SMN",  33: "SMN",   # paracentral
        34: "DMN",  35: "DMN",   # parahippocampal
        36: "FPN",  37: "FPN",   # parsopercularis
        38: "FPN",  39: "FPN",   # parsorbitalis
        40: "FPN",  41: "FPN",   # parstriangularis
        42: "VIS",  43: "VIS",   # pericalcarine
        44: "SMN",  45: "SMN",   # postcentral
        46: "DMN",  47: "DMN",   # posteriorcingulate
        48: "SMN",  49: "SMN",   # precentral
        50: "DMN",  51: "DMN",   # precuneus
        52: "DMN",  53: "DMN",   # rostralanteriorcingulate
        54: "FPN",  55: "FPN",   # rostralmiddlefrontal
        56: "FPN",  57: "FPN",   # superiorfrontal
        58: "DAN",  59: "DAN",   # superiorparietal
        60: "SMN",  61: "SMN",   # superiortemporal
        62: "VAN",  63: "VAN",   # supramarginal
        64: "DMN",  65: "DMN",   # temporalpole
        66: "SMN",  67: "SMN",   # transversetemporal
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SAVING_DIR.mkdir(parents=True, exist_ok=True)

    preproc_levels = [1, 2, 3]
    spaces         = ["source"]
    modalities     = ["power", "aperiodic", "regional", "global", "graph"]
    conn_modes_all = ["pli", "plv", "coh"]

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:
                conn_loop = (conn_modes_all
                             if modality in ["regional", "global", "graph"]
                             else [None])

                for conn_mode in conn_loop:
                    suffix    = f"{modality}_{conn_mode}" if conn_mode else modality
                    model_dir = MODELS_DIR / f"preproc_{preproc_level}" / space / suffix
                    test_path = model_dir / "full_model" / "results" / "Z_test.csv"
                    loso_path = model_dir / "loso" / "loso_controls_Z.csv"

                    if not test_path.exists():
                        continue

                    print(f"\npreproc_{preproc_level} | {space} | {suffix}")

                    # ── Load Z-scores ─────────────────────────────────────────
                    df_tin = pd.read_csv(test_path).drop(
                        columns=["observations", "subject_ids", "Unnamed: 0"],
                        errors="ignore")
                    groups: dict[str, pd.DataFrame] = {"tinnitus": df_tin}

                    if loso_path.exists():
                        df_ctrl = pd.read_csv(loso_path).drop(
                            columns=["subject_id", "Unnamed: 0"], errors="ignore")
                        groups["control"] = df_ctrl

                    # ── Build list of (band_key, df_filtered) per group ───────
                    n_rois = _get_n_labels()
                    out_dir = SAVING_DIR / suffix
                    out_dir.mkdir(exist_ok=True)

                    if modality == "aperiodic":
                        # No freq bands — plot exponent and offset separately
                        band_specs = []
                        for param in ["exponent", "offset"]:
                            cols = [c for c in df_tin.columns
                                    if c.endswith(f"_{param}")]
                            if cols:
                                band_specs.append(param)
                        freq_bands_loop = band_specs if band_specs else []

                    else:
                        try:
                            freq_bands_loop = _discover_freq_bands(test_path)
                        except Exception as e:
                            print(f"  freq-band discovery failed: {e}")
                            continue

                    if not freq_bands_loop:
                        print(f"  no bands found — skipping")
                        continue

                    stem_base = f"preproc{preproc_level}_{space}_{suffix}"

                    for freq_band in freq_bands_loop:

                        # ── Per-group brain maps ──────────────────────────────
                        roi_means: dict[str, np.ndarray] = {}

                        for group_name, df_g in groups.items():

                            if modality == "aperiodic":
                                # freq_band is actually the param name here
                                param = freq_band
                                cols  = [c for c in df_g.columns
                                         if c.endswith(f"_{param}")]
                                if not cols:
                                    continue
                                df_band = df_g[cols].apply(
                                    pd.to_numeric, errors="coerce"
                                ).dropna(axis=1, how="all")
                            else:
                                df_band = _filter_band(
                                    df_g, freq_band, conn_mode, modality)
                                if df_band.empty:
                                    continue

                            try:
                                mat = _to_roi_matrix(df_band, n_rois)
                            except Exception as e:
                                print(f"  _to_roi_matrix failed "
                                      f"({group_name}, {freq_band}): {e}")
                                continue

                            mean_z  = mat.mean(axis=0)
                            pct_dev = (np.abs(mat) > THR).mean(axis=0) * 100
                            roi_means[group_name] = mean_z

                            stem = f"{stem_base}_{freq_band}_{group_name}"
                            for vals, label, tag in [
                                (mean_z,  "Mean Z-score",  "meanZ"),
                                (pct_dev, "% |Z| > 1.96", "pctDev"),
                            ]:
                                try:
                                    fig = plot_brain(vals, PALETTE,
                                                     cbar_label=label)
                                    fpath = out_dir / f"{stem}_{tag}.pdf"
                                    fig.savefig(fpath, **FIG_KW)
                                    plt.close(fig)
                                    print(f"  Saved → {fpath}")
                                except Exception as e:
                                    print(f"  brain plot failed "
                                          f"({stem}_{tag}): {e}")

                        # ── Difference map (tinnitus − control) ───────────────
                        if ("tinnitus" in roi_means and "control" in roi_means
                                and modality != "aperiodic"):
                            diff  = roi_means["tinnitus"] - roi_means["control"]
                            fstem = f"{stem_base}_{freq_band}_diff_meanZ"
                            try:
                                fig = plot_brain(diff, PALETTE_DIV,
                                                 cbar_label="ΔMean Z (tin − ctrl)")
                                fpath = out_dir / f"{fstem}.pdf"
                                fig.savefig(fpath, **FIG_KW)
                                plt.close(fig)
                                print(f"  Saved → {fpath}")
                            except Exception as e:
                                print(f"  diff map failed ({freq_band}): {e}")


# maybe kinda standardize and plot