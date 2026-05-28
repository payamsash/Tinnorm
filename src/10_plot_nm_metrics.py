"""
10_plot_nm_metrics.py

Visualize pcntoolkit normative model evaluation outputs.

Plot types
----------
plot_dist         KDE of EXPV / MSLL / SMSE across features, for one freq band.
plot_brain        Source space: metric mapped to fsaverage cortex (4 views).
plot_topomap      Sensor space: metric mapped to EEG scalp topography (MNE).
plot_calibration  Histogram + Q-Q of held-out control Z-scores vs N(0,1).

Run from src/: python 10_plot_nm_metrics.py
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
from scipy import stats


# ── Shared helper ─────────────────────────────────────────────────────────────

def _load_stats(ev_fname: Path, metric: str = None, freq_band: str = None) -> pd.DataFrame:
    """
    Load a pcntoolkit statistics CSV and optionally filter by metric row
    and/or freq_band columns.

    Column filter matches  _{freq_band}  at end, OR  _{freq_band}_{suffix}
    (e.g. _pli) at end — covers both power/aperiodic and connectivity outputs.
    """
    df = pd.read_csv(ev_fname)
    if "statistic" not in df.columns:
        raise ValueError(f"'statistic' column not found in {ev_fname}")
    if metric is not None:
        df = df[df["statistic"] == metric]
        if df.empty:
            raise ValueError(f"Metric '{metric}' not found in {ev_fname}")
    if freq_band is not None:
        pattern = rf'_{re.escape(freq_band)}(_\w+)?$'
        band_cols = [c for c in df.columns if re.search(pattern, c)]
        if not band_cols:
            raise ValueError(f"No columns match freq_band='{freq_band}' in {ev_fname}")
        df = df[["statistic"] + band_cols]
    return df


# ── Plot 1: KDE distribution of evaluation metrics ───────────────────────────

def plot_dist(ev_fname: Path, freq_band: str):
    """
    KDE of EXPV, MSLL, SMSE across all features for a given frequency band.
    Red dashed lines mark the ideal value for each metric.
    """
    metrics = ["EXPV", "MSLL", "SMSE"]
    df = _load_stats(ev_fname, freq_band=freq_band)
    df = df[df["statistic"].isin(metrics)]
    if df.empty:
        raise ValueError(f"No rows for {metrics} in {ev_fname}")

    # Wide → long
    df = (df.set_index("statistic")
            .T
            .reset_index(names="feature")
            .melt(id_vars="feature", var_name="metric", value_name="value"))
    df["value"] = pd.to_numeric(df["value"], errors="coerce").dropna()

    pal = {
        "EXPV": sns.color_palette("Blues_d",   3)[1],
        "MSLL": sns.color_palette("Oranges_d", 3)[1],
        "SMSE": sns.color_palette("Greens_d",  3)[1],
    }
    # Ideal reference: EXPV=1 (perfect), MSLL=0 (chance baseline), SMSE=1 (chance)
    ideal = {"EXPV": 1.0, "MSLL": 0.0, "SMSE": 1.0}

    g = sns.FacetGrid(
        data=df, row="metric", hue="metric",
        aspect=3.2, height=1.5, sharey=False, sharex=False,
        row_order=metrics, palette=pal,
    )
    g.map(sns.kdeplot, "value", fill=True, alpha=0.45, linewidth=1.5)
    g.map(sns.kdeplot, "value", color="k", lw=0.8)
    g.refline(y=0, linewidth=0.4, linestyle="-", color="k", clip_on=False)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    for ax, m in zip(g.axes[:, 0], metrics):
        ax.axvline(ideal[m], color="red", lw=1.0, ls="--", alpha=0.7)
        ax.text(0.98, 0.72, m, transform=ax.transAxes,
                ha="right", va="center", fontstyle="italic",
                color=pal[m], fontsize=9)

    g.axes[-1, 0].set_xlabel(f"Metric value  [{freq_band}]")
    g.figure.suptitle(f"Normative model evaluation — {freq_band}", y=1.02, fontsize=10)
    g.tight_layout()
    return g.figure


# ── Plot 2: Source-space brain surface map ────────────────────────────────────

def plot_brain(ev_fname: Path, metric: str, freq_band: str, palette,
               surf: str = "inflated", subject: str = "fsaverage",
               subjects_dir=None, alpha: float = 0.8):
    """
    Map a per-ROI scalar metric onto the fsaverage cortical surface.
    Renders 4 views (LH/RH × lateral/medial) and returns a matplotlib figure.
    """
    from mne import read_labels_from_annot
    from mne.viz import Brain

    df = _load_stats(ev_fname, metric=metric, freq_band=freq_band)
    metric_vals = (df.drop(columns="statistic", errors="ignore")
                     .values.squeeze().astype(float))
    if metric_vals.ndim != 1:
        raise ValueError(f"Expected 1-D metric values, got shape {metric_vals.shape}")

    labels = read_labels_from_annot(
        subject=subject, subjects_dir=subjects_dir, verbose=False)[:-1]
    if len(labels) != len(metric_vals):
        raise ValueError(f"Expected {len(labels)} values, got {len(metric_vals)}")

    # Robust range: 2nd–98th percentile so outlier ROIs don't compress the colormap.
    # Values outside the range still get the nearest palette edge color (not clipped to white).
    vmin = float(np.percentile(metric_vals, 2))
    vmax = float(np.percentile(metric_vals, 98))
    if vmin == vmax:          # degenerate (all identical values)
        vmin -= 1e-6
        vmax += 1e-6
    metric_clipped = np.clip(metric_vals, vmin, vmax)
    norm = (metric_clipped - vmin) / (vmax - vmin)
    colors_hex = np.asarray(palette.as_hex())
    label_colors = colors_hex[np.round(norm * (len(colors_hex) - 1)).astype(int)]

    brain_kw = dict(subject=subject, subjects_dir=subjects_dir,
                    surf=surf, background="white",
                    cortex=["#c0bcb5", "#c0bcb5"])

    def _render(hemi, view):
        brain = Brain(hemi=hemi, views=view, **brain_kw)
        for label, color in zip(labels, label_colors):
            if label.hemi == hemi:
                brain.add_label(label, hemi=hemi, color=color,
                                borders=False, alpha=alpha)
        brain.add_annotation("aparc", borders=True, color="white")
        img = brain.screenshot()
        brain.close()
        nonwhite = (img != 255).any(axis=-1)
        return img[nonwhite.any(axis=1)][:, nonwhite.any(axis=0)]

    view_specs = [
        ("lh", "lateral"), ("rh", "lateral"),
        ("lh", "medial"),  ("rh", "medial"),
    ]
    imgs = [_render(h, v) for h, v in view_specs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img)
        ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap=mcolors.ListedColormap(palette),
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.06, shrink=0.45)
    cbar.set_label(f"{metric}")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.3g}", f"{vmax:.3g}"])

    return fig


# ── Plot 3: Sensor-space EEG topomap ─────────────────────────────────────────

def plot_topomap(ev_fname: Path, metric: str, freq_band: str,
                 cmap: str = "cubehelix"):
    """
    Map a per-channel scalar metric onto a scalp topography.
    Channels absent from the standard_1020 montage are silently dropped.
    Returns a matplotlib figure.
    """
    import mne

    df = _load_stats(ev_fname, metric=metric, freq_band=freq_band)
    feature_cols = [c for c in df.columns if c != "statistic"]
    metric_vals = df[feature_cols].values.squeeze().astype(float)

    # Strip _{freq_band} suffix to recover channel names
    ch_names = [c[: -len(f"_{freq_band}")] for c in feature_cols]

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_set = set(montage.ch_names)
    mask = np.array([n in montage_set for n in ch_names])
    ch_valid = [n for n, m in zip(ch_names, mask) if m]
    vals_valid = metric_vals[mask]

    n_dropped = mask.size - mask.sum()
    if n_dropped:
        print(f"  plot_topomap: {n_dropped} channels not in standard_1020 — dropped.")
    if not ch_valid:
        raise ValueError("No channels matched standard_1020 montage.")

    info = mne.create_info(ch_valid, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)

    # Symmetric limits for diverging colormaps, data-driven for sequential
    if cmap in ("RdBu_r", "coolwarm", "bwr", "seismic"):
        vabs = np.abs(vals_valid).max()
        vlim = (-vabs, vabs)
    else:
        vlim = (vals_valid.min(), vals_valid.max())

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im, _ = mne.viz.plot_topomap(
        vals_valid, info, axes=ax,
        cmap=cmap, vlim=vlim,
        contours=6, sensors=True, show=False,
    )
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.8, pad=0.05)
    cbar.set_label(metric, fontsize=9)
    fig.tight_layout()
    return fig


# ── Plot 4: Calibration check (Z-score distribution) ─────────────────────────

def plot_calibration(z_fname: Path, freq_band: str = None,
                     n_sample: int = 10_000, seed: int = 0):
    """
    Check that held-out control Z-scores from for_eval are approximately N(0,1).

    Left panel : histogram of Z-scores overlaid with the N(0,1) PDF.
    Right panel: Q-Q plot against N(0,1) theoretical quantiles.

    A well-calibrated model should show:
      - histogram centered on 0 with SD ≈ 1
      - Q-Q points along the diagonal
    """
    df = pd.read_csv(z_fname)
    df = df.rename(columns={"subject_ids": "subject_id"})
    df = df.drop(columns=["observations", "subject_id"], errors="ignore")

    if freq_band is not None:
        pattern = rf'_{re.escape(freq_band)}(_\w+)?$'
        cols = [c for c in df.columns if re.search(pattern, c)]
        if not cols:
            raise ValueError(f"No columns match freq_band='{freq_band}'")
        df = df[cols]

    z_vals = df.values.astype(float).ravel()
    z_vals = z_vals[np.isfinite(z_vals)]

    rng = np.random.default_rng(seed)
    if len(z_vals) > n_sample:
        z_vals = rng.choice(z_vals, size=n_sample, replace=False)

    mu, sd = z_vals.mean(), z_vals.std()
    x = np.linspace(-5, 5, 400)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Histogram vs N(0,1) ──
    ax = axes[0]
    ax.hist(z_vals, bins=70, density=True,
            color="#5b9bd5", alpha=0.55, edgecolor="white", linewidth=0.3,
            label=f"Z-scores  (μ={mu:.2f}, σ={sd:.2f})")
    ax.plot(x, stats.norm.pdf(x), "r-", lw=1.5, label="N(0, 1)")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title("Z-score distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(-5, 5)

    # ── Q-Q plot ──
    ax = axes[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(z_vals, dist="norm")
    ax.scatter(osm, osr, s=2, alpha=0.25, color="#5b9bd5", rasterized=True)
    ref = np.array([osm.min(), osm.max()])
    ax.plot(ref, slope * ref + intercept, "r-", lw=1.2, label="N(0,1) reference")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q plot")
    ax.legend(frameon=False, fontsize=8)

    band_str = f"[{freq_band}]" if freq_band else "[all bands]"
    fig.suptitle(
        f"Normative model calibration {band_str} — "
        f"μ = {mu:.3f}, σ = {sd:.3f}  (ideal: 0 / 1)",
        fontsize=10, y=1.02,
    )
    sns.despine(fig=fig)
    fig.tight_layout()
    return fig


# ── Freq-band discovery ───────────────────────────────────────────────────────

def _discover_freq_bands(ev_fname: Path) -> list:
    """Return unique freq-band names found in a statistics CSV, in physiological order."""
    df = pd.read_csv(ev_fname, nrows=1)
    feature_cols = [c for c in df.columns if c != "statistic"]
    band_re = re.compile(
        r'_((?:delta|theta|alpha|beta|gamma|broadband)(?:_\d+)?)(?:_\w+)?$'
    )
    bands = set()
    for c in feature_cols:
        m = band_re.search(c)
        if m:
            bands.add(m.group(1))
    order = ["delta", "theta",
             "alpha_0", "alpha_1", "alpha_2",
             "beta_0",  "beta_1",  "beta_2",
             "gamma_0", "gamma_1", "gamma_2", "broadband"]
    return [b for b in order if b in bands] + sorted(bands - set(order))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    models_dir  = tinnorm_dir / "models"
    plots_dir   = tinnorm_dir / "plots" / "evaluation"

    # ── Configuration ─────────────────────────────────────────────────────────
    preproc_levels  = [1, 2, 3]
    spaces          = ["sensor", "source"]
    modalities      = ["power", "conn", "aperiodic", "global", "regional", "graph"]
    conn_modes      = ["pli", "plv", "coh"]
    data_mode       = "test"                            # statistics_{data_mode}.csv

    pal_brain  = sns.color_palette("ch:", n_colors=200)
    cmap_topo  = "cubehelix"

    def _ev_path(md):
        return md / "for_eval" / "results" / f"statistics_{data_mode}.csv"

    def _z_path(md):
        return md / "for_eval" / "results" / f"Z_{data_mode}.csv"

    def _save(fig, subdir, stem):
        out = plots_dir / subdir
        out.mkdir(parents=True, exist_ok=True)
        fpath = out / f"{stem}.pdf"
        fig.savefig(fpath, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {fpath}")

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:

                conn_loop = (conn_modes
                             if modality in ["conn", "global", "regional", "graph"]
                             else [None])

                for conn_mode in conn_loop:

                    # ── Resolve paths ──────────────────────────────────────────
                    suffix = f"{modality}_{conn_mode}" if conn_mode else modality
                    model_dir = models_dir / f"preproc_{preproc_level}" / space / suffix
                    ev_fname  = _ev_path(model_dir)

                    if not ev_fname.is_file():
                        continue

                    stem_base = f"preproc{preproc_level}_{space}_{suffix}"

                    # Discover all freq bands present in this stats file
                    try:
                        eval_freq_bands = _discover_freq_bands(ev_fname)
                    except Exception as e:
                        print(f"  freq-band discovery failed ({stem_base}): {e}")
                        continue

                    # ── Calibration (once per modality × conn_mode) ────────────
                    z_fname = _z_path(model_dir)
                    if z_fname.is_file():
                        try:
                            fig = plot_calibration(z_fname, freq_band=None)
                            _save(fig, "calibration", f"{stem_base}_calibration")
                        except Exception as e:
                            print(f"  calibration failed ({stem_base}): {e}")

                    # ── Per-frequency-band plots ───────────────────────────────
                    for freq_band in eval_freq_bands:
                        stem = f"{stem_base}_{freq_band}"

                        # Distribution of EXPV / MSLL / SMSE
                        try:
                            fig = plot_dist(ev_fname, freq_band)
                            _save(fig, "dist", f"{stem}_dist")
                        except Exception as e:
                            print(f"  plot_dist failed ({stem}): {e}")

                        # Spatial maps — EXPV, MSLL, SMSE
                        for metric in ["EXPV", "MSLL", "SMSE"]:
                            try:
                                if space == "source" and modality in ["power", "aperiodic"]:
                                    fig = plot_brain(ev_fname, metric, freq_band,
                                                     palette=pal_brain)
                                    _save(fig, "brain", f"{stem}_{metric}_brain")

                                elif space == "sensor" and modality == "power":
                                    fig = plot_topomap(ev_fname, metric, freq_band,
                                                       cmap=cmap_topo)
                                    _save(fig, "topomap", f"{stem}_{metric}_topomap")

                            except Exception as e:
                                print(f"  spatial plot failed ({stem}, {metric}): {e}")
