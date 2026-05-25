"""
EEG signal quality metrics per subject and site — resting-state specific.

Operates on preproc_2-epo.fif (10-second clean epochs, already filtered and
artifact-rejected). This reflects what actually enters the analysis pipeline,
not raw acquisition quality.

Five metrics calibrated for resting-state EEG:

  alpha_snr      Alpha-band SNR = P(8–13 Hz) / mean[P(4–8 Hz), P(20–30 Hz)]
                 The gold-standard resting-state quality indicator.
                 High → clear eyes-closed alpha; low → noisy or eyes-open.

  aperiodic_exp  1/f exponent from a log-log fit of the broadband PSD (2–40 Hz).
                 Healthy resting EEG: exponent ≈ 2–3.
                 Extreme values flag preprocessing failures or artifacts.

  muscle_idx     log₁₀(P(65–80 Hz) / P(20–30 Hz))
                 EMG contamination index — lower is better.
                 Lower band starts at max(line_freq+5, 65) Hz to stay above
                 line-noise harmonics.

  line_noise     P(line_freq ± 0.5 Hz) / mean[P(flanks ± 2 Hz)]
                 Residual powerline artifact after notch filtering.
                 Should be ≈ 1.0 for well-notched data.

  scree_ratio    Proportion of total covariance explained by the top-3 PCs.
                 Very high → signal dominated by one artifact source.

PSDs are computed per epoch (Welch, 4-s window, 50% overlap) and averaged
across epochs to minimise boundary artefacts from epoch concatenation.

Saves:
  plots/diagnosis/tables/eeg_quality_preproc2.csv
  plots/diagnosis/tables/eeg_quality_preproc2_summary.csv
  plots/diagnosis/figures/eeg_quality_{metric}.pdf
  plots/diagnosis/figures/eeg_quality_combined.pdf

Run from src/:  python 02d_eeg_quality.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from scipy.integrate import trapezoid
import mne

# ── Config ────────────────────────────────────────────────────────────────────

TINNORM_DIR = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
BIDS_DIR    = TINNORM_DIR / "BIDS"
RESULTS_DIR = TINNORM_DIR / "plots" / "diagnosis"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"
CACHE_CSV   = TABLES_DIR / "eeg_quality_preproc2.csv"

SITE_MAP = {
    "1": "austin",
    "2": "dublin",
    "3": "ghent",
    "4": "illinois",
    "5": "regensburg",
    "6": "tuebingen",
    "7": "zuerich",
}
LINE_FREQ = {
    "austin":     60,
    "dublin":     50,
    "ghent":      50,
    "illinois":   60,
    "regensburg": 50,
    "tuebingen":  50,
    "zuerich":    50,
}

CTRL_COLOR = "#1f77b4"
TIN_COLOR  = "#C99700"

mne.set_log_level("ERROR")


# ── PSD helper ────────────────────────────────────────────────────────────────

def _band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Trapezoid integration of mean-channel PSD in [fmin, fmax].

    psd shape: (n_channels, n_freqs) — mean across channels first, then integrate.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return np.nan
    return float(trapezoid(psd[:, mask].mean(axis=0), freqs[mask]))


# ── Per-subject quality computation ──────────────────────────────────────────

def _compute_quality(fif_path: Path, subject_id: str, site: str) -> dict | None:
    try:
        epo   = mne.read_epochs(fif_path, preload=True, verbose=False)
        sfreq = epo.info["sfreq"]
        nyq   = sfreq / 2.0

        # (n_epochs, n_channels, n_times)
        data_3d = epo.get_data(picks="eeg")
        n_ep, n_ch, _ = data_3d.shape

        # Compute Welch PSD per epoch then average — avoids boundary
        # discontinuity effects that occur when epochs are concatenated first.
        nperseg  = int(4 * sfreq)   # 4-second analysis window
        noverlap = int(2 * sfreq)   # 50% overlap
        psds = []
        for ep in data_3d:
            f, p = welch(ep, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)
            psds.append(p)
        freqs = f                            # (n_freqs,)
        psd   = np.mean(psds, axis=0)       # (n_ch, n_freqs)

        # Concatenated epochs needed for covariance (scree ratio)
        data_cat = data_3d.transpose(1, 0, 2).reshape(n_ch, -1)

        # ── Alpha SNR ─────────────────────────────────────────────────────
        p_alpha    = _band_power(freqs, psd, 8.0,  13.0)
        p_theta    = _band_power(freqs, psd, 4.0,   8.0)
        p_low_beta = _band_power(freqs, psd, 20.0, 30.0)
        denom      = (p_theta + p_low_beta) / 2.0
        alpha_snr  = float(p_alpha / denom) if denom > 0 else np.nan

        # ── Aperiodic 1/f exponent (log-log linear fit, 2–40 Hz) ─────────
        mask_ap = (freqs >= 2) & (freqs <= 40)
        if mask_ap.sum() > 3:
            log_f   = np.log10(freqs[mask_ap])
            log_psd = np.log10(psd[:, mask_ap].mean(axis=0) + 1e-30)
            slope, _ = np.polyfit(log_f, log_psd, 1)
            aperiodic_exp = float(-slope)   # positive convention
        else:
            aperiodic_exp = np.nan

        # ── Muscle artifact index ──────────────────────────────────────────
        lf = LINE_FREQ.get(site, 50)
        muscle_fmin = max(lf + 5.0, 65.0)   # stay above line-noise harmonics
        muscle_fmax = min(80.0, nyq - 1.0)
        if muscle_fmax > muscle_fmin:
            p_muscle   = _band_power(freqs, psd, muscle_fmin, muscle_fmax)
            p_ref      = _band_power(freqs, psd, 20.0, 30.0)
            muscle_idx = float(np.log10(p_muscle / p_ref + 1e-9)) if p_ref > 0 else np.nan
        else:
            muscle_idx = np.nan

        # ── Line-noise index ───────────────────────────────────────────────
        p_line  = _band_power(freqs, psd, lf - 0.5, lf + 0.5)
        p_left  = _band_power(freqs, psd, lf - 3.0, lf - 1.0)
        p_right = _band_power(freqs, psd, lf + 1.0, lf + 3.0)
        flank   = (p_left + p_right) / 2.0
        line_noise = float(p_line / flank) if flank > 0 else np.nan

        # ── Scree ratio (top-3 PCA on concatenated epochs) ────────────────
        cov     = np.cov(data_cat)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        scree   = float(np.sum(eigvals[:3]) / np.sum(eigvals))

        return {
            "subject_id":    subject_id,
            "site":          site,
            "n_epochs":      n_ep,
            "n_channels":    n_ch,
            "alpha_snr":     alpha_snr,
            "aperiodic_exp": aperiodic_exp,
            "muscle_idx":    muscle_idx,
            "line_noise":    line_noise,
            "scree_ratio":   scree,
        }
    except Exception as e:
        print(f"  Error {subject_id}: {e}")
        return None


# ── Batch computation ─────────────────────────────────────────────────────────

def compute_all_quality(force: bool = False) -> pd.DataFrame:
    if CACHE_CSV.exists() and not force:
        print(f"Loading cached quality metrics from {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    print("Computing EEG quality metrics from preproc_2-epo.fif …")
    rows = []
    for folder in sorted(BIDS_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("sub-"):
            continue
        subject_id = folder.name[4:]
        site_key   = subject_id[0]
        site       = SITE_MAP.get(site_key, site_key)
        fif_path   = folder / "ses-01" / "eeg" / "preproc_2-epo.fif"
        if not fif_path.exists():
            continue
        print(f"  sub-{subject_id} ({site})")
        res = _compute_quality(fif_path, f"sub-{subject_id}", site)
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_CSV, index=False)
    print(f"Quality CSV → {CACHE_CSV}")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

METRIC_META = {
    "alpha_snr":     ("Alpha SNR  (α / mean[θ, β])",
                      "Higher = cleaner resting-state signal"),
    "aperiodic_exp": ("Aperiodic 1/f exponent",
                      "Expected ≈ 2–3 for healthy EEG"),
    "muscle_idx":    ("Muscle index  log₁₀(P₆₅₋₈₀ / P₂₀₋₃₀)",
                      "Lower = less EMG contamination"),
    "line_noise":    ("Line-noise ratio  P_line / P_flanks",
                      "≈ 1.0 = notch filter effective"),
    "scree_ratio":   ("Scree ratio  (top-3 PCs / total variance)",
                      "Very high = atypical, one dominant component"),
}


def _strip_boxen(df: pd.DataFrame, metric: str, save_path: Path):
    label, subtitle = METRIC_META[metric]
    site_order = sorted(df["site"].unique())

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    sns.stripplot(data=df, x="site", y=metric, order=site_order,
                  hue="group", palette={0: CTRL_COLOR, 1: TIN_COLOR},
                  dodge=True, alpha=0.55, size=5, jitter=0.15, ax=ax)
    sns.boxenplot(data=df, x="site", y=metric, order=site_order,
                  hue="group", palette={0: CTRL_COLOR, 1: TIN_COLOR},
                  dodge=True, alpha=0.22, showfliers=False, legend=False, ax=ax)
    ax.set_xlabel("Site")
    ax.set_ylabel(label)
    ax.set_title(f"{label}\n{subtitle}", style="italic", fontsize=10)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Control", "Tinnitus"], frameon=False,
              loc="best", fontsize="small")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_all_metrics(df: pd.DataFrame):
    for metric in METRIC_META:
        if metric not in df.columns:
            continue
        fpath = FIGURES_DIR / f"eeg_quality_{metric}.pdf"
        _strip_boxen(df, metric, fpath)

    # Combined panel
    metrics_to_plot = [m for m in METRIC_META if m in df.columns]
    n     = len(metrics_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    site_order = sorted(df["site"].unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes_flat = axes.flatten() if nrows > 1 else (list(axes) if ncols > 1 else [axes])

    for i, (ax, metric) in enumerate(zip(axes_flat, metrics_to_plot)):
        label, _ = METRIC_META[metric]
        show_legend = (i == 0)
        sns.violinplot(data=df, x="site", y=metric, order=site_order,
                       hue="group", palette={0: CTRL_COLOR, 1: TIN_COLOR},
                       dodge=True, inner="quartile", alpha=0.7,
                       legend=show_legend, ax=ax)
        ax.set_title(label, style="italic", fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.spines[["right", "top"]].set_visible(False)
        if show_legend:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["Control", "Tinnitus"],
                      frameon=False, fontsize="small")
        elif ax.get_legend():
            ax.get_legend().remove()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Resting-state EEG quality — preproc_2 epochs", style="italic")
    fpath = FIGURES_DIR / "eeg_quality_combined.pdf"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def print_site_summary(df: pd.DataFrame):
    metrics = [m for m in METRIC_META if m in df.columns]
    summary = df.groupby("site")[metrics].agg(["mean", "std"]).round(3)
    print("\nPer-site quality summary:")
    print(summary.to_string())
    out = TABLES_DIR / "eeg_quality_preproc2_summary.csv"
    summary.to_csv(out)
    print(f"\nSummary table → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = compute_all_quality(force=False)

    df_master = pd.read_csv("../material/master_clean.csv")
    df_master["subject_id"] = "sub-" + df_master["subject_id"].astype(str)
    df = df.merge(df_master[["subject_id", "group"]], on="subject_id", how="left")
    df["group"] = df["group"].fillna(-1).astype(int)
    df_valid = df[df["group"].isin([0, 1])].copy()

    print(f"\n{len(df_valid)} subjects with quality metrics and group labels.")
    print_site_summary(df_valid)
    plot_all_metrics(df_valid)

    print("\nDone.")
