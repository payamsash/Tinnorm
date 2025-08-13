import numpy as np
from mne.time_frequency import psd_array_welch
from fooof import FOOOF

def compute_band_power_paf_fooof_fast(data, sfreq, bands):
    """
    Vectorized Welch PSD -> band power, PAF, and FOOOF aperiodic params.

    Parameters
    ----------
    data : np.ndarray
        Shape = (n_epochs, n_labels, n_times)
    sfreq : float
        Sampling frequency.
    bands : dict
        Example: {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), ...}

    Returns
    -------
    dict
        Keys: absolute, relative, paf, aperiodic_slope, aperiodic_intercept
    """
    n_epochs, n_labels, _ = data.shape
    band_names = list(bands.keys())

    # --- Compute PSD for all epochs & labels at once ---
    data_reshaped = data.reshape(-1, data.shape[-1])  # (epochs*labels, timepoints)
    freqs, psd = psd_array_welch(
        data_reshaped, sfreq=sfreq, fmin=1, fmax=45, n_fft=1024, average='mean'
    )
    psd = psd.reshape(n_epochs, n_labels, len(freqs))

    # --- Total power for relative band power ---
    total_power = np.trapz(psd, freqs, axis=-1)

    # --- Absolute & relative power ---
    abs_power = np.zeros((n_epochs, n_labels, len(bands)))
    rel_power = np.zeros_like(abs_power)

    for i, (bname, (fmin, fmax)) in enumerate(bands.items()):
        idx = (freqs >= fmin) & (freqs <= fmax)
        bp = np.trapz(psd[:, :, idx], freqs[idx], axis=-1)
        abs_power[:, :, i] = bp
        rel_power[:, :, i] = bp / total_power

    # --- Peak Alpha Frequency (PAF) ---
    alpha_min, alpha_max = bands.get("alpha", (8, 12))
    idx_alpha = (freqs >= alpha_min) & (freqs <= alpha_max)
    paf = freqs[idx_alpha][np.argmax(psd[:, :, idx_alpha], axis=-1)]

    # --- FOOOF aperiodic params ---
    slopes = np.zeros((n_epochs, n_labels))
    intercepts = np.zeros((n_epochs, n_labels))
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1)

    for ep in range(n_epochs):
        for lbl in range(n_labels):
            fm.fit(freqs, psd[ep, lbl, :], [1, 40])
            intercepts[ep, lbl] = fm.aperiodic_params_[0]
            slopes[ep, lbl] = fm.aperiodic_params_[1]

    return {
        "absolute": abs_power,               # (epochs, labels, n_bands)
        "relative": rel_power,               # (epochs, labels, n_bands)
        "paf": paf,                           # (epochs, labels)
        "aperiodic_slope": slopes,           # (epochs, labels)
        "aperiodic_intercept": intercepts,   # (epochs, labels)
        "bands": band_names,
        "freqs": freqs
    }


eps = 1e-12
log_abs = np.log10(abs_power_median + eps)

'''
Primary NM features: log_abs_power (per band × ROI) plus aperiodic_slope, aperiodic_intercept, and log_total_power as covariates or extra targets. Also include age, sex, PTA, site.
'''



from mne_connectivity import EpochSpectralConnectivity

con = EpochSpectralConnectivity(data, freqs, n_nodes, names=None, indices='all', method=None, spec_method=None)
con_data = con.get_data()