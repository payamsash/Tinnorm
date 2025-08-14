import numpy as np
from mne.time_frequency import psd_array_welch
from fooof import FOOOF
from tqdm import tqdm
from mne_connectivity import EpochSpectralConnectivity

def compute_spectral_features(
                                data,
                                sfreq,
                                bands,
                                fmin=1,
                                fmax=45,
                                method="welch",
                                n_fft=256,
                                n_overlap=0,
                                n_per_seg=None,
                                relative=False,
                                eps=1e-12
                            ):
    """
    Compute band power, PAF, and 1/f slope/intercept using FOOOF.

    Parameters
    ----------
    data : ndarray, shape (epochs, labels, times)
        EEG time series.
    sfreq : float
        Sampling frequency in Hz.
    bands : dict
        e.g. {"delta": (1, 4), "theta": (4, 8), ...}
    fmin, fmax : float
        Min and max freq for PSD.
    relative : bool
        Return relative power instead of absolute.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    features : dict of np.ndarray
        Keys: band power per band, 'slope', 'intercept'
    """
    
    n_epochs, n_labels, _ = data.shape
    n_bands = len(bands)

    # PSD
    psd, freqs = psd_array_welch(
        data.reshape(-1, data.shape[-1]),
        sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg,
        average="mean"
    )

    psd = psd.reshape(n_epochs, n_labels, -1)
    total_power = np.trapz(psd, freqs, axis=-1)
    band_powers = {b: np.zeros((n_epochs, n_labels)) for b in bands}
    slope = np.zeros((n_epochs, n_labels))
    intercept = np.zeros((n_epochs, n_labels))
    
    for e in range(n_epochs):
        for l in range(n_labels):
            this_psd = psd[e, l]

            # Compute band powers
            for b, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs <= hi)
                bp = np.trapz(this_psd[mask], freqs[mask])
                if relative:
                    bp /= (total_power[e, l] + eps)
                band_powers[b][e, l] = bp

            fm = FOOOF(peak_width_limits=[0.5, 12], verbose=False)
            fm.fit(freqs, this_psd, [fmin, fmax])

            # 1/f slope & intercept
            ap = fm.get_params('aperiodic_params')
            intercept[e, l] = ap[0]
            slope[e, l] = ap[1]

    features = {f"{b}_power": band_powers[b] for b in bands}
    features["slope"] = slope
    features["intercept"] = intercept

    for b in bands:
        features[f"{b}_power"] = np.log10(features[f"{b}_power"] + eps)

    return features





con = EpochSpectralConnectivity(data, freqs, n_nodes, names=None, indices='all', method=None, spec_method=None)
con_data = con.get_data()





if __name__ == "__main__":

    bands = {
            "delta": (1, 6),
            "theta": (6.5, 8.5),
            "alpha_1": (8.5, 10.5),
            "alpha_2": (10.5, 12.5),
            "beta_1": (12.5, 18.5),
            "beta_2": (18.5, 21),
            "beta_3": (21, 30),
            "gamma": (30, 40),
            }

    # compute_spectral_features()

    