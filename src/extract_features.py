import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
from tqdm import tqdm
from fooof import FOOOF
from mne.time_frequency import psd_array_welch
from mne_connectivity import spectral_connectivity_time
from tools.learn_graph import log_degree_barrier

def compute_spectral_features(
                                site,
                                subject_id,
                                atlas,
                                bands,
                                fmin=1,
                                fmax=45,
                                method="welch",
                                n_fft=256,
                                n_overlap=0,
                                n_per_seg=None,
                                relative=False,
                                eps=1e-12,
                                saving_dir
                            ):
    
    data = np.load("...", allow_pickle=True)
    n_epochs, n_labels, _ = data.shape
    n_bands = len(bands)

    # PSD
    psd, freqs = psd_array_welch(
        data.reshape(-1, data.shape[-1]),
        sfreq=250, fmin=fmin, fmax=fmax,
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
        fname_save
        np.save()

    return features


def compute_connectivity_features(
                                    site,
                                    subject_id,
                                    atlas,
                                    bands,
                                    methods,
                                    saving_dir
                                    )

    data = np.load("....", allow_pickle=True)
    n_labels = {"dk": 68, "sch7": 7, "sch17": 17}
    n_label = n_labels[atlas]
    tril_idx = np.tril_indices(n_label, k=-1)

    con_dict = {}
    for band, fminmax in bands.items():
        cons = spectral_connectivity_time(
                                        data,
                                        freqs=np.linspace(fminmax[0], fminmax[1], 5),
                                        method=methods,
                                        average=False,
                                        sfreq=250,
                                        fmin=1,
                                        fmax=45,
                                        faverage=True,
                                        mode="multitaper",
                                    )
        for con, method in zip(cons, methods):
            con_data = con.get_data()[:, :, 0]
            con_data_smaller = con_data[:, np.ravel_multi_index(tril_idx, (n_label, n_label))]
            fname_save = saving_dir / "connectivity" / method / f"{site}_{subject_id}_{atlas}_{band}.npy"
            np.save(fname_save, con_data_smaller)




def compute_graph_features(
                            site,
                            subject_id,
                            atlas,
                            bands,
                            saving_dir
                            )







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
    methods = ["coh", "plv"]

    # compute_spectral_features()
    # compute_connectivity_features()
    # compute_graph_features()
    