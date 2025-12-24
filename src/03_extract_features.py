import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.signal import welch

from mne import (
    set_log_level,
    read_epochs,
    read_labels_from_annot,
    extract_label_time_course,
    make_forward_solution,
    make_ad_hoc_cov,
)
from mne.channels import make_standard_montage
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import (
    read_inverse_operator,
    apply_inverse_epochs,
    make_inverse_operator,
    write_inverse_operator,
)
from mne.time_frequency import psd_array_multitaper
from mne_connectivity import spectral_connectivity_time
from fooof import FOOOF

set_log_level("ERROR")

"""
Feature extraction from preprocessed EEG data in BIDS format.

This script computes sensor- and source-level features from preprocessed
resting-state EEG epochs stored in a BIDS-compliant structure. Depending on
the selected mode, features are extracted either at the sensor level or after
source reconstruction using an fsaverage template and cortical parcellations.

Computed features include:
- Band-limited spectral power across predefined frequency bands
- Time-resolved functional connectivity metrics (PLI, PLV, coherence)
- Aperiodic (1/f) spectral parameters estimated using FOOOF

Source-level analysis employs minimum-norm inverse solutions and atlas-based
label time series extraction. Outputs are saved as compressed CSV files to
reduce storage requirements and facilitate large-scale downstream analyses,
such as machine learning or normative modeling.

"""

def compute_features(
                    subject_id,
                    bids_root,
                    preproc_level,
                    freq_bands,
                    con_methods,
                    mode="sensor",
                    atlas="aparc",
                    compute_power=True,
                    compute_conn=True,
                    compute_aperiodic=True
                    ):

    overwrite = True
    site_map = {
        "1": "austin",
        "2": "dublin",
        "3": "ghent",
        "4": "illinois",
        "5": "regensburg",
        "6": "tuebingen",
        "7": "zuerich"
        }

    site = site_map.get(str(subject_id)[0], "unknown")
    standard_montage = make_standard_montage("easycap-M1")

    ## read epochs
    subject_dir = bids_root / f"sub-{subject_id}"  # / "ses-01" / "eeg" add it when in cloud
    epochs_fname = subject_dir / f"preproc_{preproc_level}-epo.fif"
    epochs = read_epochs(epochs_fname, preload=True)
    epochs.set_eeg_reference("average", projection=True)
    
    ## read channels
    if mode == "sensor":
        epochs.pick_types(eeg=True)
        epochs_int = epochs.copy().interpolate_to(standard_montage)
        epochs_ts = epochs_int.get_data(picks="eeg")
        ch_names = epochs_int.info["ch_names"]
    
    ## read labels
    if mode == "source":
        inv_dir = bids_root.parent / "invs"
        os.makedirs(inv_dir, exist_ok=True)
        inv_fname = inv_dir / f"{site}-inv.fif"

        if atlas == "aparc":
            labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-1]
        if atlas == "aparc.a2009s":
            labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-2]

        ## read_inverse_operator
        if inv_fname.is_file():
            inverse_operator = read_inverse_operator(inv_fname)
    
        else:
            noise_cov = make_ad_hoc_cov(epochs.info)
            fs_dir = fetch_fsaverage()
            trans = fs_dir / "bem" / "fsaverage-trans.fif"
            src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
            bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
            fwd = make_forward_solution(
                                        epochs.info,
                                        trans=trans,
                                        src=src,
                                        bem=bem,
                                        meg=False,
                                        eeg=True
                                        )
            inverse_operator = make_inverse_operator(
                                                    epochs.info,
                                                    fwd,
                                                    noise_cov
                                                    )
            write_inverse_operator(
                                fname=inv_fname,
                                inv=inverse_operator,
                                overwrite=overwrite
                                )

        ## extract brain label signals
        stcs = apply_inverse_epochs(
                                    epochs,
                                    inverse_operator,
                                    lambda2=1.0 / (1.0 ** 2),
                                    method="dSPM",
                                    label=None,
                                    pick_ori="normal",
                                    return_generator=False
                                    )
        label_ts = extract_label_time_course(
                                            stcs,
                                            labels,
                                            inverse_operator["src"],
                                            mode="mean_flip",
                                            return_generator=False,
                                            )
        label_ts = np.array(label_ts)
        lb_names = [lb.name for lb in labels]

    ## compute power in channels/labels
    if compute_power:
        if mode == "sensor":
            print("Computing band powers in sensor space ...")
            psd_chs, freqs = epochs_int.compute_psd(
                                                    fmin=freq_bands["delta"][0],
                                                    fmax=freq_bands["gamma"][1]
                                                    ).get_data(return_freqs=True)
            
            ## mask to get each
            columns = []
            all_band_powers = []
            for band_name, (min_freq, max_freq) in freq_bands.items():
                band_mask = (freqs >= min_freq) & (freqs <= max_freq)
                band_powers = np.trapz(
                    psd_chs[:, :, band_mask],
                    freqs[band_mask],
                    axis=-1
                )
                all_band_powers.append(band_powers)
                columns.extend([f"{ch}_{band_name}" for ch in ch_names])

            chs_power = np.concatenate(all_band_powers, axis=1)
            df_power = pd.DataFrame(chs_power, columns=columns)

        if mode == "source":
            print("Computing band powers ...")
            n_epochs, n_labels, n_times = label_ts.shape
            reshaped_data = label_ts.reshape(-1, n_times)

            psd, freqs = psd_array_multitaper(
                                            reshaped_data,
                                            sfreq=epochs.info["sfreq"],
                                            fmin=freq_bands["delta"][0],
                                            fmax=freq_bands["gamma"][1]
                                            )
            columns = []
            labels_power = []
            for band_name, (min_freq, max_freq) in freq_bands.items():
                band_mask = (freqs >= min_freq) & (freqs <= max_freq)

                # integrate PSD over band
                band_powers = np.trapz(
                                        psd[:, band_mask],
                                        freqs[band_mask],
                                        axis=-1
                                    )
                labels_power.append(band_powers.reshape(n_epochs, n_labels))
                columns.extend([f"{lb.name}_{band_name}" for lb in labels])

            labels_power = np.concatenate(labels_power, axis=1)
            df_power = pd.DataFrame(labels_power, columns=columns)

        ## save it
        print("Saving band powers in sensor space ...")
        csv_fname = subject_dir / f"power_{mode}_preproc_{preproc_level}.csv"
        df_power.to_csv(csv_fname)
        del df_power
        
    ## compute connectivity (pli, plv, coh)
    if compute_conn:

        if mode == "sensor":
            data_ts = epochs_ts
            names = ch_names
        elif mode == "source":
            data_ts = label_ts
            names = lb_names
        else:
            raise ValueError("mode must be 'sensor' or 'source'")

        n_nodes = data_ts.shape[1]
        i_lower, j_lower = np.tril_indices(n_nodes, k=-1)

        columns = []
        freq_cons = []

        for key, value in freq_bands.items():
            n_cycles = value[1] / 6 if key == "delta" else 7

            for con_method in con_methods:
                print(f"Computing {con_method} connectivity values for {key} frange...")

                con = spectral_connectivity_time(
                    data_ts,
                    freqs=np.arange(value[0], value[1], 5),
                    method=con_method,
                    average=False,
                    sfreq=epochs.info["sfreq"],
                    mode="cwt_morlet",
                    fmin=value[0],
                    fmax=value[1],
                    faverage=True,
                    n_cycles=n_cycles,
                )

                con_matrix = np.squeeze(con.get_data(output="dense"))  # n_epochs × n_nodes × n_nodes

                cons = []
                for ep_con in con_matrix:
                    cons.append(ep_con[i_lower, j_lower])

                freq_cons.append(np.array(cons))

                columns += [
                    f"{names[i]}_vs_{names[j]}_{key}_{con_method}"
                    for i, j in zip(i_lower, j_lower)
                ]

        freq_cons = np.concatenate(freq_cons, axis=-1)
        df_conn = pd.DataFrame(freq_cons, columns=columns).T

        print("Saving connectivity values ...")
        df_conn.to_csv(subject_dir / f"conn_{mode}_preproc_{preproc_level}.csv")
        del df_conn

    # compute aperiodic param per whole recording
    if compute_aperiodic:

        if mode == "sensor":
            data_ts = epochs_ts
            names = ch_names
        elif mode == "source":
            data_ts = label_ts
            names = lb_names
        else:
            raise ValueError("mode must be 'sensor' or 'source'")

        print("Computing aperiodic values ...")

        fmin, fmax = 1, 40
        ep_psds, freqs = psd_array_multitaper(
            data_ts, epochs.info["sfreq"], fmin, fmax
        )

        avg_psd = ep_psds.mean(axis=0)
        fm = FOOOF()
        row_dict = {}

        for idx, name in enumerate(names):
            print(f"Processing {mode} {idx + 1} / {len(names)} ...")
            fm.fit(
                freqs=freqs,
                power_spectrum=avg_psd[idx],
                freq_range=[fmin, fmax],
            )
            offset, exponent = fm.aperiodic_params_
            row_dict[f"{name}_offset"] = offset
            row_dict[f"{name}_exponent"] = exponent

        df_aperiodic = pd.DataFrame([row_dict])

        ## save it
        print("Saving aperiodic values ...")
        csv_fname = subject_dir / f"aperiodic_{mode}_preproc_{preproc_level}.csv"
        df_aperiodic.to_csv(csv_fname)
        del df_aperiodic

    ## reduce size
    for csv_mod in ["power", "conn", "aperiodic"]:
        csv_fname = subject_dir / f"{csv_mod}_{mode}_preproc_{preproc_level}.csv"
        if csv_fname.is_file():
            with zipfile.ZipFile(csv_fname.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_fname, os.path.basename(csv_fname))
            os.remove(csv_fname)


if __name__ == "__main__":
    bids_root = Path("/home/ubuntu/volume/Tinnorm/BIDS")
    subject_ids = sorted([f.name[4:] for f in bids_root.iterdir() if f.is_dir()])
    preproc_levels = [1, 2, 3]
    con_methods = ["pli", "plv", "coh"]
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

    ## running on all subjects and preproc levels
    for subject_id in subject_ids:
        for preproc_level in preproc_levels: 
            if (bids_root / f"sub-{subject_id}" / "ses-01" / "eeg" / "preproc_3-epo.fif").is_file():
                print(f"Working on subject {subject_id} and preproc level {preproc_level} ...")
                compute_features(
                                    subject_id,
                                    bids_root,
                                    preproc_level,
                                    freq_bands,
                                    con_methods,
                                    mode="sensor",
                                    atlas="aparc",
                                    compute_power=True,
                                    compute_conn=True,
                                    compute_aperiodic=True
                                    )