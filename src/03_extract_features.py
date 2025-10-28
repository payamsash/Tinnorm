import os
from mne import read_epochs
from mne.datasets import fetch_fsaverage
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import welch
from mne_connectivity import spectral_connectivity_time

from mne import (set_log_level,
                    read_epochs,
                    read_labels_from_annot,
                    extract_label_time_course,
                    make_forward_solution,
                    make_ad_hoc_cov
                    )
from mne.minimum_norm import (read_inverse_operator,
                                apply_inverse_epochs,
                                make_inverse_operator,
                                write_inverse_operator)
from mne.time_frequency import psd_array_multitaper
from fooof import FOOOF

set_log_level("Error")

def compute_source_features(
                            subject_id,
                            bids_root,
                            preproc_level,
                            freq_bands,
                            con_methods,
                            atlas="aparc"
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
    inv_dir = bids_root.parent / "invs"
    os.makedirs(inv_dir, exist_ok=True)
    inv_fname = inv_dir / f"{site}-inv.fif"

    ## read epochs
    subject_dir = bids_root / f"sub-{subject_id}" / "ses-01" / "eeg"
    epochs_fname = subject_dir / f"preproc_{preproc_level}-epo.fif"
    epochs = read_epochs(epochs_fname, preload=True)
    epochs.set_eeg_reference("average", projection=True)

    ## read labels
    if atlas == "aparc":
        labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-1]
    if atlas == "aparc.a2009s":
        labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-2]

    ## read_inverse_operator
    if inv_fname.isdir():
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
    
    ## compute power in labels
    label_ts = np.array(label_ts)
    n_epochs, n_labels, n_times = label_ts.shape
    reshaped_data = label_ts.reshape(-1, n_times)
    freqs, psd = welch(reshaped_data, epochs.info["sfreq"], axis=-1, nperseg=min(256, n_times))

    columns = []
    labels_power = []
    for key, value in freq_bands.items():
        min_freq, max_freq = list(value)
        band_mask = (freqs >= min_freq) & (freqs <= max_freq)
        band_powers = np.trapz(psd[:, band_mask], freqs[band_mask], axis=-1)
        labels_power.append(band_powers.reshape(n_epochs, n_labels))
        columns += [f"{lb.name}_{key}" for lb in labels]

    labels_power = np.concatenate(labels_power, axis=1)
    df_power = pd.DataFrame(labels_power, columns=columns)

    ## save it
    csv_fname = subject_dir / f"power_preproc_{preproc_level}.csv"
    df_power.to_csv(csv_fname)
    del df_power
        
    ## compute connectivity (pli, plv, coh)
    lb_names = [lb.name for lb in labels]
    i_lower, j_lower = np.tril_indices_from(np.zeros(shape=(label_ts.shape[1], label_ts.shape[1])), k=-1)
    columns = []
    freq_cons = []
    for key, value in freq_bands.items(): 
        for con_method in con_methods:
            con = spectral_connectivity_time(
                                            label_ts,
                                            freqs=np.arange(value[0], value[1], 5),
                                            method=con_method,
                                            average=False,
                                            sfreq=epochs.info["sfreq"],
                                            mode="cwt_morlet",
                                            fmin=value[0],
                                            fmax=value[1],
                                            faverage=True
                                            )
            con_matrix = np.squeeze(con.get_data(output="dense")) # n_epochs * n_labels * n_labels

            cons = []
            for ep_con in con_matrix:
                ep_con_value = ep_con[i_lower, j_lower]
                cons.append(ep_con_value)
            cons = np.array(cons)
            freq_cons.append(cons)

            con_labels = [f"{lb_names[i]}_vs_{lb_names[j]}_{key}_{con_method}" for i, j in zip(i_lower, j_lower)]
            columns += con_labels

    freq_cons = np.concatenate(freq_cons, axis=-1)
    df_conn = pd.DataFrame(freq_cons, columns=columns).T

    ## save it
    csv_fname = subject_dir / f"conn_preproc_{preproc_level}.csv"
    df_conn.to_csv(csv_fname)
    del df_conn

    # compute aperiodic param per whole recording
    fmin, fmax = 1, 40
    ep_psds, freqs = psd_array_multitaper(label_ts, epochs.info["sfreq"], fmin, fmax)
    avg_psd = ep_psds.mean(axis=0)
    fm = FOOOF()
    columns = []
    for lb_idx, lb_name in enumerate(lb_names):
        fm.fit(freqs=freqs, power_spectrum=avg_psd[lb_idx], freq_range=[fmin, fmax])
        offset, exponent = fm.aperiodic_params_
        columns.append({
                        f'{lb_name}_offset': offset,
                        f'{lb_name}_exponent': exponent
                        })
    df_aperiodic = pd.DataFrame(columns)

    ## save it
    csv_fname = subject_dir / f"aperiodic_preproc_{preproc_level}.csv"
    df_aperiodic.to_csv(csv_fname)
    del df_aperiodic

    ## reduce size
    for csv_mod in ["power", "conn", "aperiodic"]:
        csv_fname = subject_dir / f"{csv_mod}_preproc_{preproc_level}.csv"
        with zipfile.ZipFile(csv_fname.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_fname, os.path.basename(csv_fname))
        os.remove(csv_fname)



if __name__ == "__main__":
    preproc_level = 1
    bids_root = Path("/Users/payamsadeghishabestari/temp_folder/tide_subjects")
    subject_ids = sorted([f.name[4:] for f in bids_root.iterdir() if f.is_dir()])


    freq_bands = {"alpha": [8, 13], "theta": [4, 8]}
    con_methods = ["pli", "plv", "coh"]
    
    compute_source_features(subject_id, bids_root, preproc_level, atlas="aparc")