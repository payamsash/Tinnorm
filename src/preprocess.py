from pathlib import Path
import pickle
import numpy as np
import mne
from mne import Report
from mne_icalabel import label_components
from mne.preprocessing import ICA
from autoreject import AutoReject
from mne.preprocessing import find_bad_channels_lof
from tools.tools import read_vhdr_input_fname


with open("../material/fname_dict.pkl", "rb") as f:
    mappings = pickle.load(f)

sites = ["dublin", "illinois", "regensburg", "tuebingen", "zuerich"]

main_dir = Path("/Users/payamsadeghishabestari/Tinnorm/material/raws")
montage = mne.channels.make_standard_montage("easycap-M1")
ec_trigger_code, eo_trigger_code = 1, 3
cutoff = 3 # seconds
ica_label_thr = 0.7
overwrite = False

line_freqs = dict(zip(sites, [50, 60, 50, 50, 50]))

for site in sites[3:4]:
    for subject_id, fname in mappings[site].items():

        fname_save_0 = main_dir / "orig" / f"{site}_{subject_id}.fif"
        fname_save_1 = main_dir / "preproc_1" / f"{site}_{subject_id}.fif"
        fname_save_2 = main_dir / "preproc_2" / f"{site}_{subject_id}.fif"
        fname_save_3 = main_dir / "preproc_3" / f"{site}_{subject_id}.fif"
        fname_save_4 = main_dir / "reports" / f"{site}_{subject_id}.html"
        for dir_name in [fname_save_0, fname_save_1, fname_save_2, fname_save_3, fname_save_4]:
            dir_name.parent.mkdir(exist_ok=True)

        if fname_save_4.is_file():
            continue
        
        if site == "dublin":
            raw = mne.io.read_raw(fname, exclude=("EXG"))
            
            tmin = raw.first_samp / raw.info["sfreq"]
            tmax = raw.last_samp / raw.info["sfreq"]
            raw.crop(tmin=tmin + cutoff, tmax=tmax - cutoff)

        elif site == "illinois":
            dpo_file = fname.with_suffix('.cdt.dpo')
            dpa_file = fname.with_suffix('.cdt.dpa')
            if dpo_file.exists():
                dpo_file.rename(dpa_file)

            raw = mne.io.read_raw(fname)
            raw.drop_channels(["F11", "F12", "FT11", "FT12", "M1", "M2", "Cb1", "Cb2"])
            transform = mne.read_trans("../material/Illinois-trans.fif")
            raw.info["dev_head_t"] = transform
            
            tmin = raw.first_samp / raw.info["sfreq"]
            tmax = raw.last_samp / raw.info["sfreq"]
            raw.crop(tmin=tmin + cutoff, tmax=tmax - cutoff)

        elif site in ["regensburg", "tuebingen"]:
            raw = read_vhdr_input_fname(fname)

            tmin = raw.first_samp / raw.info["sfreq"]
            tmax = raw.last_samp / raw.info["sfreq"]
            raw.crop(tmin=tmin + cutoff, tmax=tmax - cutoff)

        elif site == "zuerich":
            ch_types = {
                "O1": "eog",
                "O2": "eog",
                "PO7": "eog",
                "PO8": "eog",
                "Pulse": "ecg",
                "Resp": "ecg",
                "Audio": "stim"
            }
            raw = read_vhdr_input_fname(fname)
            raw.set_channel_types(ch_types)

            ## take only eyes open part
            events, events_dict = mne.events_from_annotations(raw)
            eo_events = events[events[:, 2] == eo_trigger_code] 
            ec_events = events[events[:, 2] == ec_trigger_code]

            eo_events = eo_events[:, 0] + cutoff * raw.info["sfreq"]
            ec_events = ec_events[:, 0] - cutoff * raw.info["sfreq"]
            cropped_raws = []
            for start_idx, end_idx in zip(eo_events, ec_events):
                cropped_raws.append(raw.copy().crop(
                                                    tmin=start_idx / raw.info["sfreq"],
                                                    tmax=end_idx / raw.info["sfreq"])
                                                    )
            raw = mne.concatenate_raws(cropped_raws)

        report = Report(title=f"report_subject_{subject_id}")

        ## orig
        raw.pick(["eeg"])
        raw.set_montage(montage)
        raw.load_data()
        raw.resample(sfreq=250)
        raw.save(fname_save_0, overwrite=overwrite)
        report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=True)

        ## preproc 1
        raw.filter(1, 100)
        raw.notch_filter(freqs=line_freqs[site], picks="eeg", notch_widths=1)
        raw.set_eeg_reference("average", projection=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)
        epochs.save(fname_save_1, overwrite=overwrite)
        
        ## preproc 2
        noisy_chs, lof_scores = find_bad_channels_lof(raw, threshold=3, return_scores=True)
        epochs.info["bads"] = noisy_chs
        report.add_epochs(epochs=epochs, title="Epochs Info", psd=False)
        
        epochs = epochs.interpolate_bads()
        ar = AutoReject(
                        n_interpolate=np.array([1, 4, 8]),
                        consensus=np.linspace(0, 1.0, 11),
                        cv=5,
                        n_jobs=1,
                        random_state=11,
                        verbose=True
                        )
        ar.fit(epochs)
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)
        epochs_ar.save(fname_save_2, overwrite=overwrite)
        fig_ar = reject_log.plot(show_names=1)
        report.add_figure(fig=fig_ar, title="Autoreject Log", image_format="PNG")

        ## preproc 3
        ica = ICA(n_components=0.95, max_iter=800, method='infomax', fit_params=dict(extended=True))
        try:
            ica.fit(raw)
        except:
            ica = ICA(n_components=5, max_iter=800, method='infomax', fit_params=dict(extended=True))
            ica.fit(epochs_ar)

        ic_dict = label_components(epochs_ar, ica, method="iclabel")
        ic_labels = ic_dict["labels"]
        ic_probs = ic_dict["y_pred_proba"]
        eog_indices = [idx for idx, label in enumerate(ic_labels) \
                        if label == "eye blink" and ic_probs[idx] > ica_label_thr]
        eog_indices_fil = [x for x in eog_indices if x <= 10]
        
        if len(eog_indices) > 0:
            eog_components = ica.plot_properties(epochs_ar,
                                                    picks=eog_indices_fil,
                                                    show=False,
                                                    )
            ica.apply(epochs_ar, exclude=eog_indices_fil)
        
        epochs_ar.save(fname_save_3, overwrite=overwrite)
        if len(eog_indices) > 0:
            report.add_figure(fig=eog_components, title="EOG Components", image_format="PNG")
        report.save(fname=fname_save_4, open_browser=False, overwrite=overwrite)
