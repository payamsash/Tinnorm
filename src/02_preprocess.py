from pathlib import Path
import numpy as np
from mne.io import read_raw
from mne.channels import make_standard_montage
from mne import make_fixed_length_epochs
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import AutoReject
from mne import Report, set_log_level
set_log_level("Error")

def preprocess(subject_id, bids_root):

    site_map = {
        "1": "austin",
        "2": "dublin",
        "3": "ghent",
        "4": "illinois",
        "5": "regensburg",
        "6": "tuebingen",
        "7": "zuerich"
        }

    site_params = {
        "austin": {
            "suffix": "vhdr",
            "chs_to_drop": ["VREF"],
            "montage": "GSN-HydroCel-64_1.0",
        },
        "dublin": {
            "suffix": "bdf",
            "chs_to_drop": [f"EXG{i}" for i in range(9)] + ["Status"],
            "montage": "easycap-M1",
        },
        "ghent": {
            "suffix": "vhdr",
            "chs_to_drop": ["M1", "M2", "PO5", "PO6", "EOG"],
            "montage": "easycap-M1",
        },
        "illinois": {
            "suffix": "vhdr",
            "chs_to_drop": ["VEOG", "HEOG", "Trigger", "F11", "F12",
                            "FT11", "FT12", "M1", "M2", "Cb1", "Cb2"],
            "montage": "easycap-M1",
        },
        "regensburg": {
            "suffix": "vhdr",
            "chs_to_drop": ["audio"],
            "montage": "easycap-M1",
        },
        "tuebingen": {
            "suffix": "vhdr",
            "chs_to_drop": ["audio", "aux1"],
            "montage": "easycap-M1",
        },
        "zuerich": {
            "suffix": "vhdr",
            "chs_to_drop": ['Pulse', 'GSR', 'X', 'Y', 'Z', 'Resp', 'Photo', 'Audio',
                            "O1", "O2", "PO7", "PO8"],
            "montage": "easycap-M1",
        },
    }

    
    site = site_map.get(str(subject_id)[0], "unknown")
    folder = bids_root / f"sub-{subject_id}" / "ses-01" / "eeg"
    params = site_params.get(site, {"suffix": "vhdr", "chs_to_drop": [], "montage": "easycap-M1"})
    suffix = params["suffix"]
    chs_to_drop = params["chs_to_drop"]
    montage_name = params["montage"]
    
    fname = folder / f"sub-{subject_id}_ses-01_task-rest_desc-{site}_eeg.{suffix}"
    sfreq = 250.0
    l_freq = 1.0
    h_freq = 100.0
    crop_duration = 5.0
    epoch_duration = 10.0
    overwrite = True

    raw = read_raw(fname, preload=True)
    raw.drop_channels(ch_names=chs_to_drop, on_missing="warn")
    montage = make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")

    ## preproc_1
    raw.crop(tmin=crop_duration, tmax=raw.times[-1] - crop_duration)
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.resample(sfreq=sfreq)
    raw.set_eeg_reference("average", projection=False)
    epochs_1 = make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
    epochs_1.save(folder / "preproc_1-epo.fif", overwrite=overwrite)

    ## preproc_2
    ar = AutoReject(
                    n_interpolate=np.array([1, 4, 8]),
                    consensus=np.linspace(0, 1.0, 11),
                    cv=5,
                    n_jobs=1,
                    random_state=11,
                    verbose=True
                    )
    ar.fit(epochs_1)
    epochs_2, reject_log = ar.transform(epochs_1, return_log=True)
    epochs_2.save(folder / "preproc_2-epo.fif", overwrite=overwrite)

    ## preproc 3
    ica = ICA(n_components=0.999, method='fastica')
    try:
        ica.fit(epochs_2)
    except:
        ica = ICA(n_components=5, method='fastica')
        ica.fit(epochs_2)

    ic_labels = label_components(epochs_2, ica, method="iclabel")["labels"]
    artifact_idxs = [idx for idx, label in enumerate(ic_labels) \
                    if not label in ["brain", "other"]]
    epochs_3 = ica.apply(epochs_2.copy(), exclude=artifact_idxs)
    epochs_3.save(folder / "preproc_3-epo.fif", overwrite=overwrite)

    ## create report
    ## preproc_1
    report = Report(title="10001")
    report.add_epochs(epochs=epochs_1, image_kwargs={}, psd=False, title='Epochs_preproc1')

    ## preproc_2
    fig_reject = reject_log.plot(show=False)
    report.add_figure(fig=fig_reject, title="autoreject log", image_format="PNG")
    fig_drop = epochs_2.plot_drop_log()
    report.add_figure(fig=fig_drop, title="epochs drop log", image_format="PNG")

    ## preproc_3
    report.add_ica(
        ica=ica,
        title="ICA cleaning",
        picks=artifact_idxs,
        inst=None,
        eog_evoked=None,
        eog_scores=None,
        n_jobs=None
    )
    artifact_labels = np.array(ic_labels)[artifact_idxs].tolist()
    label_html = "<p>ICLabel classification results:</p>\n<ol>\n"
    for label in artifact_labels:
        label_html += f"  <li>{label}</li>\n"
    label_html += "</ol>"
    report.add_html(title="ICLabel Results", html=label_html)
    report.save(fname=folder / "report.html", open_browser=False, overwrite=overwrite)

if __name__ == "__main__":
    bids_root = Path("/Volumes/Extreme_SSD/payam_data/Tide_data/BIDS")
    subject_ids = sorted([f.name[4:] for f in bids_root.iterdir() if f.is_dir()])
    subject_ids = [sub for sub in subject_ids if sub.startswith("1")]
    
    for subject_id in subject_ids:
        print(f"working on subject {subject_id} ...")
        preprocess(subject_id, bids_root)