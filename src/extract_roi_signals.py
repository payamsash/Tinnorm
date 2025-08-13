from pathlib import Path
import numpy as np
from mne.minimum_norm import apply_inverse_epochs
from mne.minimum_norm import read_inverse_operator
from mne import read_epochs, read_labels_from_annot

def extract_roi_signals():
    main_dir = Path("/Users/payamsadeghishabestari/Tinnorm/material/raws")
    snr = 3
    sites = ["dublin", "illinois", "regensburg", "tuebingen", "zuerich"]

    labels_dk = read_labels_from_annot(subject="fsaverage", parc="aparc")[:-1] 
    labels_sh_7 = read_labels_from_annot(subject="fsaverage", parc="Yeo2011_7Networks_N1000")[:-2] # removing medial wall
    labels_sh_17 = read_labels_from_annot(subject="fsaverage", parc="Yeo2011_17Networks_N1000")[:-2]

    for site in sites:

        fname_inv = main_dir / "invs" / f"{site}-inv.fif"
        inv_operator = read_inverse_operator(fname_inv)

        for preproc_level in [1, 2, 3]:
            epochs_dir = main_dir / f"preproc_{preproc_level}"
            for ep_fname in epochs_dir.iterdir():
                if site in ep_fname.stem:
                    subject_id = ep_fname.stem.rsplit("_", 1)[1]

                    ## compute source estimate object
                    epochs = read_epochs(ep_fname, preload=True)
                    epochs.set_eeg_reference("average", projection=True)
                    stcs = apply_inverse_epochs(
                                        epochs,
                                        inv_operator,
                                        lambda2=1.0 / snr**2,
                                        pick_ori='normal',
                                        )
                    
                    ## compute label time series
                    for atlas, atlas_name in zip([labels_dk, labels_sh_7, labels_sh_17], ["dk", "sch7", "sch17"]):
                        labels_ts = []
                        for stc in stcs:
                            label_ts = stc.extract_label_time_course(
                                                                    atlas,
                                                                    inv_operator["src"],
                                                                    mode="mean_flip",
                                                                    allow_empty=True
                                                                    )
                            labels_ts.append(label_ts)

                        fname_save = main_dir / "labels_ts" / f"preproc_{preproc_level}" / f"{site}_{subject_id}_{atlas_name}.npy"
                        fname_save.parent.mkdir(exist_ok=True)
                        np.save(fname_save, np.array(labels_ts))

if __name__ == "__main__":
    extract_roi_signals()