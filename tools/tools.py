from pathlib import Path
import random

import mne
from mne.io import read_raw
from mne import make_ad_hoc_cov
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne import (
                setup_source_space,
                make_bem_model,
                make_bem_solution,
                make_forward_solution,
                make_ad_hoc_cov
                )

def read_vhdr_input_fname(fname):
    """
    Checks .vhdr and .vmrk data to have same names, otherwise fix them.
    """
    try:
        raw = read_raw(fname)
    except:
        with open(fname, "r") as file:
            lines = file.readlines()
        
        lines[5] = f'DataFile={fname.stem}.eeg\n'
        lines[6] = f'MarkerFile={fname.stem}.vmrk\n'

        with open(fname, "w") as file:
            file.writelines(lines)
        with open(f"{fname.with_suffix('')}.vmrk", "r") as file:
            lines = file.readlines()
        lines[4] = f'DataFile={fname.stem}.eeg\n'
        with open(f"{fname.with_suffix('')}.vmrk", "w") as file:
            file.writelines(lines)
        raw = read_raw(fname)
    return raw


def create_inv_operator(site):
    main_dir = Path("/Users/payamsadeghishabestari/Tinnorm/material/raws")
    epochs_dir = main_dir / f"preproc_3"

    fnames = []
    for fname in epochs_dir.iterdir():
        if site in fname.stem:
            fnames.append(fname)
    
    ep_fname = random.choice(fnames)
    epochs = mne.read_epochs(ep_fname)
    noise_cov = make_ad_hoc_cov(epochs.info)

    fs_dir = fetch_fsaverage()
    trans = fs_dir / "bem" / "fsaverage-trans.fif"
    src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
    bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    kwargs = {
            "subject": "fsaverage",
            "subjects_dir": None
            }

    src = setup_source_space(**kwargs)
    bem_model = make_bem_model(**kwargs)  
    bem = make_bem_solution(bem_model)

    coreg = Coregistration(epochs.info, fiducials='auto', **kwargs)
    coreg.fit_fiducials()
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0) 
    coreg.omit_head_shape_points(distance=5.0 / 1000)
    coreg.fit_icp(n_iterations=40, nasion_weight=10)
    trans = coreg.trans

    fwd = make_forward_solution(epochs.info,
                                trans=trans,
                                src=src,
                                bem=bem,
                                meg=False,
                                eeg=True
                                )
    inverse_operator = make_inverse_operator(epochs.info,
                                            fwd,
                                            noise_cov
                                            )
    fname_save = main_dir / "invs" / f"{site}-inv.fif"
    fname_save.parent.mkdir(exist_ok=True)
    write_inverse_operator(fname_save, inverse_operator)