from pathlib import Path
from mne.io import read_raw, read_raw_ant
from mne_bids import BIDSPath, write_raw_bids
from tqdm.contrib import tzip
from tqdm import tqdm
from mne import set_log_level
set_log_level("Error")

"""
Convert multi-site EEG datasets to BIDS format.

This script standardizes EEG data collected from multiple sites by converting
various native file formats (e.g., BrainVision, FIF, BDF, CNT) into a unified
BIDS-compliant structure using MNE-BIDS. Site-specific handling is implemented
to accommodate differences in file formats, naming conventions, and sampling
rates, with optional resampling applied where required.

The resulting BIDS dataset enables consistent and reproducible downstream
EEG analysis across cohorts.
"""

def convert_to_bids():
    root_dir = Path("/Volumes/Extreme_SSD/payam_data/Tide_rest_data")
    bids_dir = root_dir / "BIDS"


    sites = ["ghent"]

    for site in sites:
        data_dir = root_dir / site
        if site in ["regensburg", "tuebingen"]:
            files = [
                f for f in data_dir.rglob("*.vhdr")
                if not any(p.startswith('.') for p in f.parts)
            ]

        if site == "illinois":
            files = [
                f for f in data_dir.rglob("*.cdt")
                if not any(p.startswith('.') for p in f.parts)
            ]
            for fname in files:
                dpo_file = fname.with_suffix('.cdt.dpo')
                dpa_file = fname.with_suffix('.cdt.dpa')
                if dpo_file.exists():
                    dpo_file.rename(dpa_file)
        
        if site == "austin":
            files = [
                f for f in data_dir.rglob("*.fif")
                if not any(p.startswith('.') for p in f.parts)
            ]
        
        if site == "dublin":
            files = [
                f for f in data_dir.rglob("*_open.bdf")
                if not any(p.startswith('.') for p in f.parts)
            ]

        if site == "ghent":
            files = [
                f for f in data_dir.rglob("*.cnt")
                if not any(p.startswith('.') for p in f.parts)
            ]
            files = [f for f in files if "ses-1" in f.stem]
            
            ## convert to fif
            for fname in tqdm(files):
                raw = read_raw_ant(fname)
                raw.save(fname.with_suffix(".fif"), overwrite=True)

            files = [
                f for f in data_dir.rglob("*.fif")
                if not any(p.startswith('.') for p in f.parts)
            ]
        
        if site == "zuerich":
            bids_dir = Path("/Volumes/Extreme_SSD/payam_data/Tide_rest_data/BIDS")
            data_dir = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/tide")
            folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
            subject_ids = [f.name for f in folders]
            files = []
            for folder in folders:
                files.append(folder / "eeg" / f"{folder.name}_rest.fif")
        
        if not site == "zuerich": 
            subject_ids = [f.stem[:5] for f in files]
        
        for fname, subject_id in tzip(files, subject_ids,
                                        total=len(subject_ids),
                                        desc=f"Converting {site}"):
            print(subject_id)
            if subject_id == "70005": # tlyd
                continue
            
            raw = read_raw(fname)
            bids_path = BIDSPath(
                subject=subject_id,
                session="01",
                task="rest",
                datatype="eeg",
                description=site,
                root=bids_dir
            )
            write_raw_bids(raw, bids_path=bids_path, overwrite=False)

if __name__ == "__main__":
    convert_to_bids()