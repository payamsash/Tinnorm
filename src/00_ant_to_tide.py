import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import mne
mne.set_log_level("ERROR")

input_dir = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/eeg")
output_dir = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data/tide")

sfreq = 1000.0
paradigms = ("omi", "gpias", "rest", "rest_2", "xxxxx", "xxxxy")
mapping = {
            "antinomics_id": [],
            "paradigm": [],
            "tide_id": []
            }

files = os.listdir(input_dir)
subjects_dict = {}
for fname in sorted(files, key=lambda f: os.path.getctime(input_dir / f)):
    cond1 = fname.endswith(".vhdr")
    cond2 = any(sub in fname for sub in paradigms)

    if cond1 and cond2:
        file_path = input_dir / fname
        sub_id = file_path.stem[:4]
        paradigm = file_path.stem[5:]

        if not sub_id in subjects_dict:
            subjects_dict[sub_id] = []
        
        subjects_dict[sub_id].append(paradigm)


##########
for key, val in subjects_dict.items():
    for par in paradigms:
        if par not in val:
            print(f"{key} ----> {par}")
##########

sfreq = 1000.0
mapping = {
    "antinomics_id": [],
    "paradigm": [],
    "tide_id": [],
}

tide_id = 70001
for sub_id in tqdm(subjects_dict.keys()):
    print(f"\033[92mworking on subject {sub_id} | {tide_id}\033[0m")
    
    subject_tide_folder = output_dir / str(tide_id)
    eeg_dir = subject_tide_folder / "eeg"
    audiometry_dir = subject_tide_folder / "audiometry"

    eeg_dir.mkdir(parents=True, exist_ok=True)
    audiometry_dir.mkdir(parents=True, exist_ok=True)

    for par in paradigms:
        input_fname = input_dir / f"{sub_id}_{par}.vhdr"
        output_fname = eeg_dir / f"{tide_id}_{par}.fif"

        if output_fname.exists():
            print(f"Skipping existing {output_fname.name}")

            mapping["antinomics_id"].append(sub_id)
            mapping["paradigm"].append(par)
            mapping["tide_id"].append(tide_id)

            continue

        if not input_fname.exists():
            print(f"\033[91mMissing input file: {input_fname.name}\033[0m")
            continue

        raw = mne.io.read_raw_brainvision(input_fname, preload=False)
        if raw.info["sfreq"] != sfreq:  
            print(f"{sub_id} | {par}: resampling {raw.info['sfreq']} → {sfreq}")
            raw.load_data()
            raw.resample(sfreq)

        print(output_fname)
        raw.save(output_fname, overwrite=False) # just to make sure
        del raw

        mapping["antinomics_id"].append(sub_id)
        mapping["paradigm"].append(par)
        mapping["tide_id"].append(tide_id)


    mapping["antinomics_id"].append(sub_id)
    mapping["paradigm"].append(par)
    mapping["tide_id"].append(tide_id)

    tide_id += 1

df = pd.DataFrame(mapping)
df.to_csv("ant_to_tide.csv")