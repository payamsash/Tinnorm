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


# Collect available paradigms per subject
files = os.listdir(input_dir)
subjects_dict = {}

for fname in sorted(files, key=lambda f: os.path.getctime(input_dir / f)):
    if not fname.endswith(".vhdr"):
        continue

    if not any(p in fname for p in paradigms):
        continue

    file_path = input_dir / fname
    sub_id = file_path.stem[:4]
    paradigm = file_path.stem[5:]

    subjects_dict.setdefault(sub_id, []).append(paradigm)

# Print missing paradigms per subject (diagnostic)
print("\nMissing paradigms:")
for sub_id, present in subjects_dict.items():
    for par in paradigms:
        if par not in present:
            print(f"{sub_id} ----> {par}")


# Processing + mapping
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

        
        # Case 1: output already exists → mark as present
        if output_fname.exists():
            print(f"Skipping existing {output_fname.name}")

            mapping["antinomics_id"].append(sub_id)
            mapping["paradigm"].append(par)
            mapping["tide_id"].append(tide_id)
            continue

        # Case 2: input missing → do NOT mark as present
        if not input_fname.exists():
            print(f"\033[91mMissing input file: {input_fname.name}\033[0m")
            continue

        # Case 3: process and save
        raw = mne.io.read_raw_brainvision(input_fname, preload=False)

        if raw.info["sfreq"] != sfreq:
            print(f"{sub_id} | {par}: resampling {raw.info['sfreq']} → {sfreq}")
            raw.load_data()
            raw.resample(sfreq)

        raw.save(output_fname, overwrite=False)
        del raw

        mapping["antinomics_id"].append(sub_id)
        mapping["paradigm"].append(par)
        mapping["tide_id"].append(tide_id)

    tide_id += 1

# Build wide CSV
df = pd.DataFrame(mapping)

wide_df = (
    df
    .drop_duplicates(subset=["antinomics_id", "tide_id", "paradigm"])
    .assign(exists=True)
    .pivot_table(
        index=["antinomics_id", "tide_id"],
        columns="paradigm",
        values="exists",
        fill_value=False,
        aggfunc="any",
    )
    .reset_index()
    .sort_values(by="tide_id")
)

wide_df.to_csv("../material/ant_to_tide.csv", index=False)