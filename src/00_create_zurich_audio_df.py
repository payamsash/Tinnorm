import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat
from warnings import warn

data_dir = Path("/Volumes/G_USZ_ORL$/Research/ANTINOMICS/data")
audio_dir = data_dir / "audiometry"
ant_subjects = [p.stem for p in audio_dir.iterdir() if p.is_dir()]

ant_to_tide_fname = "../material/ant_to_tide.csv"
df_map = pd.read_csv(ant_to_tide_fname)

# The frequencies you want as columns
target_freqs = [125, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000]

freq_cols = []
for hemi in ["L", "R"]:
    for f in target_freqs:
        freq_cols.append(f"A{hemi}E_{f}")

rows = []
for subject in tqdm(ant_subjects):
    
    subject_audio_dir = audio_dir / subject
    if subject not in list(df_map["antinomics_id"]):
        warn(f"{subject} is not a valid antinomic ID!")
        continue

    tide_id = df_map.loc[df_map['antinomics_id'] == subject, 'tide_id'].values[0]
    row = {"antinomics_id": subject, "tide_id": tide_id}

    # Initialize all possible columns with NaN so missing ones remain NaN
    for col in freq_cols:
        row[col] = np.nan

    if subject_audio_dir.exists():
        for fname in subject_audio_dir.iterdir():
            for hemi in ["L", "R"]:
                if fname.name.lower().startswith(f"{subject.lower()} {hemi.lower()}"):

                    data = loadmat(fname)
                    freqs = data["betweenRuns"]["var1Sequence"][0][0][0]
                    thrs  = data["betweenRuns"]["thresholds"][0][0][0]

                    # Sort
                    order = np.argsort(freqs)
                    freqs_sorted = freqs[order]
                    thrs_sorted  = thrs[order]

                    for f, thr in zip(freqs_sorted, thrs_sorted):
                        f_int = int(f)
                        col_name = f"A{hemi}E_{f_int}"
                        if col_name in row:
                            row[col_name] = thr

    rows.append(row)

df = pd.DataFrame(rows)

## fix 2 frequency calibration
for hemi in ["L", "R"]:
    df[f"A{hemi}E_250"] = df[f"A{hemi}E_250"] - 8
    df[f"A{hemi}E_2000"] = df[f"A{hemi}E_2000"] + 3

df = df[["antinomics_id", "tide_id"] + freq_cols]
df.to_csv("../material/zurich_audio.csv")