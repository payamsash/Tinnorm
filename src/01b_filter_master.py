"""
01b_filter_master.py

Applies two exclusion criteria to master_new.csv and writes master_clean.csv:

  1. No feature file  — subject must have a power_sensor_preproc_1.zip file
                        in the features directory (EEG pipeline ran successfully).
  2. Missing PTA4_HF  — subject must have a valid bilateral high-frequency PTA
                        value (PTA4_HF = mean of PTA_HF_ARE and PTA_HF_ALE from
                        the TIDE audiogram file).  PTA4_HF is used as a covariate
                        in ComBat harmonization and normative modelling; NaN values
                        cause those steps to fail.

Run order: after 01_create_tinnorm_df.py, before 02_plot_demographics.py.
All downstream scripts (02 onward) should read master_clean.csv.
"""
from pathlib import Path
import pandas as pd

tinnorm_dir  = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
features_dir = tinnorm_dir / "features"
material_dir = Path("../material")

CHECK_SUFFIX = "power_sensor_preproc_1.zip"

df = pd.read_csv(material_dir / "master_new.csv")
df["subject_id"] = df["subject_id"].astype(str)
n_start = len(df)
print(f"master_new.csv   : {n_start} subjects")

# ── Exclusion 1: no feature file ──────────────────────────────────────────────
ids_with_features = {
    fname.stem[4:9]
    for fname in features_dir.iterdir()
    if fname.name.endswith(CHECK_SUFFIX)
}

ids_in_csv       = set(df["subject_id"])
only_in_csv      = sorted(ids_in_csv      - ids_with_features)
only_in_features = sorted(ids_with_features - ids_in_csv)

print(f"Features directory: {len(ids_with_features)} subjects with '{CHECK_SUFFIX}'")

print(f"\nExclusion 1 — in CSV but no feature file ({len(only_in_csv)}):")
if only_in_csv:
    dropped = df[df["subject_id"].isin(only_in_csv)][["subject_id", "site", "group", "age", "sex"]]
    print(dropped.to_string(index=False))
else:
    print("  None")

print(f"\nNo CSV entry but feature file exists ({len(only_in_features)}):")
print(f"  {only_in_features if only_in_features else 'None'}")

df = df[df["subject_id"].isin(ids_with_features)].reset_index(drop=True)

# ── Exclusion 2: missing PTA4_HF ─────────────────────────────────────────────
# PTA4_HF = bilateral mean of high-frequency PTA (PTA_HF_ARE, PTA_HF_ALE).
# Subjects without it cannot be included in ComBat harmonization or normative
# modelling (PTA4_HF is a required biological covariate in both).
if "PTA4_HF" in df.columns:
    mask_nan_pta = df["PTA4_HF"].isna()
    n_nan_pta = mask_nan_pta.sum()
    print(f"\nExclusion 2 — missing PTA4_HF ({n_nan_pta}):")
    if n_nan_pta:
        dropped_pta = df[mask_nan_pta][["subject_id", "site", "group", "age", "sex"]]
        print(dropped_pta.to_string(index=False))
        df = df[~mask_nan_pta].reset_index(drop=True)
    else:
        print("  None")
else:
    print("\nPTA4_HF column not found — skipping exclusion 2.")

# ── Summary ───────────────────────────────────────────────────────────────────
out_path = material_dir / "master_clean.csv"
df.to_csv(out_path, index=False)

print(f"\n{'─'*50}")
print(f"Started with    : {n_start} subjects")
print(f"master_clean.csv: {len(df)} subjects  ({n_start - len(df)} excluded)")
print(f"  Groups : {dict(df['group'].value_counts().sort_index())}")
print(f"  Sites  : {dict(df.groupby('site').size().sort_index())}")
print(f"Saved → {out_path}")
