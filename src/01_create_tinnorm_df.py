"""
Build master_new.csv from the two aggregate TIDE files.

Sources
-------
  material/Aggregated_Questionnaire_TIDE_Final_260326.xlsx
      sheet: Questionnaire_data_TIDE_ALL
      → demographics, group label, THI score

  material/TIDE-Compiled-Version2_3-26-2026.xlsx
      sheet: Sheet1
      → audiometry (PTA4, PTA_HF per ear)

Output
------
  material/master_new.csv
      subject_id, site, group (0=control, 1=tinnitus),
      age, sex (1=male, 2=female),
      PTA4_mean, PTA4_HF, THI

Run from src/:  python 01_create_tinnorm_df.py
"""

from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

MATERIAL_DIR = Path(__file__).resolve().parent.parent / "material"
QUEST_FILE   = MATERIAL_DIR / "Aggregated_Questionnaire_TIDE_Final_260326.xlsx"
AUDIO_FILE   = MATERIAL_DIR / "TIDE-Compiled-Version2_3-26-2026.xlsx"
OUT_FILE     = MATERIAL_DIR / "master_new.csv"

# Questionnaire site abbreviation → canonical lowercase name
SITE_MAP_QUEST = {
    "Aus": "austin",
    "Dub": "dublin",
    "Ghe": "ghent",
    "UIL": "illinois",
    "Rgb": "regensburg",
    "UKT": "tuebingen",
    "ZUR": "zuerich",
}

# Subjects excluded for clinical reasons (high HADS scores)
EXCLUDE_IDS = {"70072", "70079"}

# ── Load questionnaire ────────────────────────────────────────────────────────

print("Loading questionnaire …")
df_q = pd.read_excel(QUEST_FILE, sheet_name="Questionnaire_data_TIDE_ALL")
print(f"  {len(df_q)} rows, {df_q.shape[1]} columns")

quest_keep = {
    "study_id":    "subject_id",
    "Site":        "site",
    "Group":       "group",
    "intro_gender": "sex",
    "esit_a1":     "age",
    "THI_SCORE":   "THI",
    "TFI_SCORE":   "TFI",
}
missing_q = [c for c in quest_keep if c not in df_q.columns]
if missing_q:
    raise ValueError(f"Questionnaire is missing columns: {missing_q}")

df_q = df_q[list(quest_keep)].copy()
df_q.rename(columns=quest_keep, inplace=True)
df_q["subject_id"] = df_q["subject_id"].astype(str)
df_q["site"]       = df_q["site"].map(SITE_MAP_QUEST)

missing_site = df_q["site"].isna().sum()
if missing_site:
    warn(f"{missing_site} questionnaire rows have an unrecognised site code — they will be dropped.")
df_q.dropna(subset=["site"], inplace=True)

# Group: 'T' → 1 (tinnitus), 'C' → 0 (control)
df_q["group"] = df_q["group"].map({"T": 1, "C": 0})

print(f"  Groups: {dict(df_q['group'].value_counts(dropna=False).sort_index())}")
print(f"  Sites:  {sorted(df_q['site'].dropna().unique())}")

# ── Load audiometry ───────────────────────────────────────────────────────────

print("\nLoading audiometry …")
df_a = pd.read_excel(AUDIO_FILE, sheet_name="Sheet1")
print(f"  {len(df_a)} rows, {df_a.shape[1]} columns")

audio_keep = ["Subject ID", "PTA4_ARE", "PTA4_ALE", "PTA_HF_ARE", "PTA_HF_ALE"]
missing_a = [c for c in audio_keep if c not in df_a.columns]
if missing_a:
    raise ValueError(f"Audiometry is missing columns: {missing_a}")

df_a = df_a[audio_keep].copy()
df_a.rename(columns={"Subject ID": "subject_id"}, inplace=True)
df_a["subject_id"] = df_a["subject_id"].astype(str)

# Compute bilateral means
df_a["PTA4_mean"] = df_a[["PTA4_ARE", "PTA4_ALE"]].mean(axis=1)
df_a["PTA4_HF"]   = df_a[["PTA_HF_ARE", "PTA_HF_ALE"]].mean(axis=1)
df_a.drop(columns=["PTA4_ARE", "PTA4_ALE", "PTA_HF_ARE", "PTA_HF_ALE"], inplace=True)

# ── Merge ─────────────────────────────────────────────────────────────────────

print("\nMerging on subject_id …")
q_ids = set(df_q["subject_id"])
a_ids = set(df_a["subject_id"])
only_q = sorted(q_ids - a_ids)
only_a = sorted(a_ids - q_ids)
if only_q:
    print(f"  In questionnaire but NOT in audio ({len(only_q)}): {only_q}")
if only_a:
    print(f"  In audio but NOT in questionnaire ({len(only_a)}): {only_a}")

df = df_q.merge(df_a, on="subject_id", how="inner")
print(f"  After inner join: {len(df)} subjects")

# ── Exclusions ────────────────────────────────────────────────────────────────

n_before = len(df)
df = df[~df["subject_id"].isin(EXCLUDE_IDS)]
n_excluded = n_before - len(df)
if n_excluded:
    print(f"\n  Excluded {n_excluded} subject(s) (clinical reasons): {EXCLUDE_IDS}")

# ── Quality checks ────────────────────────────────────────────────────────────

required_cols = ["sex", "age", "PTA4_mean", "site", "group"]
n_before = len(df)
df.dropna(subset=required_cols, inplace=True)
n_dropped = n_before - len(df)
if n_dropped:
    warn(f"Dropped {n_dropped} row(s) with NaN in required columns {required_cols}.")

print(f"\n{'='*50}")
print(f"Final dataset: {len(df)} subjects")
print(f"  Groups : {dict(df['group'].value_counts().sort_index())}")
print(f"  Sites  : {dict(df.groupby('site').size().sort_index())}")
print(f"{'='*50}")

# ── Validate sex / group values ───────────────────────────────────────────────

bad_sex = df[~df["sex"].isin([1, 2])]
if len(bad_sex):
    print(f"\nSubjects with unexpected sex values ({len(bad_sex)}):")
    for _, row in bad_sex.iterrows():
        print(f"  subject_id={row['subject_id']} site={row['site']} sex={row['sex']}")
    df = df[df["sex"].isin([1, 2])]
    print(f"  → Removed. Remaining: {len(df)}")

bad_group = df[~df["group"].isin([0, 1])]
if len(bad_group):
    print(f"\nSubjects with unexpected group values ({len(bad_group)}):")
    for _, row in bad_group.iterrows():
        print(f"  subject_id={row['subject_id']} site={row['site']} group={row['group']}")
    df = df[df["group"].isin([0, 1])]
    print(f"  → Removed. Remaining: {len(df)}")

# ── Final column order and save ───────────────────────────────────────────────

col_order = ["subject_id", "site", "group", "age", "sex", "PTA4_mean", "PTA4_HF", "THI", "TFI"]
col_order = [c for c in col_order if c in df.columns]
df = df[col_order]
df.sort_values(["site", "subject_id"], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}")
print(f"Columns: {list(df.columns)}")
print(df.head(5).to_string(index=False))
