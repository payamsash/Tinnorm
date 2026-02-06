

############################################## the old one
from pathlib import Path
from warnings import warn
import numpy as np
import pandas as pd

## add documentation

behavioural_dir = Path.cwd().parent / "material" 
site_map = {
        "1": "austin",
        "2": "dublin",
        "3": "ghent",
        "4": "illinois",
        "5": "regensburg",
        "6": "tuebingen",
        "7": "zuerich"
        }

quest_cols = [ # maybe add hearing loss
            "study_id",
            "intro_gender",
            "esit_a1", # age
            "hq_score",
            "hq_attentional_score",
            "hq_social_score",
            "hq_emotional_score",
            "hads_d_score",
            "hads_a_score",
            "esit_a17", # group
            "tfi_score",
            "thi_score",
            "psq_score" # stress level
            ]

audio_cols = [
            "Subject ID",
            "PTA4_ARE",
            "PTA4_ALE",
            "PTA_TIDE_ARE",
            "PTA_TIDE_ALE",
            "PTA_HF_ARE",
            "PTA_HF_ALE",
            # "RE_TinFreq", # for now remove them
            # "RE_TinLoudness",
            # "LE_TinFreq",
            # "LE_TinLoudness"
        ]
avg_pairs = {
            "PTA4_mean": ["PTA4_ARE", "PTA4_ALE"],
            "PTA_HF_mean": ["PTA_HF_ARE", "PTA_HF_ALE"],
            "PTA_TIDE_mean": ["PTA_TIDE_ARE", "PTA_TIDE_ALE"]
            }

tinnitus_thr = 4
dfs = []
for site in site_map.values():

    site_code = site.upper()[:3]

    if site_code == "ZUE":
        print(f"{site} not ready yet ...")
        continue

    quest_fname = behavioural_dir / "questionnaires" / f"Questionnaire_data_TIDE_{site_code}.csv"
    audio_fname = behavioural_dir / "audiograms" / f"Audiometry_data_TIDE_{site_code}.csv"
    
    if not quest_fname.is_file() or not audio_fname.is_file():
        warn(f"Missing data for site '{site}': "
            f"{'questionnaire' if not quest_fname.is_file() else ''} "
            f"{'audiometry' if not audio_fname.is_file() else ''}. Skipping this site.")
        continue
    kwargs = dict(sep=None, engine="python", index_col=None, encoding="utf-8-sig")
    df_q = pd.read_csv(quest_fname, **kwargs)
    df_a = pd.read_csv(audio_fname, **kwargs)

    df_q["site"] = site

    # some site spcific fixes
    if site_code == "ILL":
        df_q.rename(columns={"Subject ID": "study_id",
                            "esit_a2": "intro_gender"}, inplace=True)
    
        df_a.rename(columns={"RE_TinFreq(k)": "RE_TinFreq",
                            "LE_TinFreq(k)": "LE_TinFreq",
                            "RE_TinLoudness(dBSL)": "RE_TinLoudness"},
                            inplace=True) # must not be converted but diff scales but both are dBSL
        df_a[["RE_TinFreq", "LE_TinFreq"]] = df_a[["RE_TinFreq", "LE_TinFreq"]] * 1000
        
    if site_code == "REG":
        df_a.rename(columns={"PT4_HF_ARE": "PTA_HF_ARE",
                            "PT4_TIDE_ARE": "PTA_TIDE_ARE"},
                            inplace=True)

    missing_quest_cols = set(quest_cols) - set(df_q.columns)
    if missing_quest_cols:
        warn(f"Site '{site}': questionnaire is missing columns {sorted(missing_quest_cols)}. Skipping this site.")
        continue

    missing_audio_cols = set(audio_cols) - set(df_a.columns)
    if missing_audio_cols:
        warn(f"Site '{site}': audiometry is missing columns {sorted(missing_audio_cols)}. Skipping this site.")
        continue

    df_a = df_a[audio_cols]
    df_a.rename(columns={"Subject ID": "study_id"}, inplace=True)
    df_q["study_id"] = df_q["study_id"].astype(str)
    df_a["study_id"] = df_a["study_id"].astype(str)

    if site_code == "GHE":
        df_a["study_id"] = (
                            pd.to_numeric(df_a["study_id"], errors="coerce")
                            .astype("Int64")
                            .astype(str)
                        )

    for new_col, cols in avg_pairs.items():
        df_a[new_col] = df_a[cols].mean(axis=1)
    df_a.drop(columns=[col for cols in avg_pairs.values() for col in cols], inplace=True)

    df = df_q.merge(
                    df_a,
                    on="study_id",
                    how="left",
                )

    dfs.append(df[["site"] + quest_cols + list(avg_pairs.keys())])

df_all = pd.concat(dfs, ignore_index=True)
mapping = {
            "study_id": "subject_id",
            "intro_gender": "sex",
            "esit_a1":"age",
            "esit_a17": "group"
            } 
df_all.rename(columns=mapping, inplace=True)

cols_required = [
                "age",
                "sex",
                "site",
                "hq_score",
                "group",
                "PTA4_mean",
                "PTA_TIDE_mean"
            ]

df_all.dropna(subset=cols_required, inplace=True)
df_all["group"] = np.where(df_all["group"] <= tinnitus_thr, 1, 0)
df_all.sort_values(by=["site", "subject_id"], inplace=True)
df_all.to_csv("../material/master.csv", index=False)