from pathlib import Path
import pandas as pd

## read the files
quest_dir = Path("../material/questionnaires")
fname_zu = "/Users/payamsadeghishabestari/Desktop/data_project_995463_2026_01_14 (1).csv" # unipark dir
fname_reg = quest_dir / "Questionnaire_data_TIDE_REG.csv"
fname_ant_to_tide = "../material/ant_to_tide.csv"

df_zu = pd.read_csv(
    fname_zu,
    sep=";",
    engine="python"
)
df_reg = pd.read_csv(
    fname_reg,
    engine="python"
)
df_map = pd.read_csv(fname_ant_to_tide)

## add esit and tschq
df_zu.columns = df_zu.columns.str.lower()
extra_cols = df_zu.columns[
                            df_zu.columns.str.startswith(("esit", "tschq"))
                            ]

## convert the unipark to regensburg style
df_zu.rename(
            columns={
                    "studiennummer": "study_id",
                    "alter" : "esit_a1",
                    "geschlecht": "intro_gender"
                    },
            inplace=True
            )

common_cols = df_zu.columns.intersection(df_reg.columns)
final_cols = list(common_cols) + list(extra_cols)
df_zu = df_zu[final_cols]

## check what is. miising
df_zu["study_id"] = (
                        df_zu["study_id"]
                        .astype(str)
                        .str.lower()
                        .str.replace(r"[()\[\]{}]", "", regex=True)
                        .str.strip()
                    )

in_df2_not_df1 = set(df_map["antinomics_id"]) - set(df_zu["study_id"])
print("In Tide but not in unipark:", sorted(in_df2_not_df1))

in_df1_not_df2 = set(df_zu["study_id"]) - set(df_map["antinomics_id"])
print("In unipark but not in Tide:", sorted(in_df1_not_df2))

## add Tide ids
df_zu = df_zu.merge(
                            df_map[["antinomics_id", "tide_id"]],
                            left_on="study_id",
                            right_on="antinomics_id",
                            how="left"
                        )
df_zu.rename(columns={
                            "study_id": "TEMP",
                            "tide_id": "study_id",
                            "TEMP": "tide_id"}, 
                            inplace=True
                            )
df_zu.drop(columns=["antinomics_id", "TEMP"], inplace=True)

## fix age column
df_zu["esit_a1"] = df_zu["esit_a1"].astype(str).str.extract(r"(\d+)")[0].astype(float)
df_zu.loc[df_zu["esit_a1"] > 1000, "esit_a1"] = 2025 - df_zu["esit_a1"]
df_zu["esit_a1"] = df_zu["esit_a1"].astype(int)

## reorder and clean
cols = list(df_zu.columns)
cols = [cols[-1]] + cols[:-1]
df_zu = df_zu[cols]
df_zu = df_zu.dropna(subset=["study_id"]).reset_index(drop=True)
df_zu["study_id"] = df_zu["study_id"].astype(int)
df_zu.sort_values(by="study_id", inplace=True)
df_zu.to_csv("../material/questionnaires/Questionnaire_data_TIDE_ZUE.csv")