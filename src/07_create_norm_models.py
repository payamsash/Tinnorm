## just run to create models and save the results not models



df = pd.read_csv('/Volumes/G_USZ_ORL$/Research/ANT/tinnorm/harmonized/power_sensor_preproc_2_hm.csv')
df_master = pd.read_csv("../material/master.csv")

df["subject_id"] = df["subject_id"].astype(str)
df_master["subject_id"] = df_master["subject_id"].astype(str)

df = df.merge(
                    df_master[["subject_id", "group", "PTA4_mean"]],
                    on="subject_id",
                    how="inner"
                )
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")


## Creating a NormData object
kwargs = {
            "covariates": ["age", "sex", "PTA"],
            "batch_effects": ["SITE"],
            "response_vars": [c for c in df.columns[:-5] if c.endswith("alpha_1")], # list(df.columns[:-5]),
            "subject_ids": "subject_id"
            }

# create NormData objects
norm_train = NormData.from_dataframe(name="train",
                                    dataframe=df.query('group == 0'),
                                    **kwargs)
norm_test = NormData.from_dataframe(name="test",
                                    dataframe=df.query('group == 1'),
                                    **kwargs)