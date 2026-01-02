import numpy as np
import pandas as pd

from pcntoolkit import (
    NormData,
    BLR,
    BsplineBasisFunction,
    NormativeModel,
    plot_centiles_advanced
)

## fix the path
saving_dir = "."
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

## creating a NormData object
kwargs = {
            "covariates": ["age", "sex", "PTA4_mean"],
            "batch_effects": ["SITE"],
            "response_vars": list(df.columns[:-6]), # [c for c in df.columns[:-6] if c.endswith("alpha_1")]
            "subject_ids": "subject_id"
            }

norm_train = NormData.from_dataframe(name="train",
                                    dataframe=df.query('group == 0'),
                                    **kwargs)
norm_test = NormData.from_dataframe(name="test",
                                    dataframe=df.query('group == 1'),
                                    **kwargs)

## creating and save the norm model
template_blr = BLR(
                    name="power_source",
                    basis_function_mean=BsplineBasisFunction(degree=3, nknots=5), 
                    fixed_effect=True,  
                    heteroskedastic=True,  
                    warp_name="warpsinharcsinh"
                    )

model = NormativeModel(
                        template_regression_model=template_blr,
                        savemodel=False,
                        evaluate_model=True,
                        saveresults=True,
                        saveplots=False,
                        save_dir=saving_dir,
                        inscaler="standardize",
                        outscaler="standardize",
                    )
model.fit_predict(norm_train, norm_test)
model.save(saving_dir)