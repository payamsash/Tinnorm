import numpy as np
import pandas as pd

from pcntoolkit import (
    NormData,
    BLR,
    BsplineBasisFunction,
    NormativeModel
)

## making dataframes ready for norm data structure
random_state = 42
saving_dir = "." # should be based on modality
df = pd.read_csv('/Volumes/G_USZ_ORL$/Research/ANT/tinnorm/harmonized/power_sensor_preproc_2_hm.csv') ## fix the path
df_master = pd.read_csv("../material/master.csv")

df["subject_id"] = df["subject_id"].astype(str)
df_master["subject_id"] = df_master["subject_id"].astype(str)

df = df.merge(
            df_master[["subject_id", "group", "PTA4_mean"]],
            on="subject_id",
            how="inner"
            )
df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

## creating a NormData objects
kwargs = {
            "covariates": ["age", "sex", "PTA4_mean"],
            "batch_effects": ["SITE"],
            "response_vars": list(df.columns[:-6]), # [c for c in df.columns[:-6] if c.endswith("alpha_1")]
            "subject_ids": "subject_id"
            }

norm_train_all = NormData.from_dataframe(name="train_all",
                                    dataframe=df.query('group == 0'),
                                    **kwargs)
norm_test_tinnitus = NormData.from_dataframe(name="test_tinnitus",
                                    dataframe=df.query('group == 1'),
                                    **kwargs)
norm_train_control, norm_test_control \
                    = norm_train_all.train_test_split(
                                                        splits=0.8,
                                                        random_state=random_state
                                                        )

## creating and save the norm model
template_blr = BLR(
                    name="power_source",
                    basis_function_mean=BsplineBasisFunction(degree=3, nknots=5), 
                    fixed_effect=True,  
                    heteroskedastic=True,  
                    warp_name="warpsinharcsinh"
                    )

model_1 = NormativeModel(
                        template_regression_model=template_blr,
                        savemodel=False,
                        evaluate_model=True,
                        saveresults=True,
                        saveplots=False,
                        save_dir=saving_dir / "for_eval",
                        inscaler="standardize",
                        outscaler="standardize",
                    )
model_1.fit_predict(norm_train_control, norm_test_control)
# model_1.save(saving_dir / "for_eval")

del model_1

model_2 = NormativeModel(
                        template_regression_model=template_blr,
                        savemodel=False,
                        evaluate_model=True,
                        saveresults=True,
                        saveplots=False,
                        save_dir=saving_dir / "full_model",
                        inscaler="standardize",
                        outscaler="standardize",
                    )
model_2.fit_predict(norm_train_all, norm_test_tinnitus)
model_2.save(saving_dir / "full_model")


'''
we will need this for plotting later ...
plot_centiles_advanced(
                        model,
                        centiles=[0.05, 0.5, 0.95],
                        covariate="age",
                        batch_effects={"SITE": ["austin", "illinois", "regensburg", "tuebingen"]},
                        scatter_data=train,
                        show_other_data=False,
                        harmonize_data=True,
                        show_yhat=True
                        )
'''