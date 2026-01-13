import os
from pathlib import Path
import pandas as pd

from pcntoolkit import (
    NormData,
    BLR,
    BsplineBasisFunction,
    NormativeModel
)


def run_nm(
            fname_feature,
            model_dir,
            random_state=42
        ):

    ## read and manipulate dfs
    df = pd.read_csv(fname_feature)
    df_master = pd.read_csv("../material/master.csv")

    df["subject_id"] = df["subject_id"].astype(str)
    df_master["subject_id"] = df_master["subject_id"].astype(str)

    df = df.merge(
                df_master[["subject_id"]],
                on="subject_id",
                how="inner"
                )
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    ## creating a NormData objects
    kwargs = {
                "covariates": ["age", "sex", "PTA4_mean"],
                "batch_effects": ["SITE"],
                "response_vars": [c for c in df.columns[:-6] if c.endswith("alpha_1_coh")], # list(df.columns[:-6]),
                "subject_ids": "subject_id"
                }

    norm_train_all = NormData.from_dataframe(
                                            name="train",
                                            dataframe=df.query('group == 0'),
                                            **kwargs
                                            )
    norm_test_tinnitus = NormData.from_dataframe(
                                            name="test",
                                            dataframe=df.query('group == 1'),
                                            **kwargs
                                            )
    norm_train_control, norm_test_control \
                        = norm_train_all.train_test_split(
                                                            splits=0.8,
                                                            random_state=random_state
                                                            )

    ## creating the norm model
    configs = [
                {
                    "save_dir": model_dir / "for_eval",
                    "train": norm_train_control,
                    "test": norm_test_control,
                },
                {
                    "save_dir": model_dir / "full_model",
                    "train": norm_train_all,
                    "test": norm_test_tinnitus,
                },
                ]

    for cfg in configs:
        template_blr = BLR(
            name="payam_blr",
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
            save_dir=cfg["save_dir"],
            inscaler="standardize",
            outscaler="none",
        )

        model.fit_predict(cfg["train"], cfg["test"])
        # model.save(cfg["save_dir"])

        del model
        del template_blr


if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    preproc_levels = [2]
    spaces = ["sensor", "source"][1:]
    modalities = ["power", "conn", "aperiodic", "global", "regional"][-2:]
    conn_modes = ["pli", "plv", "coh"][2:]

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:
                for conn_mode in conn_modes:
                    
                    if modality in ["conn", "global", "regional"]:
                        fname_feature = hm_dir / f"{modality}_{space}_preproc_{preproc_level}_{conn_mode}_hm.csv"
                        model_dir = models_dir / f"{modality}_{space}_preproc_{preproc_level}_{conn_mode}"

                    if modality in ["aperiodic", "power"]:
                        fname_feature = hm_dir / f"{modality}_{space}_preproc_{preproc_level}_hm.csv"
                        model_dir = models_dir / f"{modality}_{space}_preproc_{preproc_level}"

                    if fname_feature.is_file() and not model_dir.is_dir():
                        run_nm(
                                fname_feature,
                                model_dir,
                                random_state=42
                                )


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