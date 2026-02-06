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
            outscaler="none",
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
    demographic_cols = ["subject_id", "SITE", "group", "age", "sex", "PTA4_mean"]
    response_cols = df.columns.difference(demographic_cols).tolist()
    # response_cols = [c for c in response_cols if c.endswith(f"alpha_1")]

    kwargs = {
                "covariates": demographic_cols[-3:],
                "batch_effects": ["SITE"],
                "response_vars": response_cols, 
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
                    "savemodel": False,
                },
                {
                    "save_dir": model_dir / "full_model",
                    "train": norm_train_all,
                    "test": norm_test_tinnitus,
                    "savemodel": True,
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
            savemodel=cfg["savemodel"],
            evaluate_model=True,
            saveresults=True,
            saveplots=False,
            save_dir=str(cfg["save_dir"]),
            inscaler="standardize",
            outscaler=outscaler,
        )
        model.fit_predict(cfg["train"], cfg["test"])

        del model
        del template_blr


if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    preproc_levels = [2]
    spaces = ["sensor", "source"][1:]
    modalities = ["aperiodic", "conn", "power", "global", "regional"][-2:]
    conn_modes = ["pli", "plv", "coh"][2:]

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:
                for conn_mode in conn_modes:
                    
                    if modality in ["conn", "global", "regional"]:
                        fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_hm.csv"
                        model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}"

                    if modality in ["aperiodic", "power"]:
                        fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_hm.csv"
                        model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}"

                    if fname_feature.is_file() and not model_dir.is_dir():
                        run_nm(
                                fname_feature,
                                model_dir,
                                outscaler="none",
                                random_state=42
                                )
                        
    ## only for centile plotting I set outscaler to standardize:
    preproc_level = 2
    space = "source"
    conn_mode = "coh"

    for modality in modalities:
        if modality in ["conn", "global", "regional"]:
            fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_hm.csv"
            model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}_{conn_mode}_centile_plot"

        if modality in ["aperiodic", "power"]:
            fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{modality}_hm.csv"
            model_dir = models_dir / f"preproc_{preproc_level}" / space / f"{modality}_centile_plot"

        if fname_feature.is_file() and not model_dir.is_dir():
            run_nm(
                    fname_feature,
                    model_dir,
                    outscaler="standardize",
                    random_state=42
                    )