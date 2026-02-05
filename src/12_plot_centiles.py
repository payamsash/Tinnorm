from pathlib import Path
import pandas as pd
from pcntoolkit import (
    NormativeModel,
    NormData,
    plot_centiles_advanced
    )


def run_centile_plot(
                    brain_labels,
                    frequency,
                    covar,
                    data_mode,
                    mode,
                    space,
                    preproc_level
                    ):  

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    model_dir = models_dir / f"preproc_{preproc_level}" / space / mode / "full_model"
    fname_feature = hm_dir / f"preproc_{preproc_level}" / space / f"{mode}_hm.csv"

    ## load model and re-create norm data 
    model = NormativeModel.load(str(model_dir))
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
    demographic_cols = ["subject_id", "SITE", "group", "age", "sex", "PTA4_mean"]
    response_cols = df.columns.difference(demographic_cols).tolist()

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

    if data_mode == "train":
        scatter_data = norm_train_all
    if data_mode == "test":
        scatter_data = norm_test_tinnitus

    if mode == "aperiodic":
        bls_1 = [f"{bl}_exponent" for bl in brain_labels]
        bls_2 = [f"{bl}_offset" for bl in brain_labels]
        bls = bls_1 + bls_2
    
    elif mode == "conn": ## could be modified
        bls = []
    else: 
        bls = [f"{bl}_{frequency}" for bl in brain_labels]

    figs, re_vars = plot_centiles_advanced(
                            model,
                            centiles=[0.05, 0.5, 0.95],
                            covariate=covar,
                            batch_effects={"SITE": df["SITE"].unique().tolist()},
                            scatter_data=scatter_data,
                            show_other_data=False,
                            harmonize_data=True,
                            show_yhat=True,
                            brain_labels=bls
                            )

    for fig, re_var in zip(figs, re_vars):
        saving_dir = tinnorm_dir / "plots" / "centiles"
        saving_dir.mkdir(parents=True, exist_ok=True)

        fname_save = saving_dir / f"{mode}_{covar}_{data_mode}_{re_var}_preproc_level_{preproc_level}.pdf"
        fig.savefig(
                    fname_save,
                    format="pdf",
                    dpi=300,
                    bbox_inches="tight"
                    )
    

if __name__ == "__main__":

    mode = "aperiodic"
    space = "source"
    preproc_level = 2
    covar = "age" # "PTA4_mean"

    data_mode = "train"
    brain_labels = ["transversetemporal-lh", "transversetemporal-rh"]
    frequency = "alpha_1"

    run_centile_plot(
                    brain_labels,
                    frequency,
                    covar,
                    data_mode,
                    mode,
                    space,
                    preproc_level
                    )