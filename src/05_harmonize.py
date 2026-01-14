import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from neuroHarmonize import (
    harmonizationLearn,
    harmonizationApply
    )

def harmonize(                
                preproc_level,
                space,
                modality,
                conn_mode,
                saving_dir,
                features_dir
                ):

    ## get file names
    fnames = []
    for fname in features_dir.iterdir():
        text = f"{modality}_{space}_preproc_{preproc_level}.zip" # fix this
        if str(fname).endswith(text):
            fnames.append(fname)

    ## read and create data matrix
    print(f"reading the {modality} in {space} features ...")
    dfs_list = []
    for fname in tqdm(fnames):
        subject_id = fname.stem[4:9]
        df_subject = pd.read_csv(fname, index_col=None)
            
        if modality == "conn":
            df_subject = df_subject.set_index(df_subject.columns[0]).T
            df_subject.columns.name = None
            df_subject = df_subject.filter(regex=rf'_{conn_mode}$')
        
        df_subject.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
        df_mean = df_subject.mean(axis=0).to_frame().T # per subject
        df_mean["subject_id"] = subject_id
        dfs_list.append(df_mean)

    df_data = pd.concat(dfs_list)
    df_data['subject_id'] = df_data['subject_id'].astype(str)

    ## create covariates df and match 2 dfs
    df_q = pd.read_csv("../material/master.csv")
    cols = ["site", "age", "sex", "subject_id", "PTA4_mean", "group"]
    df_covars = df_q[cols]
    df_covars.rename(columns={"site": "SITE"}, inplace=True)
    df_covars['subject_id'] = df_covars['subject_id'].astype(str)
    df_merged = pd.merge(df_data, df_covars, on='subject_id', how='inner')
    df_merged.sort_values(by="subject_id", inplace=True)

    dropped_from_df1 = set(df_data['subject_id']) - set(df_covars['subject_id'])
    print(f"subjects missing in covar: {list(sorted(dropped_from_df1))}")

    dropped_from_df2 = set(df_covars['subject_id']) - set(df_data['subject_id'])
    print(f"subjects missing in data: {list(sorted(dropped_from_df2))}")

    ## harmonize
    if modality == "aperiodic":
        fname_save = saving_dir / f"{modality}_{space}_preproc_{preproc_level}_hm.csv"
        df_merged.to_csv(fname_save)

    else:
        
        data_full = df_merged.iloc[:, :-len(cols)].to_numpy()
        df_cov_full = df_merged[["SITE", "age",	"sex", "PTA4_mean"]]

        ## get the residuals (s_data)
        _, _, s_data = harmonizationLearn(
                                            data_full,
                                            df_cov_full,
                                            eb=True,
                                            seed=0,
                                            return_s_data=True
                                            )

        ## learn the model on controls
        df_train = df_merged.query('group == 0')
        data_controls = df_train.iloc[:, :-len(cols)].to_numpy()
        df_cov_controls = df_train[["SITE",	"age", "sex", "PTA4_mean"]]
        
        hm_model, _ = harmonizationLearn(
                                            data_controls,
                                            df_cov_controls,
                                            eb=True,
                                            seed=0,
                                            return_s_data=False
                                            )
        
        ## apply the model on all subjects
        bayes_data = harmonizationApply(
                                        data_full,
                                        df_cov_full,
                                        hm_model
                                        )
        
        ## replace and save
        column_names = df_merged.columns[:-len(cols)]
        for data, title in zip([bayes_data, s_data], ["hm", "residual"]):
            df_hm = pd.DataFrame(data, columns=column_names)
            df_hm = df_hm.round(4)
            df_hm = pd.concat([df_hm, df_merged[df_merged.columns[-len(cols):]].reset_index(drop=True)], axis=1)
            
            if modality == "conn":
                fname_save = saving_dir / f"{modality}_{space}_preproc_{preproc_level}_{conn_mode}_{title}.csv"
            else:
                fname_save = saving_dir / f"{modality}_{space}_preproc_{preproc_level}_{title}.csv"

            df_hm.to_csv(fname_save)

if __name__ == "__main__":

    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    features_dir = tinnorm_dir / "features"
    hm_dir = tinnorm_dir / "harmonized"
    os.makedirs(hm_dir, exist_ok=True)
    
    preproc_levels = [2]
    spaces = ["sensor", "source"][1:]
    modalities = ["power", "conn", "aperiodic"][2:]
    conn_modes = ["pli", "plv", "coh"][2:]

    for preproc_level in preproc_levels:
        for space in spaces:
            for modality in modalities:
                for conn_mode in conn_modes:
                    fname_save_1 = hm_dir / f"{modality}_{space}_preproc_{preproc_level}_hm.csv"
                    fname_save_2 = hm_dir / f"{modality}_{space}_preproc_{preproc_level}_{conn_mode}_hm.csv"
                    
                    if fname_save_1.exists() or fname_save_2.exists():
                        continue
                    else:
                        harmonize(
                                    preproc_level,
                                    space,
                                    modality,
                                    conn_mode,
                                    hm_dir,
                                    features_dir
                                    )