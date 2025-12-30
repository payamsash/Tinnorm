from pathlib import Path
import pandas as pd
from tqdm import tqdm
from neuroHarmonize import harmonizationLearn

def harmonize(                
                preproc_level,
                mode,
                modality,
                saving_dir,
                features_dir
                ):

    ## get file names
    fnames = []
    for fname in features_dir.iterdir():
        text = f"{modality}_preproc_{preproc_level}.zip" # fix this
        if str(fname).endswith(text):
            fnames.append(fname)

    ## read and create data matrix
    dfs_list = []
    for fname in tqdm(fnames):
        subject_id = fname.stem[4:9]
        df_subject = pd.read_csv(fname, index_col=None)
            
        if modality == "conn":
            df_subject = df_subject.set_index(df_subject.columns[0]).T
            df_subject.columns.name = None
        
        df_subject.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
        df_mean = df_subject.mean(axis=0).to_frame().T # per subject
        df_mean["subject_id"] = subject_id
        dfs_list.append(df_mean)

    df_data = pd.concat(dfs_list)
    df_data['subject_id'] = df_data['subject_id'].astype(str)

    ## create covariates df and match 2 dfs
    df_q = pd.read_csv("../material/master.csv")
    df_covars = df_q[["site", "age", "sex",	"subject_id"]]
    df_covars.rename(columns={"site": "SITE"}, inplace=True)
    df_covars['subject_id'] = df_covars['subject_id'].astype(str)
    df_merged = pd.merge(df_data, df_covars, on='subject_id', how='inner')
    df_merged.sort_values(by="subject_id", inplace=True)

    dropped_from_df1 = set(df_data['subject_id']) - set(df_covars['subject_id'])
    print(f"subjects missing in covar: {list(sorted(dropped_from_df1))}")

    dropped_from_df2 = set(df_covars['subject_id']) - set(df_data['subject_id'])
    print(f"subjects missing in data: {list(sorted(dropped_from_df2))}")

    ## harmonize
    data_matrix = df_merged.iloc[:, :-4].to_numpy()
    df_cov = df_merged[["SITE",	"age",	"sex"]]
    hm_model, my_data_adj = harmonizationLearn(data_matrix, df_cov, eb=True, seed=0)

    ## replace and save
    column_names = df_merged.columns[:-4]
    df_hm = pd.DataFrame(my_data_adj, columns=column_names)
    df_hm = pd.concat([df_hm, df_merged[['SITE', 'subject_id', 'age', 'sex']].reset_index(drop=True)], axis=1)
    # write saving directory here

if __name__ == "__main__":
    
    features_dir = Path("/Volumes/G_USZ_ORL$/Research/ANT/tinnorm/features")
    saving_dir = Path(".")
    preproc_level = 2
    mode = "sensor"
    modality = "power"
    
    harmonize(
                preproc_level,
                mode,
                modality,
                saving_dir,
                features_dir
                )
    ## add saving path for models and residulas
