import pandas as pd
from pathlib import Path
import pickle

def get_sorted_mapping(
    site_dir, 
    name_filter=None, 
    suffix_filter=None, 
    exclude_substr=None, 
    key_length=5, 
    file_level="nested",
    id_map_df=None,
):
    if isinstance(name_filter, str):
        name_filter = [name_filter]

    fnames = []
    if file_level == "nested":
        for subject_folder in site_dir.iterdir():
            if subject_folder.is_dir():
                for fname in subject_folder.iterdir():
                    if not fname.name.startswith("._"):
                        if name_filter and not any(nf in fname.name for nf in name_filter):
                            continue
                        if exclude_substr and exclude_substr in fname.name:
                            continue
                        if suffix_filter and fname.suffix != suffix_filter:
                            continue
                        fnames.append(fname)
    else:
        for fname in site_dir.iterdir():
            if fname.is_file() and not fname.name.startswith("._"):
                if name_filter and not any(nf in fname.name for nf in name_filter):
                    continue
                if exclude_substr and exclude_substr in fname.name:
                    continue
                if suffix_filter and fname.suffix != suffix_filter:
                    continue
                fnames.append(fname)

    keys = [fname.stem[:key_length] for fname in fnames]

    if id_map_df is not None:
        id_map = dict(zip(id_map_df["old_id"].astype(str), id_map_df["new_id"].astype(str)))
        keys = [id_map.get(k, k) for k in keys]

    mapping = dict(sorted(zip(keys, fnames)))
    return mapping

if __name__ == "__main__":

    main_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    df = pd.read_csv("/Users/payamsadeghishabestari/Downloads/ant_to_tide (3).csv")
    df = pd.read_csv("/Users/payamsadeghishabestari/Downloads/ant_to_tide (3).csv")
    df.loc[len(df)] = [280, "dsno", 70053, "rest"]
    df.loc[len(df)] = [281, "mkbb", 70054, "rest"]
    df.loc[len(df)] = [282, "sgwy", 70055, "rest"]

    mappings = {
        "dublin": get_sorted_mapping(
            main_dir / "dublin",
            name_filter="open",
            key_length=5,
            file_level="nested",
        ),
        "illinois": get_sorted_mapping(
            main_dir / "illinois",
            name_filter="ses-1_rest",
            key_length=5,
            file_level="nested",
        ),
        "regensburg": get_sorted_mapping(
            main_dir / "regensburg",
            exclude_substr="ses",
            suffix_filter=".vhdr",
            key_length=5,
            file_level="nested",
        ),
        "tuebingen": get_sorted_mapping(
            main_dir / "tuebingen",
            suffix_filter=".vhdr",
            key_length=5,
            file_level="nested",
        ),
        "zuerich": get_sorted_mapping(
            main_dir / "zuerich",
            name_filter="rest.vhdr",
            key_length=4,
            file_level="flat",
            id_map_df=df,
        ),
    }

    with open("../material/fname_dict.pkl", "wb") as f:
        pickle.dump(mappings, f)