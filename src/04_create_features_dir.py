from pathlib import Path
import shutil


base_dir = Path("/home/ubuntu/volume/Tinnorm")
bids_dir = base_dir / "BIDS"
features_dir = base_dir / "features"
features_dir.mkdir(exist_ok=True)

for subject_folder in bids_dir.iterdir():
    if subject_folder.is_dir():
        for mode in ["power", "aperiodic", "conn"]:
            for preproc_level in [1, 2, 3]:
                zip_file = subject_folder / "ses-01" / "eeg" / f"{mode}_preproc_{preproc_level}.zip"
                if zip_file.exists():
                    new_name = f"{subject_folder.name}_{mode}_preproc_{preproc_level}.zip"
                    dest_file = features_dir / new_name
                    shutil.move(str(zip_file), dest_file)
                    print(f"Moved: {zip_file} → {dest_file}")
                else:
                    print(f"No zip file found in {subject_folder}")