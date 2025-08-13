from mne.io import read_raw

def read_vhdr_input_fname(fname):
    """
    Checks .vhdr and .vmrk data to have same names, otherwise fix them.
    """
    try:
        raw = read_raw(fname)
    except:
        with open(fname, "r") as file:
            lines = file.readlines()
        
        lines[5] = f'DataFile={fname.stem}.eeg\n'
        lines[6] = f'MarkerFile={fname.stem}.vmrk\n'

        with open(fname, "w") as file:
            file.writelines(lines)
        with open(f"{fname.with_suffix('')}.vmrk", "r") as file:
            lines = file.readlines()
        lines[4] = f'DataFile={fname.stem}.eeg\n'
        with open(f"{fname.with_suffix('')}.vmrk", "w") as file:
            file.writelines(lines)
        raw = read_raw(fname)
    return raw