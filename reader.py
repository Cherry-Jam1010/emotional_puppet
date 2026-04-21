import io
import zipfile
from pathlib import Path
from typing import List

import scipy.io as sio


ZIP_PATH = Path(r"F:\SEED_EEG_3D\SEED-VII.zip")
OUTPUT_DIR = Path(r"F:\SEED_EEG_3D\SEED-VII_EEG_preprocessed")

# Keep only preprocessed EEG .mat files.
EEG_PREFIX = "SEED-VII/EEG_preprocessed/"

# Set to True when you want to extract the filtered EEG files.
EXTRACT_EEG_FILES = True

# Preview one EEG .mat file after filtering/extracting.
PREVIEW_MAT_FILE = "SEED-VII/EEG_preprocessed/1.mat"


def is_eeg_file(member_name: str) -> bool:
    return member_name.endswith(".mat") and member_name.startswith(EEG_PREFIX)


def list_eeg_files(zf: zipfile.ZipFile) -> List[str]:
    return [name for name in zf.namelist() if is_eeg_file(name)]


def extract_members(zf: zipfile.ZipFile, members: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for member in members:
        zf.extract(member, path=output_dir)


def preview_mat_from_zip(zf: zipfile.ZipFile, target_file: str) -> None:
    if target_file not in zf.namelist():
        print(f"Preview file not found: {target_file}")
        return

    with zf.open(target_file) as f:
        mat_data = sio.loadmat(io.BytesIO(f.read()))

    valid_keys = [key for key in mat_data.keys() if not key.startswith("__")]

    print(f"\nPreview: {target_file}")
    print("Variables:")
    for key in valid_keys[:10]:
        value = mat_data[key]
        shape = getattr(value, "shape", None)
        print(f"  {key}: shape={shape}")

    if len(valid_keys) > 10:
        print(f"  ... total {len(valid_keys)} variables")


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip file not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        eeg_files = list_eeg_files(zf)

        print(f"Found {len(eeg_files)} preprocessed EEG .mat files in {ZIP_PATH.name}")
        print(f"  {EEG_PREFIX}: {len(eeg_files)}")

        print("\nFirst 10 EEG files:")
        for name in eeg_files[:10]:
            print(f"  {name}")

        if EXTRACT_EEG_FILES:
            print(f"\nExtracting EEG files to: {OUTPUT_DIR}")
            extract_members(zf, eeg_files, OUTPUT_DIR)
            print("Extraction finished.")

        preview_mat_from_zip(zf, PREVIEW_MAT_FILE)


if __name__ == "__main__":
    main()
