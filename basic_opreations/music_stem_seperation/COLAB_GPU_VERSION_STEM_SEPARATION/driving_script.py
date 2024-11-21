# driving_script.py
import logging
import pathlib
import re
from batch_stem_separation import process_directories, get_model, DEFAULT_DEVICE

# Configuration
INPUT_BASE_PATH = "MUSIC_STAGE_02"
OUTPUT_BASE_PATH = "ADD_MUSIC_STAGE_03"

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def get_sorted_directories(base_path):
    base_path = pathlib.Path(base_path)
    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return []

    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    return sorted(dirs, key=lambda x: (
        x.name.split('_')[1],  # Category (A, B, C, D)
        natural_sort_key(x.name)
    ))

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load model once
    model = get_model('htdemucs')
    model.to(DEFAULT_DEVICE)
    
    # Process directories in order
    directories = get_sorted_directories(INPUT_BASE_PATH)
    for directory in directories:
        logging.info(f"Processing directory: {directory}")
        process_directories([directory], model)

if __name__ == "__main__":
    main()