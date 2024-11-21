#!/usr/bin/env python3
from pathlib import Path
import logging
from datetime import datetime
import time
import subprocess
import json
from typing import Dict, List, Optional
import pandas as pd
from pytubefix import YouTube
import unicodedata
import re

PREFIX = input("Enter the prefix for the videos: ")
path_to_dataset = input("Enter the path to the dataset: ")


# Time-stamped directories and files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path(f"{PREFIX}_{TIMESTAMP}")
DOWNLOADS_DIR = RESULTS_DIR / "downloads"
LOG_FILE = RESULTS_DIR / "download.log"
HISTORY_FILE = RESULTS_DIR / "download_history.json"
METADATA_FILE = RESULTS_DIR / "metadata.csv"

# Configuration
GCP_BUCKET = f"gs://gh-september-2024/MUSIC/MUSIC_{TIMESTAMP}/"
BATCH_SIZE = 10
DELAY_SECONDS = 5
MAX_RETRIES = 3
RETRY_DELAY = 10


# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)


def sanitize_filename(title: str) -> str:
    """
    Create a safe filename from the video title.
    Handles UTF-8 characters and special characters.
    """
    # Step 1: Normalize unicode characters
    title = unicodedata.normalize("NFKD", title)

    # Step 2: Convert to ASCII, removing diacritics
    title = title.encode("ASCII", "ignore").decode("ASCII")

    # Step 3: Replace spaces and unwanted characters
    title = re.sub(
        r"[^\w\s-]", "", title
    )  # Remove all non-word chars except spaces and hyphens
    title = re.sub(
        r"[-\s]+", "_", title
    )  # Replace spaces and hyphens with single underscore
    title = title.strip("_")  # Remove leading/trailing underscores

    # Step 4: Ensure reasonable length and lowercase
    title = title[:100].lower()  # Limit length and convert to lowercase

    return title


def load_download_history() -> Dict:
    """Load the download history from JSON file if it exists."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"successful": {}, "failed": {}}


def save_download_history(history: Dict) -> None:
    """Save the download history to JSON file."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def download_audio(video_id: str, retry_count: int = 0) -> Optional[Dict]:
    """
    Download audio from a YouTube video with retry logic.
    Returns metadata dictionary if successful, None if failed.
    """
    url = f"https://youtube.com/watch?v={video_id}"

    try:
        yt = YouTube(url)
        audio_stream = yt.streams.get_audio_only()

        if not audio_stream:
            raise ValueError("No audio stream found")

        # Prepare metadata with sanitized filename
        original_title = yt.title
        safe_title = sanitize_filename(original_title)
        filename = f"{safe_title}_{video_id}.mp3"
        file_path = DOWNLOADS_DIR / filename

        # Download the audio
        logging.info(f"Downloading: {original_title}")
        audio_stream.download(output_path=str(DOWNLOADS_DIR), filename=filename)

        metadata = {
            "video_id": video_id,
            "title": original_title,
            "author": yt.author,
            "length_seconds": yt.length,
            "filename": filename,
            "download_time": datetime.now().isoformat(),
        }

        logging.info(f"Successfully downloaded: {filename}")
        return metadata

    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")

        if retry_count < MAX_RETRIES:
            logging.info(
                f"Retrying in {RETRY_DELAY} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})"
            )
            time.sleep(RETRY_DELAY)
            return download_audio(video_id, retry_count + 1)

        return None


def upload_to_gcp(files: List[Path]) -> bool:
    """Upload files to GCP bucket with improved error handling for filenames."""
    try:
        # Handle each file individually to better manage failures
        for file_path in files:
            upload_cmd = f'gsutil cp "{file_path}" "{GCP_BUCKET}"'
            logging.info(f"Uploading file to GCP: {file_path}")

            result = subprocess.run(
                upload_cmd, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                logging.error(f"Failed to upload {file_path}: {result.stderr}")
                continue

            logging.info(f"Successfully uploaded: {file_path}")

        return True

    except Exception as e:
        logging.error(f"Error during GCP upload: {str(e)}")
        return False


def process_videos(video_ids: List[str]) -> None:
    """Main function to process videos with batch uploading."""
    history = load_download_history()
    current_batch: List[Path] = []
    metadata_list = []

    for index, video_id in enumerate(video_ids, 1):
        if video_id in history["successful"]:
            logging.info(f"Skipping {video_id} - already downloaded")
            continue

        metadata = download_audio(video_id)

        if metadata:
            file_path = DOWNLOADS_DIR / metadata["filename"]
            current_batch.append(file_path)
            metadata_list.append(metadata)
            history["successful"][video_id] = metadata
        else:
            history["failed"][video_id] = {
                "timestamp": datetime.now().isoformat(),
                "attempts": MAX_RETRIES,
            }

        # Upload batch if size reached or last item
        if len(current_batch) >= BATCH_SIZE or index == len(video_ids):
            if current_batch:
                if upload_to_gcp(current_batch):
                    current_batch = []

        save_download_history(history)

        if index < len(video_ids):
            logging.info(f"Waiting {DELAY_SECONDS} seconds before next download...")
            time.sleep(DELAY_SECONDS)

    # Save and upload metadata
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv(METADATA_FILE, index=False, encoding="utf-8")
        upload_to_gcp([METADATA_FILE])

    # Upload log files
    upload_to_gcp([LOG_FILE, HISTORY_FILE])


if __name__ == "__main__":
    df = pd.read_csv(path_to_dataset, encoding="utf-8")
    # Assuming the dataset has a column named 'video_id'
    VIDEO_IDS = df["youtubeId"].tolist()

    logging.info("Starting download process...")
    process_videos(VIDEO_IDS)
    logging.info("Download process completed!")
