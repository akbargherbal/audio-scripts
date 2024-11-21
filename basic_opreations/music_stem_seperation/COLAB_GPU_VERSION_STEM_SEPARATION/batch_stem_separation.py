# batch_stem_separaton.py BASIC Version
import logging
import os
import pathlib
import sys
import typing as tp
from fractions import Fraction
import subprocess
import io
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import soundfile
import soxr
import torch
import torch.nn as nn
from demucs.pretrained import get_model
from demucs.separate import *

print("batch_stem_separation.py BASIC Version")


# Configuration Constants
MODEL_NAME = "htdemucs"  # Options: mdx_extra, mdx_extra_q, htdemucs, htdemucs_ft
SEPARATION_MODE = (
    "basic"  # Options: 'basic' (vocals/instruments) or 'full' (drums/bass/vocals/other)
)
INPUT_PATHS = ["/content/MUSIC_STAGE_02"]  # List of input paths (files or directories)
OUTPUT_TEMPLATE = "ADD_MUSIC_STAGE_03/{track}/{stem}.{ext}"
OUTPUT_FORMAT = "mp3"  # Options: 'wav', 'flac', 'mp3'
MP3_BITRATE = "320k"  # Bitrate for MP3 encoding
RECURSIVE_PROCESSING = True
NUM_WORKERS = 16
SEGMENT_LENGTH = 10.0  # seconds
OVERLAP = 0.1
NUM_SHIFTS = 2

# Device selection
DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def find_audio_files(paths, recursive=False):
    """Find all audio files in given paths"""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}
    audio_files = []

    for path in paths:
        path = pathlib.Path(path)
        if path.is_file() and path.suffix.lower() in audio_extensions:
            audio_files.append(path)
        elif path.is_dir():
            if recursive:
                for root, _, files in os.walk(path):
                    for file in files:
                        file_path = pathlib.Path(root) / file
                        if file_path.suffix.lower() in audio_extensions:
                            audio_files.append(file_path)
            else:
                for file in path.iterdir():
                    if file.is_file() and file.suffix.lower() in audio_extensions:
                        audio_files.append(file)

    return audio_files


def read_with_ffmpeg(path: pathlib.Path, target_sr: int = 44100) -> np.ndarray:
    """Read audio file using FFmpeg with specific sample rate."""
    import io
    import subprocess
    import soundfile

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found in PATH. Please install FFmpeg to handle this audio format."
        )

    logging.info(f"Reading audio with FFmpeg: {path}")
    command = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "wav",
        "-ar",
        str(target_sr),
        "-ac",
        "2",  # Force stereo output
        "-acodec",
        "pcm_f32le",
        "-",
    ]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode()
            logging.error(f"FFmpeg processing failed: {error_msg}")
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        audio, sr = soundfile.read(io.BytesIO(stdout), dtype="float32", always_2d=True)
        logging.info(f"Successfully read audio via FFmpeg: {sr}Hz, shape={audio.shape}")
        return audio

    except Exception as e:
        logging.error(f"Error reading audio with FFmpeg: {str(e)}")
        raise


def read_with_soundfile(path: pathlib.Path, target_sr: int = 44100) -> np.ndarray:
    """Read audio file using soundfile with optional resampling."""
    import soundfile
    import soxr

    logging.info(f"Reading audio with soundfile: {path}")
    try:
        audio, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
        logging.info(f"Successfully read audio: {sr}Hz, shape={audio.shape}")

        # Resample if necessary
        if sr != target_sr:
            logging.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            audio = soxr.resample(audio, sr, target_sr, "VHQ")

        return audio

    except Exception as e:
        logging.error(f"Error reading audio with soundfile: {str(e)}")
        raise


def read_audio(path: pathlib.Path, target_sr: int = 44100) -> np.ndarray:
    """
    Smart audio file reader that chooses the appropriate method based on file format.

    Args:
        path: Path to the audio file
        target_sr: Target sample rate for the output audio

    Returns:
        numpy.ndarray: Audio data as a 2D array (channels x samples)

    Raises:
        RuntimeError: If neither soundfile nor FFmpeg can read the file
    """
    # Dictionary of format-specific readers
    format_readers = {
        ".mp3": read_with_ffmpeg,
        ".m4a": read_with_ffmpeg,
        ".aac": read_with_ffmpeg,
        ".wav": read_with_soundfile,
        ".flac": read_with_soundfile,
        ".aiff": read_with_soundfile,
        ".aif": read_with_soundfile,
    }

    file_ext = path.suffix.lower()

    # Get the appropriate reader for the file format
    reader = format_readers.get(file_ext, None)

    if reader is None:
        logging.warning(f"Unknown format {file_ext}, attempting FFmpeg first...")
        reader = read_with_ffmpeg

    try:
        return reader(path, target_sr)
    except Exception as first_error:
        # If the preferred reader fails, try the other method as fallback
        fallback_reader = (
            read_with_ffmpeg if reader == read_with_soundfile else read_with_soundfile
        )
        try:
            logging.info(f"Primary reader failed, attempting fallback method...")
            return fallback_reader(path, target_sr)
        except Exception as second_error:
            logging.error(f"Both readers failed for {path}")
            logging.error(f"Primary reader error: {str(first_error)}")
            logging.error(f"Fallback reader error: {str(second_error)}")
            raise RuntimeError(f"Failed to read audio file {path} with both readers")


def get_audio_duration(path: pathlib.Path) -> float:
    """
    Get the duration of an audio file in seconds using FFmpeg.
    Useful for logging and progress tracking.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        output = subprocess.check_output(command).decode().strip()
        return float(output)
    except:
        return 0.0  # Return 0 if duration cannot be determined


def save_audio(path: pathlib.Path, audio: np.ndarray, sr: int, format: str):
    """Save audio using soundfile or FFmpeg for MP3"""
    logging.info(f"Saving audio to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "mp3":
        # Create a temporary WAV file
        temp_wav = path.with_suffix(".tmp.wav")
        try:
            # Save as temporary WAV
            soundfile.write(str(temp_wav), audio, sr, subtype="FLOAT")

            # Convert to MP3 using FFmpeg
            if not shutil.which("ffmpeg"):
                raise RuntimeError(
                    "FFmpeg not found in PATH. Required for MP3 encoding."
                )

            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_wav),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                MP3_BITRATE,
                "-q:a",
                "0",  # Highest quality
                str(path),
            ]

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg MP3 encoding failed: {stderr.decode()}")

        finally:
            # Clean up temporary file
            if temp_wav.exists():
                temp_wav.unlink()

    else:
        # Use soundfile for WAV and FLAC
        subtype = "FLOAT" if format == "wav" else "PCM_24"
        soundfile.write(str(path), audio, sr, subtype=subtype)


def combine_sources(sources: torch.Tensor, source_indices: list) -> torch.Tensor:
    """Combine multiple sources into a single stem"""
    return torch.sum(sources[source_indices], dim=0)


def process_file(input_path: pathlib.Path, model) -> None:
    """Process a single audio file"""
    try:
        logging.info(f"Processing file: {input_path}")

        # Read input audio
        wav = read_audio(input_path, model.samplerate)

        # Convert to tensor
        wav = torch.from_numpy(wav)
        wav = wav.t()
        wav = wav.unsqueeze(0)

        # Separate audio
        ref = wav.mean(1)
        wav = (wav - ref.mean()) / ref.std()

        segment = int(SEGMENT_LENGTH * model.samplerate)

        sources = apply_model(
            model,
            wav,
            shifts=NUM_SHIFTS,
            split=segment,
            overlap=OVERLAP,
            progress=False,  # Disable internal progress bar
            device=DEFAULT_DEVICE,
        )[0]

        sources = sources * ref.std() + ref.mean()

        # Define source mappings
        source_map = {name: idx for idx, name in enumerate(model.sources)}

        if SEPARATION_MODE == "basic":
            # Basic mode: vocals and instruments
            vocals = sources[source_map["vocals"]]
            non_vocal_indices = [
                idx for idx, name in enumerate(model.sources) if name != "vocals"
            ]
            instruments = combine_sources(sources, non_vocal_indices)
            output_stems = {"vocals": vocals, "instruments": instruments}
        else:
            # Full mode: all stems
            output_stems = {
                source_name: sources[idx]
                for idx, source_name in enumerate(model.sources)
            }

        # Save stems
        for stem_name, source in output_stems.items():
            source = source.cpu().numpy()

            output_path = OUTPUT_TEMPLATE.format(
                model=MODEL_NAME,
                track=input_path.stem,
                stem=stem_name,
                ext=OUTPUT_FORMAT,
            )
            save_audio(
                pathlib.Path(output_path), source.T, model.samplerate, OUTPUT_FORMAT
            )

        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        return False


# [Keep all helper functions the same until main()]


def process_directories(input_paths=None, model=None):
    """Process a list of directories. This function allows external control of processing."""
    if input_paths is None:
        input_paths = INPUT_PATHS

    if model is None:
        model = get_model(MODEL_NAME)
        model.to(DEFAULT_DEVICE)

    audio_files = find_audio_files(input_paths, RECURSIVE_PROCESSING)
    if not audio_files:
        logging.error("No audio files found!")
        return False

    logging.info(f"Found {len(audio_files)} audio files to process")

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_file, file, model) for file in audio_files]

        for file, future in tqdm(
            zip(audio_files, futures), total=len(audio_files), desc="Processing files"
        ):
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Failed to process {file}: {str(e)}")
                failed += 1

    logging.info(f"Processing completed: {successful} successful, {failed} failed")
    return failed == 0


def main():
    """Main execution function"""
    setup_logging()

    try:
        # Load model
        logging.info(f"Loading model {MODEL_NAME}...")
        model = get_model(MODEL_NAME)
        model.to(DEFAULT_DEVICE)
        logging.info(f"Successfully loaded model {MODEL_NAME}")
        logging.info(f"Model sources: {model.sources}")

        # Process using default settings
        success = process_directories(model=model)

        if not success:
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
