#!/usr/bin/env python3

"""
Demucs CLI - Audio Source Separation Tool with Batch Processing

Dependencies:
    - torch
    - soundfile>=0.9.0
    - soxr>=0.3.6
    - numpy<2
    - demucs>=4.0.0
    - ffmpeg (system installation)

Installation:
    1. Install FFmpeg for your system
    2. Install Python dependencies:
       pip install torch soundfile soxr numpy demucs
"""

import argparse
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


def setup_logging():
    """Initialize logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Demucs CLI - Audio Source Separation Tool"
    )

    # Model selection
    parser.add_argument(
        "--model",
        default="htdemucs",
        help="Model name: mdx_extra, mdx_extra_q, htdemucs, htdemucs_ft, etc. (default: htdemucs)",
    )

    # Separation mode
    parser.add_argument(
        "--mode",
        choices=["basic", "full"],
        default="basic",
        help="Separation mode: basic (vocals/instruments) or full (drums/bass/vocals/other) (default: full)",
    )

    # Input/Output
    parser.add_argument(
        "input", type=str, nargs="+", help="Input audio files or directories"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="separated/{model}/{track}/{stem}.{ext}",
        help="Output path template (default: separated/{model}/{track}/{stem}.{ext})",
    )
    parser.add_argument(
        "--format",
        choices=["wav", "flac", "mp3"],
        default="mp3",
        help="Output format (default: wav)",
    )

    # Batch processing options
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively process directories"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for file reading/writing (default: 1)",
    )

    # Separation parameters
    parser.add_argument(
        "--segment",
        type=float,
        default=10,
        help="Segment length in seconds (default: 10)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.1,
        help="Overlap between segments (default: 0.1)",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=2,
        help="Number of random shifts for better separation (default: 2)",
    )

    # Device selection
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument(
        "--device",
        default=default_device,
        help=f"Device to use (cpu, cuda, mps) (default: {default_device})",
    )

    # Add batch size parameter
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process simultaneously on GPU (default: 6)",
    )

    return parser.parse_args()


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


def read_audio(path: pathlib.Path, target_sr: int = 44100) -> np.ndarray:
    """Read audio file using soundfile with FFmpeg fallback"""
    logging.info(f"Reading audio file: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Try reading with soundfile first
        audio, sr = soundfile.read(str(path), dtype="float32", always_2d=True)

        # Resample if necessary
        if sr != target_sr:
            logging.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            audio = soxr.resample(audio, sr, target_sr, "VHQ")

        return audio

    except Exception as e:
        logging.info(f"Soundfile read failed, using FFmpeg fallback: {e}")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found in PATH. Please install FFmpeg.")

        # Create a temporary WAV file
        temp_wav = path.parent / f"{path.stem}_temp.wav"
        try:
            # Convert to WAV using FFmpeg
            command = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-i",
                str(path),
                "-ar",
                str(target_sr),  # Set sample rate
                "-ac",
                "2",  # Force stereo
                "-acodec",
                "pcm_f32le",  # Use 32-bit float
                str(temp_wav),
            ]

            process = subprocess.run(
                command, capture_output=True, text=True, check=True
            )

            # Read the temporary WAV file
            audio, sr = soundfile.read(str(temp_wav), dtype="float32", always_2d=True)
            return audio

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
        finally:
            # Clean up temporary file
            if temp_wav.exists():
                temp_wav.unlink()


def save_audio(path: pathlib.Path, audio: np.ndarray, sr: int, format: str):
    """Save audio using soundfile with FFmpeg fallback for MP3"""
    logging.info(f"Saving audio to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "mp3":
        # Create temporary WAV file
        temp_wav = path.parent / f"{path.stem}_temp.wav"
        try:
            # Save as WAV first
            soundfile.write(str(temp_wav), audio, sr, subtype="FLOAT")

            # Convert to MP3 using FFmpeg
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_wav),
                "-acodec",
                "libmp3lame",
                "-q:a",
                "2",  # High quality VBR
                str(path),
            ]

            process = subprocess.run(
                command, capture_output=True, text=True, check=True
            )

        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {e}")
        finally:
            # Clean up temporary file
            if temp_wav.exists():
                temp_wav.unlink()
    else:
        # For WAV and FLAC, use soundfile directly
        subtype = "FLOAT" if format == "wav" else "PCM_24"
        soundfile.write(str(path), audio, sr, subtype=subtype)


def combine_sources(sources: torch.Tensor, source_indices: list) -> torch.Tensor:
    """Combine multiple sources into a single stem"""
    return torch.sum(sources[source_indices], dim=0)


def process_batch(batch_paths: list, model, args) -> list:
    """Process a batch of audio files simultaneously"""
    try:
        # Read and prepare all files in the batch
        wavs = []
        for path in batch_paths:
            wav = read_audio(path, model.samplerate)
            wav = torch.from_numpy(wav)
            wav = wav.t()
            wav = wav.unsqueeze(0)
            wavs.append(wav)

        # Stack all audio files into a single batch tensor
        wav_batch = torch.cat(wavs, dim=0)
        
        # Normalize batch
        ref = wav_batch.mean(1)
        wav_batch = (wav_batch - ref.mean()) / ref.std()

        segment = int(args.segment * model.samplerate)

        # Process the entire batch at once
        sources_batch = apply_model(
            model,
            wav_batch,
            shifts=args.shifts,
            split=segment,
            overlap=args.overlap,
            progress=False,
            device=args.device,
        )

        results = []
        source_map = {name: idx for idx, name in enumerate(model.sources)}

        # Process each result in the batch
        for batch_idx, (sources, ref) in enumerate(zip(sources_batch, ref)):
            sources = sources * ref.std() + ref.mean()

            if args.mode == "basic":
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
                output_path = str(args.output).format(
                    model=args.model,
                    track=batch_paths[batch_idx].stem,
                    stem=stem_name,
                    ext=args.format,
                )
                save_audio(
                    pathlib.Path(output_path), source.T, model.samplerate, args.format
                )

            results.append(True)

        return results

    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        return [False] * len(batch_paths)


def main():
    """Main execution function"""
    args = parse_arguments()
    setup_logging()

    try:
        # Load model
        logging.info(f"Loading model {args.model}...")
        model = get_model(args.model)
        model.to(args.device)
        logging.info(f"Successfully loaded model {args.model}")
        logging.info(f"Model sources: {model.sources}")

        # Find all audio files
        audio_files = find_audio_files(args.input, args.recursive)
        if not audio_files:
            logging.error("No audio files found!")
            sys.exit(1)

        logging.info(f"Found {len(audio_files)} audio files to process")

        # Process files with progress bar
        successful = 0
        failed = 0

        # Determine batch size based on device
        batch_size = args.batch_size if args.device == "cuda" else 1

        # Process files in batches
        for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing batches"):
            batch_paths = audio_files[i:i + batch_size]
            results = process_batch(batch_paths, model, args)
            
            successful += sum(results)
            failed += len(results) - sum(results)

        logging.info(f"Processing completed: {successful} successful, {failed} failed")

        if failed > 0:
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error during batch processing: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
