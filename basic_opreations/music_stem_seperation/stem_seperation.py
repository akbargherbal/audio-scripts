#!/usr/bin/env python3

"""
Demucs CLI - Audio Source Separation Tool

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
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Demucs CLI - Audio Source Separation Tool')
    
    # Model selection
    parser.add_argument('--model', default='htdemucs', 
                       help='Model name: mdx_extra, mdx_extra_q, htdemucs, htdemucs_ft, etc. (default: htdemucs)')
    
    # Separation mode
    parser.add_argument('--mode', choices=['basic', 'full'], default='full',
                       help='Separation mode: basic (vocals/instruments) or full (drums/bass/vocals/other) (default: full)')
    
    # Input/Output
    parser.add_argument('input', type=pathlib.Path, help='Input audio file')
    parser.add_argument('--output', type=pathlib.Path, default='separated/{model}/{track}/{stem}.{ext}',
                       help='Output path template (default: separated/{model}/{track}/{stem}.{ext})')
    parser.add_argument('--format', choices=['wav', 'flac'], default='wav',
                       help='Output format (default: wav)')
    
    # Separation parameters
    parser.add_argument('--segment', type=float, default=10,
                       help='Segment length in seconds (default: 10)')
    parser.add_argument('--overlap', type=float, default=0.1,
                       help='Overlap between segments (default: 0.1)')
    parser.add_argument('--shifts', type=int, default=2,
                       help='Number of random shifts for better separation (default: 2)')
    
    # Device selection
    default_device = 'cpu'
    if torch.cuda.is_available():
        default_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = 'mps'
    
    parser.add_argument('--device', default=default_device,
                       help=f'Device to use (cpu, cuda, mps) (default: {default_device})')
    
    return parser.parse_args()

def read_audio(path: pathlib.Path, target_sr: int = 44100) -> np.ndarray:
    """Read audio file using soundfile with optional resampling"""
    logging.info(f"Reading audio file: {path}")
    
    try:
        # Try reading with soundfile first
        audio, sr = soundfile.read(str(path), dtype='float32', always_2d=True)
        logging.info(f"Loaded audio: {sr}Hz, shape={audio.shape}")
        
        # Resample if necessary
        if sr != target_sr:
            logging.info(f"Resampling from {sr}Hz to {target_sr}Hz")
            audio = soxr.resample(audio, sr, target_sr, 'VHQ')
        
        return audio
        
    except Exception as e:
        logging.error(f"Failed to read audio with soundfile: {e}")
        if not shutil.which('ffmpeg'):
            raise RuntimeError("FFmpeg not found in PATH. Please install FFmpeg to handle this audio format.")
            
        # Fallback to FFmpeg
        logging.info("Trying FFmpeg fallback...")
        command = [
            'ffmpeg', '-i', str(path),
            '-f', 'wav',
            '-ar', str(target_sr),
            '-ac', '2',
            '-acodec', 'pcm_f32le',
            '-'
        ]
        
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
            
        audio, sr = soundfile.read(io.BytesIO(stdout), dtype='float32', always_2d=True)
        return audio

def save_audio(path: pathlib.Path, audio: np.ndarray, sr: int, format: str):
    """Save audio using soundfile"""
    logging.info(f"Saving audio to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    subtype = 'FLOAT' if format == 'wav' else 'PCM_24'
    soundfile.write(str(path), audio, sr, subtype=subtype)

def combine_sources(sources: torch.Tensor, source_indices: list) -> torch.Tensor:
    """Combine multiple sources into a single stem"""
    return torch.sum(sources[source_indices], dim=0)

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
        
        # Read input audio
        wav = read_audio(args.input, model.samplerate)
        
        # Convert to tensor with shape (channels, samples)
        wav = torch.from_numpy(wav)
        wav = wav.t()  # Now shape is (channels, samples)
        
        # Add batch dimension: (1, channels, samples)
        wav = wav.unsqueeze(0)
        
        logging.info(f"Audio shape: {wav.shape}")
        
        # Separate audio
        logging.info("Starting separation...")
        ref = wav.mean(1)  # mono reference
        wav = (wav - ref.mean()) / ref.std()
        
        # Convert segment length from seconds to samples
        segment = int(args.segment * model.samplerate)
        
        sources = apply_model(
            model,
            wav,
            shifts=args.shifts,
            split=segment,
            overlap=args.overlap,
            progress=True,
            device=args.device
        )[0]
        
        # Apply normalization
        sources = sources * ref.std() + ref.mean()
        
        # Define source mappings
        source_map = {name: idx for idx, name in enumerate(model.sources)}
        
        if args.mode == 'basic':
            # Basic mode: combine sources into vocals and instruments
            logging.info("Processing basic mode (vocals/instruments)...")
            
            # Extract vocals
            vocals = sources[source_map['vocals']]
            
            # Combine all other sources into instruments
            non_vocal_indices = [idx for idx, name in enumerate(model.sources) if name != 'vocals']
            instruments = combine_sources(sources, non_vocal_indices)
            
            # Save vocals and instruments
            output_stems = {
                'vocals': vocals,
                'instruments': instruments
            }
            
        else:  # full mode
            # Full mode: save all stems separately
            logging.info("Processing full mode (all stems)...")
            output_stems = {
                source_name: sources[idx]
                for idx, source_name in enumerate(model.sources)
            }
        
        # Save all stems
        for stem_name, source in output_stems.items():
            source = source.cpu().numpy()
            
            output_path = str(args.output).format(
                model=args.model,
                track=args.input.stem,
                stem=stem_name,
                ext=args.format
            )
            save_audio(pathlib.Path(output_path), source.T, model.samplerate, args.format)
            logging.info(f"Saved {stem_name} stem")
        
        logging.info("Separation completed successfully")
        
    except Exception as e:
        logging.error(f"Error during separation: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
