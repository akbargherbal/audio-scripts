import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def plot_audio_with_breaks(vocals, vocals_sr, silent_regions, rms, hop_length, threshold):
    """Plot the vocals waveform and RMS energy with detected breaks highlighted"""
    plt.figure(figsize=(15, 8))
    
    # Plot waveform
    time = np.arange(len(vocals)) / vocals_sr
    plt.subplot(2, 1, 1)
    plt.plot(time, vocals, color='blue', alpha=0.5, label='Vocals')
    
    # Highlight silent regions
    for start_sample, end_sample in silent_regions:
        start_time = start_sample / vocals_sr
        end_time = end_sample / vocals_sr
        plt.axvspan(start_time, end_time, color='red', alpha=0.2)
    
    plt.grid(True)
    plt.legend()
    plt.title('Vocal Waveform with Detected Breaks')
    
    # Plot RMS energy
    plt.subplot(2, 1, 2)
    frames_time = np.arange(len(rms)) * hop_length / vocals_sr
    plt.plot(frames_time, rms, label='RMS Energy')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    # Highlight silent regions
    for start_sample, end_sample in silent_regions:
        start_time = start_sample / vocals_sr
        end_time = end_sample / vocals_sr
        plt.axvspan(start_time, end_time, color='red', alpha=0.2)
    
    plt.grid(True)
    plt.legend()
    plt.title('RMS Energy with Threshold')
    
    plt.tight_layout()
    plt.savefig('breaks_detection_visualization.png')
    plt.close()

def detect_vocal_breaks(vocals_file, instrument_file, min_silence_duration=5, silence_threshold=0.015):
    """
    Detect periods of silence in vocals and extract corresponding instrumental segments.
    """
    # Load audio files using librosa
    print(f"Loading vocals from: {vocals_file}")
    vocals, vocals_sr = librosa.load(vocals_file, sr=None)
    
    print(f"Loading instrumental from: {instrument_file}")
    instrument, instrument_sr = librosa.load(instrument_file, sr=None)
    
    # Print some debug information
    print(f"\nAudio Information:")
    print(f"Vocals duration: {len(vocals)/vocals_sr:.2f} seconds")
    print(f"Vocals sample rate: {vocals_sr} Hz")
    print(f"Vocals max amplitude: {np.max(np.abs(vocals)):.4f}")
    print(f"Instrumental duration: {len(instrument)/instrument_sr:.2f} seconds")
    
    # Ensure same sample rate
    if vocals_sr != instrument_sr:
        print(f"Warning: Different sample rates detected. Resampling instrumental to match vocals.")
        instrument = librosa.resample(instrument, orig_sr=instrument_sr, target_sr=vocals_sr)
        instrument_sr = vocals_sr
    
    # Calculate RMS energy in small windows
    frame_length = int(0.1 * vocals_sr)  # 100ms windows (increased from 50ms)
    hop_length = frame_length // 4  # Increased overlap for smoother detection
    rms = librosa.feature.rms(y=vocals, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Print silence detection debug info
    print(f"\nSilence Detection Info:")
    print(f"Threshold: {silence_threshold}")
    print(f"Number of frames: {len(rms)}")
    print(f"Max RMS value: {np.max(rms):.4f}")
    print(f"Min RMS value: {np.min(rms):.4f}")
    
    # Find segments where vocals are silent
    is_silent = rms < silence_threshold
    print(f"Number of silent frames: {np.sum(is_silent)}")
    
    # Find contiguous silent regions with hysteresis
    silent_regions = []
    start = None
    frames_needed = int(min_silence_duration * vocals_sr / hop_length)
    
    # Add hysteresis: require slightly higher threshold to exit silence
    exit_threshold = silence_threshold * 1.2
    
    for i in range(len(rms)):
        if rms[i] < silence_threshold and start is None:
            start = i
        elif start is not None and (rms[i] > exit_threshold or i == len(rms)-1):
            duration_frames = i - start
            if duration_frames >= frames_needed:
                # Convert frame indices to samples
                start_sample = start * hop_length
                end_sample = min(i * hop_length, len(vocals))
                silent_regions.append((start_sample, end_sample))
            start = None
    
    # Plot the detection results
    plot_audio_with_breaks(vocals, vocals_sr, silent_regions, rms, hop_length, silence_threshold)
    
    # Extract and save instrumental breaks
    for i, (start_sample, end_sample) in enumerate(silent_regions):
        start_time = start_sample / vocals_sr
        end_time = end_sample / vocals_sr
        
        # Add small margins to avoid cutting off abruptly
        margin_samples = int(0.1 * vocals_sr)  # 100ms margin
        start_sample = max(0, start_sample - margin_samples)
        end_sample = min(len(instrument), end_sample + margin_samples)
        
        # Extract the corresponding segment from the instrumental
        break_segment = instrument[start_sample:end_sample]
        
        # Apply fade in/out
        fade_samples = int(0.05 * vocals_sr)  # 50ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        break_segment[:fade_samples] *= fade_in
        break_segment[-fade_samples:] *= fade_out
        
        # Save the segment
        output_filename = f'instrument_break_{i+1}.wav'
        sf.write(output_filename, break_segment, vocals_sr)
        
        print(f"\nFound break {i+1}: {start_time:.2f}s - {end_time:.2f}s")
        print(f"Break duration: {(end_time - start_time):.2f}s")
        print(f"Saved as: {output_filename}")
    
    return silent_regions

def main():
    vocals_file = "./separated/htdemucs/malik_tasooq_12/vocals.wav"
    instrument_file = "./separated/htdemucs/malik_tasooq_12/instruments.wav"
    
    print("Detecting instrumental breaks...")
    silent_regions = detect_vocal_breaks(
        vocals_file,
        instrument_file,
        min_silence_duration=7,     # 5 seconds minimum silence
        silence_threshold=0.05    # Adjust this based on visualization
    )
    
    print(f"\nFound {len(silent_regions)} instrumental breaks")
    print("\nCreated 'breaks_detection_visualization.png' for visual inspection")

if __name__ == "__main__":
    main()
