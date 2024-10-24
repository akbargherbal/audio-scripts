# 🎧 Audio Processing Scripts

A collection of Python scripts for automating common audio processing tasks. Built for podcasters, musicians, and content creators who need to process audio files efficiently.

## 🎯 What's This For?

These scripts help you automate repetitive audio tasks like:
- Splitting long recordings into separate sessions
- Converting between audio formats
- Normalizing audio levels
- Reducing noise
- Adding effects
- Managing audio metadata

## 📦 Currently Available Scripts

### Basic Operations
- `session_splitter.py`: Automatically detects and splits different recording sessions in a long audio file
- `format_converter.py`: Converts audio between common formats (WAV, MP3, etc.)

### Audio Enhancement
- `noise_reducer.py`: Reduces background noise in recordings
- `loudness_normalizer.py`: Normalizes audio levels for consistent volume

### Effects Processing
- `reverb.py`: Adds customizable reverb effects
- `delay.py`: Adds delay effects

### Quality Control
- `level_checker.py`: Checks audio levels and identifies potential issues
- `silence_detector.py`: Finds long silences in audio files

### Metadata Management
- `tag_editor.py`: Bulk edit audio file metadata
- `file_renamer.py`: Rename audio files based on their metadata

## 🚀 Quick Start

1. Clone this repository:
```bash
git clone https://github.com/yourusername/audio-scripts.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run any script:
```bash
python 01_basic_operations/session_splitter.py input.wav output_prefix
```

## 📖 Example: Splitting Recording Sessions

```bash
# Split a long recording into separate sessions
python 01_basic_operations/session_splitter.py podcast_recording.wav episode5

# This will create:
# - episode5_session_001.wav
# - episode5_session_002.wav
# etc.
```

## ⚙️ Requirements

- Python 3.7 or higher
- librosa
- numpy
- soundfile

## 📁 Repository Structure

```
audio-scripts/
├── 01_basic_operations/    # Format conversion, splitting, joining
├── 02_audio_enhancement/   # Noise reduction, normalization
├── 03_effects/            # Audio effects processing
├── 04_quality_control/    # Level checking, error detection
├── 05_metadata/           # Tag editing, file organization
└── utils/                 # Shared helper functions
```

## 💡 Usage Tips

1. All scripts can be run directly from the command line
2. Use `--help` with any script to see available options:
   ```bash
   python any_script.py --help
   ```
3. Scripts save output files in the same directory by default
4. Most scripts support batch processing of multiple files

## 🤝 Contributing

Found a bug? Want to add a new script? Feel free to:
1. Open an issue
2. Submit a pull request
3. Suggest improvements

## 📝 Script Naming Convention

- Use clear, descriptive names
- Include the main function in the name
- Examples: `session_splitter.py`, `noise_reducer.py`

## 📄 License

MIT License - Feel free to use these scripts for any purpose.

---
Created by Akbar Gherbal - For audio professionals who value their time 🎵
