# Smart Audio Session Splitter

## The Problem: Identifying Distinct Recording Sessions

When creating long-form audio content (like audiobooks, podcasts, or lectures), it's common to record different sections at different times. Each recording session might have slightly different audio characteristics due to:

### Common Causes of Level Variations
1. **Recording Setup Changes**
   - Microphone gain adjustments
   - Different microphone positioning
   - Equipment changes between sessions
   - Room acoustics variations

2. **Session-Specific Factors**
   - Different recording locations
   - Changes in background noise levels
   - Variations in recording software settings

3. **Human Factors**
   - Different speaking distances from microphone
   - Voice energy variations between sessions
   - Starting new sessions without referencing previous levels

### Impact on Audio Quality
- Noticeable level jumps between recording sessions
- Inconsistent listening experience
- Need for manual audio splitting and normalization
- Increased post-processing complexity

## The Solution: Human-Inspired Session Detection

Unlike traditional approaches that focus on detecting short-term variations or using complex statistical analysis, our solution mimics how a human audio engineer would identify different recording sessions by looking at the waveform:

### Key Principles

1. **Session-Based Analysis**
   - Look at the overall level profile of the entire recording
   - Identify sustained changes in audio levels
   - Focus on finding natural session boundaries
   - Avoid splitting on brief variations or natural dynamics

2. **Sustained Level Detection**
   - Use long analysis windows with overlap
   - Look for significant level changes that persist
   - Require minimum session lengths
   - Smooth out short-term variations

3. **Practical Thresholds**
   - Use dB differences that reflect real session changes
   - Set minimum durations that match typical recording patterns
   - Avoid over-splitting due to natural level variations
   - Maintain original audio integrity

### Technical Implementation

1. **Smooth Level Analysis**
   ```python
   # Calculate RMS with overlapping windows
   window_samples = int(analysis_window * sr)
   hop_length = window_samples // 4  # 75% overlap
   rms = librosa.feature.rms(y=y, frame_length=window_samples, 
                            hop_length=hop_length)
   ```

2. **Session Boundary Detection**
   ```python
   # Look for sustained level changes
   current_mean = np.mean(db_profile[i-window_size:i])
   next_mean = np.mean(db_profile[i:i+window_size])
   
   if abs(next_mean - current_mean) > level_threshold_db:
       # Validate minimum session length
       if duration >= min_session_length:
           sessions.append(...)
   ```

### Key Features

1. **Intelligent Analysis**
   - Focuses on finding genuine session boundaries
   - Uses overlapping windows for smooth analysis
   - Avoids false splits from natural dynamics
   - Maintains minimum session lengths

2. **Professional Output**
   - Splits audio at natural session boundaries
   - Preserves original audio quality
   - Creates clearly labeled session files
   - Provides detailed analysis logs

3. **Configurable Parameters**
   - Analysis window size (default: 5 minutes)
   - Level difference threshold (default: 2.0 dB)
   - Minimum session length (default: 5 minutes)
   - Output file naming and format

## Usage

### Basic Usage
```bash
python session_splitter.py input.mp3 output_prefix
```

### Custom Parameters
```bash
python session_splitter.py input.mp3 output_prefix --window 300 --threshold 2.0
```

### Output Structure
```
output/
├── prefix_session_001.wav
├── prefix_session_002.wav
└── prefix_session_003.wav

logs/
└── audio_splitter_20240424_123456.log
```

## Installation

### Prerequisites
```bash
pip install librosa numpy scipy soundfile
```

### Dependencies
- Python 3.7+
- librosa
- numpy
- scipy
- soundfile

## Benefits of This Approach

1. **Natural Session Detection**
   - Matches how humans identify session boundaries
   - Avoids artificial splitting points
   - Preserves natural audio dynamics
   - Handles varying content types

2. **Robust Analysis**
   - Resilient to brief level variations
   - Identifies genuine session boundaries
   - Maintains minimum session lengths
   - Smooth transition detection

3. **Professional Output**
   - Clean session splitting
   - Detailed analysis logs
   - Preserved audio quality
   - Organized file structure

4. **Efficient Processing**
   - Single-pass analysis
   - Memory-efficient operations
   - Clear progress logging
   - Robust error handling

## Future Improvements

1. **Analysis Enhancements**
   - Additional session characteristics detection
   - Spectral analysis integration
   - Advanced boundary refinement
   - Multiple threshold detection

2. **User Interface**
   - Visual waveform display
   - Interactive session boundary adjustment
   - Real-time analysis preview
   - Batch processing interface

3. **Processing Features**
   - Automatic level normalization
   - Cross-session level matching
   - Metadata preservation
   - Multiple format support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please ensure:
- Clear code documentation
- Comprehensive error handling
- Detailed logging
- Unit test coverage
- Updated documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
