import librosa
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import os
from datetime import datetime

@dataclass
class AudioSession:
    """Represents a distinct recording session within the audio file"""
    start_time: float
    end_time: float
    rms_db: float
    session_number: int

class AudioSessionSplitter:
    def __init__(self,
                 analysis_window: float = 300.0,  # 5-minute windows for smoothing
                 level_threshold_db: float = 2.0,  # Minimum dB difference to consider a new session
                 min_session_length: float = 300.0):  # Minimum session length (5 minutes)
        self.analysis_window = analysis_window
        self.level_threshold_db = level_threshold_db
        self.min_session_length = min_session_length
        self.logger = logging.getLogger(__name__)

    def _calculate_smooth_rms_profile(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate a smoothed RMS level profile of the entire audio file.
        Uses overlapping windows to create a smoother profile.
        """
        window_samples = int(self.analysis_window * sr)
        hop_length = window_samples // 4  # 75% overlap for smoother profile
        
        # Calculate RMS values
        rms = librosa.feature.rms(y=y, 
                                frame_length=window_samples,
                                hop_length=hop_length)[0]
        
        # Convert to dB with fixed reference
        db = librosa.amplitude_to_db(rms, ref=1.0)
        
        # Get timestamps for each measurement
        timestamps = librosa.times_like(db, sr=sr, hop_length=hop_length)
        
        return db, timestamps

    def _find_session_boundaries(self, db_profile: np.ndarray, timestamps: np.ndarray) -> List[AudioSession]:
        """
        Find recording session boundaries by looking for sustained level changes.
        """
        sessions = []
        current_session_start = 0
        current_session_db = db_profile[0]
        min_samples = int(self.min_session_length / (timestamps[1] - timestamps[0]))
        
        # Calculate the mean level for each potential session
        # using a sliding window approach
        window_size = min_samples
        
        i = window_size
        while i < len(db_profile) - window_size:
            # Calculate mean levels of current and next windows
            current_mean = np.mean(db_profile[i-window_size:i])
            next_mean = np.mean(db_profile[i:i+window_size])
            
            # If we detect a significant sustained change in level
            if abs(next_mean - current_mean) > self.level_threshold_db:
                # Only create a new session if the current one is long enough
                if timestamps[i] - timestamps[current_session_start] >= self.min_session_length:
                    sessions.append(AudioSession(
                        start_time=timestamps[current_session_start],
                        end_time=timestamps[i],
                        rms_db=np.mean(db_profile[current_session_start:i]),
                        session_number=len(sessions) + 1
                    ))
                    current_session_start = i
                    current_session_db = next_mean
            
            i += 1
        
        # Add the final session
        if len(timestamps) - current_session_start >= min_samples:
            sessions.append(AudioSession(
                start_time=timestamps[current_session_start],
                end_time=timestamps[-1],
                rms_db=np.mean(db_profile[current_session_start:]),
                session_number=len(sessions) + 1
            ))
        
        return sessions

    def _log_analysis_results(self, sessions: List[AudioSession]) -> None:
        """Log the analysis results in a human-readable format."""
        self.logger.info("\n=== Audio Analysis Results ===")
        
        if len(sessions) == 1:
            self.logger.info("Found single recording session:")
            session = sessions[0]
            self.logger.info(
                f"Duration: {session.start_time/60:.1f}m to {session.end_time/60:.1f}m "
                f"(RMS level: {session.rms_db:.1f} dB)"
            )
        else:
            self.logger.info(f"Found {len(sessions)} distinct recording sessions:")
            for session in sessions:
                self.logger.info(
                    f"Session {session.session_number}: "
                    f"{session.start_time/60:.1f}m to {session.end_time/60:.1f}m "
                    f"(RMS level: {session.rms_db:.1f} dB)"
                )

    def _save_audio_session(self, y: np.ndarray, sr: int, session: AudioSession, 
                          output_dir: Path, output_prefix: str) -> None:
        """Save an individual session to file."""
        start_idx = int(session.start_time * sr)
        end_idx = int(session.end_time * sr)
        session_audio = y[start_idx:end_idx]
        
        output_path = output_dir / f"{output_prefix}_session_{session.session_number:03d}.wav"
        sf.write(str(output_path), session_audio, sr)
        
        self.logger.info(f"Saved session {session.session_number} to {output_path}")

    def process_audio_file(self, audio_path: str, output_prefix: str) -> Optional[List[AudioSession]]:
        """
        Process an audio file and split it into separate recording sessions.
        """
        self.logger.info(f"Analyzing audio file: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Get smoothed RMS profile
        db_profile, timestamps = self._calculate_smooth_rms_profile(y, sr)
        
        # Find session boundaries
        sessions = self._find_session_boundaries(db_profile, timestamps)
        
        # Log results
        self._log_analysis_results(sessions)
        
        # If we found multiple sessions, save them
        if len(sessions) > 1:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            for session in sessions:
                self._save_audio_session(y, sr, session, output_dir, output_prefix)
            
            return sessions
        
        return None

def setup_logging(log_dir: str = "logs") -> None:
    """Configure logging with both file and console handlers."""
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"audio_splitter_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main(audio_path: str, output_prefix: str):
    """Main function to split audio file into sessions."""
    setup_logging()
    
    splitter = AudioSessionSplitter()
    
    try:
        sessions = splitter.process_audio_file(audio_path, output_prefix)
        
        if sessions is None:
            print("\nConclusion: Single consistent recording session detected.")
        else:
            print(f"\nConclusion: Split into {len(sessions)} distinct recording sessions.")
            
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio recording session splitter")
    parser.add_argument("audio_path", help="Path to the input audio file")
    parser.add_argument("output_prefix", help="Prefix for output files")
    parser.add_argument("--window", type=float, default=300.0,
                      help="Analysis window size in seconds (default: 300)")
    parser.add_argument("--threshold", type=float, default=2.0,
                      help="Level difference threshold in dB (default: 2.0)")
    
    args = parser.parse_args()
    main(args.audio_path, args.output_prefix)
