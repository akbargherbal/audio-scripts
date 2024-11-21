import librosa
import numpy as np

def load_audio_data(path):
    try:
        audio_data, sample_rate = librosa.load(path, sr=None)
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def analyze_audio(path):
    """
    Analyze audio file and return tempo and key information
    
    Args:
        path (str): Path to audio file
        
    Returns:
        dict: Dictionary containing tempo, key, and alternative key
    """
    # Initialize result dictionary
    result = {
        'tempo': None,
        'key': None,
        'alt_key': None
    }
    
    # Load audio data
    audio_data, sample_rate = load_audio_data(path)
    
    if audio_data is not None:
        try:
            # Get tempo
            tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            # Handle tempo as a scalar value, not an array
            result['tempo'] = round(float(tempo), 1)
            
            # Get key information
            # Compute chromagram with better frequency resolution
            chromagram = librosa.feature.chroma_cqt(
                y=audio_data, 
                sr=sample_rate,
                hop_length=512,
                n_chroma=12,
                bins_per_octave=36
            )
            
            # Apply median filtering to reduce noise
            chromagram = librosa.decompose.nn_filter(
                chromagram,
                aggregate=np.median,
                metric='cosine'
            )
            
            # Sum over time to get key profile
            chroma_sum = np.sum(chromagram, axis=1)
            
            # Normalize the sum
            chroma_sum = chroma_sum / np.max(chroma_sum)
            
            # Get the two highest values (key and alternative key)
            key_indices = np.argsort(chroma_sum)[-2:]
            
            # Map indices to musical keys (C, C#, D, etc.)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            result['key'] = key_names[key_indices[1]]
            result['alt_key'] = key_names[key_indices[0]]
            
            # Add confidence scores for keys
            result['key_confidence'] = float(chroma_sum[key_indices[1]])
            result['alt_key_confidence'] = float(chroma_sum[key_indices[0]])
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            print("Full error details:", e.__class__.__name__)
    
    return result

# Example usage:
def print_analysis(path):
    result = analyze_audio(path)
    print(f"\nAnalysis Results for: {path}")
    print(f"Tempo: {result['tempo']} BPM" if result['tempo'] else "Tempo: Could not determine")
    print(f"Key: {result['key']} (confidence: {result['key_confidence']:.2f})" if result['key'] else "Key: Could not determine")
    print(f"Alternative Key: {result['alt_key']} (confidence: {result['alt_key_confidence']:.2f})" if result['alt_key'] else "Alternative Key: Could not determine")

# Example usage:
if __name__ == "__main__":
    print_analysis(path)