import os
import hashlib
from collections import defaultdict
from pathlib import Path

def get_file_hash(filepath, block_size=65536):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()

def find_audio_duplicates(directory):
    """Find duplicate audio files in a directory"""
    # Dictionary to store file hashes and their paths
    hash_dict = defaultdict(list)
    # Common audio extensions
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if Path(filename).suffix.lower() in audio_extensions:
                filepath = os.path.join(root, filename)
                try:
                    file_hash = get_file_hash(filepath)
                    hash_dict[file_hash].append(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    # Return only the duplicate files (where hash has more than one file)
    return {k: v for k, v in hash_dict.items() if len(v) > 1}

def print_duplicates(duplicates):
    """Print duplicate files in a readable format"""
    if not duplicates:
        print("No duplicates found.")
        return
        
    print("\nDuplicate files found:")
    for hash_value, file_list in duplicates.items():
        print(f"\nDuplicate group (Hash: {hash_value[:8]}...):")
        for file_path in file_list:
            print(f"  - {file_path}")
        print(f"  Size: {os.path.getsize(file_list[0]) / 1024 / 1024:.2f} MB")

# Example usage
if __name__ == "__main__":
    directory = input("Enter the directory path to scan: ")
    duplicates = find_audio_duplicates(directory)
    print_duplicates(duplicates)
