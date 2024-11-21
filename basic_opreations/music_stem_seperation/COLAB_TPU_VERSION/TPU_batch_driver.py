import multiprocessing as mp
from multiprocessing import Pool, Lock, Manager
import os
import time
import logging
import psutil
from datetime import datetime
import shutil
from instrumental_breaks import detect_vocal_breaks
import traceback
import tempfile
import uuid
import pandas as pd
import gc

# Load YouTube ID mapping
dict_ytid_folder = pd.read_pickle("./STAGE_04A_INPUT.pkl")
dict_ytid_folder = dict(
    zip(dict_ytid_folder["folder_name"], dict_ytid_folder["youtube_id"])
)

def give_id(folder_name):
    """Convert folder name to YouTube ID format"""
    return f"START_HERE{dict_ytid_folder[folder_name]}END_HERE"

def setup_logging(base_dir):
    """Configure logging with timestamps and proper formatting"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, 'processing_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'processing_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return log_dir

def get_optimal_processes():
    cpu_count = psutil.cpu_count(logical=True)  # Will return 96
    # Use 85% of cores, no artificial cap
    return max(1, int(cpu_count * 0.85))  # Returns ~81 processes


def safe_move_file(src, dst, max_retries=3, retry_delay=1):
    """Safely move a file with retries and proper error handling"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                logging.warning(f"Failed to move file after {max_retries} attempts: {src} -> {dst}: {str(e)}")
                return False
            time.sleep(retry_delay)
    return False

def process_song(args):
    """Process a single song directory"""
    song_dir, base_dir, script_dir, shared_stats = args
    
    start_time = time.time()
    process_name = mp.current_process().name
    
    # Create a temporary directory for this process
    temp_dir = tempfile.mkdtemp(prefix=f"song_processing_{uuid.uuid4().hex[:8]}_")
    original_cwd = os.getcwd()
    
    try:
        logging.info(f"[{process_name}] Starting processing of {song_dir}")
        
        # Construct absolute paths
        song_path = os.path.join(base_dir, 'ADD_MUSIC_STAGE_03', song_dir)
        vocals_file = os.path.join(song_path, "vocals.mp3")
        instrument_file = os.path.join(song_path, "instruments.mp3")
        
        # Verify files exist
        if not os.path.exists(vocals_file) or not os.path.exists(instrument_file):
            raise FileNotFoundError(f"Required files not found in {song_path}")
        
        # Change to temp directory for processing
        os.chdir(temp_dir)
        
        # Process the song
        silent_regions = detect_vocal_breaks(
            vocals_file,
            instrument_file,
            min_silence_duration=7,
            silence_threshold=0.05
        )
        
        # Create output directory
        output_dir = os.path.join(base_dir, "instrumental_breaks", song_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Move files from temp directory to final location
        num_breaks = len(silent_regions)
        
        # Handle visualization file
        viz_file = "breaks_detection_visualization.png"
        if os.path.exists(viz_file):
            safe_move_file(
                os.path.join(temp_dir, viz_file),
                os.path.join(output_dir, f"breaks_detection_{uuid.uuid4().hex[:8]}.png")
            )
        
        # Handle break files
        youtube_id_prefix = give_id(song_dir)
        for i in range(num_breaks):
            break_file = f"instrument_break_{i+1}.wav"
            if os.path.exists(break_file):
                new_name = f"{youtube_id_prefix}_IB_{i+1}.wav"
                safe_move_file(
                    os.path.join(temp_dir, break_file),
                    os.path.join(output_dir, new_name)
                )
        
        processing_time = time.time() - start_time
        
        # Update shared statistics
        with Lock():
            shared_stats['processed_songs'] += 1
            shared_stats['total_breaks'] += num_breaks
            shared_stats['total_time'] += processing_time
            shared_stats['successful_songs'].append(song_dir)
        
        logging.info(f"[{process_name}] Completed {song_dir}: {num_breaks} breaks in {processing_time:.2f}s")
        return True, song_dir, num_breaks, processing_time
        
    except Exception as e:
        error_msg = f"Error processing {song_dir}: {str(e)}"
        logging.error(f"[{process_name}] {error_msg}")
        
        with Lock():
            shared_stats['failed_songs'].append((song_dir, str(e)))
        
        return False, song_dir, str(e), time.time() - start_time
        
    finally:
        try:
            os.chdir(original_cwd)
            shutil.rmtree(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

def process_batch(batch_songs, base_dir, script_dir, batch_num):
    """Process a batch of songs with its own process pool"""
    logging.info(f"Starting batch {batch_num} with {len(batch_songs)} songs")
    
    # Create a new manager for this batch
    manager = Manager()
    shared_stats = manager.dict({
        'processed_songs': 0,
        'total_breaks': 0,
        'total_time': 0,
        'successful_songs': manager.list(),
        'failed_songs': manager.list()
    })
    
    # Prepare arguments
    song_args = [(song, base_dir, script_dir, shared_stats) for song in batch_songs]
    
    # Process this batch
    num_processes = get_optimal_processes()
    start_time = time.time()
    
    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_song, song_args)
            
            # Filter out None results
            results = [r for r in results if r is not None]
            
            batch_time = time.time() - start_time
            successful = [r for r in results if r[0]]
            failed = [r for r in results if not r[0]]
            
            batch_report = f"""
Batch {batch_num} Complete!
===================
Batch Runtime: {batch_time:.2f} seconds
Songs Processed: {len(successful)} of {len(batch_songs)}
Total Breaks Found: {shared_stats['total_breaks']}
Average Processing Time: {shared_stats['total_time']/max(1, len(successful)):.2f} seconds per song

Failed Songs in Batch:
------------
{chr(10).join(f"- {song}: {error}" for song, error in shared_stats['failed_songs'])}
"""
            logging.info(batch_report)
            
            return results, shared_stats['total_breaks'], batch_time
            
    except Exception as e:
        logging.error(f"Batch {batch_num} failed: {str(e)}")
        return [], 0, time.time() - start_time
    finally:
        pool.close()
        pool.join()
        manager.shutdown()
        gc.collect()

def main():
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.curdir)
    
    # Set up logging
    log_dir = setup_logging(base_dir)
    logging.info(f"Starting batched audio processing from {base_dir}")
    
    # Get list of songs to process
    sample_music_dir = os.path.join(base_dir, "ADD_MUSIC_STAGE_03")
    song_dirs = []
    
    for song_dir in os.listdir(sample_music_dir):
        song_path = os.path.join(sample_music_dir, song_dir)
        if not os.path.isdir(song_path):
            continue
        
        vocals_file = os.path.join(song_path, "vocals.mp3")
        instrument_file = os.path.join(song_path, "instruments.mp3")
        
        if os.path.exists(vocals_file) and os.path.exists(instrument_file):
            song_dirs.append(song_dir)
        else:
            logging.warning(f"Skipping {song_dir}: Missing required files")
    
    # Process in batches
    BATCH_SIZE = 200 # ON TPU 200; on CPU 50
    all_results = []
    total_breaks = 0
    total_time = 0
    
    for batch_num, i in enumerate(range(0, len(song_dirs), BATCH_SIZE), 1):
        batch = song_dirs[i:i + BATCH_SIZE]
        results, breaks, batch_time = process_batch(batch, base_dir, script_dir, batch_num)
        
        all_results.extend(results)
        total_breaks += breaks
        total_time += batch_time
        
        # Cleanup between batches
        gc.collect()
        time.sleep(1)
        
        logging.info(f"Completed batch {batch_num}/{(len(song_dirs) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    # Generate final report
    successful = [r for r in all_results if r[0]]
    failed = [r for r in all_results if not r[0]]
    
    final_report = f"""
Processing Complete!
===================
Total Runtime: {total_time:.2f} seconds
Total Songs Processed: {len(successful)} of {len(song_dirs)}
Total Breaks Found: {total_breaks}
Average Processing Time: {total_time/max(1, len(successful)):.2f} seconds per song
Average Breaks per Song: {total_breaks/max(1, len(successful)):.1f}

Successful Songs:
----------------
{chr(10).join(f"- {song}: {breaks} breaks in {time:.2f}s" for _, song, breaks, time in successful)}

Failed Songs:
------------
{chr(10).join(f"- {song}: {error}" for _, song, error in [(r[1], r[2]) for r in failed])}
"""
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(log_dir, f'final_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    logging.info("All processing complete! Final report saved to: " + report_path)
    print(final_report)

if __name__ == "__main__":
    main()
