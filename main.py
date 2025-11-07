import os
import io
import sys
import csv
import argparse
import tarfile
import logging
import shutil
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Optional, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from PIL import Image

# Global lock for thread-safe file operations
log_lock = Lock()
progress_lock = Lock()
position_lock = Lock()

# Position manager for tqdm progress bars
class PositionManager:
    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.available = list(range(max_workers))
        self.in_use = {}
        self.lock = Lock()
    
    def acquire(self, task_id):
        with self.lock:
            if self.available:
                pos = self.available.pop(0)
                self.in_use[task_id] = pos
                return pos
            return 0
    
    def release(self, task_id):
        with self.lock:
            if task_id in self.in_use:
                pos = self.in_use.pop(task_id)
                self.available.append(pos)
                self.available.sort()

# Configure logging - only to file, not to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('imagenet_downloader.log')
    ]
)
logger = logging.getLogger(__name__)


def create_session_with_retries(max_retries: int = 10, pool_connections: int = 10, pool_maxsize: int = 20) -> requests.Session:
    """
    Creates a requests session with connection pooling and retry strategy.
    
    Args:
        max_retries: Maximum number of retries for failed requests
        pool_connections: Number of connection pools to cache
        pool_maxsize: Maximum number of connections to save in the pool
        
    Returns:
        Configured requests.Session object
    """
    session = requests.Session()
    
    # Retry strategy for transient failures
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # HTTP adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def extract_tar(part_id: str, tar_path: str, target_path: str = None) -> None:
    """
    Extracts a .tar file into a structured directory and resizes all images to 256x256.

    If target_path is given:
        -> target_path/<part_id>/
    Otherwise:
        -> <directory_of_tar>/<part_id>/

    Args:
        part_id (str): ImageNet class ID (e.g., 'n02124075')
        tar_path (str): Path to the .tar file
        target_path (str, optional): Root directory for extraction
    """
    # Determine extraction directory
    if target_path:
        dest_folder = os.path.join(target_path, part_id)
    else:
        dest_folder = os.path.join(os.path.dirname(tar_path), part_id)
    os.makedirs(dest_folder, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            total_files = len(members)

            with tqdm(total=total_files, unit="files", desc=f"Extracting {part_id}", ncols=100, leave=False) as pbar:
                for member in members:
                    member_path = os.path.join(dest_folder, member.name)

                    # Security: block unsafe paths
                    if not os.path.commonpath([dest_folder, member_path]).startswith(dest_folder):
                        continue

                    # Extract file into memory
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        pbar.update(1)
                        continue

                    # Handle images safely
                    try:
                        img = Image.open(io.BytesIO(extracted.read()))
                        img = img.convert("RGB")
                        img = img.resize((256, 256), Image.Resampling.LANCZOS)

                        # Ensure parent folder exists
                        os.makedirs(os.path.dirname(member_path), exist_ok=True)
                        img.save(member_path, format="JPEG", quality=95)
                    except Exception:
                        # Fallback: just write raw file if not an image
                        os.makedirs(os.path.dirname(member_path), exist_ok=True)
                        with open(member_path, "wb") as f:
                            extracted.seek(0)
                            f.write(extracted.read())

                    pbar.update(1)
    except Exception as e:
        logger.error(f"Failed to extract {tar_path}: {e}")
        raise

def download_tar(
    part_id: str, 
    session: requests.Session,
    position_manager: 'PositionManager',
    target_path: str = None, 
    main_link: str = "https://image-net.org/data/winter21_whole", 
    delete_tar: bool = False
) -> Dict[str, any]:
    """
    Downloads a .tar file from ImageNet (winter21_whole) and extracts it.
    Thread-safe with connection pooling and robust error handling.

    Args:
        part_id (str): ImageNet part ID (e.g., 'n02124075').
        session (requests.Session): Reusable session with connection pooling.
        position_manager (PositionManager): Manager for progress bar positions.
        target_path (str, optional): Root directory where everything should be saved. 
                                     If None, uses current working directory.
        main_link (str): Base URL of ImageNet archive.
        delete_tar (bool): Whether to delete the .tar file after extraction. Default is False.
        position (int): Progress bar position for multi-threading display.

    Returns:
        Dict with status information: {'part_id': str, 'success': bool, 'error': str or None}
    """
    # Determine base directory for all operations
    if target_path:
        base_dir = target_path
    else:
        base_dir = os.getcwd()
    
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, "failed_downloads.log")

    tar_url = f"{main_link}/{part_id}.tar"
    filename = f"{part_id}.tar"
    tar_path = os.path.join(base_dir, filename)

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36",
        "Accept": "application/octet-stream",
        "Connection": "keep-alive",
    }

    try:
        # Check if already extracted and completed
        extract_dir = os.path.join(base_dir, part_id)
        completion_marker = os.path.join(extract_dir, ".download_complete")
        
        # Resume functionality: Skip if already completed
        if os.path.exists(completion_marker):
            logger.info(f"Skipping {part_id} - already completed")
            return {'part_id': part_id, 'success': True, 'error': None, 'skipped': True}
        
        # If extraction folder exists but no completion marker, it was interrupted
        if os.path.exists(extract_dir):
            logger.info(f"Resuming {part_id} - cleaning up incomplete extraction")
            import shutil
            shutil.rmtree(extract_dir)
        
        # Clean up any partial tar file
        if os.path.exists(tar_path):
            logger.info(f"Removing partial tar file: {tar_path}")
            os.remove(tar_path)
        
        # Acquire a unique position for this download's progress bar
        position = position_manager.acquire(part_id)
        
        try:
            with session.get(tar_url, stream=True, timeout=520, headers=HEADERS) as r:
                r.raise_for_status()

                total_size = int(r.headers.get("Content-Length", 0))
                chunk_size = 65536

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"[{position:02d}] {part_id}",
                    leave=False,
                    ncols=100,
                    position=position,
                    dynamic_ncols=True,
                    smoothing=0.1,
                ) as pbar:
                    with open(tar_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            
            # Extract the tar file
            extract_tar(part_id, tar_path, target_path)
            
            # Create completion marker for resume functionality
            extract_dir = os.path.join(base_dir, part_id)
            completion_marker = os.path.join(extract_dir, ".download_complete")
            with open(completion_marker, 'w') as f:
                f.write(f"Completed at: {__import__('datetime').datetime.now().isoformat()}\n")
            
            # Delete tar file if requested
            if delete_tar:
                try:
                    os.remove(tar_path)
                    logger.debug(f"Deleted {filename}")
                except Exception as e:
                    logger.warning(f"Could not delete {filename}: {e}")
            
            logger.info(f"✓ Successfully processed {part_id}")
            return {'part_id': part_id, 'success': True, 'error': None, 'skipped': False}
        
        finally:
            # Always release the position back to the pool
            position_manager.release(part_id)

    except Exception as e:
        error_msg = f"{part_id} - {str(e)}"
        
        # Thread-safe logging
        with log_lock:
            with open(log_file, "a") as log:
                log.write(f"{error_msg}\n")
        
        logger.error(f"Failed to download {part_id}: {e}")
        
        # Clean up failed tar file
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
            except:
                pass
        
        return {'part_id': part_id, 'success': False, 'error': str(e), 'skipped': False}

def load_part_ids_from_csv(csv_path: str) -> List[str]:
    """
    Loads part IDs (wnid column) from a CSV file.
    
    Args:
        csv_path: Path to the CSV file with 'wnid' column
        
    Returns:
        List of part IDs (wnids)
    """
    part_ids = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if 'wnid' column exists
            if 'wnid' not in reader.fieldnames:
                logger.error(f"CSV file must contain 'wnid' column. Found: {reader.fieldnames}")
                return []
            
            for row in reader:
                wnid = row.get('wnid', '').strip()
                if wnid:
                    part_ids.append(wnid)
        
        logger.info(f"Loaded {len(part_ids)} part IDs from {csv_path}")
        return part_ids
        
    except Exception as e:
        logger.error(f"Failed to load CSV file {csv_path}: {e}")
        return []


def download_batch(
    part_ids: List[str],
    target_path: str,
    num_workers: int = 10,
    delete_tar: bool = False,
    main_link: str = "https://image-net.org/data/winter21_whole"
) -> Dict[str, int]:
    """
    Downloads and extracts multiple ImageNet classes in parallel using thread pool.
    
    Args:
        part_ids: List of ImageNet class IDs (wnids)
        target_path: Root directory for downloads and extraction
        num_workers: Number of parallel download threads
        delete_tar: Whether to delete tar files after extraction
        main_link: Base URL for ImageNet downloads
        
    Returns:
        Dictionary with statistics: {'success': int, 'failed': int, 'skipped': int}
    """
    if not part_ids:
        logger.error("No part IDs provided")
        return {'success': 0, 'failed': 0, 'skipped': 0}
    
    # Print clean start message to terminal
    print("\n" + "="*80)
    print(f"ImageNet Downloader Started")
    print(f"Classes to download: {len(part_ids)}")
    print(f"Parallel workers: {num_workers}")
    print(f"Target directory: {target_path}")
    print(f"Delete tar files: {delete_tar}")
    print("="*80 + "\n")
    
    # Log to file only
    logger.info(f"Starting batch download of {len(part_ids)} classes with {num_workers} workers")
    logger.info(f"Target directory: {target_path}")
    logger.info(f"Delete tar files: {delete_tar}")
    
    # Create shared session with connection pooling
    session = create_session_with_retries(
        pool_connections=num_workers,
        pool_maxsize=num_workers * 2
    )
    
    # Create position manager for progress bars
    position_manager = PositionManager(num_workers)
    
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    results = []
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all download tasks
        future_to_id = {
            executor.submit(
                download_tar,
                part_id,
                session,
                position_manager,
                target_path,
                main_link,
                delete_tar
            ): part_id
            for idx, part_id in enumerate(part_ids)
        }
        
        # Process completed tasks
        for future in as_completed(future_to_id):
            part_id = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                
                if result.get('skipped'):
                    stats['skipped'] += 1
                elif result['success']:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {part_id}: {e}")
                stats['failed'] += 1
    
    # Print clean summary to terminal
    print("\n" + "="*80)
    print("Download Completed!")
    print(f"✓ Success: {stats['success']}")
    print(f"⊘ Skipped: {stats['skipped']}")
    print(f"✗ Failed:  {stats['failed']}")
    print(f"Total:     {len(part_ids)}")
    print("="*80 + "\n")    
    return stats


def check_resume_status(csv_path: str, target_path: str) -> None:
    """
    Checks and displays the resume status - what's completed, pending, and failed.
    
    Args:
        csv_path: Path to the CSV file with part IDs
        target_path: Target directory where files are being downloaded
    """
    part_ids = load_part_ids_from_csv(csv_path)
    
    if not part_ids:
        print("✗ Error: No valid part IDs found in CSV file")
        return
    
    completed = []
    pending = []
    failed = []
    
    # Check status of each part ID
    for part_id in part_ids:
        extract_dir = os.path.join(target_path, part_id)
        completion_marker = os.path.join(extract_dir, ".download_complete")
        
        if os.path.exists(completion_marker):
            completed.append(part_id)
        else:
            pending.append(part_id)
    
    # Check failed downloads log
    log_file = os.path.join(target_path, "failed_downloads.log")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            failed_lines = f.readlines()
            failed = [line.split(' - ')[0].strip() for line in failed_lines if line.strip()]
    
    # Print status
    print("\n" + "="*80)
    print("Resume Status Report")
    print("="*80)
    print(f"Total classes in CSV: {len(part_ids)}")
    print(f"✓ Completed:          {len(completed)}")
    print(f"⧗ Pending:            {len(pending)}")
    print(f"✗ Failed:             {len(failed)}")
    print("="*80)
    
    if pending and len(pending) <= 20:
        print("\nPending downloads:")
        for part_id in pending[:20]:
            print(f"  - {part_id}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
    
    if failed and len(failed) <= 10:
        print("\nFailed downloads (check failed_downloads.log for details):")
        for part_id in failed[:10]:
            print(f"  - {part_id}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    print("\n" + "="*80 + "\n")
    
    completion_pct = (len(completed) / len(part_ids)) * 100 if part_ids else 0
    print(f"Progress: {completion_pct:.1f}% complete\n")


def main():
    """
    Main entry point with argument parsing for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description='ImageNet Dataset Downloader - multi-threaded downloader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--fname',
        type=str,
        default='data.csv',
        help='CSV file containing ImageNet class IDs (must have "wnid" column)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel download threads'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default="/home/ramachandra/Pictures/DataSet/Imagenet21K",
        help='Target directory for downloads (default: current directory)'
    )
    
    parser.add_argument(
        '--delete-tar',
        action='store_true',
        default=True,
        help='Delete tar files after extraction'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default='https://image-net.org/data/winter21_whole',
        help='Base URL for ImageNet downloads'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check resume status (completed, pending, failed)'
    )
    
    args = parser.parse_args()
    
    # Status check mode
    if args.status:
        target_path = args.target if args.target else os.getcwd()
        check_resume_status(args.fname, target_path)
        return
    
    # Validate CSV file
    if not os.path.exists(args.fname):
        print(f"✗ Error: CSV file not found: {args.fname}")
        logger.error(f"CSV file not found: {args.fname}")
        sys.exit(1)
    
    # Load part IDs from CSV
    part_ids = load_part_ids_from_csv(args.fname)
    
    if not part_ids:
        print(f"✗ Error: No valid part IDs found in CSV file")
        logger.error("No valid part IDs found in CSV file")
        sys.exit(1)
    
    # Set target path
    target_path = args.target if args.target else os.getcwd()
    
    # Run batch download
    download_batch(
        part_ids=part_ids,
        target_path=target_path,
        num_workers=args.workers,
        delete_tar=args.delete_tar,
        main_link=args.url
    )


if __name__ == "__main__":
    main()
