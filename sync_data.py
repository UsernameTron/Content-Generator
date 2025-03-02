"""
Data synchronization script to ensure consistent data files.
"""

import os
import json
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def sync_data_files():
    """
    Synchronize data files between data/ and src/data/ directories.
    """
    # Get paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    src_data_dir = project_dir / "src" / "data"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    src_data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Syncing data between {data_dir} and {src_data_dir}")
    
    # List of files to sync (add more as needed)
    files_to_sync = [
        "writing_style.json",
        "custom_templates.json"
    ]
    
    # For each file, ensure it exists in both locations and is up-to-date
    for filename in files_to_sync:
        data_file = data_dir / filename
        src_data_file = src_data_dir / filename
        
        # Check if the file exists in either location
        data_exists = data_file.exists()
        src_data_exists = src_data_file.exists()
        
        # If neither exists, create a default file
        if not data_exists and not src_data_exists:
            logger.warning(f"{filename} not found in either location. Skipping.")
            continue
        
        # If file exists in data/ but not in src/data/, copy it to src/data/
        if data_exists and not src_data_exists:
            logger.info(f"Copying {filename} from data/ to src/data/")
            shutil.copy2(data_file, src_data_file)
            continue
        
        # If file exists in src/data/ but not in data/, copy it to data/
        if not data_exists and src_data_exists:
            logger.info(f"Copying {filename} from src/data/ to data/")
            shutil.copy2(src_data_file, data_file)
            continue
        
        # If file exists in both locations, compare modification times
        data_mtime = data_file.stat().st_mtime
        src_data_mtime = src_data_file.stat().st_mtime
        
        # If data/ has newer version, copy to src/data/
        if data_mtime > src_data_mtime:
            logger.info(f"Updating {filename} in src/data/ from data/")
            shutil.copy2(data_file, src_data_file)
        # If src/data/ has newer version, copy to data/
        elif src_data_mtime > data_mtime:
            logger.info(f"Updating {filename} in data/ from src/data/")
            shutil.copy2(src_data_file, data_file)
        # If both files have the same modification time, check content
        else:
            # Open both files and compare their content
            with open(data_file, 'r') as f1, open(src_data_file, 'r') as f2:
                content1 = f1.read()
                content2 = f2.read()
                
                # If content is different, update using data/ as source
                if content1 != content2:
                    logger.info(f"Content differs for {filename}, syncing from data/ to src/data/")
                    shutil.copy2(data_file, src_data_file)
                else:
                    logger.info(f"{filename} is already synchronized")
    
    logger.info("Data synchronization completed successfully")
    return True

if __name__ == "__main__":
    print("Synchronizing data files...")
    success = sync_data_files()
    
    if success:
        print("\nData synchronization completed successfully!")
        print("All data files are now consistent between data/ and src/data/ directories.")
    else:
        print("\nData synchronization encountered errors.")
        print("Check the log for more information.")
