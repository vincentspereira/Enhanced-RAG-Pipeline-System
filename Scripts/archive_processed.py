import os
import json
import shutil
from datetime import datetime

def archive_processed_documents():
    # Load processed files log
    with open("processed_files.json", 'r') as f:
        processed_files = json.load(f)
    
    # Create archive directory with timestamp
    archive_dir = f"Processed_Documents_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(archive_dir, exist_ok=True)
    
    # Move processed files to archive
    for file_path in processed_files:
        if os.path.exists(file_path):
            # Create subdirectories in archive if needed
            relative_path = os.path.relpath(file_path, "Documents")
            archive_path = os.path.join(archive_dir, relative_path)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            
            # Move file to archive
            shutil.move(file_path, archive_path)
            print(f"Archived: {file_path}")

if __name__ == "__main__":
    archive_processed_documents()