import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rag_pipeline import RAGPipeline
import os
import threading
from datetime import datetime
import mimetypes
import magic  # For better file type detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("document_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.processing_files = set()
        
        # Define supported extensions
        self.supported_extensions = {
            # Document Formats
            '.pdf', '.docx', '.txt', '.rtf',
            # Data Formats
            '.xlsx', '.csv', '.parquet', '.sql',
            # Structured Data Formats
            '.json', '.xml', '.yaml', '.yml',
            # Web Formats
            '.html', '.htm', '.md',
            # Presentation
            '.pptx'
        }
        
        # Initialize mime type detection
        mimetypes.init()
        
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate file before processing
        Returns True if file is valid for processing
        """
        try:
            # Check if file still exists
            if not file_path.exists():
                logger.warning(f"File no longer exists: {file_path}")
                return False
                
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Unsupported file format: {file_path}")
                return False
                
            # Check file size (100MB limit)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_path.stat().st_size > max_size:
                logger.warning(f"File too large (>100MB): {file_path}")
                return False
                
            # Check if file is readable
            if not os.access(str(file_path), os.R_OK):
                logger.warning(f"File not readable: {file_path}")
                return False
                
            # Check file type using magic numbers
            mime = magic.from_file(str(file_path), mime=True)
            expected_mime = mimetypes.guess_type(str(file_path))[0]
            
            if expected_mime and mime != expected_mime:
                logger.warning(f"File type mismatch for {file_path}. Expected: {expected_mime}, Got: {mime}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            return False
    
    def is_file_ready(self, file_path: Path, timeout: int = 30) -> bool:
        """
        Check if file is ready for processing by monitoring size changes
        Returns True when file size remains stable for specified timeout
        """
        initial_size = -1
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                current_size = file_path.stat().st_size
                if current_size == initial_size:
                    return True
                initial_size = current_size
                time.sleep(1)
            except FileNotFoundError:
                logger.warning(f"File disappeared while checking: {file_path}")
                return False
            except Exception as e:
                logger.error(f"Error checking file size for {file_path}: {str(e)}")
                return False
        
        logger.warning(f"File size not stabilized after {timeout}s: {file_path}")
        return False
    
    def process_file(self, file_path: Path):
        """Process a file with proper validation and error handling"""
        if str(file_path) not in self.processing_files:
            self.processing_files.add(str(file_path))
            try:
                logger.info(f"New file detected: {file_path}")
                
                # Validate file
                if not self.validate_file(file_path):
                    return
                    
                # Wait for file to be ready
                if not self.is_file_ready(file_path):
                    return
                
                logger.info(f"Processing file: {file_path}")
                success = self.rag_pipeline.process_document(str(file_path))
                
                if success:
                    logger.info(f"Successfully processed: {file_path}")
                    
                    # Move to archive if successful
                    archive_path = self.rag_pipeline.archive_dir / file_path.name
                    if archive_path.exists():
                        # Append timestamp if file exists
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        archive_path = self.rag_pipeline.archive_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    
                    file_path.rename(archive_path)
                    logger.info(f"Moved to archive: {archive_path}")
                else:
                    logger.error(f"Failed to process: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
            finally:
                self.processing_files.remove(str(file_path))
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        # Start processing in a separate thread
        threading.Thread(target=self.process_file, args=(file_path,)).start()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if str(file_path) not in self.processing_files:
            # Start processing in a separate thread
            threading.Thread(target=self.process_file, args=(file_path,)).start()

def start_monitoring(documents_dir: str, 
                     archive_dir: str, 
                     batch_size: int = 32,
                     max_workers: int = 16,
                     chunk_size: int = 512):
    """
    Start monitoring the documents directory for new files
    
    Args:
        documents_dir: Directory to monitor for new documents
        archive_dir: Directory where processed documents will be archived
        batch_size: Batch size for parallel processing
        max_workers: Number of parallel workers
        chunk_size: Size of text chunks for embedding
    """
    # Ensure directories exist
    Path(documents_dir).mkdir(parents=True, exist_ok=True)
    Path(archive_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline(
        documents_dir=documents_dir,
        archive_dir=archive_dir,
        batch_size=batch_size,
        max_workers=max_workers,
        chunk_size=chunk_size
    )
    
    # Set up file monitoring
    event_handler = DocumentHandler(rag_pipeline)
    observer = Observer()
    observer.schedule(event_handler, documents_dir, recursive=False)
    observer.start()
    
    logger.info(f"Started monitoring {documents_dir}")
    logger.info("Configuration:")
    logger.info(f"- Documents directory: {documents_dir}")
    logger.info(f"- Archive directory: {archive_dir}")
    logger.info(f"- Batch size: {batch_size}")
    logger.info(f"- Max workers: {max_workers}")
    logger.info(f"- Chunk size: {chunk_size}")
    logger.info(f"- Supported formats: {', '.join(sorted(event_handler.supported_extensions))}")
    
    def print_stats():
        """Print periodic statistics"""
        while True:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Count files in directories
                doc_count = len([f for f in Path(documents_dir).glob('*') 
                               if f.is_file() and f.suffix.lower() in event_handler.supported_extensions])
                archive_count = len([f for f in Path(archive_dir).glob('*') if f.is_file()])
                processing_count = len(event_handler.processing_files)
                
                logger.info("Current Statistics:")
                logger.info(f"- Files waiting to be processed: {doc_count}")
                logger.info(f"- Files currently processing: {processing_count}")
                logger.info(f"- Files in archive: {archive_count}")
                
            except Exception as e:
                logger.error(f"Error in stats reporting: {str(e)}")
    
    # Start stats reporting thread
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Monitoring stopped")
    
    observer.join()

if __name__ == "__main__":
    DOCUMENTS_DIR = r"C:\Users\Vincent_Pereira\Qdrant\Documents"
    ARCHIVE_DIR = r"C:\Users\Vincent_Pereira\Qdrant\Archived Documents"
    
    start_monitoring(
        documents_dir=DOCUMENTS_DIR,
        archive_dir=ARCHIVE_DIR,
        batch_size=32,
        max_workers=16,
        chunk_size=512
    )
