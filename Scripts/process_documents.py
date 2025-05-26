import os
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import traceback
import hashlib
from datetime import datetime
import pytz
import logging
from dataclasses import dataclass
import gc

# Hardware and Processing
import torch
import torch.cuda.amp as amp
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import psutil

# Document Processing Libraries
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd
import pyarrow.parquet as pq
import csv
import xml.etree.ElementTree as ET
import yaml
import re
import tiktoken
import pptx
from bs4 import BeautifulSoup
import xml.dom.minidom

# System Configuration
class SystemConfig:
    # Time and User Settings
    CURRENT_TIME = "2025-05-21 14:48:29"
    CURRENT_USER = "vincentspereira"
    TIMEZONE = pytz.UTC
    
    # API Endpoints
    OLLAMA_BASE_URL = "http://localhost:11434"
    QDRANT_BASE_URL = "http://localhost:6333"
    COLLECTION_NAME = "documents"
    EMBED_MODEL = "snowflake-arctic-embed2:latest"
    
    # Processing Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    PROCESSED_FILES_LOG = "processed_files.json"
    
    # Hardware Configuration
    @dataclass
    class Hardware:
        # GPU
        CUDA_AVAILABLE = torch.cuda.is_available()
        GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
        if CUDA_AVAILABLE:
            torch.cuda.init()
            GPU_PROPERTIES = torch.cuda.get_device_properties(0)
            VRAM_TOTAL = int(GPU_PROPERTIES.total_memory / 1024 / 1024)  # Total VRAM in MB
        else:
            VRAM_TOTAL = 0
        VRAM_RESERVE = 1024  # Reserve 1GB
        BATCH_SIZE = 32
        
        # CPU - Ryzen 7 5800H
        CPU_CORES = multiprocessing.cpu_count()
        CPU_THREADS = 16
        MAX_WORKERS = CPU_THREADS - 2
        
        # RAM - 64GB DDR4
        RAM_TOTAL = psutil.virtual_memory().total / (1024**3)
        RAM_SPEED = 3200
        RAM_CL = 20
        
        @classmethod
        def get_optimal_batch_size(cls):
            if not cls.CUDA_AVAILABLE:
                return cls.BATCH_SIZE
            
            try:
                available_vram = cls.VRAM_TOTAL - cls.VRAM_RESERVE
                embedding_size = 1536
                memory_per_embedding = embedding_size * 4 / 1024  # KB
                optimal_batch = int(available_vram * 1024 / memory_per_embedding)
                return min(optimal_batch, cls.BATCH_SIZE)
            except:
                return cls.BATCH_SIZE

# Initialize Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/document_processing_{SystemConfig.CURRENT_TIME.replace(":", "-")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Performance Monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        
    def start(self):
        self.start_time = time.time()
        if SystemConfig.Hardware.CUDA_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            
    def stop(self) -> Dict[str, Any]:
        end_time = time.time()
        duration = end_time - self.start_time
        
        self.metrics = {
            "timestamp": SystemConfig.CURRENT_TIME,
            "duration": duration,
            "cpu_percent": psutil.cpu_percent(),
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if SystemConfig.Hardware.CUDA_AVAILABLE:
            self.metrics.update({
                "gpu_memory_used_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2)
            })
            
        return self.metrics


# Hardware Optimizer Class
class HardwareOptimizer:
    def __init__(self):
        self.current_time = "2025-05-21 14:49:27"
        self.current_user = "vincentspereira"
        self.config = SystemConfig.Hardware
        
        # GPU Optimization
        self.scaler = amp.GradScaler()
        self.autocast = amp.autocast(device_type='cuda')
        self.streams = []
        
        # Initialize hardware
        self._init_gpu()
        self._init_cpu()
        self._init_memory()
        logger.info(f"Hardware optimizer initialized by {self.current_user} at {self.current_time}")
        
    def _init_gpu(self):
        """Initialize GPU optimizations for RTX 3060"""
        if self.config.CUDA_AVAILABLE:
            # Set memory fraction for 6GB VRAM
            torch.cuda.set_per_process_memory_fraction(0.7)
            
            # Enable TF32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            
            # Create CUDA streams for parallel processing
            self.streams = [torch.cuda.Stream() for _ in range(4)]
            
            logger.info("GPU optimizations initialized")
            
    def _init_cpu(self):
        """Initialize CPU optimizations for Ryzen 7 5800H"""
        try:
            process = psutil.Process()
            # Use all 16 threads
            process.cpu_affinity(list(range(self.config.CPU_THREADS)))
            
            # Set process priority
            if os.name == 'nt':  # Windows
                process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:  # Linux/Unix
                process.nice(-10)
                
            logger.info("CPU optimizations initialized")
            
        except Exception as e:
            logger.error(f"CPU optimization error: {e}")
            
    def _init_memory(self):
        """Initialize memory optimizations for 64GB RAM"""
        try:
            # Enable garbage collection
            gc.enable()
            
            # Set memory allocator settings
            if os.name == 'nt':  # Windows
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                
            logger.info("Memory optimizations initialized")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
            
    def process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts using optimized hardware"""
        ollama_client = OllamaClient()
        if self.config.CUDA_AVAILABLE:
            return ollama_client._gpu_batch_process(texts)
        return ollama_client._cpu_batch_process(texts)
        
    def _gpu_batch_process(self, texts: List[str]) -> List[List[float]]:
        """Process batch using GPU optimizations"""
            
    def cleanup(self):
        """Cleanup resources"""
        if self.config.CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        gc.collect()

class OllamaClient:
    def _gpu_batch_process(self, texts: List[str]) -> List[List[float]]:
        """Process batch using GPU optimizations"""
        try:
            with torch.cuda.stream(self.streams[0]):
                with self.autocast:
                    response = requests.post(
                        f"{SystemConfig.OLLAMA_BASE_URL}/api/embeddings",
                        json={
                            "model": SystemConfig.EMBED_MODEL,
                            "prompt": texts,
                            "batch_size": self.config.get_optimal_batch_size()
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        embeddings = response.json()["embeddings"]
                        # Clear cache if memory usage is high
                        if torch.cuda.memory_allocated() > (4 * 1024 * 1024 * 1024):
                            torch.cuda.empty_cache()
                        return embeddings
                    
            return []
            
        except Exception as e:
            logger.error(f"GPU batch processing error: {e}")
            return self._cpu_batch_process(texts)
            
    def _cpu_batch_process(self, texts: List[str]) -> List[List[float]]:
        """Process batch using CPU optimizations"""
        try:
            with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        lambda t: requests.post(
                            f"{SystemConfig.OLLAMA_BASE_URL}/api/embeddings",
                            json={"model": SystemConfig.EMBED_MODEL, "prompt": t}
                        ).json()["embedding"],
                        text
                    )
                    for text in texts
                ]
                return [f.result() for f in futures]
                
        except Exception as e:
            logger.error(f"CPU batch processing error: {e}")
            return []

def load_processed_files() -> Dict:
    """Load processed files log"""
    try:
        if os.path.exists(SystemConfig.PROCESSED_FILES_LOG):
            with open(SystemConfig.PROCESSED_FILES_LOG, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading processed files log: {e}")
        return {}

def save_processed_files(processed_files: Dict):
    """Save processed files log"""
    try:
        with open(SystemConfig.PROCESSED_FILES_LOG, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving processed files log: {e}")

# Document Processing Class
class DocumentProcessor:
    def __init__(self):
        self.current_time = "2025-05-21 14:49:27"
        self.current_user = "vincentspereira"
        self.hardware = HardwareOptimizer()
        self.performance = PerformanceMonitor()
        
    def process_document(self, file_path: str, total_files: int, current_file: int) -> List[Dict]:
        """Process a single document"""
        logger.info(f"Processing document {current_file}/{total_files}: {file_path}")
        self.performance.start()
        
        try:
            # Extract text
            text = get_document_text(file_path)
            if not text:
                logger.warning(f"No text extracted from {file_path}")
                return []
                
            # Get document metadata
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            file_size = os.path.getsize(file_path)
            
            # Split into chunks
            text_chunks = chunk_text(text)
            logger.info(f"Split document into {len(text_chunks)} chunks")
            
            # Process chunks in batches
            chunks = []
            batch_size = SystemConfig.Hardware.get_optimal_batch_size()
            
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i:i + batch_size]
                
                # Get embeddings for batch
                embeddings = self.hardware.process_batch(batch)
                
                # Create chunk objects
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunks.append({
                        "text": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "source": file_name,
                            "chunk_index": i + j,
                            "file_extension": file_extension,
                            "file_size_bytes": file_size,
                            "processing_time": self.current_time,
                            "embedding_status": "success" if embedding else "failed",
                            "processor": self.current_user,
                            "hardware": {
                                "gpu": SystemConfig.Hardware.GPU_NAME,
                                "batch_size": batch_size
                            },
                            "performance": self.performance.metrics
                        }
                    })
                    
            # Record performance metrics
            metrics = self.performance.stop()
            logger.info(f"Document processing metrics: {metrics}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            traceback.print_exc()
            return []

# Utility Functions for Text Processing and File Handling
def chunk_text(text: str) -> List[str]:
    """Split text into chunks with overlap"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), SystemConfig.CHUNK_SIZE):
        chunk = words[i:i + SystemConfig.CHUNK_SIZE + SystemConfig.CHUNK_OVERLAP]
        chunks.append(" ".join(chunk))
    
    return chunks

def get_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file"""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()

def get_document_text(file_path: str) -> str:
    """Extract text from various document formats"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.pdf':
            return _extract_pdf_text(file_path)
        elif file_extension == '.docx':
            return _extract_docx_text(file_path)
        elif file_extension in ['.txt', '.sql', '.md']:
            return _extract_text_file(file_path)
        elif file_extension in ['.csv', '.xlsx']:
            return _extract_tabular_text(file_path)
        elif file_extension == '.json':
            return _extract_json_text(file_path)
        elif file_extension in ['.xml', '.html', '.htm']:
            return _extract_markup_text(file_path)
        elif file_extension in ['.yaml', '.yml']:
            return _extract_yaml_text(file_path)
        elif file_extension == '.pptx':
            return _extract_pptx_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_extension}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def _extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF"""
    text = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)

def _extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX"""
    doc = DocxDocument(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def _extract_text_file(file_path: str) -> str:
    """Extract text from plain text files"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def _extract_tabular_text(file_path: str) -> str:
    """Extract text from CSV/Excel files"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df.to_string()

def _extract_json_text(file_path: str) -> str:
    """Extract text from JSON"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return json.dumps(data, indent=2)

def _extract_markup_text(file_path: str) -> str:
    """Extract text from XML/HTML"""
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return soup.get_text()

def _extract_yaml_text(file_path: str) -> str:
    """Extract text from YAML"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return yaml.dump(data, default_flow_style=False)

def _extract_pptx_text(file_path: str) -> str:
    """Extract text from PowerPoint"""
    prs = pptx.Presentation(file_path)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def needs_processing(file_path: str, processed_files: Dict) -> bool:
    """Check if file needs processing"""
    file_hash = get_file_hash(file_path)
    if file_path in processed_files:
        if processed_files[file_path]["hash"] == file_hash:
            logger.info(f"File '{file_path}' already processed and up to date")
            return False
        else:
            logger.info(f"File '{file_path}' already processed but outdated")
            return True
    else:
        logger.info(f"File '{file_path}' is new and needs processing")
        return True

def process_documents_parallel(file_paths: List[str]) -> List[Dict]:
    """Process documents in parallel"""
    all_chunks = []
    total_files = len(file_paths)
    
    with tqdm(total=total_files, desc="Processing documents") as pbar:
        with ProcessPoolExecutor(max_workers=SystemConfig.Hardware.MAX_WORKERS) as executor:
            futures = [
                executor.submit(DocumentProcessor().process_document, file_path, total_files, i + 1)
                for i, file_path in enumerate(file_paths)
            ]
            
            for future in futures:
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
                pbar.update(1)
                
    return all_chunks

def update_processed_files(self, new_files: List[str], processed_files: Dict):
    """Update processed files log"""
    for file_path in new_files:
        valid_chunks = [c for c in chunks if len(c["embedding"]) > 0]
        
        if not valid_chunks:
            logger.warning("No valid chunks to add to Qdrant")
            return
            
        total_batches = (len(valid_chunks) + batch_size - 1) // batch_size
        
        with tqdm(total=total_batches, desc="Adding to Qdrant") as pbar:
            for i in range(0, len(valid_chunks), batch_size):
                batch = valid_chunks[i:i + batch_size]
                
                try:
                    points = [
                        {
                            "id": str(hash(f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_index']}")),
                            "vector": chunk["embedding"],
                            "payload": {
                                "text": chunk["text"],
                                "metadata": chunk["metadata"]
                            }
                        }
                        for chunk in batch
                    ]
                    
                    response = requests.put(
                        f"{SystemConfig.QDRANT_BASE_URL}/collections/{SystemConfig.COLLECTION_NAME}/points",
                        json={"points": points}
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Error adding batch to Qdrant: {response.text}")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error adding batch to Qdrant: {e}")
                    
        metrics = self.performance.stop()
        logger.info(f"Qdrant upload metrics: {metrics}")



# Removing main function from process_documents.py
if __name__ == "__main__":
    main()