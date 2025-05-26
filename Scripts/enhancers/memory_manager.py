from typing import Dict, Any, List, Optional
import os
import psutil
import logging
import numpy as np
from dataclasses import dataclass
import tempfile
import json
import mmap
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    max_memory_percentage: float = 0.8  # Max % of system memory to use
    chunk_memory_limit_mb: int = 1024   # Max memory per chunk in MB
    use_memory_mapping: bool = True     # Use memory mapping for large files
    temp_dir: Optional[str] = None      # Directory for temporary files
    cleanup_threshold: float = 0.9      # Cleanup when memory usage exceeds this %

class MemoryManager:
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.temp_files: List[Path] = []
        self._setup_temp_dir()

    def _setup_temp_dir(self):
        """Set up temporary directory for memory management."""
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "rag_pipeline"
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        system = psutil.virtual_memory()

        return {
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'system_memory_used_percent': system.percent,
            'available_memory_mb': system.available / (1024 * 1024),
            'total_memory_mb': system.total / (1024 * 1024)
        }

    def should_use_disk_offload(self, data_size_bytes: int) -> bool:
        """Determine if data should be offloaded to disk."""
        memory_usage = self.get_memory_usage()
        available_memory = memory_usage['available_memory_mb'] * 1024 * 1024
        
        # Add safety margin
        available_memory *= 0.9
        
        return data_size_bytes > available_memory

    def cleanup_memory(self):
        """Clean up temporary files and free memory."""
        # Remove temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        self.temp_files.clear()

    async def process_large_document(
        self,
        document: Dict[str, Any],
        chunk_processor: Any,
        embedding_function: Any
    ) -> List[Dict[str, Any]]:
        """Process a large document in chunks with memory management."""
        text = document.get('text', '')
        doc_size = len(text.encode('utf-8'))
        
        if self.should_use_disk_offload(doc_size):
            return await self._process_with_disk_offload(
                text, chunk_processor, embedding_function
            )
        else:
            return await self._process_in_memory(
                text, chunk_processor, embedding_function
            )

    async def _process_with_disk_offload(
        self,
        text: str,
        chunk_processor: Any,
        embedding_function: Any
    ) -> List[Dict[str, Any]]:
        """Process document using disk offload for memory management."""
        # Create temporary file for memory mapping
        temp_file = self.temp_dir / f"doc_{id(text)}.mmap"
        self.temp_files.append(temp_file)
        
        chunks = []
        
        try:
            # Write text to memory-mapped file
            with open(temp_file, 'wb') as f:
                f.write(text.encode('utf-8'))
            
            with open(temp_file, 'r+b') as f:
                # Memory map the file
                mm = mmap.mmap(f.fileno(), 0)
                
                # Process in smaller chunks
                chunk_size = self.config.chunk_memory_limit_mb * 1024 * 1024
                for i in range(0, len(mm), chunk_size):
                    chunk = mm[i:i + chunk_size].decode('utf-8')
                    
                    # Process chunk
                    processed_chunks = await chunk_processor.process_chunk(chunk)
                    
                    # Generate embeddings
                    for proc_chunk in processed_chunks:
                        embedding = await embedding_function(proc_chunk['text'])
                        proc_chunk['embedding'] = embedding
                        chunks.append(proc_chunk)
                
                mm.close()
        
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
            self.temp_files.remove(temp_file)
        
        return chunks

    async def _process_in_memory(
        self,
        text: str,
        chunk_processor: Any,
        embedding_function: Any
    ) -> List[Dict[str, Any]]:
        """Process document entirely in memory."""
        chunks = await chunk_processor.process_chunk(text)
        
        # Generate embeddings
        for chunk in chunks:
            chunk['embedding'] = await embedding_function(chunk['text'])
        
        return chunks

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup_memory()
