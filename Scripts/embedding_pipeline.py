import os
import sys
from pathlib import Path
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import shutil
import json
from typing import List, Dict, Any, Optional
import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import pandas as pd
from tqdm import tqdm
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRTFLoader
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = ThreadPoolExecutor(max_workers=16)  # Adjust based on CPU cores
        self.batch_size = config.get('batch_size', 32)
        self.chunk_size = config.get('chunk_size', 1000)
        self.chunk_overlap = config.get('chunk_overlap', 200)
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=config['qdrant_url'],
            api_key=config.get('qdrant_api_key')
        )
        
        # Ensure collection exists
        self._init_collection()
        
        # Initialize document loader mapping
        self.loader_mapping = {
            '.pdf': PyMuPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.xml': UnstructuredXMLLoader,
            '.html': UnstructuredHTMLLoader,
            '.htm': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pptx': UnstructuredPowerPointLoader,
            '.rtf': UnstructuredRTFLoader
        }
        
    def _init_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections()
        if not any(c.name == self.config['collection_name'] for c in collections.collections):
            self.qdrant.create_collection(
                collection_name=self.config['collection_name'],
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama API with GPU acceleration."""
        async with httpx.AsyncClient() as client:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks = []
                for text in batch:
                    response = await client.post(
                        "http://localhost:11434/api/embeddings",
                        json={
                            "model": "snowflake-arctic-embed2",
                            "prompt": text
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        embeddings.append(data['embedding'])
                    else:
                        logger.error(f"Error getting embeddings: {response.text}")
                        raise Exception(f"Failed to get embeddings: {response.status_code}")
            return embeddings
    
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single document and return chunks with metadata."""
        try:
            # Get appropriate loader
            loader_class = self.loader_mapping.get(file_path.suffix.lower())
            if not loader_class:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            # Load document
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            chunks = []
            for doc in documents:
                doc_chunks = text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "file_type": file_path.suffix,
                            "created_at": datetime.utcnow().isoformat(),
                            **doc.metadata
                        }
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    async def process_directory(self, input_dir: Path, archive_dir: Path):
        """Process all documents in a directory."""
        try:
            # Get all files
            files = []
            for ext in self.loader_mapping.keys():
                files.extend(input_dir.glob(f"**/*{ext}"))
            
            if not files:
                logger.info("No supported files found")
                return
            
            # Process files in parallel
            chunk_futures = []
            for file_path in files:
                chunk_futures.append(
                    self.executor.submit(self.process_document, file_path)
                )
            
            # Collect all chunks
            all_chunks = []
            for future in tqdm(chunk_futures, desc="Processing documents"):
                chunks = future.result()
                all_chunks.extend(chunks)
            
            # Get embeddings in batches
            all_texts = [chunk["text"] for chunk in all_chunks]
            embeddings = await self.get_embeddings(all_texts)
            
            # Upload to Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                points.append({
                    "id": i,
                    "vector": embedding,
                    "payload": {
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                })
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant.upsert(
                    collection_name=self.config['collection_name'],
                    points=batch
                )
            
            # Move processed files to archive
            for file_path in files:
                relative_path = file_path.relative_to(input_dir)
                archive_path = archive_dir / relative_path
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(archive_path))
                logger.info(f"Moved {file_path} to archive")
            
            logger.info(f"Successfully processed {len(files)} files, created {len(points)} vectors")
            
        except Exception as e:
            logger.error(f"Error in process_directory: {str(e)}")
            raise

async def main():
    # Load configuration
    config = {
        "qdrant_url": "http://localhost:6333",
        "collection_name": "documents",
        "batch_size": 32,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(config)
    
    # Set up directories
    input_dir = Path("C:/Users/Vincent_Pereira/Qdrant/Documents")
    archive_dir = Path("C:/Users/Vincent_Pereira/Qdrant/Archived Documents")
    
    # Process documents
    await pipeline.process_directory(input_dir, archive_dir)

if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU not available, using CPU")
    
    asyncio.run(main())
