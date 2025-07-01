import torch
from typing import List, Dict, Any, Union
import logging
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import concurrent.futures
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # CPU thread configuration
        # This assumes that the 'config' object passed to this class or accessible globally
        # has a model.cpu_thread_count attribute.
        # For now, let's make it an optional constructor argument,
        # which RAGPipeline would populate from app_config.model.cpu_thread_count.
        # This will be passed during __init__ by RAGPipeline after this change.
        # For direct instantiation in tests or standalone, it might be None.
        self.cpu_thread_count = kwargs.get("cpu_thread_count") # Will be passed by RAGPipeline

        if self.device == "cpu" and self.cpu_thread_count is not None and self.cpu_thread_count > 0:
            torch.set_num_threads(self.cpu_thread_count)
            logger.info(f"Using device: {self.device} with {torch.get_num_threads()} threads for PyTorch.")
        else:
            logger.info(f"Using device: {self.device}. Default PyTorch threads for CPU, or GPU active.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        
        # Enable automatic mixed precision for faster inference
        self.use_amp = self.device == "cuda"
        
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts using batching and GPU acceleration."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize all texts in the batch
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings with automatic mixed precision
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
                model_output = self.model(**encoded_input)
                # Use mean pooling to get text embeddings
                attention_mask = encoded_input["attention_mask"]
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process document chunks and add embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts, batch_size)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        
        return chunks

    def save_embeddings(self, embeddings: np.ndarray, output_path: Union[str, Path]):
        """Save embeddings to disk."""
        output_path = Path(output_path)
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")

    def load_embeddings(self, input_path: Union[str, Path]) -> np.ndarray:
        """Load embeddings from disk."""
        input_path = Path(input_path)
        embeddings = np.load(input_path)
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings
