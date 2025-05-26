"""
Embeddings module supporting multiple embedding providers.
"""
import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import logging
import asyncio
import httpx
from openai import AsyncOpenAI

import torch
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Generate embeddings for input texts.
        
        Args:
            texts: Single text string or list of text strings to embed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embeddings as float arrays
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings produced by this provider."""
        pass

class OllamaEmbeddings(EmbeddingProvider):
    """Embeddings using local Ollama models."""
    
    def __init__(self, model: str = "snowflake-arctic-embed2:latest", batch_size: int = 32,
                 api_base: str = "http://localhost:11434"):
        self.model = model
        self.batch_size = batch_size 
        self.api_base = api_base.rstrip('/')

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await asyncio.gather(
                *[self._embed_single(text) for text in batch]
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_single(self, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Failed to get embeddings: {response.text}")
                
            return response.json()["embedding"]

    def get_embedding_dim(self) -> int:
        # Fixed dimension for snowflake-arctic-embed2
        return 1536

class OpenAIEmbeddings(EmbeddingProvider):
    """Embeddings using OpenAI's API."""
    
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 32,
                api_key: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize OpenAI embeddings provider.
        
        Args:
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in parallel
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI org ID (defaults to OPENAI_ORGANIZATION env var)
        """
        self.model = model
        self.batch_size = batch_size
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization or os.getenv("OPENAI_ORGANIZATION")
        )
        self._dim_cache = None

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch = texts[i:i + self.batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _get_embedding_dim(self) -> int:
        """Get embedding dimension by making a test API call."""
        if self._dim_cache is None:
            response = await self.client.embeddings.create(
                model=self.model,
                input="test",
                encoding_format="float"
            )
            self._dim_cache = len(response.data[0].embedding)
        return self._dim_cache

    def get_embedding_dim(self) -> int:
        """Get embedding dimension (requires async context)."""
        if self._dim_cache is None:
            # Run async call in sync context
            self._dim_cache = asyncio.run(self._get_embedding_dim())
        return self._dim_cache

class CacheEmbeddings(EmbeddingProvider):
    """Caching wrapper for embedding providers."""
    
    def __init__(self, provider: EmbeddingProvider, cache_dir: str = "embeddings_cache"):
        self.provider = provider
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cache if it exists
        self.cache = {}
        self._load_cache()

    def _get_cache_path(self, text_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{text_hash}.npy")

    def _load_cache(self):
        """Load all cached embeddings."""
        for f in os.listdir(self.cache_dir):
            if f.endswith(".npy"):
                text_hash = f[:-4]
                self.cache[text_hash] = np.load(os.path.join(self.cache_dir, f))

    def _save_to_cache(self, text_hash: str, embedding: List[float]):
        """Save embedding to cache."""
        np.save(os.path.join(self.cache_dir, f"{text_hash}.npy"), embedding)
        self.cache[text_hash] = embedding

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        # Generate hashes for input texts
        text_hashes = [str(hash(text)) for text in texts]
        
        # Find texts that need embedding
        missing_indices = [i for i, h in enumerate(text_hashes) if h not in self.cache]
        texts_to_embed = [texts[i] for i in missing_indices]
        
        if texts_to_embed:
            # Generate embeddings for missing texts
            new_embeddings = await self.provider.generate_embeddings(texts_to_embed, **kwargs)
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                text_hash = str(hash(text))
                self._save_to_cache(text_hash, embedding)

        # Return embeddings in original order
        return [self.cache[h] for h in text_hashes]

    def get_embedding_dim(self) -> int:
        return self.provider.get_embedding_dim()

def create_embedding_provider(provider: str = "ollama", **kwargs) -> EmbeddingProvider:
    """Factory function to create embedding providers.
    
    Args:
        provider: One of 'ollama' or 'openai'
        **kwargs: Provider-specific configuration options
        
    Returns:
        Configured embedding provider
    """
    providers = {
        "ollama": OllamaEmbeddings,
        "openai": OpenAIEmbeddings
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
        
    base_provider = providers[provider](**kwargs)
    
    # Wrap with caching if enabled
    if kwargs.get("enable_cache", True):
        cache_dir = kwargs.get("cache_dir", "embeddings_cache")
        return CacheEmbeddings(base_provider, cache_dir=cache_dir)
        
    return base_provider