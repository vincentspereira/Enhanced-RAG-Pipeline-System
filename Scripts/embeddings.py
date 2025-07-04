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

class CohereEmbeddings(EmbeddingProvider):
    """Embeddings using Cohere's API."""
    def __init__(self, model: str = "embed-english-v3.0", input_type: str = "search_document",
                 api_key: Optional[str] = None, batch_size: int = 96): # Cohere batch limit often 96
        self.model = model
        self.input_type = input_type # "search_document", "search_query", "classification", "clustering"
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not provided or COHERE_API_KEY not set.")
        self.batch_size = batch_size
        self._dim_cache = None

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        async with httpx.AsyncClient(timeout=30.0) as client: # Added timeout
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Cohere Embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                try:
                    response = await client.post(
                        "https://api.cohere.ai/v1/embed",
                        json={
                            "texts": batch_texts,
                            "model": self.model,
                            "input_type": kwargs.get("input_type", self.input_type),
                            "truncate": kwargs.get("truncate", "END")
                        },
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    response.raise_for_status() # Raise an exception for HTTP error codes
                    data = response.json()
                    all_embeddings.extend(data['embeddings'])
                except httpx.HTTPStatusError as e:
                    logger.error(f"Cohere API request failed: {e.response.status_code} - {e.response.text}")
                    # Fill with dummy embeddings of correct (guessed) dimension or raise
                    # For now, let's assume a common dimension or fetch it. If first batch fails, this is an issue.
                    # Fallback for failed batches: add list of zeros based on known dim or last successful batch.
                    dim = self.get_embedding_dim() # Try to get it
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 1024) )

                except Exception as e:
                    logger.error(f"Error during Cohere embedding generation: {e}")
                    dim = self.get_embedding_dim()
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 1024) )
        return all_embeddings

    def get_embedding_dim(self) -> int:
        if self._dim_cache is None:
            # Common dimensions for Cohere models, can be fetched dynamically too.
            # embed-english-v3.0 -> 1024, embed-multilingual-v3.0 -> 1024
            # embed-english-light-v3.0 -> 384
            if "v3.0" in self.model and "light" not in self.model: self._dim_cache = 1024
            elif "light-v3.0" in self.model: self._dim_cache = 384
            elif "v2.0" in self.model: self._dim_cache = 4096 # embed-english-v2.0
            else: # Fallback, or make a test call
                logger.warning(f"Unknown Cohere model '{self.model}' for dimension, defaulting to 1024. Please verify.")
                self._dim_cache = 1024
        return self._dim_cache

class VoyageAIEmbeddings(EmbeddingProvider):
    """Embeddings using Voyage AI's API."""
    def __init__(self, model: str = "voyage-2", input_type: Optional[str] = None,
                 api_key: Optional[str] = None, batch_size: int = 8): # Voyage docs suggest small batches
        self.model = model
        self.input_type = input_type # e.g. "document" or "query"
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Voyage AI API key not provided or VOYAGE_API_KEY not set.")
        self.batch_size = batch_size
        self._dim_cache = None

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str): texts = [texts]

        all_embeddings = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in tqdm(range(0, len(texts), self.batch_size), desc="VoyageAI Embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                json_payload = {"input": batch_texts, "model": self.model}
                if self.input_type: json_payload["input_type"] = self.input_type
                if kwargs.get("truncate") is not None: json_payload["truncation"] = kwargs.get("truncate")

                try:
                    response = await client.post(
                        "https://api.voyageai.com/v1/embeddings",
                        json=json_payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    all_embeddings.extend([item['embedding'] for item in data['data']])
                except httpx.HTTPStatusError as e:
                    logger.error(f"VoyageAI API request failed: {e.response.status_code} - {e.response.text}")
                    dim = self.get_embedding_dim()
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 1024) )
                except Exception as e:
                    logger.error(f"Error during VoyageAI embedding generation: {e}")
                    dim = self.get_embedding_dim()
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 1024) )
        return all_embeddings

    def get_embedding_dim(self) -> int:
        if self._dim_cache is None:
            # voyage-2 -> 1024, voyage-lite -> 1024, voyage-code-2 -> 1536
            if self.model in ["voyage-2", "voyage-02", "voyage-large-2"]: self._dim_cache = 1024 # voyage-large-2 is 1024 from docs
            elif "code-2" in self.model: self._dim_cache = 1536
            elif "lite" in self.model: self._dim_cache = 1024 # voyage-lite is 1024
            else: # Fallback
                logger.warning(f"Unknown VoyageAI model '{self.model}' for dimension, defaulting to 1024. Please verify.")
                self._dim_cache = 1024
        return self._dim_cache

class JinaAIEmbeddings(EmbeddingProvider):
    """Embeddings using Jina AI's API."""
    # Jina has v2 models: jina-embeddings-v2-base-en (768), jina-embeddings-v2-small-en (512)
    # Also multilingual models.
    def __init__(self, model: str = "jina-embeddings-v2-base-en",
                 api_key: Optional[str] = None, batch_size: int = 32):
        self.model = model
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        # Jina might require API key for higher rate limits or specific models,
        # but their main embedding endpoint is often usable without one for basic use.
        # if not self.api_key:
        #     raise ValueError("Jina AI API key not provided or JINA_API_KEY not set.")
        self.batch_size = batch_size
        self._dim_cache = None

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str): texts = [texts]

        all_embeddings = []
        headers = {"Content-Type": "application/json"}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in tqdm(range(0, len(texts), self.batch_size), desc="JinaAI Embeddings"):
                batch_texts = texts[i:i + self.batch_size]
                try:
                    response = await client.post(
                        "https://api.jina.ai/v1/embeddings", # Standard Jina API endpoint
                        json={"input": batch_texts, "model": self.model},
                        headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    all_embeddings.extend([item['embedding'] for item in data['data']])
                except httpx.HTTPStatusError as e:
                    logger.error(f"JinaAI API request failed: {e.response.status_code} - {e.response.text}")
                    dim = self.get_embedding_dim()
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 768) )
                except Exception as e:
                    logger.error(f"Error during JinaAI embedding generation: {e}")
                    dim = self.get_embedding_dim()
                    for _ in batch_texts: all_embeddings.append([0.0] * (dim if dim else 768) )
        return all_embeddings

    def get_embedding_dim(self) -> int:
        if self._dim_cache is None:
            if "v2-base" in self.model: self._dim_cache = 768
            elif "v2-small" in self.model: self._dim_cache = 512
            # Add other Jina model dimensions here
            else:
                logger.warning(f"Unknown JinaAI model '{self.model}' for dimension, defaulting to 768. Please verify.")
                self._dim_cache = 768
        return self._dim_cache

class LocalHuggingFaceEmbeddings(EmbeddingProvider):
    """Embeddings using local HuggingFace SentenceTransformer models."""

    def __init__(self, model_path: str, batch_size: int = 32, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer # Import here to keep it optional
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = SentenceTransformer(model_path, device=self.device)
            self._dim_cache = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded local SentenceTransformer model from: {model_path} on device {self.device}. Dim: {self._dim_cache}")
        except Exception as e:
            logger.error(f"Failed to load local SentenceTransformer model from {model_path}: {e}")
            raise

    async def generate_embeddings(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        # SentenceTransformer.encode is not async, run in executor or ensure it's okay in current event loop.
        # For simplicity here, direct call. In a heavy async app, use asyncio.to_thread
        # For batching, SentenceTransformer handles it internally if a list is passed.
        # The `batch_size` in `encode` is for internal processing, not API batching like OpenAI/Cohere.
        try:
            # Normalize to numpy array then to list of lists
            embeddings_np = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
            return embeddings_np.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings with local model {self.model_path}: {e}")
            # Return empty embeddings of correct dimension for failed ones
            dim = self.get_embedding_dim()
            return [[0.0] * dim for _ in texts]

    def get_embedding_dim(self) -> int:
        if self._dim_cache is None:
            # This should have been set in __init__
            raise RuntimeError("Embedding dimension not initialized for LocalHuggingFaceEmbeddings.")
        return self._dim_cache


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
        "openai": OpenAIEmbeddings,
        "cohere": CohereEmbeddings,
        "voyage": VoyageAIEmbeddings,
        "jina": JinaAIEmbeddings,
        "local_hf": LocalHuggingFaceEmbeddings, # Added new provider type
    }
    
    provider_key = provider.lower()
    if provider_key not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")

    # Specific argument handling for LocalHuggingFaceEmbeddings
    # It expects 'model_path' instead of 'model' like others.
    # We can standardize by checking kwargs or make `create_embedding_provider` smarter.
    # For now, let's assume if provider is 'local_hf', 'model' kwarg is actually the path.
    provider_kwargs = kwargs.copy()
    if provider_key == "local_hf":
        if "model" in provider_kwargs and "model_path" not in provider_kwargs:
            provider_kwargs["model_path"] = provider_kwargs.pop("model")
        elif "model_path" not in provider_kwargs:
            raise ValueError("For 'local_hf' provider, 'model_path' (or 'model' as alias) must be specified in kwargs.")

    base_provider = providers[provider_key](**provider_kwargs)
    
    # Wrap with caching if enabled
    if kwargs.get("enable_cache", True):
        cache_dir = kwargs.get("cache_dir", "embeddings_cache")
        return CacheEmbeddings(base_provider, cache_dir=cache_dir)
        
    return base_provider