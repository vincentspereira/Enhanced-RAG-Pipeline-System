from typing import List, Dict, Any, Optional, Union
import openai
from openai import OpenAI
import numpy as np
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time
import backoff
from datetime import datetime, timedelta
import tiktoken
import hashlib
import os
import pickle

logger = logging.getLogger(__name__)

@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    timeout: int = 30
    cache_dir: Optional[str] = "embeddings_cache"
    enable_caching: bool = True
    cache_ttl: int = 86400  # 24 hours
    dimension_size: int = 1536
    normalize_embeddings: bool = True
    strip_newlines: bool = True
    enable_batching: bool = True
    max_tokens: int = 8191
    use_async: bool = True

class OpenAIEmbeddings:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)
        self._setup_cache()
        self.tokenizer = tiktoken.encoding_for_model(config.model)

    def _setup_cache(self):
        """Initialize embedding cache"""
        if self.config.enable_caching:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(
                self.config.cache_dir,
                f"embeddings_cache_{self.config.model}.pkl"
            )
            self._load_cache()

    def _load_cache(self):
        """Load embeddings cache from disk"""
        self.cache = {}
        if self.config.enable_caching and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Filter out expired cache entries
                    current_time = datetime.now()
                    self.cache = {
                        k: v for k, v in cached_data.items()
                        if current_time - v['timestamp'] < timedelta(seconds=self.config.cache_ttl)
                    }
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save embeddings cache to disk"""
        if self.config.enable_caching:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def _compute_hash(self, text: str) -> str:
        """Compute hash for text to use as cache key"""
        return hashlib.md5(text.encode()).hexdigest()

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before computing embeddings"""
        if not text:
            return text
            
        if self.config.strip_newlines:
            text = ' '.join(text.split())
            
        return text

    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in text"""
        return len(self.tokenizer.encode(text))

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError),
        max_tries=3
    )
    async def _get_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
            )
            
            embeddings = [data.embedding for data in response.data]
            
            if self.config.normalize_embeddings:
                embeddings = [
                    self._normalize_embedding(embedding)
                    for embedding in embeddings
                ]
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to get embeddings batch: {e}")
            raise

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (np.array(embedding) / norm).tolist()
        return embedding

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True
    ) -> Union[List[float], List[List[float]]]:
        """
        Get embeddings for one or more texts
        Returns a single embedding vector for a string input,
        or a list of embedding vectors for a list input
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Initialize results list
        embeddings = []
        texts_to_embed = []
        cache_keys = []

        # Check cache and collect texts that need embedding
        for text in processed_texts:
            if not text:  # Handle empty text
                embeddings.append([0.0] * self.config.dimension_size)
                continue

            cache_key = self._compute_hash(text)
            cache_keys.append(cache_key)
            
            if use_cache and self.config.enable_caching and cache_key in self.cache:
                cached_item = self.cache[cache_key]
                if datetime.now() - cached_item['timestamp'] < timedelta(seconds=self.config.cache_ttl):
                    embeddings.append(cached_item['embedding'])
                    continue
                    
            texts_to_embed.append(text)

        # Get embeddings for uncached texts
        if texts_to_embed:
            if self.config.enable_batching:
                # Process in batches
                all_new_embeddings = []
                for i in range(0, len(texts_to_embed), self.config.batch_size):
                    batch = texts_to_embed[i:i + self.config.batch_size]
                    batch_embeddings = await self._get_embeddings_batch(batch)
                    all_new_embeddings.extend(batch_embeddings)
            else:
                # Process one at a time
                all_new_embeddings = []
                for text in texts_to_embed:
                    embedding = await self._get_embeddings_batch([text])
                    all_new_embeddings.extend(embedding)

            # Update cache with new embeddings
            if self.config.enable_caching:
                for text, embedding in zip(texts_to_embed, all_new_embeddings):
                    cache_key = self._compute_hash(text)
                    self.cache[cache_key] = {
                        'embedding': embedding,
                        'timestamp': datetime.now()
                    }
                self._save_cache()

            # Merge cached and new embeddings in correct order
            final_embeddings = []
            new_embedding_idx = 0
            
            for cache_key in cache_keys:
                if cache_key in self.cache:
                    final_embeddings.append(self.cache[cache_key]['embedding'])
                else:
                    final_embeddings.append(all_new_embeddings[new_embedding_idx])
                    new_embedding_idx += 1

            embeddings = final_embeddings

        return embeddings[0] if single_input else embeddings

    async def similarity(
        self,
        text1: str,
        text2: str,
        use_cache: bool = True
    ) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = await self.get_embeddings([text1, text2], use_cache=use_cache)
        return self.cosine_similarity(embeddings[0], embeddings[1])

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def clear_cache(self):
        """Clear the embeddings cache"""
        self.cache = {}
        if self.config.enable_caching:
            self._save_cache()

    async def bulk_embed(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[List[float]]:
        """Efficiently embed a large number of texts"""
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        from tqdm import tqdm
        
        # Process in batches
        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not show_progress,
            desc="Computing embeddings"
        ):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.get_embeddings(
                batch,
                use_cache=use_cache
            )
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding model and cache"""
        return {
            "model": self.config.model,
            "dimension_size": self.config.dimension_size,
            "cache_enabled": self.config.enable_caching,
            "cache_size": len(self.cache) if self.config.enable_caching else 0,
            "cache_ttl": self.config.cache_ttl,
            "normalized": self.config.normalize_embeddings,
            "max_batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens
        }
