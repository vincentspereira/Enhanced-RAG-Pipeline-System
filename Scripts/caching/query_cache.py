from typing import Any, Dict, Optional, List, Union
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import aioredis
import diskcache
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    cache_type: str = "memory"  # "memory", "redis", or "disk"
    redis_url: Optional[str] = None
    disk_cache_dir: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory_size: int = 1024 * 1024 * 1024  # 1GB
    compression: bool = True
    enable_stats: bool = True
    min_query_frequency: int = 2
    max_cache_size: int = 10000
    update_threshold: float = 0.1  # 10% change triggers update

class CacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size = 0
        self.query_frequencies: Dict[str, int] = {}
        self.last_cleanup = datetime.now()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

class CacheBase(ABC):
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> bool:
        pass

class MemoryCache(CacheBase):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() < entry['expiry']:
                self.stats.hits += 1
                return entry['value']
            else:
                await self.delete(key)
        self.stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        try:
            ttl = ttl or self.config.default_ttl
            self.cache[key] = {
                'value': value,
                'expiry': datetime.now() + timedelta(seconds=ttl)
            }
            self.stats.size = len(self.cache)
            await self._cleanup_if_needed()
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            del self.cache[key]
            self.stats.size = len(self.cache)
            return True
        except KeyError:
            return False

    async def clear(self) -> bool:
        try:
            self.cache.clear()
            self.stats.size = 0
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def _cleanup_if_needed(self):
        """Remove expired entries and enforce size limits"""
        if len(self.cache) > self.config.max_cache_size:
            # Remove expired entries
            now = datetime.now()
            expired = [
                k for k, v in self.cache.items()
                if v['expiry'] < now
            ]
            for key in expired:
                await self.delete(key)
                self.stats.evictions += 1

            # If still too large, remove least frequently accessed
            if len(self.cache) > self.config.max_cache_size:
                sorted_keys = sorted(
                    self.stats.query_frequencies.items(),
                    key=lambda x: x[1]
                )
                for key, _ in sorted_keys[:len(self.cache) - self.config.max_cache_size]:
                    await self.delete(key)
                    self.stats.evictions += 1

class RedisCache(CacheBase):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis = aioredis.from_url(config.redis_url)
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        try:
            value = await self.redis.get(key)
            if value:
                self.stats.hits += 1
                return pickle.loads(value)
            self.stats.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        try:
            ttl = ttl or self.config.default_ttl
            serialized = pickle.dumps(value)
            await self.redis.set(key, serialized, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def clear(self) -> bool:
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

class DiskCache(CacheBase):
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = diskcache.Cache(config.disk_cache_dir)
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        try:
            value = self.cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
            self.stats.misses += 1
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        try:
            ttl = ttl or self.config.default_ttl
            self.cache.set(key, value, expire=ttl)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            del self.cache[key]
            return True
        except KeyError:
            return False

    async def clear(self) -> bool:
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

class QueryCache:
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        if self.config.cache_type == "redis":
            self.cache = RedisCache(self.config)
        elif self.config.cache_type == "disk":
            self.cache = DiskCache(self.config)
        else:
            self.cache = MemoryCache(self.config)

    def _generate_key(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique cache key for a query"""
        key_data = {
            'query': query,
            'params': params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get_cached_result(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached result for a query"""
        key = self._generate_key(query, params)
        return await self.cache.get(key)

    async def cache_result(
        self,
        query: str,
        result: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache a query result"""
        key = self._generate_key(query, params)
        return await self.cache.set(key, result, ttl)

    async def invalidate(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Invalidate cached result for a query"""
        key = self._generate_key(query, params)
        return await self.cache.delete(key)

    def cache_query(self, ttl: Optional[int] = None):
        """Decorator for caching query results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract query and params from function arguments
                query = args[0] if args else kwargs.get('query')
                params = kwargs.get('params', {})
                
                # Check cache first
                cached_result = await self.get_cached_result(query, params)
                if cached_result is not None:
                    return cached_result
                
                # Execute query if not cached
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.cache_result(query, result, params, ttl)
                
                return result
            return wrapper
        return decorator

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.cache.stats
