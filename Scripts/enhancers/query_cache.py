from typing import Any, Dict, List, Optional, Union
import redis
from datetime import datetime, timedelta
import json
import hashlib
import logging
from dataclasses import dataclass
import time
import threading
from collections import OrderedDict
import pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory_mb: int = 512
    enable_memory_cache: bool = True
    enable_redis_cache: bool = True
    max_memory_items: int = 10000
    compression_threshold: int = 1024  # bytes
    background_refresh: bool = True
    refresh_interval: int = 300  # 5 minutes

class QueryCache:
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._memory_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        
        if self.config.enable_redis_cache:
            self._init_redis()
            
        if self.config.background_refresh:
            self._start_background_refresh()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False  # We'll handle encoding ourselves
            )
            # Configure Redis
            self.redis.config_set('maxmemory', f'{self.config.max_memory_mb}mb')
            self.redis.config_set('maxmemory-policy', 'allkeys-lru')
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.config.enable_redis_cache = False

    def _compute_key(self, query: Union[str, Dict]) -> str:
        """Compute a stable hash key for the query"""
        if isinstance(query, dict):
            # Sort dictionary to ensure stable hash
            query = json.dumps(query, sort_keys=True)
        return hashlib.sha256(query.encode('utf-8')).hexdigest()

    def _compress_value(self, value: Any) -> bytes:
        """Compress value if it exceeds threshold"""
        try:
            pickled = pickle.dumps(value)
            if len(pickled) > self.config.compression_threshold:
                import zlib
                return zlib.compress(pickled)
            return pickled
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return pickle.dumps(value)

    def _decompress_value(self, value: bytes) -> Any:
        """Decompress value if it was compressed"""
        try:
            import zlib
            return pickle.loads(zlib.decompress(value))
        except zlib.error:
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None

    def get(self, query: Union[str, Dict]) -> Optional[Any]:
        """Get value from cache"""
        key = self._compute_key(query)
        
        # Try memory cache first
        if self.config.enable_memory_cache:
            with self._cache_lock:
                if key in self._memory_cache:
                    value, expiry = self._memory_cache[key]
                    if expiry > datetime.now():
                        self._memory_cache.move_to_end(key)  # LRU update
                        return value
                    else:
                        del self._memory_cache[key]

        # Try Redis cache
        if self.config.enable_redis_cache:
            try:
                value = self.redis.get(key)
                if value is not None:
                    value = self._decompress_value(value)
                    if self.config.enable_memory_cache:
                        self.set_memory_cache(key, value)
                    return value
            except Exception as e:
                logger.error(f"Redis get failed: {e}")

        return None

    def set(self, query: Union[str, Dict], value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        key = self._compute_key(query)
        ttl = ttl or self.config.default_ttl
        
        success = True
        
        # Set in memory cache
        if self.config.enable_memory_cache:
            self.set_memory_cache(key, value, ttl)
            
        # Set in Redis cache
        if self.config.enable_redis_cache:
            try:
                compressed = self._compress_value(value)
                self.redis.setex(key, ttl, compressed)
            except Exception as e:
                logger.error(f"Redis set failed: {e}")
                success = False
                
        return success

    def set_memory_cache(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache"""
        with self._cache_lock:
            if len(self._memory_cache) >= self.config.max_memory_items:
                self._memory_cache.popitem(last=False)  # Remove oldest item
            expiry = datetime.now() + timedelta(seconds=ttl or self.config.default_ttl)
            self._memory_cache[key] = (value, expiry)

    def delete(self, query: Union[str, Dict]) -> bool:
        """Delete value from cache"""
        key = self._compute_key(query)
        success = True
        
        # Delete from memory cache
        if self.config.enable_memory_cache:
            with self._cache_lock:
                self._memory_cache.pop(key, None)
                
        # Delete from Redis cache
        if self.config.enable_redis_cache:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.error(f"Redis delete failed: {e}")
                success = False
                
        return success

    def clear(self) -> bool:
        """Clear all cache entries"""
        success = True
        
        # Clear memory cache
        if self.config.enable_memory_cache:
            with self._cache_lock:
                self._memory_cache.clear()
                
        # Clear Redis cache
        if self.config.enable_redis_cache:
            try:
                self.redis.flushdb()
            except Exception as e:
                logger.error(f"Redis flush failed: {e}")
                success = False
                
        return success

    def _refresh_cache(self):
        """Background task to refresh cache entries"""
        while True:
            try:
                # Get all keys from Redis
                if self.config.enable_redis_cache:
                    all_keys = self.redis.keys('*')
                    for key in all_keys:
                        ttl = self.redis.ttl(key)
                        # Refresh if TTL is less than refresh interval
                        if 0 < ttl < self.config.refresh_interval:
                            value = self._decompress_value(self.redis.get(key))
                            if value is not None:
                                self.redis.expire(key, self.config.default_ttl)
                
                # Clean expired memory cache entries
                if self.config.enable_memory_cache:
                    with self._cache_lock:
                        current_time = datetime.now()
                        expired_keys = [
                            k for k, (_, exp) in self._memory_cache.items()
                            if exp <= current_time
                        ]
                        for k in expired_keys:
                            del self._memory_cache[k]
                            
                time.sleep(self.config.refresh_interval)
            except Exception as e:
                logger.error(f"Cache refresh failed: {e}")
                time.sleep(60)  # Wait before retrying

    def _start_background_refresh(self):
        """Start background refresh thread"""
        if self.config.background_refresh:
            refresh_thread = threading.Thread(
                target=self._refresh_cache,
                daemon=True
            )
            refresh_thread.start()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache": {
                "enabled": self.config.enable_memory_cache,
                "items": len(self._memory_cache) if self.config.enable_memory_cache else 0,
                "max_items": self.config.max_memory_items
            },
            "redis_cache": {
                "enabled": self.config.enable_redis_cache
            }
        }
        
        if self.config.enable_redis_cache:
            try:
                info = self.redis.info()
                stats["redis_cache"].update({
                    "used_memory_mb": info["used_memory"] / 1024 / 1024,
                    "max_memory_mb": self.config.max_memory_mb,
                    "connected_clients": info["connected_clients"],
                    "total_connections_received": info["total_connections_received"],
                    "total_commands_processed": info["total_commands_processed"]
                })
            except Exception as e:
                logger.error(f"Failed to get Redis stats: {e}")
                
        return stats

    def __del__(self):
        """Cleanup Redis connection"""
        if hasattr(self, 'redis'):
            try:
                self.redis.close()
            except:
                pass
