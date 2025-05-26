import json
from typing import Optional, Any, Dict, List
import redis
from datetime import timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 3600):
        """Initialize Redis cache manager.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time to live for cache entries in seconds (default: 1 hour)
        """
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from the data."""
        if isinstance(data, str):
            data_str = data
        else:
            data_str = json.dumps(data, sort_keys=True)
        
        key_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get_from_cache(self, prefix: str, data: Any) -> Optional[Any]:
        """Get data from cache."""
        if not self.redis_client:
            return None
            
        try:
            key = self._generate_key(prefix, data)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                logger.debug(f"Cache hit for key: {key}")
                return json.loads(cached_data)
                
            logger.debug(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Error getting data from cache: {e}")
            return None

    def set_in_cache(self, prefix: str, data: Any, value: Any) -> bool:
        """Set data in cache."""
        if not self.redis_client:
            return False
            
        try:
            key = self._generate_key(prefix, data)
            self.redis_client.setex(
                key,
                timedelta(seconds=self.ttl),
                json.dumps(value)
            )
            logger.debug(f"Successfully cached data with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error setting data in cache: {e}")
            return False

    def delete_from_cache(self, prefix: str, data: Any) -> bool:
        """Delete data from cache."""
        if not self.redis_client:
            return False
            
        try:
            key = self._generate_key(prefix, data)
            self.redis_client.delete(key)
            logger.debug(f"Successfully deleted cache key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting data from cache: {e}")
            return False

    def clear_cache_by_prefix(self, prefix: str) -> bool:
        """Clear all cache entries with the given prefix."""
        if not self.redis_client:
            return False
            
        try:
            keys = self.redis_client.keys(f"{prefix}:*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Successfully cleared cache for prefix: {prefix}")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_cache_stats(self, prefix: str) -> Dict[str, int]:
        """Get cache statistics for a prefix."""
        if not self.redis_client:
            return {"total_keys": 0, "memory_used": 0}
            
        try:
            keys = self.redis_client.keys(f"{prefix}:*")
            memory_used = sum(
                self.redis_client.memory_usage(key) or 0 
                for key in keys
            )
            return {
                "total_keys": len(keys),
                "memory_used": memory_used
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"total_keys": 0, "memory_used": 0}
