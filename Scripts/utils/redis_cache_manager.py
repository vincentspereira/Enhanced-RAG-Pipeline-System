import redis
import json
import logging
import os
from typing import Optional, Any, Union

# Configure logger
logger = logging.getLogger(__name__)

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    # This adjustment assumes redis_cache_manager.py is in Scripts/utils/
    # and config_loader.py is also in Scripts/utils/
    # If structure is different, this path might need to be .. to go up one level.
    # For safety, let's assume they could be in different subdirs of Scripts
    base_scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_scripts_dir not in sys.path:
         sys.path.append(base_scripts_dir)
    try:
        from utils.config_loader import get_config_value # Try direct if Scripts is in PYTHONPATH
    except ImportError:
        logger.error(f"Critical: Failed to import get_config_value for RedisCacheManager. Ensure Scripts/utils/config_loader.py exists. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

class RedisCacheManager:
    _instance_map = {} # Store instances by connection params string to allow multiple connections if needed

    def __new__(cls, host: Optional[str] = None, port: Optional[int] = None, db: Optional[int] = None, password: Optional[str] = None, *args, **kwargs):
        # Configuration for connection (allows multiple uniquely configured instances if needed)
        conf_host = host or get_config_value("REDIS_HOST", yaml_path="caching.redis.host", default="localhost")
        conf_port = port or int(get_config_value("REDIS_PORT", yaml_path="caching.redis.port", default=6379))
        conf_db = db or int(get_config_value("REDIS_DB", yaml_path="caching.redis.db", default=0))
        # Password can be None
        conf_password = password or get_config_value("REDIS_PASSWORD", yaml_path="caching.redis.password")

        instance_key = f"{conf_host}:{conf_port}:{conf_db}" # Key by host, port, db

        if instance_key not in cls._instance_map:
            instance = super(RedisCacheManager, cls).__new__(cls)
            cls._instance_map[instance_key] = instance
            # Initialization logic will be in __init__ and only run once per instance
            instance._initialized_once = False
        return cls._instance_map[instance_key]

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, db: Optional[int] = None, password: Optional[str] = None):
        if hasattr(self, '_initialized_once') and self._initialized_once:
            return

        self.host = host or get_config_value("REDIS_HOST", yaml_path="caching.redis.host", default="localhost")
        self.port = port or int(get_config_value("REDIS_PORT", yaml_path="caching.redis.port", default=6379))
        self.db = db or int(get_config_value("REDIS_DB", yaml_path="caching.redis.db", default=0))
        self.password = password or get_config_value("REDIS_PASSWORD", yaml_path="caching.redis.password")

        self.redis_client: Optional[redis.Redis] = None
        self._initialized_connection = False # Tracks if the current connection attempt was successful
        self._connect()

        self._initialized_once = True # Mark that __init__ has run for this instance

    def _connect(self):
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_connect_timeout=3, # Shorter timeout for quicker feedback
                socket_timeout=3,
                decode_responses=False
            )
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {self.host}:{self.port}, DB: {self.db}")
            self._initialized_connection = True
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Failed to connect to Redis at {self.host}:{self.port}, DB: {self.db}. Caching will be disabled. Error: {e}")
            self.redis_client = None
            self._initialized_connection = False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis connection: {e}", exc_info=True)
            self.redis_client = None
            self._initialized_connection = False

    def is_available(self) -> bool:
        if not self.redis_client or not self._initialized_connection:
            # Optionally, try to reconnect if not available
            # logger.debug("Redis client not available, attempting to reconnect...")
            # self._connect() # Be careful with reconnect logic in is_available to avoid loops
            return False
        try:
            return self.redis_client.ping()
        except redis.exceptions.ConnectionError:
            self._initialized_connection = False # Mark as disconnected
            logger.warning("Redis ping failed. Marking as unavailable.")
            return False

    def set_json(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> bool:
        if not self.is_available():
            logger.debug(f"Redis not available. Cannot set key '{key}'.")
            return False
        try:
            json_data = json.dumps(data)
            self.redis_client.set(key, json_data, ex=ttl_seconds)
            logger.debug(f"Set JSON for key '{key}' with TTL {ttl_seconds}s.")
            return True
        except Exception as e:
            logger.error(f"Error setting JSON for key '{key}': {e}", exc_info=True)
            return False

    def get_json(self, key: str) -> Optional[Any]:
        if not self.is_available():
            logger.debug(f"Redis not available. Cannot get key '{key}'.")
            return None
        try:
            cached_data_bytes = self.redis_client.get(key)
            if cached_data_bytes:
                logger.debug(f"Cache hit for key '{key}'.")
                return json.loads(cached_data_bytes.decode('utf-8'))
            logger.debug(f"Cache miss for key '{key}'.")
            return None
        except Exception as e:
            logger.error(f"Error getting JSON for key '{key}': {e}", exc_info=True)
            return None

    def set_string(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> bool:
        if not self.is_available():
            logger.debug(f"Redis not available. Cannot set key '{key}'.")
            return False
        try:
            self.redis_client.set(key, value, ex=ttl_seconds)
            logger.debug(f"Set string for key '{key}' with TTL {ttl_seconds}s.")
            return True
        except Exception as e:
            logger.error(f"Error setting string for key '{key}': {e}", exc_info=True)
            return False

    def get_string(self, key: str) -> Optional[str]:
        if not self.is_available():
            logger.debug(f"Redis not available. Cannot get key '{key}'.")
            return None
        try:
            value_bytes = self.redis_client.get(key)
            if value_bytes:
                logger.debug(f"Cache hit for key '{key}'.")
                return value_bytes.decode('utf-8')
            logger.debug(f"Cache miss for key '{key}'.")
            return None
        except Exception as e:
            logger.error(f"Error getting string for key '{key}': {e}", exc_info=True)
            return None

    def delete(self, key: str) -> bool:
        if not self.is_available():
            logger.warning(f"Redis not available. Cannot delete key '{key}'.")
            return False
        try:
            self.redis_client.delete(key)
            logger.debug(f"Deleted key '{key}'.")
            return True
        except Exception as e:
            logger.error(f"Error deleting key '{key}': {e}", exc_info=True)
            return False

    def close(self):
        # For redis-py, close() on a client with a connection pool returns the connection to the pool.
        # If not using a pool explicitly (like this simple client), it closes the connection.
        instance_key = f"{self.host}:{self.port}:{self.db}"
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info(f"Redis client for {instance_key} closed.")
            except Exception as e:
                logger.error(f"Error closing Redis client for {instance_key}: {e}")

        if instance_key in RedisCacheManager._instance_map:
            del RedisCacheManager._instance_map[instance_key]

        self._initialized_connection = False
        self._initialized_once = False # Allow re-init if another instance with same params is requested later


# Example Usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Ensure Redis is running locally on default port 6379 for this example.
    # Docker: docker run -d -p 6379:6379 redis:alpine

    cache_manager1 = RedisCacheManager() # Default instance

    if not cache_manager1.is_available():
        logger.error("Cannot run RedisCacheManager example: Redis (default instance) is not available.")
    else:
        logger.info("--- RedisCacheManager Example (Default Instance) ---")

        json_key = "user:1"
        user_data = {"name": "Jules", "id": 1, "active": True}
        cache_manager1.set_json(json_key, user_data, ttl_seconds=10)
        retrieved = cache_manager1.get_json(json_key)
        logger.info(f"Retrieved user:1 -> {retrieved}")
        assert retrieved == user_data

        str_key = "config:feature_flag_x"
        cache_manager1.set_string(str_key, "enabled", ttl_seconds=5)
        retrieved_str = cache_manager1.get_string(str_key)
        logger.info(f"Retrieved config:feature_flag_x -> {retrieved_str}")
        assert retrieved_str == "enabled"

        cache_manager1.delete(str_key)
        assert cache_manager1.get_string(str_key) is None
        logger.info(f"Key '{str_key}' deleted and verified.")

        logger.info("Waiting for 'user:1' to expire (10s TTL)...")
        import time
        time.sleep(11)
        retrieved_expired = cache_manager1.get_json(json_key)
        logger.info(f"Retrieved user:1 after TTL: {retrieved_expired}")
        assert retrieved_expired is None

        logger.info("--- Testing another instance with different DB (if Redis supports it) ---")
        # This assumes your Redis server allows selecting different DBs.
        # If REDIS_DB_CACHE_RESULTS and REDIS_DB_LLM_ANSWERS are set to different values
        # os.environ["REDIS_DB_CACHE_RESULTS"] = "1" # Hypothetical config
        # cache_manager2 = RedisCacheManager(db=1) # Or use a different config key

        # For this example, let's simulate a different config by passing it directly
        # Note: This will create a new client instance if params differ.
        cache_manager2 = RedisCacheManager(db=1) # Connect to DB 1
        if cache_manager2.is_available():
            cache_manager2.set_string("other_db_key", "value_in_db_1", 60)
            val_db1 = cache_manager2.get_string("other_db_key")
            logger.info(f"Value from DB 1 ('other_db_key'): {val_db1}")
            assert val_db1 == "value_in_db_1"

            # Check that default instance (DB 0) doesn't see this key
            val_db0 = cache_manager1.get_string("other_db_key")
            logger.info(f"Value from DB 0 ('other_db_key'): {val_db0}")
            assert val_db0 is None

            cache_manager2.close()
        else:
            logger.warning("Could not connect to Redis DB 1 for second instance test.")

        cache_manager1.close()
        logger.info("--- RedisCacheManager Example Finished ---")
