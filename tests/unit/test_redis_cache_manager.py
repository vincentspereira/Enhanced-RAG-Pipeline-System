import pytest
from unittest.mock import MagicMock, patch
import json

# Adjust the import path based on your project structure
# This assumes that 'Scripts' is a top-level directory in PYTHONPATH or tests are run from root
from Scripts.utils.redis_cache_manager import RedisCacheManager
from Scripts.utils.config_loader import get_config_value # To mock its usage

# Mock 'get_config_value' to avoid dependency on actual config files/env vars during unit tests
# We can patch it per test or globally for this test module.
# Global patch is simpler here if all tests use similar mock config.
@pytest.fixture(autouse=True)
def mock_get_config_value(mocker):
    mock = mocker.patch('Scripts.utils.redis_cache_manager.get_config_value')
    # Default mock values for Redis connection, can be overridden in tests
    mock.side_effect = lambda key, yaml_path=None, default=None: {
        "REDIS_HOST": "mockredis",
        "REDIS_PORT": 6379,
        "REDIS_DB": 0,
        "REDIS_PASSWORD": None
    }.get(key, default)
    return mock

@pytest.fixture
def mock_redis_client(mocker):
    """Fixture to mock the redis.Redis client instance."""
    mock_client = MagicMock()
    # Mock specific methods used by RedisCacheManager
    mock_client.ping.return_value = True
    mock_client.get.return_value = None # Default to cache miss
    mock_client.set.return_value = True
    mock_client.delete.return_value = 1 # Indicates 1 key deleted
    mock_client.close.return_value = True

    # Patch 'redis.Redis' to return this mock_client when instantiated
    mocker.patch('redis.Redis', return_value=mock_client)
    return mock_client

@pytest.fixture
def cache_manager(mock_redis_client): # Depends on mock_redis_client to ensure redis.Redis is patched
    # Clear the singleton instance map for clean tests if RedisCacheManager uses it
    RedisCacheManager._instance_map = {}
    manager = RedisCacheManager()
    # Ensure the mocked client is actually assigned
    manager.redis_client = mock_redis_client
    manager._initialized_connection = True # Assume connection was successful with mock
    return manager

def test_redis_cache_manager_singleton_behavior():
    """Test that multiple calls to constructor with same params yield same instance."""
    # Clear instance map before this specific test
    RedisCacheManager._instance_map = {}
    manager1 = RedisCacheManager(host="localhost", port=6379, db=0)
    manager2 = RedisCacheManager(host="localhost", port=6379, db=0)
    assert manager1 is manager2

    # Clear again for different params
    RedisCacheManager._instance_map = {}
    manager3 = RedisCacheManager(host="localhost", port=6379, db=1)
    assert manager1 is not manager3 # Should be a different instance due to different db

    # Clean up for other tests
    RedisCacheManager._instance_map = {}


def test_connection_success(cache_manager, mock_redis_client):
    """Test successful connection and initialization."""
    assert cache_manager.is_available() is True
    mock_redis_client.ping.assert_called_once()

def test_connection_failure(mocker):
    """Test connection failure during initialization."""
    RedisCacheManager._instance_map = {} # Ensure fresh instance attempt
    mocker.patch('redis.Redis', side_effect=redis.exceptions.ConnectionError("Mock connection error"))

    manager = RedisCacheManager() # This will try to connect
    assert manager.is_available() is False
    assert manager.redis_client is None

def test_set_json_success(cache_manager, mock_redis_client):
    """Test setting a JSON value successfully."""
    key = "test_json_key"
    data = {"name": "Jules", "value": 123}
    ttl = 60

    result = cache_manager.set_json(key, data, ttl_seconds=ttl)

    assert result is True
    mock_redis_client.set.assert_called_once_with(key, json.dumps(data), ex=ttl)

def test_set_json_redis_unavailable(cache_manager, mock_redis_client):
    """Test set_json when Redis is unavailable."""
    mock_redis_client.ping.return_value = False # Simulate ping failure
    cache_manager._initialized_connection = False # Also mark as not connected

    result = cache_manager.set_json("key", {"data": "value"})
    assert result is False
    mock_redis_client.set.assert_not_called()

def test_get_json_cache_hit(cache_manager, mock_redis_client):
    """Test getting a JSON value that exists in cache."""
    key = "hit_key"
    expected_data = {"name": "Jules", "value": 123}
    mock_redis_client.get.return_value = json.dumps(expected_data).encode('utf-8')

    retrieved_data = cache_manager.get_json(key)

    assert retrieved_data == expected_data
    mock_redis_client.get.assert_called_once_with(key)

def test_get_json_cache_miss(cache_manager, mock_redis_client):
    """Test getting a JSON value not in cache."""
    key = "miss_key"
    mock_redis_client.get.return_value = None # Simulate cache miss

    retrieved_data = cache_manager.get_json(key)

    assert retrieved_data is None
    mock_redis_client.get.assert_called_once_with(key)

def test_get_json_redis_unavailable(cache_manager, mock_redis_client):
    """Test get_json when Redis is unavailable."""
    mock_redis_client.ping.return_value = False
    cache_manager._initialized_connection = False

    retrieved_data = cache_manager.get_json("key")
    assert retrieved_data is None
    mock_redis_client.get.assert_not_called() # Should not attempt get if ping fails

def test_set_string_success(cache_manager, mock_redis_client):
    key = "test_string_key"
    value = "Hello Redis"
    ttl = 120
    result = cache_manager.set_string(key, value, ttl_seconds=ttl)
    assert result is True
    mock_redis_client.set.assert_called_once_with(key, value, ex=ttl)

def test_get_string_cache_hit(cache_manager, mock_redis_client):
    key = "hit_string_key"
    expected_value = "Hello from cache"
    mock_redis_client.get.return_value = expected_value.encode('utf-8')
    retrieved_value = cache_manager.get_string(key)
    assert retrieved_value == expected_value
    mock_redis_client.get.assert_called_once_with(key)

def test_delete_success(cache_manager, mock_redis_client):
    """Test deleting a key successfully."""
    key = "delete_key"
    result = cache_manager.delete(key)
    assert result is True
    mock_redis_client.delete.assert_called_once_with(key)

def test_delete_redis_unavailable(cache_manager, mock_redis_client):
    """Test delete when Redis is unavailable."""
    mock_redis_client.ping.return_value = False
    cache_manager._initialized_connection = False

    result = cache_manager.delete("key")
    assert result is False
    mock_redis_client.delete.assert_not_called()

def test_close_method(cache_manager, mock_redis_client):
    """Test the close method."""
    # Ensure it's marked as initialized before close
    cache_manager._initialized_connection = True
    # For singleton with instance_map
    instance_key = f"{cache_manager.host}:{cache_manager.port}:{cache_manager.db}"
    RedisCacheManager._instance_map[instance_key] = cache_manager

    cache_manager.close()

    mock_redis_client.close.assert_called_once()
    assert not cache_manager._initialized_connection
    assert not cache_manager._initialized_once # Test if it resets this flag for re-init logic
    assert instance_key not in RedisCacheManager._instance_map # Check if removed from map

# To run these tests:
# Ensure pytest and pytest-mock are installed.
# Navigate to the root of the project and run:
# pytest Scripts/tests/unit/test_redis_cache_manager.py
#
# Note: The 'redis' import in RedisCacheManager itself is not mocked here,
# so the 'redis' library needs to be installed in the test environment.
# The redis.Redis *client instance* is mocked by the mock_redis_client fixture.
# The global `get_config_value` is mocked by `mock_get_config_value` fixture.
#
# If redis library itself is an issue for CI minimal env, then redis module could be mocked too.
# Example: mocker.patch('Scripts.utils.redis_cache_manager.redis') -> this mock_redis_module
# mock_redis_module.Redis.return_value = mock_redis_client
# mock_redis_module.exceptions.ConnectionError = redis.exceptions.ConnectionError # if needed for exception checking
# For now, assuming 'redis' library is installed.
#
# The `RedisCacheManager._instance_map = {}` in fixtures helps isolate singleton tests.
# Be mindful of this if running tests in parallel within the same process without proper isolation.
# Pytest typically runs tests in separate processes or handles this, but good to be aware of for complex singletons.
