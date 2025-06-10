import pytest
import os
import tempfile
from pathlib import Path
import sqlalchemy as sa
import qdrant_client
from sqlalchemy.orm import Session
import asyncio
import shutil
# Removed asynccontextmanager as db_session will be sync

# Test configuration
@pytest.fixture(scope="session") # Changed to session scope
def test_config():
    return {
        "database_url": os.getenv("TEST_DATABASE_URL", "sqlite:///test.db"),
        "qdrant_url": os.getenv("TEST_QDRANT_URL", "http://localhost:6333"),
        "storage_path": tempfile.mkdtemp(),
        "log_dir": tempfile.mkdtemp(),
        "backup_dir": tempfile.mkdtemp(),
    }

# Database fixtures
from Scripts.security.rbac import RBACManager # Import RBACManager

@pytest.fixture(scope="session") # Scope to session to run once
def db_engine(test_config): # Making it synchronous
    engine = sa.create_engine(test_config["database_url"])

    # Create tables and initialize default roles once per test session
    rbac_manager = RBACManager(engine=engine)
    rbac_manager.create_tables_if_needed()
    # APIKeyManager tables are also created if they share the same Base.metadata
    # from Scripts.security.api_keys import APIKeyManager
    # apikey_manager = APIKeyManager(rbac_manager)
    # apikey_manager.create_tables_if_needed() # If APIKeyManager had its own Base/tables

    rbac_manager.initialize_default_roles_if_needed() # This handles its own session and commit

    yield engine
    engine.dispose()

@pytest.fixture # Making it synchronous, function scope for transactions
def db_session(db_engine): # Making it synchronous, depends on session-scoped db_engine
    # Standard SQLAlchemy session setup is synchronous
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

# Qdrant fixtures
@pytest.fixture # Assuming qdrant_client can also be sync for unit tests if not interacting with loop
def qdrant_client(test_config): # Making it synchronous
    client = qdrant_client.QdrantClient(url=test_config["qdrant_url"])
    yield client
    # Cleanup collections after tests
    collections = client.get_collections()
    for collection in collections:
        client.delete_collection(collection.name)

# Storage fixtures
@pytest.fixture
def temp_storage(test_config):
    storage_path = Path(test_config["storage_path"])
    storage_path.mkdir(parents=True, exist_ok=True)
    yield storage_path
    shutil.rmtree(storage_path)

@pytest.fixture
def temp_log_dir(test_config):
    log_path = Path(test_config["log_dir"])
    log_path.mkdir(parents=True, exist_ok=True)
    yield log_path
    shutil.rmtree(log_path)

@pytest.fixture
def temp_backup_dir(test_config):
    backup_path = Path(test_config["backup_dir"])
    backup_path.mkdir(parents=True, exist_ok=True)
    yield backup_path
    shutil.rmtree(backup_path)

# Test data fixtures
@pytest.fixture
def sample_document():
    return {
        "content": "This is a test document content",
        "metadata": {
            "title": "Test Document",
            "author": "Test Author",
            "date": "2025-05-23"
        }
    }

@pytest.fixture
def sample_user_data():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!"
    }

# Mock service fixtures
@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("openai.Embedding.create")

@pytest.fixture
def mock_s3(mocker):
    return mocker.patch("boto3.client")

# Async support
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()
