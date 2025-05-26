import pytest
import os
import tempfile
from pathlib import Path
import sqlalchemy as sa
import qdrant_client
from sqlalchemy.orm import Session
import asyncio
import shutil

# Test configuration
@pytest.fixture
def test_config():
    return {
        "database_url": os.getenv("TEST_DATABASE_URL", "sqlite:///test.db"),
        "qdrant_url": os.getenv("TEST_QDRANT_URL", "http://localhost:6333"),
        "storage_path": tempfile.mkdtemp(),
        "log_dir": tempfile.mkdtemp(),
        "backup_dir": tempfile.mkdtemp(),
    }

# Database fixtures
@pytest.fixture
async def db_engine(test_config):
    engine = sa.create_engine(test_config["database_url"])
    yield engine
    engine.dispose()

@pytest.fixture
async def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

# Qdrant fixtures
@pytest.fixture
async def qdrant_client(test_config):
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
