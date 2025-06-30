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

import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock # Added for unit test fixtures
from pytest import FixtureRequest # For typing fixtures that request other fixtures

# Core FastAPI and RAG components for integration testing
from Scripts.enhanced_api import app as actual_enhanced_app  # Import the app instance
from Scripts.config.manager import ConfigManager, SystemConfig as AppSystemConfig, APIConfig as AppAPIConfig # For unit test mock
from Scripts.rag_pipeline import RAGPipeline
from Scripts.document_processor import DocumentProcessor as ActualDocumentProcessor
from Scripts.embedding_generator import EmbeddingGenerator as ActualEmbeddingGenerator
from Scripts.auth.auth_manager import AuthManager, AuthConfig as AppAuthConfig, User as AuthUser, Role, Permission
from qdrant_client import QdrantClient, models as qdrant_models
from elasticsearch import AsyncElasticsearch, NotFoundError as ESNotFoundError

# --- Constants for Tests ---
SAMPLE_DOCS_CONTENT = [
    {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog.", "category": "animals", "timestamp": "2023-01-01T10:00:00Z"},
    {"id": "doc2", "text": "Elasticsearch is a powerful search engine for text.", "category": "software", "timestamp": "2023-01-02T12:00:00Z"},
    {"id": "doc3", "text": "Qdrant is a vector database, efficient for similarity search over fox and dog vectors.", "category": "software", "timestamp": "2023-01-03T14:00:00Z"},
    {"id": "doc4", "text": "A lazy dog also enjoys a textual challenge from a quick fox.", "category": "animals", "timestamp": "2023-01-04T16:00:00Z"},
    {"id": "doc5", "text": "Vector search helps find similar items quickly.", "category": "concepts", "timestamp": "2023-01-05T18:00:00Z"}
]


# --- Event Loop Fixture (Session Scoped) ---
@pytest.fixture(scope="session")
def event_loop():
    """Ensure a single event loop for session-scoped async fixtures."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

# --- Configuration Fixture for Integration Tests (Session Scoped) ---
@pytest.fixture(scope="session")
def test_integration_config_path(tmp_path_factory):
    """Creates a temporary YAML config file for integration tests and returns its path."""
    config_dir = tmp_path_factory.mktemp("config_integration")
    test_config_path = config_dir / "config.integration.yaml"

    # Define Qdrant and ES collection/index names for integration tests
    # These should be unique if tests run in parallel or on shared instances,
    # but for local sequential testing, fixed names are okay with proper cleanup.
    qdrant_collection_name = "test_integration_api_collection"
    es_index_name = "test_integration_api_es_index"

    test_yaml_content = f"""
version: 1

model:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu" # Use CPU for tests to avoid GPU dependencies
  batch_size: 4
  llm_service: "openai" # Dummy, not used if RAGPipeline's LLM part is mocked or not hit

vector_store:
  host: "localhost" # Assumes Qdrant is running locally for integration tests
  port: 6333
  collection_name: "{qdrant_collection_name}"
  vector_size: 384 # For all-MiniLM-L6-v2
  recreate_collection: True # Ensure clean state for tests

processing:
  chunk_size: 256
  chunk_overlap: 32
  max_workers: 2 # Keep low for tests
  use_ocr: False

api:
  host: "0.0.0.0"
  port: 8000 # This will be overridden by TestClient anyway
  workers: 1
  max_concurrent_requests: 10
  default_page_size: 10
  max_page_size: 50

paths:
  cache_dir: "{str(tmp_path_factory.mktemp('cache_integration'))}" # Use temp dir for cache
  log_dir: "{str(tmp_path_factory.mktemp('logs_integration'))}"

cache_settings:
  enabled: True
  ttl: 300 # seconds

feature_flags:
  enable_elasticsearch_fallback: True
  enable_hybrid_search: True # Enable hybrid search for RAGPipeline
  # Disable other complex enhancers for basic API integration tests if not needed
  enable_query_caching: False
  enable_advanced_search_pipeline: False
  enable_error_recovery: False
  enable_auto_scaling: False
  enable_monitoring: False

elasticsearch:
  hosts: ["http://localhost:9200"] # Assumes ES is running locally
  index_name: "{es_index_name}"
  enable_hybrid_search_in_es: False # Keep this simple for basic ES fallback tests
  username: "" # Add if your ES needs auth
  password: ""

auth: # Basic auth config for token generation in fixtures
  secret_key: "test_secret_key_for_integration_tokens"
  token_expire_minutes: 60
  refresh_token_expire_days: 1
  password_min_length: 8
  max_failed_attempts: 5
  lockout_duration_minutes: 15
  api_key_prefix: "test-rag-"
  redis_url: "redis://localhost:6379/1" # Use a different DB for test auth if needed
    """
    with open(test_config_path, "w") as f:
        f.write(test_yaml_content)
    return str(test_config_path)


@pytest.fixture(scope="session")
def test_config_manager_integration(test_integration_config_path):
    """Provides a ConfigManager instance loaded with the integration test config."""
    cm = ConfigManager(config_path=test_integration_config_path)
    # Ensure the config is loaded
    _ = cm.config
    return cm


# --- Qdrant & Elasticsearch Client Fixture (Session Scoped) ---
@pytest.fixture(scope="session")
async def qdrant_es_integration_clients(test_config_manager_integration: ConfigManager, event_loop: asyncio.AbstractEventLoop):
    """Provides Qdrant and Elasticsearch clients and handles cleanup for the session."""
    config = test_config_manager_integration.config

    q_client = QdrantClient(host=config.vector_store.host, port=config.vector_store.port)

    es_hosts = config.elasticsearch.hosts
    es_auth = None
    if config.elasticsearch.username and config.elasticsearch.password:
        es_auth = (config.elasticsearch.username, config.elasticsearch.password)
    es_client = AsyncElasticsearch(hosts=es_hosts, http_auth=es_auth)

    # Initial cleanup at the start of the session
    try:
        q_client.delete_collection(collection_name=config.vector_store.collection_name)
        print(f"SESSION SETUP: Deleted Qdrant collection {config.vector_store.collection_name} if it existed.")
    except Exception: pass

    try:
        if await es_client.indices.exists(index=config.elasticsearch.index_name):
            await es_client.indices.delete(index=config.elasticsearch.index_name)
            print(f"SESSION SETUP: Deleted ES index {config.elasticsearch.index_name} if it existed.")
    except ESNotFoundError: pass
    except Exception as e: print(f"SESSION SETUP: Error deleting ES index: {e}")


    # Recreate Qdrant collection based on config (if RAGPipeline doesn't do it or for safety)
    # RAGPipeline's async_initialize_components should handle this if recreate_collection is True.
    # For safety, we can do it here too.
    try:
        q_client.recreate_collection(
            collection_name=config.vector_store.collection_name,
            vectors_config=qdrant_models.VectorParams(size=config.vector_store.vector_size, distance=qdrant_models.Distance.COSINE)
        )
        print(f"SESSION SETUP: Recreated Qdrant collection {config.vector_store.collection_name}")
    except Exception as e:
        print(f"SESSION SETUP: Error recreating Qdrant collection {config.vector_store.collection_name}: {e}")
        # If it fails, maybe it's fine if RAGPipeline handles it, but good to know.

    yield q_client, es_client

    # Teardown at the end of the session
    try:
        q_client.delete_collection(collection_name=config.vector_store.collection_name)
        print(f"SESSION TEARDOWN: Deleted Qdrant collection {config.vector_store.collection_name}.")
    except Exception: pass

    try:
        if await es_client.indices.exists(index=config.elasticsearch.index_name):
            await es_client.indices.delete(index=config.elasticsearch.index_name)
            print(f"SESSION TEARDOWN: Deleted ES index {config.elasticsearch.index_name}.")
    except ESNotFoundError: pass
    except Exception as e: print(f"SESSION TEARDOWN: Error deleting ES index: {e}")

    q_client.close()
    await es_client.close()


# --- Authentication Fixtures for Integration Tests (Session Scoped) ---
@pytest.fixture(scope="session")
def app_auth_config_for_tokens(test_config_manager_integration: ConfigManager):
    """Provides an AppAuthConfig instance for generating test tokens."""
    auth_data = test_config_manager_integration.config.auth
    return AppAuthConfig(
        secret_key=auth_data.secret_key,
        token_expire_minutes=auth_data.token_expire_minutes,
        # Other fields can be default if not critical for token generation itself
    )

@pytest.fixture(scope="session")
def token_auth_manager(app_auth_config_for_tokens: AppAuthConfig):
    """A minimal AuthManager for generating tokens, not for full app auth simulation."""
    # This manager won't use Redis, just for minting tokens.
    # If tests require user creation via AuthManager, this needs to be more complete or use a mock Redis.
    # For now, we directly create AuthUser objects.
    manager = MagicMock(spec=AuthManager)
    manager.config = app_auth_config_for_tokens

    def create_access_token_side_effect(user: AuthUser, expires_delta=None):
        # Simplified token creation logic, copied from AuthManager
        expires = datetime.utcnow() + (
            expires_delta if expires_delta else timedelta(minutes=manager.config.token_expire_minutes)
        )
        to_encode = {
            "sub": user.id, "exp": expires,
            "permissions": [p.value for p in user.permissions], # Ensure enum values are used
            "roles": [r.value for r in user.roles]
        }
        return jwt.encode(to_encode, manager.config.secret_key, algorithm="HS256")

    manager.create_access_token = MagicMock(side_effect=create_access_token_side_effect)
    return manager


@pytest.fixture(scope="session")
def admin_user_for_tokens():
    return AuthUser(
        id="admin_integration_user", username="admin_integration", email="admin_int@test.com",
        hashed_password="hashed_password", # Not used for token minting here
        roles=[Role.ADMIN],
        permissions={p for p in Permission} # All permissions
    )

@pytest.fixture(scope="session")
def regular_user_for_tokens():
    return AuthUser(
        id="user_integration_user", username="user_integration", email="user_int@test.com",
        hashed_password="hashed_password",
        roles=[Role.USER],
        permissions={
            Permission.SEARCH_BASIC, Permission.DOCUMENT_READ,
            Permission.API_READ, Permission.API_EXECUTE, Permission.API_WRITE  # Added API_WRITE
        }
    )

@pytest.fixture(scope="session")
def admin_token(token_auth_manager: AuthManager, admin_user_for_tokens: AuthUser):
    return token_auth_manager.create_access_token(user=admin_user_for_tokens)

@pytest.fixture(scope="session")
def user_token(token_auth_manager: AuthManager, regular_user_for_tokens: AuthUser):
    return token_auth_manager.create_access_token(user=regular_user_for_tokens)


# --- Integration Test Client Fixture (Session Scoped) ---
@pytest.fixture(scope="session")
def integration_test_client(
    test_integration_config_path: str,
    qdrant_es_integration_clients: tuple, # Ensures DBs are set up and cleaned up at session level
    event_loop: asyncio.AbstractEventLoop # Ensures event loop is available
):
    """
    Provides a TestClient for enhanced_api.app, configured for integration tests.
    Uses real Qdrant and Elasticsearch. App startup events are run.
    Data seeding and per-test cleanup should be handled by a separate function-scoped fixture.
    """
    # Patch the config file path BEFORE actual_enhanced_app is used or its modules are too deeply imported.
    # This ensures that when `enhanced_api.py` creates its global `config_manager`, it uses our test config.
    with patch('Scripts.enhanced_api.config_file_path', test_integration_config_path):
        # The app instance `actual_enhanced_app` from `Scripts.enhanced_api` will now use the patched path
        # when its ConfigManager is initialized (typically at module level or early startup).

        # We need to ensure any global instances within enhanced_api are reset if they were loaded with prod config
        # For robust testing, it's often better if the app can be configured at instantiation or via env vars
        # rather than relying solely on a hardcoded module-level path.
        # Assuming the patch is effective for the TestClient's app instance.

        # The TestClient will run the app's startup events.
        client = TestClient(actual_enhanced_app)

        # At this point, actual_enhanced_app.state.rag_pipeline, .auth_manager etc.
        # should be initialized based on the test_integration_config_path.
        # We can add assertions here to verify if needed.
        # For example:
        # assert client.app.state.app_config.vector_store.collection_name == "test_integration_api_collection"

        yield client

        # TestClient's app shutdown events are typically handled by TestClient's exit.


# --- Data Seeding/Cleanup Fixture for API Integration Tests (Function Scoped) ---
@pytest.fixture(scope="function") # Changed to function scope for per-test isolation
async def seed_data_for_integration_api_test(integration_test_client: TestClient, event_loop: asyncio.AbstractEventLoop):
    """
    Seeds data into Qdrant and Elasticsearch using the RAGPipeline from the integration_test_client.
    Clears data before each test function.
    """
    app = integration_test_client.app
    rag_pipeline: RAGPipeline = app.state.rag_pipeline
    test_config: AppSystemConfig = app.state.app_config

    # --- Assert that the pipeline is using the test configuration ---
    assert rag_pipeline.collection_name == test_config.vector_store.collection_name
    if rag_pipeline.es_fallback:
        assert rag_pipeline.es_fallback.es_index_name == test_config.elasticsearch.index_name

    # --- Clear existing data in collections/indices for this test function ---
    # Qdrant: Delete all points from the collection
    try:
        # Get all points (this might be slow for large collections, but fine for test scale)
        # A better way if available is delete_points with a filter that matches all, or recreate.
        # Since recreate_collection is True in config and qdrant_es_integration_clients also does it,
        # the collection should be fresh at session start. For function scope, we delete points.
        scroll_response = rag_pipeline.client.scroll(collection_name=rag_pipeline.collection_name, limit=1000, with_payload=False, with_vectors=False)
        point_ids_to_delete = [p.id for p in scroll_response[0]]
        if point_ids_to_delete:
            rag_pipeline.client.delete_points(collection_name=rag_pipeline.collection_name, points_selector=point_ids_to_delete)
            print(f"FUNCTION SETUP: Cleared {len(point_ids_to_delete)} points from Qdrant collection {rag_pipeline.collection_name}")
    except Exception as e:
        print(f"FUNCTION SETUP: Error clearing Qdrant collection {rag_pipeline.collection_name}: {e}")
        # If collection doesn't exist yet (e.g. first test), this might fail.
        # The RAGPipeline's init should create it.

    # Elasticsearch: Delete all documents from the index
    if rag_pipeline.es_fallback and await rag_pipeline.es_fallback.es.indices.exists(index=rag_pipeline.es_fallback.es_index_name):
        try:
            await rag_pipeline.es_fallback.es.delete_by_query(
                index=rag_pipeline.es_fallback.es_index_name,
                body={"query": {"match_all": {}}},
                refresh=True,
                wait_for_completion=True, # Ensure it's done before proceeding
                ignore=[404] # Ignore if index not found (should exist from RAG init)
            )
            print(f"FUNCTION SETUP: Cleared documents from ES index {rag_pipeline.es_fallback.es_index_name}")
        except Exception as e:
            print(f"FUNCTION SETUP: Error clearing ES index {rag_pipeline.es_fallback.es_index_name}: {e}")


    # --- Seed new data ---
    # Using RAGPipeline's document processing and embedding for a true integration test
    # Create temporary files with SAMPLE_DOCS_CONTENT
    temp_docs_dir = Path(tempfile.mkdtemp(prefix="api_integ_docs_"))
    for doc_data in SAMPLE_DOCS_CONTENT:
        with open(temp_docs_dir / f"{doc_data['id']}.txt", "w") as f:
            f.write(doc_data['text'])
            # If metadata like category/timestamp needs to be in files, adjust format
            # For now, RAGPipeline's process_documents might not pick these up unless doc_processor is adapted.
            # The current RAGPipeline.process_documents takes a directory.

    print(f"FUNCTION SETUP: Seeding data from {temp_docs_dir}...")
    # Call the API endpoint for processing documents if possible, or use pipeline directly
    # For fixture setup, using pipeline directly is more straightforward.
    # The RAGPipeline instance on app.state should be fully initialized.
    await rag_pipeline.process_documents(input_dir=temp_docs_dir, batch_size=2)

    # Give ES time to index, crucial for subsequent searches in the same test function
    await asyncio.sleep(2)
    print(f"FUNCTION SETUP: Data seeding complete for {rag_pipeline.collection_name} and {rag_pipeline.es_fallback.es_index_name if rag_pipeline.es_fallback else 'N/A'}")

    yield # Test runs here

    # Cleanup temp_docs_dir
    shutil.rmtree(temp_docs_dir)
    print(f"FUNCTION TEARDOWN: Cleaned up temp docs dir {temp_docs_dir}")
    # Per-function data cleanup is handled at the start of the next test's seed_data fixture run.


# --- Unit Test Specific Mocks (from original test_enhanced_api.py, if needed globally or in other unit tests) ---
# These are more for unit tests, not integration tests, but can live in conftest.py

@pytest.fixture
def mock_rag_pipeline_unit(): # Renamed to avoid conflict if used alongside integration
    mock = MagicMock(spec=RAGPipeline)
    mock.search = AsyncMock(return_value=[{"text": "searched", "metadata": {}, "score": 0.9}])
    mock.process_documents = AsyncMock()
    mock.get_collection_info = MagicMock(return_value={"name": "test_coll", "points_count": 10})
    mock.generate_response = MagicMock(return_value={"answer": "generated_answer", "sources": []})
    mock.async_initialize_components = AsyncMock()
    return mock

@pytest.fixture
def mock_auth_manager_unit(): # Renamed
    mock = MagicMock(spec=AuthManager)
    mock.authenticate_user = AsyncMock()
    mock.create_access_token = MagicMock(return_value="test_jwt_token_unit")
    # For require_permission, the dependency itself will be mocked in specific tests
    return mock

@pytest.fixture
def mock_config_manager_unit(): # Renamed
    mock = MagicMock(spec=ConfigManager)
    # Provide a mock config object that also has Pydantic model structure if deep access is tested
    mock.config = MagicMock(spec=AppSystemConfig)
    mock.config.api = MagicMock(spec=AppAPIConfig) # Example nested mock
    mock.config.auth = MagicMock(spec=AppAuthConfig)
    mock.update_config = MagicMock()
    return mock

@pytest.fixture
def mock_copilot_agent_unit(): # Renamed
    mock = MagicMock(spec=CopilotAgent) # Assuming CopilotAgent is defined elsewhere
    mock.get_completion = AsyncMock(return_value=MagicMock(response="copilot_completed")) # Simplified

    async def mock_stream_gen(*args, **kwargs):
        yield "copilot "
        yield "stream "
        yield "response"
    mock.stream_completion = AsyncMock(return_value=mock_stream_gen())
    return mock

@pytest.fixture
def mock_integration_manager_unit(): # Renamed
    mock = AsyncMock() # Assuming IntegrationManager spec
    mock.optimize_query = AsyncMock(side_effect=lambda query, context: {"optimized_query": query, "metadata": {}})
    return mock

@pytest.fixture
def enhanced_api_client( # This is the UNIT TEST client from previous steps
    mock_rag_pipeline_unit, mock_auth_manager_unit, mock_config_manager_unit, event_loop: asyncio.AbstractEventLoop
):
    """Provides a TestClient for enhanced_api.app with key services mocked for UNIT testing."""

    # Patch config path for unit tests too, to ensure a known, minimal config if not overridden
    # This uses a different temp file than integration tests.
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", prefix="config_unit_") as tmp_config:
        # Basic YAML structure, can be expanded if unit tests need more specific config values
        tmp_config.write("""
version: 1
model: {embedding_model: "dummy-model", device: "cpu"}
vector_store: {host: "dummy_qdrant", port: 1234, collection_name: "unit_test_coll", vector_size: 10}
auth: {secret_key: "unit_test_secret", token_expire_minutes: 5}
paths: {cache_dir: "/tmp/unit_test_cache"}
# Add other minimal sections if startup complains
processing: {}
api: {}
cache_settings: {}
feature_flags: {enable_elasticsearch_fallback: False} # Keep simple for unit tests
elasticsearch: {}
        """)
        unit_test_config_path = tmp_config.name

    with patch('Scripts.enhanced_api.config_file_path', unit_test_config_path):
        # Create a new app instance for unit testing to avoid state leakage from integration tests
        # This requires enhanced_api.py to be structured to allow app creation on demand,
        # or careful management of the global `actual_enhanced_app`.
        # For simplicity, we assume TestClient(actual_enhanced_app) is okay if state is managed.

        client = TestClient(actual_enhanced_app)

        # Override app.state for unit tests
        client.app.state.rag_pipeline = mock_rag_pipeline_unit
        client.app.state.auth_manager = mock_auth_manager_unit
        client.app.state.config_manager = mock_config_manager_unit
        client.app.state.app_config = mock_config_manager_unit.config # Make sure mocked config is used

        # Mock other dependencies if necessary for unit tests of specific endpoints
        # e.g., client.app.dependency_overrides[get_some_dependency] = lambda: mock_some_dependency

        yield client # Unit test runs

    os.remove(unit_test_config_path) # Clean up unit test temp config
    # Clean up dependency_overrides if any were set directly on client.app
    actual_enhanced_app.dependency_overrides = {}

# --- Old fixtures from conftest.py (review and remove/integrate if redundant) ---
# The following are from the originally provided conftest.py.
# They might be redundant or need adaptation.

# @pytest.fixture
# def test_config(): ... # Likely replaced by test_config_manager_integration

# @pytest.fixture
# async def db_engine(test_config): ... # Not used by RAG system as per current code

# @pytest.fixture
# async def db_session(db_engine): ... # Not used

# @pytest.fixture
# async def qdrant_client(test_config): ... # Replaced by qdrant_es_integration_clients

# @pytest.fixture
# def temp_storage(test_config): ... # paths.cache_dir is now part of YAML config

# @pytest.fixture
# def temp_log_dir(test_config): ... # paths.log_dir is now part of YAML config

# @pytest.fixture
# def temp_backup_dir(test_config): ... # Not used

# @pytest.fixture
# def sample_document(): ... # Replaced by SAMPLE_DOCS_CONTENT

# @pytest.fixture
# def sample_user_data(): ... # Replaced by admin_user_for_tokens, regular_user_for_tokens

# @pytest.fixture
# def mock_openai(mocker): ... # Specific to OpenAI, embedding provider is more generic now

# @pytest.fixture
# def mock_s3(mocker): ... # If S3 integration is tested, this might be useful
