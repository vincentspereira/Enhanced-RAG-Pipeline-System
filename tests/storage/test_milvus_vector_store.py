import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType, MilvusException

from Scripts.storage.vector_store_base import DocumentChunk, SearchResult
from Scripts.storage.milvus_vector_store import MilvusVectorStore

# Mock targets
MILVUS_CONNECTIONS_TARGET = "Scripts.storage.milvus_vector_store.connections"
MILVUS_UTILITY_TARGET = "Scripts.storage.milvus_vector_store.utility"
MILVUS_COLLECTION_TARGET = "Scripts.storage.milvus_vector_store.Collection"
MILVUS_COLLECTIONSCHEMA_TARGET = "Scripts.storage.milvus_vector_store.CollectionSchema"
MILVUS_FIELDSCHEMA_TARGET = "Scripts.storage.milvus_vector_store.FieldSchema"


@pytest.fixture
def mock_milvus_dependencies(mocker):
    """Mocks all pymilvus dependencies."""
    mock_connections = mocker.patch(MILVUS_CONNECTIONS_TARGET, spec=connections)
    mock_utility = mocker.patch(MILVUS_UTILITY_TARGET, spec=utility)
    mock_Collection = mocker.patch(MILVUS_COLLECTION_TARGET, spec=Collection) # Mock the class itself

    # Mock Collection instance methods that will be returned by mock_Collection()
    mock_collection_instance = MagicMock(spec=Collection)
    mock_collection_instance.has_index.return_value = True # Assume index exists for simplicity here
    mock_collection_instance.load = MagicMock()
    mock_collection_instance.insert = MagicMock()
    mock_collection_instance.flush = MagicMock()
    mock_collection_instance.search = MagicMock()
    mock_collection_instance.delete = MagicMock()
    mock_collection_instance.schema = MagicMock(spec=CollectionSchema)
    mock_collection_instance.name = "TestCollection"
    mock_collection_instance.description = "Test Description"
    mock_collection_instance.num_entities = 0
    mock_collection_instance.primary_field = MagicMock(name="doc_id")
    mock_collection_instance.indexes = []
    mock_collection_instance.is_empty = True
    mock_collection_instance.create_index = MagicMock()


    mock_Collection.return_value = mock_collection_instance # When Collection() is called, return this mock instance

    # Mock utility functions
    mock_utility.has_collection.return_value = False # Default: collection does not exist
    mock_utility.list_connections.return_value = [("default", MagicMock())] # Simulate default connection exists
    mock_utility.list_collections.return_value = [] # Simulate no collections by default

    # Mock Schema classes if they are directly used in the connector for type hinting or instantiation
    mocker.patch(MILVUS_COLLECTIONSCHEMA_TARGET, spec=CollectionSchema)
    mocker.patch(MILVUS_FIELDSCHEMA_TARGET, spec=FieldSchema)

    return {
        "connections": mock_connections,
        "utility": mock_utility,
        "Collection": mock_Collection, # The mocked class
        "collection_instance": mock_collection_instance # The instance returned by Collection()
    }

@pytest.fixture
async def milvus_store(mock_milvus_dependencies):
    """Provides a MilvusVectorStore instance with mocked pymilvus."""
    store = MilvusVectorStore(host="mockhost", port="19530", alias="test_milvus_alias")
    yield store
    # MilvusVectorStore.close() calls utility.disconnect, which is mocked.
    # No explicit async cleanup needed here for mocks.

@pytest.mark.asyncio
async def test_milvus_store_initialization_success(milvus_store, mock_milvus_dependencies):
    """Test successful initialization of MilvusVectorStore."""
    mock_connections = mock_milvus_dependencies["connections"]

    assert milvus_store.alias == "test_milvus_alias"
    mock_connections.connect.assert_called_once_with(
        alias="test_milvus_alias", host="mockhost", port="19530", user=None, password=None, secure=False
    )

@pytest.mark.asyncio
async def test_milvus_store_initialization_failure(mock_milvus_dependencies):
    """Test initialization failure."""
    mock_connections = mock_milvus_dependencies["connections"]
    mock_connections.connect.side_effect = MilvusException("Connection failed")

    # Expect MilvusException to be caught and logged, not re-raised by __init__
    store = MilvusVectorStore(host="failedhost", port="19530", alias="fail_alias")
    assert store.alias == "fail_alias" # Init proceeds but connection fails
    # Health check would be false
    assert not await store.health_check()


@pytest.mark.asyncio
async def test_initialize_collection_creates_if_not_exists(milvus_store, mock_milvus_dependencies):
    """Test collection creation during initialize."""
    mock_utility = mock_milvus_dependencies["utility"]
    mock_Collection = mock_milvus_dependencies["Collection"] # The mocked class
    mock_collection_instance = mock_milvus_dependencies["collection_instance"]

    mock_utility.has_collection.return_value = False # Collection does not exist

    collection_name = "MyMilvusCol"
    vector_size = 768
    distance_metric = "IP"

    await milvus_store.initialize(collection_name, vector_size, distance_metric)

    mock_utility.has_collection.assert_called_with(collection_name, using="test_milvus_alias")
    # Check CollectionSchema and FieldSchema were called (indirectly via Collection constructor)
    # This requires deeper mocking of Collection's __init__ or checking args to mock_Collection
    # For now, verify Collection was instantiated and index created
    mock_Collection.assert_called_with(collection_name, schema=ANY, using="test_milvus_alias")
    mock_collection_instance.create_index.assert_called_once()
    mock_collection_instance.load.assert_called_once()
    # Check metric type in index_params
    args, kwargs = mock_collection_instance.create_index.call_args
    assert kwargs['index_params']['metric_type'] == "IP"


@pytest.mark.asyncio
async def test_add_documents(milvus_store, mock_milvus_dependencies):
    """Test adding documents."""
    mock_collection_instance = mock_milvus_dependencies["collection_instance"]
    mock_mutation_result = MagicMock()
    mock_mutation_result.insert_count = 2
    mock_mutation_result.primary_keys = ["id1_uuid_str", "id2_uuid_str"] # Milvus returns string PKs
    mock_collection_instance.insert.return_value = mock_mutation_result

    doc_id_1 = uuid.uuid4()
    docs = [
        DocumentChunk(id=doc_id_1, text="Doc 1 for Milvus", vector=[0.1]*5, metadata={"type": "A"}),
        DocumentChunk(id="doc2_plain_id", text="Doc 2 for Milvus", vector=[0.2]*5, metadata={"type": "B"})
    ]
    collection_name = "MyMilvusCol"

    added_ids = await milvus_store.add_documents(collection_name, docs)

    mock_collection_instance.insert.assert_called_once()
    # Check data format passed to insert (list of lists/rows)
    # [[doc_id_str, vector, text_str, metadata_json_str], ...]
    inserted_data = mock_collection_instance.insert.call_args[0][0]
    assert len(inserted_data) == 2
    assert inserted_data[0][0] == str(doc_id_1)
    assert inserted_data[1][0] == "doc2_plain_id"
    assert inserted_data[0][2] == "Doc 1 for Milvus"
    assert json.loads(inserted_data[1][3]) == {"type": "B"}

    mock_collection_instance.flush.assert_called_once()
    assert added_ids == [doc_id_1, "doc2_plain_id"]


@pytest.mark.asyncio
async def test_search_documents(milvus_store, mock_milvus_dependencies):
    """Test searching documents."""
    mock_collection_instance = mock_milvus_dependencies["collection_instance"]
    collection_name = "MyMilvusCol"
    query_vector = [0.3]*5
    top_k = 2

    # Mock Milvus search response structure
    # It's a list of SearchResult objects, where each SearchResult contains a list of Hit objects
    mock_hit1_entity = {"doc_id": "hit_uuid1", "text": "Found doc 1", "metadata_json": '{"topic":"milvus"}'}
    mock_hit1 = MagicMock(distance=0.8, entity=mock_hit1_entity)
    mock_hit2_entity = {"doc_id": "hit_uuid2", "text": "Found doc 2", "metadata_json": '{"topic":"vector"}'}
    mock_hit2 = MagicMock(distance=0.7, entity=mock_hit2_entity)

    milvus_search_result_mock = [[mock_hit1, mock_hit2]] # List containing one SearchResult (for one query_vector)
    mock_collection_instance.search.return_value = milvus_search_result_mock

    results = await milvus_store.search(collection_name, query_vector, top_k, filters={"metadata.topic": "milvus"})

    mock_collection_instance.search.assert_called_once()
    search_args, search_kwargs = mock_collection_instance.search.call_args
    assert search_kwargs['data'] == [query_vector]
    assert search_kwargs['anns_field'] == "vector"
    assert search_kwargs['limit'] == top_k
    assert "metadata_json like '%\"topic\": \"milvus\"%'" in search_kwargs['expr'] # Basic filter check
    assert search_kwargs['output_fields'] == ["doc_id", "text", "metadata_json"]

    assert len(results) == 2
    assert results[0].id == "hit_uuid1"
    assert pytest.approx(results[0].score) == 0.8 # Milvus returns distance
    assert results[0].payload["text"] == "Found doc 1"
    assert results[0].payload["metadata"] == {"topic": "milvus"}


@pytest.mark.asyncio
async def test_delete_documents(milvus_store, mock_milvus_dependencies):
    """Test deleting documents."""
    mock_collection_instance = mock_milvus_dependencies["collection_instance"]
    mock_delete_result = MagicMock() # Potentially a MutationResult
    mock_collection_instance.delete.return_value = mock_delete_result

    collection_name = "MyMilvusCol"
    doc_ids_to_delete = [str(uuid.uuid4()), "doc_id_xyz"]

    success = await milvus_store.delete_documents(collection_name, doc_ids_to_delete)

    assert success
    mock_collection_instance.delete.assert_called_once()
    # Check the expression for deletion
    delete_expr = mock_collection_instance.delete.call_args[0][0]
    assert f"doc_id in ['{doc_ids_to_delete[0]}','{doc_ids_to_delete[1]}']" in delete_expr
    mock_collection_instance.flush.assert_called_once()


@pytest.mark.asyncio
async def test_health_check(milvus_store, mock_milvus_dependencies):
    """Test health check functionality."""
    mock_utility = mock_milvus_dependencies["utility"]

    mock_utility.list_collections.return_value = ["some_collection"] # Simulate healthy response
    assert await milvus_store.health_check() is True
    mock_utility.list_collections.assert_called_with(using="test_milvus_alias")

    mock_utility.list_collections.side_effect = MilvusException("Cannot connect")
    assert await milvus_store.health_check() is False

    # Test ensure_connected path in health_check
    mock_utility.list_connections.return_value = [] # Simulate no active connection initially
    mock_connections = mock_milvus_dependencies["connections"]
    mock_utility.list_collections.side_effect = None # Clear side effect
    mock_utility.list_collections.return_value = ["some_collection"]

    assert await milvus_store.health_check() is True
    mock_connections.connect.assert_called() # Should have been called by _ensure_connected


# Note: get_collection_info test can be added, similar to other stores,
# mocking Collection attributes and utility.has_collection.
# For brevity, focusing on core CRUD and search.
