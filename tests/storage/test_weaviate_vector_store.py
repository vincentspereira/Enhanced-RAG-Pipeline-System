import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import weaviate # For exceptions and model types if needed for mocks
from weaviate.exceptions import WeaviateStartUpError, WeaviateQueryError

# Assuming vector_store_base and weaviate_vector_store are in Scripts/storage/
from Scripts.storage.vector_store_base import DocumentChunk, SearchResult
from Scripts.storage.weaviate_vector_store import WeaviateVectorStore

@pytest.fixture
def mock_weaviate_client_constructor():
    """Mocks the weaviate.Client constructor."""
    with patch("Scripts.storage.weaviate_vector_store.weaviate.Client") as mock_constructor:
        mock_client_instance = MagicMock(spec=weaviate.Client)
        # Simulate client being ready by default
        mock_client_instance.is_ready.return_value = True

        # Mock schema attribute and its methods
        mock_client_instance.schema = MagicMock()
        mock_client_instance.schema.exists = MagicMock(return_value=False) # Default: class does not exist
        mock_client_instance.schema.create_class = AsyncMock() # create_class is not async, but methods called by it might be
                                                             # Actually, v3 client schema methods are not async.
        mock_client_instance.schema.create_class = MagicMock()


        # Mock batch attribute and its context manager methods
        mock_batch_instance = MagicMock()
        mock_batch_instance.__enter__.return_value = mock_batch_instance # For 'with client.batch(...)'
        mock_batch_instance.__exit__.return_value = None
        mock_batch_instance.add_data_object = MagicMock()
        mock_batch_instance.num_errors = 0 # Default to no errors
        mock_client_instance.batch = MagicMock(return_value=mock_batch_instance)

        # Mock query attribute
        mock_query_instance = MagicMock()
        mock_client_instance.query = mock_query_instance
        # Chain mock for query methods like get().with_near_vector().with_limit().do()
        mock_query_instance.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do = AsyncMock(return_value={
            "data": {"Get": {"TestClass": []}} # Default empty result
        })
        # For v3 client query methods are not async.
        mock_query_instance.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do = MagicMock(return_value={
            "data": {"Get": {"TestClass": []}}
        })


        # Mock data_object attribute for deletion
        mock_client_instance.data_object = MagicMock()
        mock_client_instance.data_object.delete = MagicMock()

        mock_constructor.return_value = mock_client_instance
        yield mock_constructor, mock_client_instance

@pytest.fixture
async def weaviate_store(mock_weaviate_client_constructor):
    """Provides a WeaviateVectorStore instance with mocked client."""
    store = WeaviateVectorStore(url="http://mock-weaviate:8080", api_key="mock_key")
    # The client is initialized in __init__. We need to ensure the mock is used.
    # The mock_weaviate_client_constructor fixture already patches weaviate.Client
    # so the instance created within WeaviateVectorStore will be the mock.
    yield store
    # No explicit close in WeaviateVectorStore, so no cleanup here.

@pytest.mark.asyncio
async def test_weaviate_store_initialization_success(weaviate_store, mock_weaviate_client_constructor):
    """Test successful initialization of WeaviateVectorStore."""
    _, mock_client_instance = mock_weaviate_client_constructor
    assert weaviate_store.client == mock_client_instance
    mock_client_instance.is_ready.assert_called_once()

@pytest.mark.asyncio
async def test_weaviate_store_initialization_not_ready(mock_weaviate_client_constructor):
    """Test initialization when Weaviate instance is not ready."""
    mock_constructor, mock_client_instance = mock_weaviate_client_constructor
    mock_client_instance.is_ready.return_value = False # Simulate not ready

    with pytest.raises(WeaviateStartUpError): # Assuming constructor re-raises or specific error if not ready
         # Re-initializing to trigger the error path
         WeaviateVectorStore(url="http://mock-weaviate:8080")


@pytest.mark.asyncio
async def test_initialize_collection_creates_class_if_not_exists(weaviate_store, mock_weaviate_client_constructor):
    """Test class creation during initialize if it doesn't exist."""
    _, mock_client = mock_weaviate_client_constructor
    mock_client.schema.exists.return_value = False # Class does not exist

    class_name = "MyRagClass"
    vector_size = 128 # Not directly used by Weaviate schema if vectorizer='none'

    await weaviate_store.initialize(class_name, vector_size, distance_metric="cosine")

    mock_client.schema.exists.assert_called_once_with(class_name)
    mock_client.schema.create_class.assert_called_once()
    # More detailed check of class_obj passed to create_class
    args, _ = mock_client.schema.create_class.call_args
    created_class_obj = args[0]
    assert created_class_obj["class"] == class_name
    assert created_class_obj["vectorizer"] == "none"
    assert created_class_obj["vectorIndexConfig"]["distance"] == "cosine"

@pytest.mark.asyncio
async def test_initialize_collection_uses_existing_class(weaviate_store, mock_weaviate_client_constructor):
    """Test initialize does not try to create class if it already exists."""
    _, mock_client = mock_weaviate_client_constructor
    mock_client.schema.exists.return_value = True # Class exists

    await weaviate_store.initialize("ExistingClass", 128)

    mock_client.schema.create_class.assert_not_called()

@pytest.mark.asyncio
async def test_add_documents(weaviate_store, mock_weaviate_client_constructor):
    """Test adding documents to Weaviate."""
    _, mock_client = mock_weaviate_client_constructor
    mock_batch_instance = mock_client.batch.return_value.__enter__.return_value

    doc_id_1 = uuid.uuid4()
    docs = [
        DocumentChunk(id=doc_id_1, text="Doc 1", vector=[0.1]*10, metadata={"source": "A"}),
        DocumentChunk(id="not-a-uuid", text="Doc 2", vector=[0.2]*10, metadata={"source": "B"})
    ]
    class_name = "MyRagClass"

    added_ids = await weaviate_store.add_documents(class_name, docs)

    assert mock_batch_instance.add_data_object.call_count == 2
    # Check first call (with UUID)
    call_args1 = mock_batch_instance.add_data_object.call_args_list[0][1] # kwargs of first call
    assert call_args1['class_name'] == class_name
    assert call_args1['uuid'] == doc_id_1
    assert call_args1['vector'] == [0.1]*10
    # Check second call (generates new UUID)
    call_args2 = mock_batch_instance.add_data_object.call_args_list[1][1]
    assert isinstance(call_args2['uuid'], uuid.UUID) # Should have generated a UUID
    assert call_args2['uuid'] != "not-a-uuid"

    assert len(added_ids) == 2
    assert doc_id_1 in added_ids
    assert added_ids[1] == call_args2['uuid'] # Ensure the generated UUID is returned


@pytest.mark.asyncio
async def test_search_documents(weaviate_store, mock_weaviate_client_constructor):
    """Test searching documents."""
    _, mock_client = mock_weaviate_client_constructor
    class_name = "MyRagClass"
    query_vector = [0.5]*10
    top_k = 3

    # Mock the chained query building and response
    mock_do_method = mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value.do
    mock_do_method.return_value = {
        "data": {
            "Get": {
                class_name: [
                    {"text": "Result 1", "metadata_json": '{"src":"X"}', "_additional": {"id": "uuid1", "certainty": 0.9, "vector": [0.1]*10}},
                    {"text": "Result 2", "metadata_json": '{"src":"Y"}', "_additional": {"id": "uuid2", "certainty": 0.8}},
                ]
            }
        }
    }

    results = await weaviate_store.search(class_name, query_vector, top_k, with_vectors=True)

    mock_client.query.get.assert_called_with(class_name, ["text", "metadata_json"])
    mock_client.query.get.return_value.with_near_vector.assert_called_with({"vector": query_vector})
    mock_client.query.get.return_value.with_near_vector.return_value.with_limit.assert_called_with(top_k)
    mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.assert_called_with(["certainty", "id", "vector"])

    assert len(results) == 2
    assert results[0].id == "uuid1"
    assert pytest.approx(results[0].score) == 0.9
    assert results[0].payload["text"] == "Result 1"
    assert results[0].payload["metadata"] == {"src": "X"}
    assert results[0].vector == [0.1]*10
    assert results[1].payload["metadata"] == {"src": "Y"}


@pytest.mark.asyncio
async def test_delete_documents(weaviate_store, mock_weaviate_client_constructor):
    """Test deleting documents."""
    _, mock_client = mock_weaviate_client_constructor
    class_name = "MyRagClass"
    doc_id_uuid_str = str(uuid.uuid4())
    doc_ids_to_delete = [doc_id_uuid_str, "invalid-uuid-string", str(uuid.uuid4())]

    success = await weaviate_store.delete_documents(class_name, doc_ids_to_delete)

    assert success # Should be true if at least one valid UUID deletion was attempted
    # delete should be called for valid UUIDs
    assert mock_client.data_object.delete.call_count == 2
    mock_client.data_object.delete.assert_any_call(uuid=doc_id_uuid_str, class_name=class_name, consistency_level=ANY)
    # The third ID is also a valid UUID string
    mock_client.data_object.delete.assert_any_call(uuid=doc_ids_to_delete[2], class_name=class_name, consistency_level=ANY)


@pytest.mark.asyncio
async def test_health_check(weaviate_store, mock_weaviate_client_constructor):
    """Test health check functionality."""
    _, mock_client = mock_weaviate_client_constructor

    mock_client.is_ready.return_value = True
    assert await weaviate_store.health_check() is True

    mock_client.is_ready.return_value = False
    assert await weaviate_store.health_check() is False

    mock_client.is_ready.side_effect = Exception("Connection error")
    assert await weaviate_store.health_check() is False

# Note: Tests for get_collection_info can be added, mocking client.schema.get()
# and potentially client.query.aggregate() for object counts.
# For brevity here, focusing on core CRUD and search.
