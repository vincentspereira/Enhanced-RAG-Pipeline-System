import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection # Type hint
from chromadb.config import Settings as ChromaSettings

from Scripts.storage.vector_store_base import DocumentChunk, SearchResult
from Scripts.storage.chroma_vector_store import ChromaVectorStore

# Mock targets
CHROMA_HTTP_CLIENT_TARGET = "Scripts.storage.chroma_vector_store.chromadb.HttpClient"
CHROMA_PERSISTENT_CLIENT_TARGET = "Scripts.storage.chroma_vector_store.chromadb.PersistentClient"
CHROMA_EPHEMERAL_CLIENT_TARGET = "Scripts.storage.chroma_vector_store.chromadb.EphemeralClient"


@pytest.fixture
def mock_chroma_client_instance():
    """Mocks a Chroma client instance (generic for HttpClient, PersistentClient, EphemeralClient)."""
    mock_client = MagicMock() # Can be spec'd with one of the client types if needed
    mock_client.heartbeat = MagicMock(return_value=123456789) # Simulate successful heartbeat

    mock_collection_instance = MagicMock(spec=ChromaCollection)
    mock_collection_instance.name = "TestCollection"
    mock_collection_instance.id = uuid.uuid4()
    mock_collection_instance.count.return_value = 0
    mock_collection_instance.metadata = {"hnsw:space": "cosine"}

    mock_collection_instance.add = MagicMock()
    # Mock query to return the expected structure
    mock_collection_instance.query.return_value = {
        'ids': [[]], 'distances': [[]], 'metadatas': [[]],
        'documents': [[]], 'embeddings': [[]], 'uris': None, 'data': None
    }
    mock_collection_instance.delete = MagicMock()

    mock_client.get_or_create_collection.return_value = mock_collection_instance
    mock_client.get_collection.return_value = mock_collection_instance

    return mock_client

@pytest.fixture
async def chroma_store_ephemeral(mocker, mock_chroma_client_instance):
    """Provides a ChromaVectorStore (Ephemeral) instance with mocked client."""
    mocker.patch(CHROMA_EPHEMERAL_CLIENT_TARGET, return_value=mock_chroma_client_instance)
    store = ChromaVectorStore() # Defaults to EphemeralClient
    yield store

@pytest.fixture
async def chroma_store_persistent(mocker, mock_chroma_client_instance):
    """Provides a ChromaVectorStore (Persistent) instance with mocked client."""
    mocker.patch(CHROMA_PERSISTENT_CLIENT_TARGET, return_value=mock_chroma_client_instance)
    with patch("Scripts.storage.chroma_vector_store.os.makedirs"): # Mock makedirs for path
        store = ChromaVectorStore(path="./test_chroma_data")
    yield store

@pytest.fixture
async def chroma_store_http(mocker, mock_chroma_client_instance):
    """Provides a ChromaVectorStore (HTTP) instance with mocked client."""
    mocker.patch(CHROMA_HTTP_CLIENT_TARGET, return_value=mock_chroma_client_instance)
    store = ChromaVectorStore(host="localhost", port=8000)
    yield store


@pytest.mark.asyncio
@pytest.mark.parametrize("get_store", ["chroma_store_ephemeral", "chroma_store_persistent", "chroma_store_http"])
async def test_chroma_store_initialization(get_store, request):
    """Test successful initialization for different Chroma client types."""
    store: ChromaVectorStore = request.getfixturevalue(get_store)
    assert store.client is not None
    store.client.heartbeat.assert_called_once()

@pytest.mark.asyncio
async def test_chroma_store_initialization_failure(mocker):
    """Test initialization failure if heartbeat fails."""
    mock_failing_client = MagicMock()
    mock_failing_client.heartbeat.side_effect = Exception("Connection failed")
    mocker.patch(CHROMA_EPHEMERAL_CLIENT_TARGET, return_value=mock_failing_client)

    store = ChromaVectorStore()
    assert store.client is None # Should fail to initialize fully


@pytest.mark.asyncio
async def test_initialize_collection(chroma_store_ephemeral: ChromaVectorStore):
    """Test collection creation/retrieval during initialize."""
    store = chroma_store_ephemeral
    collection_name = "MyChromaCol"
    vector_size = 128 # Not directly used by Chroma for collection creation if providing embeddings
    distance_metric = "l2"

    await store.initialize(collection_name, vector_size, distance_metric)

    store.client.get_or_create_collection.assert_called_once_with(
        name=collection_name,
        metadata={"hnsw:space": "l2"}, # Default collection_metadata is {}
        embedding_function=None      # Default embedding_function is None
    )

@pytest.mark.asyncio
async def test_add_documents(chroma_store_ephemeral: ChromaVectorStore):
    """Test adding documents to Chroma."""
    store = chroma_store_ephemeral
    mock_collection = store.client.get_or_create_collection.return_value # Get the mocked collection

    doc_id_1 = str(uuid.uuid4())
    docs = [
        DocumentChunk(id=doc_id_1, text="Chroma Doc 1", vector=[0.1]*10, metadata={"source": "S1"}),
        DocumentChunk(id="doc2", text="Chroma Doc 2", vector=[0.2]*10, metadata={"source": "S2"})
    ]
    collection_name = "MyChromaCol" # Name used when mock_collection was set up

    added_ids = await store.add_documents(collection_name, docs)

    store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.add.assert_called_once()

    call_kwargs = mock_collection.add.call_args[1]
    assert call_kwargs['ids'] == [doc_id_1, "doc2"]
    assert len(call_kwargs['embeddings']) == 2
    assert call_kwargs['embeddings'][0] == [0.1]*10
    assert call_kwargs['metadatas'] == [{"source": "S1"}, {"source": "S2"}]
    assert call_kwargs['documents'] == ["Chroma Doc 1", "Chroma Doc 2"]

    assert added_ids == [doc_id_1, "doc2"]

@pytest.mark.asyncio
async def test_search_documents(chroma_store_ephemeral: ChromaVectorStore):
    """Test searching documents."""
    store = chroma_store_ephemeral
    mock_collection = store.client.get_collection.return_value

    collection_name = "MyChromaCol"
    query_vector = [0.5]*10
    top_k = 1
    filters = {"source": "S1"}

    # Setup mock query response
    mock_collection.query.return_value = {
        'ids': [[str(uuid.uuid4())]],
        'distances': [[0.123]],
        'metadatas': [[{"source": "S1"}]],
        'documents': [["Searched Chroma Doc"]],
        'embeddings': [[[0.15]*10]] # Example if with_vectors=True
    }

    results = await store.search(collection_name, query_vector, top_k, filters, with_vectors=True)

    store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.query.assert_called_once_with(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=filters,
        include=["metadatas", "documents", "distances", "embeddings"]
    )

    assert len(results) == 1
    res = results[0]
    assert isinstance(res.id, str)
    assert pytest.approx(res.score) == 1.0 - 0.123 # Assuming cosine, score = 1 - distance
    assert res.payload["text"] == "Searched Chroma Doc"
    assert res.payload["metadata"] == {"source": "S1"}
    assert res.vector == [0.15]*10


@pytest.mark.asyncio
async def test_delete_documents(chroma_store_ephemeral: ChromaVectorStore):
    """Test deleting documents."""
    store = chroma_store_ephemeral
    mock_collection = store.client.get_collection.return_value
    collection_name = "MyChromaCol"
    doc_ids_to_delete = [str(uuid.uuid4()), "doc_id_xyz"]

    success = await store.delete_documents(collection_name, doc_ids_to_delete)

    assert success
    store.client.get_collection.assert_called_once_with(name=collection_name)
    mock_collection.delete.assert_called_once_with(ids=doc_ids_to_delete)


@pytest.mark.asyncio
async def test_get_collection_info(chroma_store_ephemeral: ChromaVectorStore):
    """Test getting collection info."""
    store = chroma_store_ephemeral
    mock_collection = store.client.get_collection.return_value # This is the spec'd one

    # Configure the mock collection that get_collection returns
    mock_collection.name = "MyChromaColInfo"
    mock_collection.id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    mock_collection.count.return_value = 150
    mock_collection.metadata = {"hnsw:space": "l2", "custom_meta": "value"}

    info = await store.get_collection_info("MyChromaColInfo")

    assert info["name"] == "MyChromaColInfo"
    assert info["id"] == "12345678-1234-5678-1234-567812345678"
    assert info["count"] == 150
    assert info["metadata"] == {"hnsw:space": "l2", "custom_meta": "value"}

@pytest.mark.asyncio
async def test_health_check(chroma_store_ephemeral: ChromaVectorStore):
    """Test health check functionality."""
    store = chroma_store_ephemeral

    store.client.heartbeat.return_value = 987654321 # Simulate success
    assert await store.health_check() is True

    store.client.heartbeat.side_effect = Exception("Heartbeat failed")
    assert await store.health_check() is False
