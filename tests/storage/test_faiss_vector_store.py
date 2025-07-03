import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import numpy as np
import faiss # For faiss.Index and exceptions
import os
import pickle

from Scripts.storage.vector_store_base import DocumentChunk, SearchResult
from Scripts.storage.faiss_vector_store import FaissVectorStore

FAISS_READ_INDEX_TARGET = "Scripts.storage.faiss_vector_store.faiss.read_index"
FAISS_WRITE_INDEX_TARGET = "Scripts.storage.faiss_vector_store.faiss.write_index"
FAISS_INDEXFLATL2_TARGET = "Scripts.storage.faiss_vector_store.faiss.IndexFlatL2"
FAISS_INDEXFLATIP_TARGET = "Scripts.storage.faiss_vector_store.faiss.IndexFlatIP"
PICKLE_LOAD_TARGET = "Scripts.storage.faiss_vector_store.pickle.load"
PICKLE_DUMP_TARGET = "Scripts.storage.faiss_vector_store.pickle.dump"
OS_PATH_EXISTS_TARGET = "Scripts.storage.faiss_vector_store.os.path.exists"
OS_MAKEDIRS_TARGET = "Scripts.storage.faiss_vector_store.os.makedirs"


@pytest.fixture
def mock_faiss_index():
    """Mocks a faiss.Index instance."""
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 0 # Dimension, will be set in initialize
    mock_index.ntotal = 0 # Number of vectors
    mock_index.add = MagicMock()
    mock_index.search = MagicMock(return_value=(np.array([[]]), np.array([[]]))) # Distances, Indices
    mock_index.is_trained = True # For IndexFlat, it's always trained
    return mock_index

@pytest.fixture
def faiss_store_new(mocker, mock_faiss_index):
    """Provides a FaissVectorStore instance for a new (non-existent) index."""
    mocker.patch(OS_PATH_EXISTS_TARGET, return_value=False) # Simulate files not existing
    mocker.patch(OS_MAKEDIRS_TARGET) # Mock makedirs

    # Mock index constructors that might be called in initialize
    mocker.patch(FAISS_INDEXFLATL2_TARGET, return_value=mock_faiss_index)
    mocker.patch(FAISS_INDEXFLATIP_TARGET, return_value=mock_faiss_index)

    # Mock pickle dump for _save_store during placeholder creation or init
    mocker.patch(PICKLE_DUMP_TARGET)
    mocker.patch(FAISS_WRITE_INDEX_TARGET) # Mock write_index for _save_store

    store = FaissVectorStore(index_file_path="./test_faiss.index", metadata_file_path="./test_faiss_meta.pkl")
    # Initialize sets placeholder index which is None, actual index created in .initialize()
    return store

@pytest.fixture
def faiss_store_loaded(mocker, mock_faiss_index):
    """Provides a FaissVectorStore instance simulating a loaded index."""
    mocker.patch(OS_PATH_EXISTS_TARGET, return_value=True) # Simulate files exist
    mocker.patch(OS_MAKEDIRS_TARGET)

    mock_faiss_index.d = 128 # Simulate loaded dimension
    mock_faiss_index.ntotal = 10 # Simulate loaded vectors
    mocker.patch(FAISS_READ_INDEX_TARGET, return_value=mock_faiss_index)

    loaded_meta = {
        'doc_id_to_index_pos': {"doc1": 0},
        'index_pos_to_doc_id': {0: "doc1"},
        'document_texts': ["Test text"],
        'document_metadatas': [{"source": "test"}],
        'vector_size': 128,
        'distance_metric': "L2"
    }
    mocker.patch(PICKLE_LOAD_TARGET, return_value=loaded_meta)

    store = FaissVectorStore(index_file_path="./loaded.index", metadata_file_path="./loaded_meta.pkl")
    return store


@pytest.mark.asyncio
async def test_faiss_store_initialization_new(faiss_store_new: FaissVectorStore, mock_faiss_index):
    """Test initialization when index files do not exist."""
    assert faiss_store_new.index is None # Placeholder initially
    assert faiss_store_new.vector_size is None

    # Now call initialize to create the actual index
    collection_name = "new_collection"
    vector_dim = 64
    await faiss_store_new.initialize(collection_name, vector_dim, distance_metric="L2")

    assert faiss_store_new.index is not None # Should be the mocked faiss_index
    assert faiss_store_new.index.d == vector_dim
    assert faiss_store_new.vector_size == vector_dim
    assert faiss_store_new.distance_metric == "L2"
    # _save_store should have been called
    assert پتцһ(FAISS_WRITE_INDEX_TARGET).called
    assert پتцһ(PICKLE_DUMP_TARGET).called


@pytest.mark.asyncio
async def test_faiss_store_initialization_loaded(faiss_store_loaded: FaissVectorStore, mock_faiss_index):
    """Test initialization when index files exist and are loaded."""
    assert faiss_store_loaded.index == mock_faiss_index
    assert faiss_store_loaded.vector_size == 128
    assert faiss_store_loaded.index.ntotal == 10
    assert faiss_store_loaded.document_texts == ["Test text"]
    assert faiss_store_loaded.doc_id_to_index_pos == {"doc1": 0}

@pytest.mark.asyncio
async def test_initialize_collection_specific_type(faiss_store_new: FaissVectorStore, mock_faiss_index, mocker):
    """Test initialize with a specific FAISS index type (e.g., IndexFlatIP)."""
    mock_ip_index = MagicMock(spec=faiss.IndexFlatIP)
    mock_ip_index.d = 0
    mock_ip_index.ntotal = 0
    mocker.patch(FAISS_INDEXFLATIP_TARGET, return_value=mock_ip_index)

    await faiss_store_new.initialize("ip_collection", 32, distance_metric="IP", faiss_index_type="IndexFlatIP")

    assert faiss_store_new.index == mock_ip_index
    assert faiss_store_new.vector_size == 32
    assert faiss_store_new.distance_metric == "IP"

@pytest.mark.asyncio
async def test_add_documents(faiss_store_new: FaissVectorStore, mock_faiss_index):
    """Test adding documents to FAISS."""
    store = faiss_store_new
    # Initialize first to set vector_size and create index
    await store.initialize("test_add_coll", 3, distance_metric="L2")
    mock_faiss_index.d = 3 # Set dimension on the mock after it's assigned

    doc_id_1 = str(uuid.uuid4())
    docs = [
        DocumentChunk(id=doc_id_1, text="FAISS doc1", vector=[0.1,0.2,0.3], metadata={"s":1}),
        DocumentChunk(id="doc2", text="FAISS doc2", vector=[0.4,0.5,0.6], metadata={"s":2}),
    ]

    added_ids = await store.add_documents("test_add_coll", docs)

    assert len(added_ids) == 2
    mock_faiss_index.add.assert_called_once()
    # Check that the numpy array passed to add has correct shape and content
    added_vectors_np = mock_faiss_index.add.call_args[0][0]
    assert isinstance(added_vectors_np, np.ndarray)
    assert added_vectors_np.shape == (2, 3)
    assert np.array_equal(added_vectors_np[0], np.array([0.1,0.2,0.3], dtype=np.float32))

    assert store.index.ntotal == 0 # mock_faiss_index.add is mocked, so ntotal doesn't change on the mock itself
                                   # In a real scenario, ntotal would update.
                                   # We test the interaction, not faiss internal state.
    assert store.document_texts == ["FAISS doc1", "FAISS doc2"]
    assert store.doc_id_to_index_pos[doc_id_1] == 0 # Assuming ntotal was 0 before add
    assert store.doc_id_to_index_pos["doc2"] == 1


@pytest.mark.asyncio
async def test_add_documents_dimension_mismatch(faiss_store_new: FaissVectorStore, caplog):
    """Test adding document with vector dimension mismatch."""
    store = faiss_store_new
    await store.initialize("dim_mismatch_coll", 3) # Expects 3D vectors

    docs_bad_dim = [DocumentChunk(id="bad1", text="Bad dim", vector=[0.1, 0.2])] # 2D vector
    added_ids = await store.add_documents("dim_mismatch_coll", docs_bad_dim)

    assert not added_ids # No IDs should be returned
    assert "Vector dimension mismatch for doc ID bad1" in caplog.text
    assert store.index.add.call_count == 0 # add should not have been called

@pytest.mark.asyncio
async def test_search_documents(faiss_store_loaded: FaissVectorStore):
    """Test searching documents with FAISS."""
    store = faiss_store_loaded # Uses the loaded fixture with some data
    mock_index = store.index # This is the mock_faiss_index from the fixture

    query_vector = [0.11, 0.21, 0.31] * (128//3) + [0.11]*(128%3) # Dummy 128D vector
    if len(query_vector) < 128 : query_vector.extend([0.0]*(128-len(query_vector)))


    # Simulate FAISS search result: (distances_array, indices_array)
    # Indices are positions in the FAISS index.
    # Distances depend on metric (L2 for this fixture)
    mock_distances = np.array([[0.01, 0.5]], dtype=np.float32) # Two results
    mock_indices = np.array([[0, 5]], dtype=np.int64) # Index 0 and Index 5 (assuming ntotal >= 6)
    mock_index.search.return_value = (mock_distances, mock_indices)

    # Ensure index_pos_to_doc_id has entries for these indices
    store.index_pos_to_doc_id = {0: "doc1_uuid", 5: "doc6_uuid"}
    store.document_texts = ["text_for_doc1"] + [""]*4 + ["text_for_doc6"] # Ensure list is long enough
    store.document_metadatas = [{"src": "A"}] + [{}]*4 + [{"src": "B"}]

    results = await store.search("loaded_collection", query_vector, top_k=2)

    mock_index.search.assert_called_once()
    # Check query vector passed to search (should be numpy array)
    search_query_vec_arg = mock_index.search.call_args[0][0]
    assert isinstance(search_query_vec_arg, np.ndarray)
    assert search_query_vec_arg.shape == (1, 128) # (1 query, 128 dimensions)

    assert len(results) == 2
    assert results[0].id == "doc1_uuid"
    assert pytest.approx(results[0].score) == 1.0 / (1.0 + 0.01) # L2 to similarity
    assert results[0].payload["text"] == "text_for_doc1"
    assert results[0].payload["metadata"] == {"src": "A"}

@pytest.mark.asyncio
async def test_search_with_post_filtering(faiss_store_loaded: FaissVectorStore):
    store = faiss_store_loaded
    mock_index = store.index
    query_vector = [0.1] * 128 # Dummy 128D vector

    mock_distances = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_indices = np.array([[0, 1, 2]], dtype=np.int64) # Index 0, 1, 2
    mock_index.search.return_value = (mock_distances, mock_indices)

    store.index_pos_to_doc_id = {0: "id0", 1: "id1", 2: "id2"}
    store.document_texts = ["text0", "text1", "text2"]
    store.document_metadatas = [
        {"source": "A", "year": 2022},
        {"source": "B", "year": 2023},
        {"source": "A", "year": 2023}
    ]

    filters = {"source": "A", "year": 2023} # Expect only doc with id2
    results = await store.search("collection", query_vector, top_k=3, filters=filters)

    assert len(results) == 1
    assert results[0].id == "id2"
    assert results[0].payload["metadata"]["source"] == "A"
    assert results[0].payload["metadata"]["year"] == 2023


@pytest.mark.asyncio
async def test_get_collection_info(faiss_store_loaded: FaissVectorStore):
    """Test getting collection info for a loaded FAISS index."""
    store = faiss_store_loaded # Uses mock_faiss_index with d=128, ntotal=10
    store.distance_metric = "L2" # Ensure this is set as it would be during init/load

    info = await store.get_collection_info("conceptual_collection_name")

    assert info["name"] == "conceptual_collection_name"
    assert info["path"] == store.index_file_path
    assert info["vectors_count"] == 10
    assert info["vector_dimension"] == 128
    assert info["distance_metric_type"] == "L2"
    assert info["faiss_index_type"] == mock_faiss_index.__class__.__name__ # e.g. "MagicMock"

# Test for _save_store could be added, mocking faiss.write_index and pickle.dump
# Test for deletion: Current impl warns and returns False. A test can verify this behavior.
@pytest.mark.asyncio
async def test_delete_documents_logs_warning(faiss_store_loaded: FaissVectorStore, caplog):
    await faiss_store_loaded.delete_documents("any_coll", ["id1"])
    assert "FAISS deletion is complex and not fully supported" in caplog.text
    assert await faiss_store_loaded.delete_documents("any_coll", ["id1"]) is False

# Test health_check
@pytest.mark.asyncio
async def test_health_check(faiss_store_loaded: FaissVectorStore):
    assert await faiss_store_loaded.health_check() is True # Loaded store should be healthy

    store_new = FaissVectorStore("./new.idx", "./new_meta.pkl", create_if_not_exists=False) # No init
    assert await store_new.health_check() is False # Not initialized index

    store_new_init_placeholder = FaissVectorStore("./new2.idx", "./new2_meta.pkl", create_if_not_exists=True)
    assert await store_new_init_placeholder.health_check() is False # Index is None, vector_size is None until initialize()
    await store_new_init_placeholder.initialize("coll", 32) # Now index and vector_size are set
    assert await store_new_init_placeholder.health_check() is True

    # Clean up files created by this specific test if any (though mocks should prevent actual file creation)
    if os.path.exists("./new2.idx"): os.remove("./new2.idx")
    if os.path.exists("./new2_meta.pkl"): os.remove("./new2_meta.pkl")
