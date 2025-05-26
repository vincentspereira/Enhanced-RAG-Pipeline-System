import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from Scripts.vector_stores.base import (
    VectorStore,
    QdrantVectorStore,
    VectorStoreRegistry
)
from Scripts.vector_stores.weaviate_store import WeaviateVectorStore
from Scripts.vector_stores.pinecone_store import PineconeVectorStore
from Scripts.vector_stores.milvus_store import MilvusVectorStore
from Scripts.vector_stores import init_vector_stores

class TestVectorStore:
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        class ConcreteStore(VectorStore):
            pass
            
        with pytest.raises(TypeError):
            ConcreteStore()

class TestQdrantVectorStore:
    @pytest.fixture
    def mock_client(self):
        return Mock()
    
    @patch("qdrant_client.QdrantClient")
    def test_initialization(self, mock_client_class):
        """Test store initialization"""
        store = QdrantVectorStore("test-collection", 768)
        
        mock_client_class.assert_called_once()
        mock_client_class.return_value.recreate_collection.assert_called_once()
    
    def test_add_vectors(self, mock_client):
        """Test adding vectors"""
        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_client
            
            store = QdrantVectorStore("test-collection", 768)
            vectors = np.random.randn(3, 768)
            metadata = [{"id": i} for i in range(3)]
            
            store.add_vectors(vectors, metadata)
            
            mock_client.upsert.assert_called_once()
    
    def test_search(self, mock_client):
        """Test vector search"""
        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.return_value = mock_client
            
            # Mock search results
            mock_client.search.return_value = [
                Mock(id="1", score=0.9, payload={"text": "test1"}),
                Mock(id="2", score=0.8, payload={"text": "test2"})
            ]
            
            store = QdrantVectorStore("test-collection", 768)
            query_vector = np.random.randn(768)
            
            results = store.search(query_vector, top_k=2)
            
            assert len(results) == 2
            assert all(isinstance(r, dict) for r in results)
            assert all({"id", "score", "metadata"}.issubset(r.keys()) for r in results)

class TestVectorStoreRegistry:
    def test_register_and_get_store(self):
        """Test registering and retrieving stores"""
        registry = VectorStoreRegistry()
        store = Mock(spec=VectorStore)
        
        registry.register_store("test-store", store)
        retrieved_store = registry.get_store("test-store")
        
        assert retrieved_store == store
    
    def test_get_nonexistent_store(self):
        """Test getting non-existent store raises error"""
        registry = VectorStoreRegistry()
        
        with pytest.raises(KeyError):
            registry.get_store("nonexistent-store")
    
    def test_list_stores(self):
        """Test listing registered stores"""
        registry = VectorStoreRegistry()
        store1 = Mock(spec=VectorStore)
        store2 = Mock(spec=VectorStore)
        
        registry.register_store("store1", store1)
        registry.register_store("store2", store2)
          stores = registry.list_stores()
        assert set(stores) == {"store1", "store2"}

class TestWeaviateVectorStore:
    @pytest.fixture
    def mock_weaviate_client(self):
        with patch('Scripts.vector_stores.weaviate_store.weaviate.Client') as mock:
            client = mock.return_value
            client.schema.exists.return_value = True
            batch = MagicMock()
            client.batch.__enter__.return_value = batch
            yield client
    
    def test_initialization(self, mock_weaviate_client):
        """Test Weaviate store initialization"""
        store = WeaviateVectorStore("TestClass")
        
        # Check if schema exists called
        mock_weaviate_client.schema.exists.assert_called_once()
    
    def test_add_vectors(self, mock_weaviate_client):
        """Test adding vectors to Weaviate"""
        store = WeaviateVectorStore("TestClass")
        
        # Test data
        vectors = np.random.randn(2, 768)
        metadata = [{"content": "test1"}, {"content": "test2"}]
        ids = ["id1", "id2"]
        
        # Call method
        store.add_vectors(vectors, metadata, ids)
        
        # Verify
        batch = mock_weaviate_client.batch.__enter__.return_value
        assert batch.add_data_object.call_count == 2
    
    def test_search(self, mock_weaviate_client):
        """Test searching in Weaviate"""
        # Setup mock query
        mock_query = MagicMock()
        mock_weaviate_client.query.get.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "TestClass": [
                        {
                            "id": "id1",
                            "content": "test1",
                            "metadata": {"field": "value"},
                            "_additional": {"distance": 0.1}
                        }
                    ]
                }
            }
        }
        
        # Create store
        store = WeaviateVectorStore("TestClass")
        
        # Test data
        query_vector = np.random.randn(768)
        
        # Call method
        results = store.search(query_vector, top_k=1)
        
        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["content"] == "test1"

class TestPineconeVectorStore:
    @pytest.fixture
    def mock_pinecone(self):
        with patch('Scripts.vector_stores.pinecone_store.pinecone') as mock:
            mock.list_indexes.return_value = ["test_index"]
            mock_index = MagicMock()
            mock.Index.return_value = mock_index
            yield mock, mock_index
    
    def test_initialization(self, mock_pinecone):
        """Test Pinecone store initialization"""
        mock_pc, _ = mock_pinecone
        
        store = PineconeVectorStore("test_index", "fake-api-key", "us-west1-gcp")
        
        # Check init called
        mock_pc.init.assert_called_once()
        mock_pc.Index.assert_called_once_with("test_index")
    
    def test_add_vectors(self, mock_pinecone):
        """Test adding vectors to Pinecone"""
        _, mock_index = mock_pinecone
        
        # Create store
        store = PineconeVectorStore("test_index", "fake-api-key", "us-west1-gcp")
        
        # Test data
        vectors = np.random.randn(2, 768)
        metadata = [{"text": "test1"}, {"text": "test2"}]
        ids = ["id1", "id2"]
        
        # Call method
        store.add_vectors(vectors, metadata, ids)
        
        # Verify
        mock_index.upsert.assert_called_once()
    
    def test_search(self, mock_pinecone):
        """Test searching in Pinecone"""
        _, mock_index = mock_pinecone
        
        # Mock query result
        mock_match = MagicMock()
        mock_match.id = "id1"
        mock_match.score = 0.9
        mock_match.metadata = {"text": "test1"}
        
        mock_result = MagicMock()
        mock_result.matches = [mock_match]
        mock_index.query.return_value = mock_result
        
        # Create store
        store = PineconeVectorStore("test_index", "fake-api-key", "us-west1-gcp")
        
        # Test data
        query_vector = np.random.randn(768)
        
        # Call method
        results = store.search(query_vector, top_k=1)
        
        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.9
        assert results[0]["metadata"] == {"text": "test1"}

class TestMilvusVectorStore:
    @pytest.fixture
    def mock_milvus(self):
        with patch('Scripts.vector_stores.milvus_store.connections') as mock_connections:
            with patch('Scripts.vector_stores.milvus_store.utility') as mock_utility:
                with patch('Scripts.vector_stores.milvus_store.Collection') as mock_collection:
                    mock_utility.has_collection.return_value = True
                    mock_collection_instance = mock_collection.return_value
                    yield mock_connections, mock_utility, mock_collection, mock_collection_instance
    
    def test_initialization(self, mock_milvus):
        """Test Milvus store initialization"""
        mock_connections, mock_utility, mock_collection, _ = mock_milvus
        
        store = MilvusVectorStore("test_collection")
        
        # Check connections
        mock_connections.connect.assert_called_once()
        mock_utility.has_collection.assert_called_once_with("test_collection")
        mock_collection.assert_called_once_with("test_collection")
    
    def test_add_vectors(self, mock_milvus):
        """Test adding vectors to Milvus"""
        _, _, _, mock_collection_instance = mock_milvus
        
        # Create store
        store = MilvusVectorStore("test_collection")
        
        # Test data
        vectors = np.random.randn(2, 1536)
        metadata = [{"text": "test1"}, {"text": "test2"}]
        ids = ["id1", "id2"]
        
        # Call method
        store.add_vectors(vectors, metadata, ids)
        
        # Verify
        mock_collection_instance.insert.assert_called_once()

class TestVectorStoreInitialization:
    @patch('Scripts.vector_stores.base.QdrantVectorStore')
    def test_init_vector_stores_qdrant(self, mock_qdrant):
        """Test initializing Qdrant vector store from config"""
        # Test config
        config = {
            "vector_store": {
                "provider": "qdrant",
                "qdrant": {
                    "collection_name": "test_collection",
                    "vector_size": 768
                }
            }
        }
        
        # Initialize
        registry = init_vector_stores(config)
            
        # Verify
        mock_qdrant.assert_called_once()
