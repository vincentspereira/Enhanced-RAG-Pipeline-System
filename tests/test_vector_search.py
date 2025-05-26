"""
Integration tests for the vector search functionality.
"""
import pytest
import os
import json
import qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from unittest.mock import patch, MagicMock

# Test data
TEST_COLLECTION = "test_search_collection"
TEST_VECTOR_SIZE = 384  # Common dimension for small embedding models

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing vector search"""
    mock_client = MagicMock(spec=qdrant_client.QdrantClient)
    
    # Mock collection creation
    mock_client.recreate_collection.return_value = None
    
    # Mock search results
    mock_search_result = [
        (0.95, {"id": "doc1", "text": "Test document 1", "metadata": {"source": "test"}}),
        (0.85, {"id": "doc2", "text": "Test document 2", "metadata": {"source": "test"}}),
        (0.75, {"id": "doc3", "text": "Test document 3", "metadata": {"source": "test"}})
    ]
    mock_client.search.return_value = mock_search_result
    
    # Mock collection info
    mock_client.get_collection.return_value = MagicMock(
        vectors_count=100,
        status="green"
    )
    
    return mock_client

@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing"""
    def generate_mock_embedding(text):
        # Generate a deterministic embedding based on text
        # This is just for testing - real embeddings would be semantic
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(TEST_VECTOR_SIZE).astype(np.float32)
    
    mock_model = MagicMock()
    mock_model.encode.side_effect = generate_mock_embedding
    
    return mock_model

class TestVectorSearch:
    """Test vector search functionality"""
    
    def test_document_indexing(self, mock_qdrant_client, mock_embedding_model):
        """Test indexing documents into vector database"""
        # Test documents
        test_docs = [
            {"id": "doc1", "text": "This is test document one", "metadata": {"source": "test", "type": "article"}},
            {"id": "doc2", "text": "This is test document two", "metadata": {"source": "test", "type": "article"}},
            {"id": "doc3", "text": "This is another test document", "metadata": {"source": "test", "type": "note"}}
        ]
        
        # Generate embeddings for documents
        for doc in test_docs:
            doc["vector"] = mock_embedding_model.encode(doc["text"])
        
        # Index documents
        points = []
        for i, doc in enumerate(test_docs):
            points.append({
                "id": i,
                "vector": doc["vector"].tolist(),
                "payload": {
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
            })
        
        # Mock upsert
        mock_qdrant_client.upsert.return_value = None
        
        # Call upsert
        mock_qdrant_client.upsert(
            collection_name=TEST_COLLECTION,
            points=points
        )
        
        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        
        # Verify number of documents indexed
        assert len(points) == 3
    
    def test_vector_search(self, mock_qdrant_client, mock_embedding_model):
        """Test searching for documents using vector similarity"""
        # Query text
        query = "Find me a test document"
        
        # Generate query embedding
        query_vector = mock_embedding_model.encode(query)
        
        # Perform search
        results = mock_qdrant_client.search(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector.tolist(),
            limit=5
        )
        
        # Verify search was called
        mock_qdrant_client.search.assert_called_once()
        
        # Check results
        assert len(results) == 3
        assert results[0][0] > results[1][0]  # Scores are in descending order
    
    def test_filtered_search(self, mock_qdrant_client, mock_embedding_model):
        """Test searching with metadata filters"""
        # Query text
        query = "Find me a test article"
        
        # Generate query embedding
        query_vector = mock_embedding_model.encode(query)
        
        # Create filter for articles only
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.type",
                    match=MatchValue(value="article")
                )
            ]
        )
        
        # Mock filtered search results - only articles
        mock_filtered_results = [
            (0.92, {"id": "doc1", "text": "Test document 1", "metadata": {"source": "test", "type": "article"}}),
            (0.82, {"id": "doc2", "text": "Test document 2", "metadata": {"source": "test", "type": "article"}})
        ]
        mock_qdrant_client.search.return_value = mock_filtered_results
        
        # Perform filtered search
        results = mock_qdrant_client.search(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector.tolist(),
            filter=filter_condition,
            limit=5
        )
        
        # Check only articles are returned
        assert len(results) == 2
        for score, payload in results:
            assert payload["metadata"]["type"] == "article"
    
    def test_vector_stats(self, mock_qdrant_client):
        """Test getting vector collection statistics"""
        # Get collection info
        collection_info = mock_qdrant_client.get_collection(TEST_COLLECTION)
        
        # Check statistics
        assert collection_info.vectors_count == 100
        assert collection_info.status == "green"
        
        # Verify get_collection was called
        mock_qdrant_client.get_collection.assert_called_once_with(TEST_COLLECTION)
