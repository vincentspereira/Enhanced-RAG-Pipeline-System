import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import Mock, patch
from main import app

client = TestClient(app)

@pytest.fixture
def mock_pipeline():
    return Mock()

def test_process_document(mock_pipeline):
    """Test document processing endpoint"""
    with patch("main.pipeline", mock_pipeline):
        mock_pipeline.process_document.return_value = "task-123"
        
        response = client.post(
            "/documents",
            json={
                "document_id": "doc1",
                "content": "test content",
                "metadata": {"source": "test"}
            }
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "task_id" in response.json()

def test_search(mock_pipeline):
    """Test search endpoint"""
    with patch("main.pipeline", mock_pipeline):
        mock_results = [
            {
                "id": "doc1",
                "score": 0.9,
                "metadata": {"content": "test1"}
            }
        ]
        mock_pipeline.search.return_value = mock_results
        
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "top_k": 5,
                "rerank": True
            }
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert "results" in response.json()

def test_add_feedback(mock_pipeline):
    """Test feedback endpoint"""
    with patch("main.pipeline", mock_pipeline):
        response = client.post(
            "/feedback",
            json={
                "query_id": "q1",
                "document_id": "doc1",
                "is_relevant": True
            }
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_pipeline.add_relevance_feedback.assert_called_once()

def test_get_status(mock_pipeline):
    """Test status endpoint"""
    with patch("main.pipeline", mock_pipeline):
        mock_pipeline.get_document_count.return_value = 100
        mock_pipeline.get_vector_count.return_value = 500
        mock_pipeline.get_model_info.return_value = {
            "embedding_model": "test-model",
            "vector_dimension": 768
        }
        
        response = client.get("/status")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert "stats" in response.json()

def test_error_handling(mock_pipeline):
    """Test error handling"""
    with patch("main.pipeline", mock_pipeline):
        mock_pipeline.process_document.side_effect = Exception("Test error")
        
        response = client.post(
            "/documents",
            json={
                "document_id": "doc1",
                "content": "test content"
            }
        )
        
        assert response.status_code == 500
        assert "detail" in response.json()
