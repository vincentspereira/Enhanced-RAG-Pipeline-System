"""
End-to-end tests for the Qdrant RAG system.
These tests simulate real user scenarios and test the complete system flow.
"""
import pytest
import os
import json
import time
import requests
from requests.exceptions import ConnectionError
import subprocess
import signal
import shutil
import tempfile
from pathlib import Path

# Test constants
TEST_SERVER_URL = "http://localhost:8000"
TEST_DOCUMENT_DIR = "tests/e2e/test_documents"
TEST_QUERY = "What are embedding models?"

@pytest.fixture(scope="module")
def test_server():
    """Start a test server for E2E testing"""
    # Create test directories
    os.makedirs(TEST_DOCUMENT_DIR, exist_ok=True)
    
    # Create a test document
    with open(f"{TEST_DOCUMENT_DIR}/test_embeddings.txt", "w") as f:
        f.write("""
        Embedding models are neural networks that map text to vectors of real numbers.
        These vectors, or embeddings, capture semantic meaning, allowing us to measure
        similarity between texts based on their content rather than superficial characteristics.
        
        Popular embedding models include:
        - Snowflake Arctic Embed2
        - OpenAI's text-embedding-ada-002
        - Google's Universal Sentence Encoder
        - BERT-based models
        
        Embeddings are foundational to RAG (Retrieval Augmented Generation) systems,
        enabling semantic search over documents.
        """)
    
    # Start server in a separate process
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "Scripts.rag_api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Wait for server to start
    max_retries = 30
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = requests.get(f"{TEST_SERVER_URL}/docs")
            if response.status_code == 200:
                break
        except ConnectionError:
            pass
        
        time.sleep(1)
        retry_count += 1
    
    if retry_count >= max_retries:
        # Kill the server if it didn't start properly
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        pytest.fail("Failed to start test server")
    
    # Let the server initialize fully
    time.sleep(5)
    
    yield server_process
    
    # Clean up after tests
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    shutil.rmtree(TEST_DOCUMENT_DIR, ignore_errors=True)

@pytest.mark.e2e
def test_rag_workflow(test_server):
    """Test the complete RAG workflow from document processing to query"""
    # Step 1: Check system status
    response = requests.get(f"{TEST_SERVER_URL}/status")
    assert response.status_code == 200
    initial_status = response.json()
    
    # Step 2: Process documents
    response = requests.post(
        f"{TEST_SERVER_URL}/process",
        json={"directory_path": TEST_DOCUMENT_DIR, "batch_size": 2}
    )
    assert response.status_code == 200
    assert "Started processing" in response.json()["message"]
    
    # Step 3: Wait for processing to complete (poll status)
    max_retries = 60
    retry_count = 0
    status_changed = False
    
    while retry_count < max_retries and not status_changed:
        time.sleep(2)  # Wait for processing
        
        response = requests.get(f"{TEST_SERVER_URL}/status")
        assert response.status_code == 200
        current_status = response.json()
        
        # Check if vector count increased (indicating documents were processed)
        if current_status.get("vector_count", 0) > initial_status.get("vector_count", 0):
            status_changed = True
        
        retry_count += 1
    
    assert status_changed, "Document processing did not complete successfully"
    
    # Step 4: Search for documents
    response = requests.post(
        f"{TEST_SERVER_URL}/search",
        json={"query": TEST_QUERY, "limit": 5}
    )
    assert response.status_code == 200
    results = response.json()
    
    # Verify search results
    assert len(results) > 0
    assert any("embedding" in result["text"].lower() for result in results)
    
    # Step 5: Test Copilot API
    response = requests.post(
        f"{TEST_SERVER_URL}/copilot/query",
        json={"query": TEST_QUERY},
        headers={"X-Copilot-Token": "gca_test_token"}
    )
    
    # Check if Copilot API is available (may not be in all environments)
    if response.status_code == 200:
        answer = response.json()
        assert "answer" in answer
        assert "sources" in answer
        assert len(answer["sources"]) > 0
    
    # Successfully completed the full RAG workflow

@pytest.mark.e2e
def test_api_hub_e2e(test_server):
    """Test the complete API Hub workflow from service registration to API request"""
    # Step 1: Register a test service
    test_service = {
        "id": "e2e-test-api",
        "name": "E2E Test API",
        "description": "API for E2E testing",
        "base_url": "https://httpbin.org",
        "auth_type": "none",
        "endpoints": [
            {
                "path": "/get",
                "method": "GET",
                "description": "Test GET endpoint",
                "requires_auth": False
            },
            {
                "path": "/post",
                "method": "POST",
                "description": "Test POST endpoint",
                "requires_auth": False
            }
        ]
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/api-hub/services", json=test_service)
    assert response.status_code == 200
    assert response.json()["message"] == "API service 'E2E Test API' registered successfully"
    
    # Step 2: Verify service was registered
    response = requests.get(f"{TEST_SERVER_URL}/api/v1/api-hub/services/e2e-test-api")
    assert response.status_code == 200
    assert response.json()["id"] == "e2e-test-api"
    
    # Step 3: Register a webhook
    test_webhook = {
        "url": "https://webhook.test.e2e.com",
        "events": ["service.updated", "request.error"],
        "description": "E2E test webhook"
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/api-hub/webhooks", json=test_webhook)
    assert response.status_code == 200
    assert "webhook" in response.json()
    webhook_id = response.json()["webhook"]["id"]
    
    # Step 4: Make an API request
    response = requests.post(
        f"{TEST_SERVER_URL}/api/v1/api-hub/request",
        json={
            "service_id": "e2e-test-api",
            "endpoint": "/get",
            "method": "GET",
            "params": {"param1": "value1"}
        }
    )
    
    # Check the response (may be a mock/stub in test environment)
    assert response.status_code in (200, 502)  # 502 if external API is blocked
    
    # Step 5: Batch request
    response = requests.post(
        f"{TEST_SERVER_URL}/api/v1/api-hub/batch",
        json={
            "requests": [
                {
                    "service_id": "e2e-test-api",
                    "endpoint": "/get",
                    "method": "GET"
                },
                {
                    "service_id": "e2e-test-api",
                    "endpoint": "/post",
                    "method": "POST",
                    "body": {"key": "value"}
                }
            ],
            "parallel": True
        }
    )
    
    # Check the response (may be a mock/stub in test environment)
    assert response.status_code in (200, 502)  # 502 if external API is blocked
    
    # Step 6: Check analytics
    response = requests.get(f"{TEST_SERVER_URL}/api/v1/api-hub/analytics/dashboard")
    assert response.status_code == 200
    
    # Step 7: Check service health
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/api-hub/services/e2e-test-api/health-check")
    assert response.status_code in (200, 502)  # 502 if external API is blocked
    
    # Step 8: Clean up - Delete webhook
    response = requests.delete(f"{TEST_SERVER_URL}/api/v1/api-hub/webhooks/{webhook_id}")
    assert response.status_code == 200
    
    # Step 9: Clean up - Delete service
    response = requests.delete(f"{TEST_SERVER_URL}/api/v1/api-hub/services/e2e-test-api")
    assert response.status_code == 200

@pytest.mark.e2e
def test_embedding_training_workflow(test_server):
    """Test the embedding training workflow"""
    # Step 1: Check if embedding training API is available
    response = requests.get(f"{TEST_SERVER_URL}/api/v1/embedding/models")
    
    # This test may be skipped if the embedding training API is not available in the test environment
    if response.status_code != 200:
        pytest.skip("Embedding training API not available")
    
    # Step 2: Create a test dataset
    test_dataset = {
        "name": "e2e-test-dataset",
        "description": "Dataset for E2E testing",
        "texts": [
            "Embedding models convert text to vectors",
            "RAG systems use embeddings for retrieval",
            "Vector databases store embeddings efficiently",
            "Semantic search uses embedding similarity"
        ],
        "validation_split": 0.25
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/embedding/datasets", json=test_dataset)
    assert response.status_code == 200
    dataset_id = response.json()["dataset_id"]
    
    # Step 3: Start a training job
    training_config = {
        "dataset_id": dataset_id,
        "model_name": "e2e-test-model",
        "base_model": "all-MiniLM-L6-v2",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.0001
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/embedding/train", json=training_config)
    
    # Training might not actually run in the test environment
    if response.status_code == 200:
        job_id = response.json()["job_id"]
        
        # Step 4: Check training status (poll)
        max_retries = 5
        for _ in range(max_retries):
            response = requests.get(f"{TEST_SERVER_URL}/api/v1/embedding/jobs/{job_id}")
            if response.status_code == 200:
                status = response.json()["status"]
                if status in ("completed", "failed"):
                    break
            time.sleep(2)
    
    # Step 5: Clean up - Delete dataset
    response = requests.delete(f"{TEST_SERVER_URL}/api/v1/embedding/datasets/{dataset_id}")
    assert response.status_code in (200, 404)  # 404 if already cleaned up

@pytest.mark.e2e
def test_prompt_management_workflow(test_server):
    """Test the prompt management workflow"""
    # Step 1: Check if prompt management API is available
    response = requests.get(f"{TEST_SERVER_URL}/api/v1/prompts")
    
    # This test may be skipped if the prompt management API is not available in the test environment
    if response.status_code != 200:
        pytest.skip("Prompt management API not available")
    
    # Step 2: Create a test prompt template
    test_prompt = {
        "name": "e2e-test-prompt",
        "description": "Prompt template for E2E testing",
        "template": "Answer the following question based on the context: {{context}}\n\nQuestion: {{question}}",
        "variables": ["context", "question"],
        "default_values": {"question": "What is RAG?"}
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/prompts", json=test_prompt)
    assert response.status_code == 200
    prompt_id = response.json()["prompt_id"]
    
    # Step 3: Get prompt details
    response = requests.get(f"{TEST_SERVER_URL}/api/v1/prompts/{prompt_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "e2e-test-prompt"
    
    # Step 4: Create a new version
    new_version = {
        "template": "Based on the provided context, please answer: {{question}}\n\nContext: {{context}}",
        "description": "Updated version for E2E testing"
    }
    
    response = requests.post(f"{TEST_SERVER_URL}/api/v1/prompts/{prompt_id}/versions", json=new_version)
    assert response.status_code == 200
    version_id = response.json()["version_id"]
    
    # Step 5: Use the prompt template
    response = requests.post(
        f"{TEST_SERVER_URL}/api/v1/prompts/{prompt_id}/render",
        json={
            "variables": {
                "context": "RAG stands for Retrieval Augmented Generation.",
                "question": "What does RAG stand for?"
            },
            "version_id": version_id
        }
    )
    assert response.status_code == 200
    assert "RAG" in response.json()["rendered_prompt"]
    
    # Step 6: Clean up - Delete prompt
    response = requests.delete(f"{TEST_SERVER_URL}/api/v1/prompts/{prompt_id}")
    assert response.status_code == 200
