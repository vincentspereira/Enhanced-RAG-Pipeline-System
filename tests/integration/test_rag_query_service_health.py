import pytest
import httpx
import os
import logging

# Configure logger for tests
logger = logging.getLogger(__name__)

# Determine RAG Query Service URL from environment variable, default to localhost
# This allows flexibility for CI environments or different local setups.
RAG_QUERY_SERVICE_BASE_URL = os.getenv("TEST_RAG_QUERY_SERVICE_URL", "http://localhost:8001")

# Simple check to see if the target service might be running
# This is a very basic check. More sophisticated checks might involve trying a connection.
# For now, we'll rely on the test failure if the service isn't up.
# try:
#     response = httpx.get(f"{RAG_QUERY_SERVICE_BASE_URL}/health", timeout=2) # Quick check
#     SERVICE_IS_ASSUMED_RUNNING = response.status_code == 200
# except httpx.ConnectError:
#     SERVICE_IS_ASSUMED_RUNNING = False
#     logger.warning(f"Could not connect to RAG Query Service at {RAG_QUERY_SERVICE_BASE_URL} for initial check. Ensure it's running for integration tests.")


# @pytest.mark.skipif(not SERVICE_IS_ASSUMED_RUNNING, reason="RAG Query Service not running or not reachable for integration tests.")
@pytest.mark.asyncio # Pytest-asyncio for async test functions
async def test_rag_query_service_health_endpoint():
    """
    Tests the /health endpoint of the RAG Query Service.
    Assumes the RAG Query Service is running at RAG_QUERY_SERVICE_BASE_URL.
    """
    health_url = f"{RAG_QUERY_SERVICE_BASE_URL}/health"
    logger.info(f"Testing RAG Query Service health endpoint: {health_url}")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(health_url)

        logger.info(f"Health endpoint response status: {response.status_code}")
        logger.info(f"Health endpoint response body: {response.text}")

        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

        response_json = response.json()
        assert "status" in response_json, "Response JSON should contain a 'status' key"
        # The status can be "healthy" or "degraded" depending on component initialization
        assert response_json["status"] in ["healthy", "degraded"], f"Status should be 'healthy' or 'degraded', got {response_json['status']}"

        assert "components" in response_json, "Response JSON should contain a 'components' key"
        components = response_json["components"]
        assert "qdrant_accessible" in components
        assert "embedding_model_loaded" in components
        assert "llm_client_initialized" in components
        assert "ollama_service_accessible" in components
        assert "redis_cache_connected" in components

        # If all components are expected to be up for a "healthy" status in a test environment:
        # if response_json["status"] == "healthy":
        #     assert components["qdrant_accessible"] is True
        #     assert components["embedding_model_loaded"] is True
        #     assert components["llm_client_initialized"] is True # This is just client init
        #     assert components["ollama_service_accessible"] is True # This checks Ollama itself
        #     assert components["redis_cache_connected"] is True
        # (This part can be made more strict if the test environment guarantees all dependencies are up)

    except httpx.ConnectError as e:
        pytest.fail(f"Connection to RAG Query Service at {health_url} failed. Ensure the service is running. Error: {e}")
    except httpx.ReadTimeout as e:
        pytest.fail(f"Request to RAG Query Service at {health_url} timed out. Error: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred while testing health endpoint {health_url}: {e}")

# To run this test:
# 1. Ensure pytest, pytest-asyncio, and httpx are installed.
# 2. Start the RAG Query Service (e.g., `python Scripts/services/rag_query_service.py`).
#    Ensure all its dependencies (Qdrant, Ollama, Redis) are also running if you expect a "healthy" status.
# 3. (Optional) Set the TEST_RAG_QUERY_SERVICE_URL environment variable if the service is not on http://localhost:8001.
# 4. Run pytest: `pytest Scripts/tests/integration/test_rag_query_service_health.py`
#
# In a CI environment, you would typically:
# - Build the Docker image for the RAG Query Service.
# - Start containers for Qdrant, Ollama, Redis, and the RAG Query Service.
# - Run this integration test against the RAG Query Service container's exposed port.
# - Then stop/remove the containers.
# This can be managed with Docker Compose or Kubernetes test setups.
#
# For this iteration, the test assumes the service and its dependencies are manually started
# or managed by an external process for the test execution context.
# The skipif logic is commented out to allow it to run and fail if service isn't up,
# which is often the desired behavior in CI to catch deployment/startup issues.
