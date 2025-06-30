import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient

# From conftest, these will be injected:
# - integration_test_client: TestClient for enhanced_api.app with real Qdrant/ES
# - seed_data_for_integration_api_test: Handles data seeding and cleanup for each test
# - admin_token, user_token: JWT tokens for authenticated requests
# - SAMPLE_DOCS_CONTENT: The actual data that was seeded

# Expected number of documents seeded by seed_data_for_integration_api_test
EXPECTED_DOC_COUNT = 5

# Note: The seed_data_for_integration_api_test fixture is function-scoped,
# so it will run before each test function here, ensuring a clean data state.

@pytest.mark.asyncio
async def test_api_process_documents_and_status(integration_test_client: TestClient, admin_token: str, seed_data_for_integration_api_test: None):
    """
    Tests if documents seeded via RAGPipeline (in fixture) are reflected in status.
    This test implicitly relies on seed_data_for_integration_api_test having run.
    It also tests the /process_docs endpoint by adding one more document.
    """
    client = integration_test_client

    # 1. Check system status after initial seeding by the fixture
    response = client.get("/system_status_rag", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["name"] == client.app.state.app_config.vector_store.collection_name
    # Point count can be tricky due to chunking. Let's check if it's at least EXPECTED_DOC_COUNT
    # as each doc might become one or more points.
    # A more precise check would require knowing the exact chunking strategy.
    # For now, let's assume 1 chunk per doc for simplicity of SAMPLE_DOCS_CONTENT.
    assert status_data["points_count"] >= EXPECTED_DOC_COUNT
    initial_points_count = status_data["points_count"]

    # 2. Process an additional document via the API endpoint
    temp_dir = Path("temp_api_processing_docs_integration")
    temp_dir.mkdir(exist_ok=True)
    new_doc_api_file = temp_dir / "new_api_doc.txt"
    new_doc_api_text = "A brand new document processed via API for integration testing."

    with open(new_doc_api_file, "w") as f:
        f.write(new_doc_api_text)

    process_payload = {"directory_path": str(temp_dir)}
    response = client.post(
        "/process_docs",
        json=process_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"].startswith("Started processing documents from")

    # Wait for processing and indexing to complete
    await asyncio.sleep(5) # Increased sleep to allow for full processing and ES indexing

    # 3. Check system status again to see if the new document increased the count
    response = client.get("/system_status_rag", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    status_data_after_api_process = response.json()
    # Assuming the new doc adds at least one more point (or more if chunked)
    assert status_data_after_api_process["points_count"] > initial_points_count

    # Cleanup the temp file and dir for this specific test part
    new_doc_api_file.unlink()
    temp_dir.rmdir()


@pytest.mark.asyncio
async def test_api_hybrid_search_integration(integration_test_client: TestClient, admin_token: str, seed_data_for_integration_api_test: None):
    """
    Tests the /search_rag endpoint with a query designed for hybrid results.
    Relies on SAMPLE_DOCS_CONTENT being seeded by the fixture.
    """
    client = integration_test_client
    query = "quick fox and vector database" # Targets doc1/doc4 (keyword) and doc3 (semantic + keyword)

    search_payload = {"query": query, "limit": 5}
    response = client.post(
        "/search_rag",
        json=search_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    results_data = response.json()
    results = results_data["results"]

    assert len(results) > 0

    found_doc1_or_4_by_keyword = any(
        r['metadata']['source'].startswith('doc1') or r['metadata']['source'].startswith('doc4')
        # Keyword score might not be directly exposed by API, focus on finding the doc
        for r in results
    )
    found_doc3_by_semantic_and_keyword = any(
        r['metadata']['source'].startswith('doc3')
        for r in results
    )

    print(f"\nAPI Hybrid Search Results for query '{query}':")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:60]}..., Source: {r_val['metadata']['source']}, Score: {r_val['score']:.4f}")

    assert found_doc1_or_4_by_keyword, "Expected 'doc1' or 'doc4' (match for 'fox')"
    assert found_doc3_by_semantic_and_keyword, "Expected 'doc3' (match for 'vector database' and 'fox')"
    for r_val in results:
        assert 'score' in r_val and r_val['score'] > 0 # Basic check for score presence

@pytest.mark.asyncio
async def test_api_keyword_dominant_search_integration(integration_test_client: TestClient, admin_token: str, seed_data_for_integration_api_test: None):
    """Tests /search_rag with a query likely to be keyword-dominant."""
    client = integration_test_client
    query = "powerful search engine" # Targets doc2 strongly by keyword

    search_payload = {"query": query, "limit": 3}
    response = client.post(
        "/search_rag",
        json=search_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    results_data = response.json()
    results = results_data["results"]

    print(f"\nAPI Keyword Dominant Search Results for query '{query}':")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:60]}..., Source: {r_val['metadata']['source']}, Score: {r_val['score']:.4f}")

    assert len(results) > 0
    # Check if doc2 is among the top results (ideally the first)
    assert results[0]['metadata']['source'].startswith('doc2'), "Expected 'doc2' to be the top result for keyword dominant query"

@pytest.mark.asyncio
async def test_api_semantic_dominant_search_integration(integration_test_client: TestClient, admin_token: str, seed_data_for_integration_api_test: None):
    """Tests /search_rag with a query likely to be semantic-dominant."""
    client = integration_test_client
    query = "finding similar items very quickly" # Targets doc5 strongly by semantics

    search_payload = {"query": query, "limit": 3}
    response = client.post(
        "/search_rag",
        json=search_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    results_data = response.json()
    results = results_data["results"]

    print(f"\nAPI Semantic Dominant Search Results for query '{query}':")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:60]}..., Source: {r_val['metadata']['source']}, Score: {r_val['score']:.4f}")

    assert len(results) > 0
    assert results[0]['metadata']['source'].startswith('doc5'), "Expected 'doc5' to be the top result for semantic dominant query"

@pytest.mark.asyncio
async def test_api_generate_response_integration(integration_test_client: TestClient, admin_token: str, seed_data_for_integration_api_test: None):
    """Tests the /generate_response endpoint."""
    client = integration_test_client
    question = "What is Qdrant?" # This should find doc3

    # Mock the LLM call within RAGPipeline for this integration test,
    # as we don't want to make actual OpenAI/Anthropic calls.
    # The RAGPipeline instance is on `client.app.state.rag_pipeline`.
    # Its `llm_service.generate` method is what we need to mock.

    rag_pipeline_instance: RAGPipeline = client.app.state.rag_pipeline

    # Ensure llm_service is present; it's initialized in RAGPipeline's __init__
    assert rag_pipeline_instance.llm_service is not None

    # Store original and then mock
    original_llm_generate = rag_pipeline_instance.llm_service.generate
    rag_pipeline_instance.llm_service.generate = MagicMock(
        return_value="Qdrant is a vector database, as described in the provided context."
    )

    generate_payload = {"question": question, "limit": 3} # Limit for context search
    response = client.post(
        "/generate_response",
        json=generate_payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )

    # Restore original method
    rag_pipeline_instance.llm_service.generate = original_llm_generate

    assert response.status_code == 200
    response_data = response.json()

    print(f"\nAPI Generate Response for question '{question}': {response_data}")

    assert "answer" in response_data
    assert "sources" in response_data
    assert response_data["answer"] == "Qdrant is a vector database, as described in the provided context."

    # Check if the source for "Qdrant" (doc3) was likely found and passed to LLM
    # The mock doesn't use the sources, but RAGPipeline.generate_response should have found them.
    # This part is harder to assert without seeing the arguments to the mocked llm_service.generate.
    # For a more thorough test, one might check that `mock_llm_generate.assert_called_once()`
    # and inspect `call_args`.

    # Let's check if sources related to Qdrant (doc3) are present in the API response
    found_doc3_source = any(s['metadata']['source'].startswith('doc3') for s in response_data["sources"])
    assert found_doc3_source, "Expected 'doc3' to be among sources for 'What is Qdrant?'"


@pytest.mark.asyncio
async def test_api_search_unauthorized(integration_test_client: TestClient, seed_data_for_integration_api_test: None):
    """Tests /search_rag endpoint without authentication token."""
    client = integration_test_client
    search_payload = {"query": "any query", "limit": 1}
    response = client.post("/search_rag", json=search_payload) # No Authorization header

    assert response.status_code == 401 # Expecting Unauthorized
    assert "Not authenticated" in response.json().get("detail", "").lower()

@pytest.mark.asyncio
async def test_api_process_docs_forbidden_for_user_role(integration_test_client: TestClient, user_token: str, seed_data_for_integration_api_test: None):
    """Tests /process_docs endpoint with a user token that lacks permission."""
    client = integration_test_client

    # Create a dummy directory and file for the payload
    temp_dir = Path("temp_api_processing_forbidden")
    temp_dir.mkdir(exist_ok=True)
    dummy_file = temp_dir / "dummy.txt"
    with open(dummy_file, "w") as f:
        f.write("content")

    process_payload = {"directory_path": str(temp_dir)}
    response = client.post(
        "/process_docs",
        json=process_payload,
        headers={"Authorization": f"Bearer {user_token}"} # User token, not admin
    )

    # Cleanup dummy file and dir
    dummy_file.unlink()
    temp_dir.rmdir()

    # Based on current permissions, Role.USER does not have DOCUMENT_WRITE or ADMIN_WRITE
    # which are likely required for /process_docs (this needs to be set in enhanced_api.py)
    # Assuming /process_docs requires a permission like ADMIN_WRITE or a specific DOCUMENT_PROCESS.
    # If Role.USER was granted this, this test would fail.
    # Let's assume it requires a permission that 'user_token' doesn't have.
    assert response.status_code == 403 # Expecting Forbidden
    assert "Permission denied" in response.json().get("detail", "")


# Add more tests:
# - Different filter conditions for search
# - Different categories for search
# - Pagination if implemented and exposed via API
# - Error conditions (e.g., RAG pipeline not initialized - though client fixture should prevent this)
# - Malformed payloads
# - Test other endpoints if they interact with the core RAG data (e.g., specific document retrieval if added)

from unittest.mock import MagicMock # Ensure MagicMock is imported for the generate_response test
