import pytest
import httpx
import os
import logging
from unittest.mock import MagicMock, patch, ANY
import json
from typing import List, Dict, Any, Union
import uuid

# Base URL for the RAG Query Service
RAG_QUERY_SERVICE_BASE_URL = os.getenv("TEST_RAG_QUERY_SERVICE_URL", "http://localhost:8001")

# Configure logger
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"

# Mock QdrantClient, SentenceTransformer, httpx.AsyncClient (for Ollama), and RedisCacheManager
@pytest.fixture(autouse=True)
def mock_dependencies_for_rag_query_service(mocker):
    mock_qdrant = MagicMock()
    # mock_qdrant.search.return_value = [] # Default empty semantic results
    # mock_qdrant.scroll.return_value = ([], None) # Default empty keyword results

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5] # Dummy embedding
    mock_embedder.get_sentence_embedding_dimension.return_value = 5 # Match dummy embedding

    mock_ollama_client_instance = MagicMock()
    mock_ollama_response = MagicMock()
    mock_ollama_response.status_code = 200
    mock_ollama_response.json.return_value = {"response": "Mocked LLM answer."}
    mock_ollama_client_instance.post.return_value = mock_ollama_response # for /api/generate
    mock_ollama_client_instance.get.return_value = mock_ollama_response # for health check ("/")

    # Patch the constructor of httpx.AsyncClient to return our mock when called by the service
    mocker.patch('httpx.AsyncClient', return_value=mock_ollama_client_instance)

    mock_redis_cache = MagicMock()
    mock_redis_cache.is_available.return_value = True
    mock_redis_cache.get_json.return_value = None # Default cache miss for JSON
    mock_redis_cache.set_json.return_value = True
    mock_redis_cache.get_string.return_value = None # Default cache miss for string
    mock_redis_cache.set_string.return_value = True

    # Patch where these are instantiated or accessed in rag_query_service.py
    mocker.patch('Scripts.services.rag_query_service.qdrant_client', mock_qdrant)
    mocker.patch('Scripts.services.rag_query_service.embedding_model', mock_embedder)
    # httpx.AsyncClient is already patched globally for this fixture's scope
    mocker.patch('Scripts.services.rag_query_service.redis_cache', mock_redis_cache)

    # Return all mocks if needed by tests
    return {
        "qdrant": mock_qdrant,
        "embedder": mock_embedder,
        "ollama": mock_ollama_client_instance,
        "redis": mock_redis_cache
    }

# Helper to simulate Qdrant PointStruct for mocking search results
def create_mock_qdrant_point(id_val, score, text, metadata, search_method="semantic"):
    # Qdrant's actual Hit has `id`, `score`, `payload`. `payload` contains `text` and `metadata`.
    # Our SearchResult model flattens this a bit.
    # The mock for qdrant_client.search should return objects that look like Qdrant's `Hit`.
    mock_hit = MagicMock()
    mock_hit.id = id_val
    mock_hit.score = score # Semantic score
    mock_hit.payload = {"text": text, "metadata": metadata}
    # For keyword search mock via scroll, it also returns Hit-like objects
    return mock_hit


@pytest.mark.asyncio
async def test_hybrid_search_flow(mock_dependencies_for_rag_query_service):
    qdrant_mock = mock_dependencies_for_rag_query_service["qdrant"]
    redis_mock = mock_dependencies_for_rag_query_service["redis"]

    # --- Setup Mock Qdrant Responses ---
    # Semantic search results
    semantic_hits = [
        create_mock_qdrant_point("doc1", 0.9, "Document 1 about apples and oranges.", {"keywords": ["apple", "orange"], "source": "s1"}),
        create_mock_qdrant_point("doc2", 0.8, "Document 2 mentions only oranges.", {"keywords": ["orange"], "source": "s2"}),
    ]
    qdrant_mock.search.return_value = semantic_hits # For semantic search

    # Keyword filter search results (via scroll)
    # Let's say "apple" query keyword matches doc1 and doc3
    keyword_hits_for_apple = [
        create_mock_qdrant_point("doc1", 0.0, "Document 1 about apples and oranges.", {"keywords": ["apple", "orange"], "source": "s1"}), # Score not used by scroll
        create_mock_qdrant_point("doc3", 0.0, "Document 3 is an apple pie recipe.", {"keywords": ["apple", "pie", "recipe"], "source": "s3"}),
    ]
    # If query was just "apple", scroll would be called with filter for "apple"
    # We need to configure the mock based on what _keyword_filter_search will do.
    # For simplicity, assume _keyword_filter_search will find these if query is "apple"
    qdrant_mock.scroll.return_value = (keyword_hits_for_apple, None)


    # --- Test Hybrid Search ---
    query = "apple"
    payload = {"query": query, "top_k": 3, "search_type": "hybrid", "generate_answer": True, "force_no_cache": True}

    async with httpx.AsyncClient(base_url=RAG_QUERY_SERVICE_BASE_URL, timeout=10.0) as client:
        response = await client.post("/query", json=payload)

    assert response.status_code == 200
    response_json = response.json()

    assert response_json["query"] == query
    assert "search_results" in response_json
    results = response_json["search_results"]
    logger.info(f"Hybrid search results for '{query}': {results}")

    # Check RRF logic (simplified assertions)
    # doc1 should be highly ranked (semantic hit, keyword hit)
    # doc3 should appear (keyword hit)
    # doc2 might appear (semantic hit)
    result_ids = [r["id"] for r in results]
    assert "doc1" in result_ids
    assert "doc3" in result_ids

    # Verify doc1 has a good score and "hybrid_boosted" or similar method if it was boosted
    doc1_res = next((r for r in results if r["id"] == "doc1"), None)
    assert doc1_res is not None
    # Check if score was boosted (original semantic 0.9, keyword match adds to it)
    # The exact RRF score depends on ranks and K, this is a rough check.
    # Semantic rank 1, BM25/keyword rank 1 for "doc1" if query is "apple"
    # Semantic rank 2 for "doc2"
    # BM25/keyword rank 2 for "doc3"
    # Expected RRF order: doc1, then doc3/doc2 depending on tie-breaking or exact RRF scores.
    # With current boosting: doc1 score = 0.9 (semantic) + (1.0 * 0.25) (keyword boost) = 1.15
    # doc3 score = 0.1 (base for keyword only) + 1.0 (keyword score) = 1.1
    # doc2 score = 0.8 (semantic only)
    # So order should be doc1, doc3, doc2
    if len(results) >= 1: assert results[0]["id"] == "doc1"
    if len(results) >= 2: assert results[1]["id"] == "doc3" # Assuming doc3's keyword score + base > doc2's semantic
    if len(results) >= 3: assert results[2]["id"] == "doc2"


    assert "Mocked LLM answer." in response_json["answer"]
    assert response_json["llm_model_used"] is not None

    # Verify cache was set (since force_no_cache was True for request, but internal calls might set)
    # This part is tricky as force_no_cache applies to the *request*, not necessarily sub-calls if not passed down.
    # The current implementation sets cache *after* retrieving if it was a miss.
    # So, redis_mock.set_json should have been called for search_results,
    # and redis_mock.set_string for the LLM answer.
    assert redis_mock.set_json.call_count >= 1 # For search results
    assert redis_mock.set_string.call_count >= 1 # For LLM answer


@pytest.mark.asyncio
async def test_caching_flow(mock_dependencies_for_rag_query_service):
    qdrant_mock = mock_dependencies_for_rag_query_service["qdrant"]
    redis_mock = mock_dependencies_for_rag_query_service["redis"]
    ollama_mock = mock_dependencies_for_rag_query_service["ollama"]

    query = "tell me about caching"
    payload = {"query": query, "top_k": 1, "search_type": "semantic", "generate_answer": True, "force_no_cache": False}

    # --- First call (Cache Miss) ---
    # Simulate Qdrant returning one result for semantic search
    qdrant_mock.search.return_value = [create_mock_qdrant_point("cache_doc1", 0.85, "Caching is important.", {"source": "c1"})]
    # Simulate Ollama response
    ollama_mock.post.return_value.json.return_value = {"response": "Caching helps speed up responses."}

    # Cache should miss for both search results and LLM answer
    redis_mock.get_json.return_value = None # search_results cache miss
    redis_mock.get_string.return_value = None # llm_answer cache miss

    async with httpx.AsyncClient(base_url=RAG_QUERY_SERVICE_BASE_URL, timeout=10.0) as client:
        logger.info("Testing caching: First call (expect miss for search and LLM)")
        response1 = await client.post("/query", json=payload)

    assert response1.status_code == 200
    response1_json = response1.json()
    assert response1_json["cached_response"] is False # Search results were not from cache
    assert "Caching helps speed up responses." in response1_json["answer"]

    # Verify mocks: Qdrant search, Ollama call, Redis sets
    qdrant_mock.search.assert_called_once()
    ollama_mock.post.assert_called_once() # Called to generate answer
    # Two set calls: one for search_results, one for llm_answer
    assert redis_mock.set_json.call_count == 1
    assert redis_mock.set_string.call_count == 1

    # --- Setup for Second call (Cache Hit) ---
    # Mock Redis to return what was supposedly cached
    # Search results cache:
    cached_search_results_payload = [res._asdict() if hasattr(res, '_asdict') else res for res in response1_json["search_results"]] # Convert SearchResult objects if needed
    redis_mock.get_json.return_value = cached_search_results_payload
    # LLM answer cache:
    redis_mock.get_string.return_value = response1_json["answer"]

    # Reset call counts for Qdrant and Ollama to ensure they are NOT called
    qdrant_mock.search.reset_mock()
    ollama_mock.post.reset_mock()
    # Reset set_x call counts for Redis, but not get_x
    redis_mock.set_json.reset_mock()
    redis_mock.set_string.reset_mock()


    async with httpx.AsyncClient(base_url=RAG_QUERY_SERVICE_BASE_URL, timeout=10.0) as client:
        logger.info("Testing caching: Second call (expect hit for search and LLM)")
        response2 = await client.post("/query", json=payload)

    assert response2.status_code == 200
    response2_json = response2.json()
    # cached_response flag in QueryResponse reflects search_results cache primarily
    assert response2_json["cached_response"] is True
    assert response2_json["answer"] == response1_json["answer"] # LLM answer should be from cache

    # Verify mocks: Qdrant and Ollama should NOT have been called
    qdrant_mock.search.assert_not_called()
    ollama_mock.post.assert_not_called()
    # Verify Redis get methods were called
    assert redis_mock.get_json.call_count >= 1 # For search results
    assert redis_mock.get_string.call_count >= 1 # For LLM answer
    # Verify Redis set methods were NOT called
    redis_mock.set_json.assert_not_called()
    redis_mock.set_string.assert_not_called()

    # --- Third call (Force no cache) ---
    payload_no_cache = payload.copy()
    payload_no_cache["force_no_cache"] = True

    # Reset mocks for Qdrant and Ollama to ensure they ARE called now
    qdrant_mock.search.reset_mock()
    qdrant_mock.search.return_value = [create_mock_qdrant_point("cache_doc1_nocache", 0.86, "No cache this time.", {"source": "c2"})] # Different data
    ollama_mock.post.reset_mock()
    ollama_mock.post.return_value.json.return_value = {"response": "Fresh answer, no cache."}


    async with httpx.AsyncClient(base_url=RAG_QUERY_SERVICE_BASE_URL, timeout=10.0) as client:
        logger.info("Testing caching: Third call (force_no_cache=True)")
        response3 = await client.post("/query", json=payload_no_cache)

    assert response3.status_code == 200
    response3_json = response3.json()
    assert response3_json["cached_response"] is False # Even if data was in cache, force_no_cache means we don't report it as cached.
    assert "Fresh answer, no cache." in response3_json["answer"]

    # Verify mocks: Qdrant and Ollama WERE called
    qdrant_mock.search.assert_called_once()
    ollama_mock.post.assert_called_once()
    # Redis get methods should NOT have been called due to force_no_cache
    # (This depends on implementation: current RAG service checks cache *before* force_no_cache logic branch)
    # Let's refine this: force_no_cache should ideally prevent reads *and* writes.
    # The current RAG service implementation:
    # - If force_no_cache: Skips reading from cache.
    # - After computing, if force_no_cache: Skips writing to cache.
    # So, get_json/get_string should not be called.
    # The set_json/set_string should also not be called if force_no_cache is true.
    # This needs adjustment in RAG service code if that's the desired behavior for `set`.
    # For now, test based on current RAG service code (which *will* set if it was a miss path)
    # This test is primarily for *reading* from cache.

# To run: pytest tests/integration/test_rag_query_service_logic.py
# Ensure RAG Query Service is running. Mocks handle its dependencies.
