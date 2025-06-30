import pytest
import asyncio
from pathlib import Path
import os
import time # For potential sleeps if needed for ES indexing

from Scripts.config.manager import ConfigManager
from Scripts.rag_pipeline import RAGPipeline
from Scripts.document_processor import DocumentProcessor as ActualDocumentProcessor
from Scripts.embedding_generator import EmbeddingGenerator as ActualEmbeddingGenerator
from qdrant_client import QdrantClient, models as qdrant_models
from elasticsearch import AsyncElasticsearch # Using AsyncElasticsearch for setup/teardown

# Test data (same as before)
SAMPLE_DOCS_CONTENT = [
    {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog.", "category": "animals", "timestamp": "2023-01-01T10:00:00Z"},
    {"id": "doc2", "text": "Elasticsearch is a powerful search engine for text.", "category": "software", "timestamp": "2023-01-02T12:00:00Z"},
    {"id": "doc3", "text": "Qdrant is a vector database, efficient for similarity search over fox and dog vectors.", "category": "software", "timestamp": "2023-01-03T14:00:00Z"},
    {"id": "doc4", "text": "A lazy dog also enjoys a textual challenge from a quick fox.", "category": "animals", "timestamp": "2023-01-04T16:00:00Z"},
    {"id": "doc5", "text": "Vector search helps find similar items quickly.", "category": "concepts", "timestamp": "2023-01-05T18:00:00Z"}
]

@pytest.fixture(scope="session")
def event_loop():
    """Ensure a single event loop for session-scoped async fixtures."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config_manager_integration():
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(exist_ok=True)
    test_config_path = config_dir / "config.integration.yaml" # Distinct name

    test_yaml_content = """
vector_store:
  host: "localhost"
  port: 6333 # Standard Qdrant port
  collection_name: "test_integration_rag_collection"
  vector_size: 384

model:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

elasticsearch:
  hosts: ["http://localhost:9200"] # Standard Elasticsearch port
  index_name: "test_integration_rag_es_index"
  enable_hybrid_search_in_es: False

feature_flags:
  enable_elasticsearch_fallback: True

paths:
  cache_dir: "test_cache_integration_system"
# Add other minimal necessary configs for RAGPipeline
processing: {}
api: {}
cache_settings: {}
    """
    with open(test_config_path, "w") as f:
        f.write(test_yaml_content)

    cm = ConfigManager(config_path=str(test_config_path))
    return cm

@pytest.fixture(scope="session")
async def qdrant_es_clients(test_config_manager_integration):
    """Provides Qdrant and Elasticsearch clients and handles cleanup."""
    config = test_config_manager_integration.config

    q_client = QdrantClient(host=config.vector_store.host, port=config.vector_store.port)
    es_client = AsyncElasticsearch(hosts=config.elasticsearch.hosts)

    # Cleanup before tests
    try:
        q_client.delete_collection(collection_name=config.vector_store.collection_name)
        print(f"Deleted Qdrant collection {config.vector_store.collection_name} if it existed.")
    except Exception: pass # Ignore if collection doesn't exist

    if await es_client.indices.exists(index=config.elasticsearch.index_name):
        await es_client.indices.delete(index=config.elasticsearch.index_name)
        print(f"Deleted ES index {config.elasticsearch.index_name} if it existed.")

    yield q_client, es_client # Provide clients to the pipeline fixture

    # Teardown after tests
    try:
        q_client.delete_collection(collection_name=config.vector_store.collection_name)
    except Exception: pass
    if await es_client.indices.exists(index=config.elasticsearch.index_name):
        await es_client.indices.delete(index=config.elasticsearch.index_name)

    q_client.close()
    await es_client.close()


@pytest.fixture(scope="session")
async def initialized_rag_pipeline_integration(test_config_manager_integration, qdrant_es_clients, event_loop):
    config = test_config_manager_integration.config
    q_client, es_client = qdrant_es_clients # Unpack clients

    # For integration tests, we use the actual components
    doc_processor = ActualDocumentProcessor()
    embedding_generator = ActualEmbeddingGenerator(model_name=config.model.embedding_model, device=config.model.device)

    pipeline = RAGPipeline(
        app_config=config,
        doc_processor=doc_processor,
        embedding_generator=embedding_generator
    )
    await pipeline.async_initialize_components() # Await the async initialization

    # Simplified data indexing for the test
    all_texts = [doc["text"] for doc in SAMPLE_DOCS_CONTENT]
    # Embedding generation is sync, run in executor if it were a bottleneck for fixture setup
    all_embeddings = await event_loop.run_in_executor(None, embedding_generator.generate_embeddings, all_texts)

    qdrant_points = []
    es_docs_to_index = []

    for i, doc_content in enumerate(SAMPLE_DOCS_CONTENT):
        doc_id_val = doc_content["id"]
        text = doc_content["text"]
        embedding = all_embeddings[i].tolist()

        qdrant_points.append(qdrant_models.PointStruct(
            id=i, # Simple int ID for Qdrant test
            vector=embedding,
            payload={"text": text, "metadata": {"source": doc_id_val, "category": doc_content["category"], "timestamp": doc_content["timestamp"], "chunk_index": 0}}
        ))

        es_docs_to_index.append({
            "id": pipeline._get_document_id(text),
            "content": text,
            "metadata": {"source": doc_id_val, "category": doc_content["category"], "timestamp": doc_content["timestamp"]},
            "doc_id_source": f"{doc_id_val}_0"
        })

    # Index into Qdrant (using sync client method in executor)
    await event_loop.run_in_executor(None, pipeline.client.upsert, pipeline.collection_name, qdrant_points)

    if pipeline.es_fallback:
        await pipeline.es_fallback.index_documents(es_docs_to_index, embeddings=None) # Assuming no ES hybrid for this test config
        await asyncio.sleep(2) # Give ES time to index

    yield pipeline


@pytest.mark.asyncio
async def test_full_hybrid_search_integration(initialized_rag_pipeline_integration: RAGPipeline):
    pipeline = initialized_rag_pipeline_integration
    query = "quick fox and vector database"
    results = await pipeline.search(query=query, limit=5) # Use await

    assert len(results) > 0
    found_doc1_or_4_by_keyword = any(r['metadata']['source'] in ['doc1', 'doc4'] and r.get('keyword_score', 0) > 0.01 for r in results) # Lowered threshold for keyword
    found_doc3_by_semantic_and_keyword = any(r['metadata']['source'] == 'doc3' and r.get('semantic_score', 0) > 0.1 and r.get('keyword_score', 0) > 0.01 for r in results)

    print("\nIntegration Hybrid Search Results:")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:50]}..., Source: {r_val['metadata']['source']}, Final: {r_val['score']:.4f}, Sem: {r_val.get('semantic_score',0):.4f}, Key: {r_val.get('keyword_score',0):.4f}")

    assert found_doc1_or_4_by_keyword, "Expected 'doc1' or 'doc4' (keyword match for 'fox')"
    assert found_doc3_by_semantic_and_keyword, "Expected 'doc3' (semantic for 'vector database' and keyword for 'fox')"
    for r_val in results:
        assert 'score' in r_val and r_val['score'] > 0

@pytest.mark.asyncio
async def test_keyword_dominant_search_integration(initialized_rag_pipeline_integration: RAGPipeline):
    pipeline = initialized_rag_pipeline_integration
    query = "powerful search engine"

    # To make it keyword dominant for real, we don't mock _semantic_search here.
    # We rely on the query itself being a stronger keyword match for doc2 than a semantic one.
    results = await pipeline.search(query=query, limit=3)

    print("\nIntegration Keyword Dominant Results:")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:50]}..., Source: {r_val['metadata']['source']}, Final: {r_val['score']:.4f}, Key: {r_val.get('keyword_score',0):.4f}, Sem: {r_val.get('semantic_score',0):.4f}")

    assert len(results) > 0
    # Check if doc2 is the top result and its keyword score contributes significantly
    top_result_is_doc2 = results[0]['metadata']['source'] == 'doc2'
    top_result_has_strong_keyword = results[0].get('keyword_score', 0) > 0.1 # Check if keyword score is substantial

    assert top_result_is_doc2 and top_result_has_strong_keyword

@pytest.mark.asyncio
async def test_semantic_dominant_search_integration(initialized_rag_pipeline_integration: RAGPipeline):
    pipeline = initialized_rag_pipeline_integration
    query = "finding similar items very quickly"

    # To make it semantic dominant for real, we don't mock _keyword_search.
    # We rely on the query and data.
    results = await pipeline.search(query=query, limit=3)

    print("\nIntegration Semantic Dominant Results:")
    for r_idx, r_val in enumerate(results):
        print(f"Result {r_idx+1}: Text: {r_val['text'][:50]}..., Source: {r_val['metadata']['source']}, Final: {r_val['score']:.4f}, Sem: {r_val.get('semantic_score',0):.4f}, Key: {r_val.get('keyword_score',0):.4f}")

    assert len(results) > 0
    # Check if doc5 is the top result and its semantic score contributes significantly
    top_result_is_doc5 = results[0]['metadata']['source'] == 'doc5'
    top_result_has_strong_semantic = results[0].get('semantic_score', 0) > 0.1 # Check if semantic score is substantial

    assert top_result_is_doc5 and top_result_has_strong_semantic

@pytest.mark.asyncio
async def test_process_documents_and_search_integration(initialized_rag_pipeline_integration: RAGPipeline, test_config_manager_integration, event_loop):
    """Test processing a new document and then searching for it."""
    pipeline = initialized_rag_pipeline_integration
    config = test_config_manager_integration.config

    # Create a dummy file for processing
    temp_dir = Path("temp_test_docs_integration")
    temp_dir.mkdir(exist_ok=True)
    new_doc_file = temp_dir / "new_integration_doc.txt"
    new_doc_text = "Unique content about quantum widgets and flux capacitors for integration."
    new_doc_id_source = "new_integ_doc.txt" # This is the 'source' metadata

    with open(new_doc_file, "w") as f:
        f.write(new_doc_text)

    # Process this new document (now async)
    # Ensure DocumentProcessor and EmbeddingGenerator are correctly configured for this temp path
    # For this test, we'll assume they can pick up files from this path.
    # We need to re-initialize pipeline's doc_processor or mock it to point to temp_dir
    # Simplification: Create a new DP for this test.

    # This test is becoming complex because process_documents takes a dir.
    # A more focused test would be to manually craft the "processed_chunks" and call the indexing parts.
    # However, to test `process_documents` itself:

    # We need to ensure the pipeline's doc_processor and embedding_generator are set up
    # for the test context if they are not passed in or reconfigured.
    # The fixture `initialized_rag_pipeline_integration` uses Actual... classes.
    # We assume `pipeline.doc_processor.process_directory` can handle `temp_dir`.

    await pipeline.process_documents(input_dir=temp_dir, batch_size=1)
    await asyncio.sleep(3) # Give ES time to index

    # Test 1: Keyword search for the new document
    keyword_query = "quantum widgets flux"
    results_keyword = await pipeline.search(query=keyword_query, limit=1)
    print(f"\nKeyword search for new doc ('{keyword_query}'): {results_keyword}")
    assert len(results_keyword) >= 1, "New doc not found by keyword"
    assert results_keyword[0]['metadata']['source'] == new_doc_id_source
    assert results_keyword[0].get('keyword_score', 0) > 0.1

    # Test 2: Semantic search for the new document
    semantic_query = "advanced physics components"
    results_semantic = await pipeline.search(query=semantic_query, limit=1)
    print(f"\nSemantic search for new doc ('{semantic_query}'): {results_semantic}")
    assert len(results_semantic) >= 1, "New doc not found by semantic search"
    assert results_semantic[0]['metadata']['source'] == new_doc_id_source
    assert results_semantic[0].get('semantic_score', 0) > 0.1

    # Cleanup dummy file and dir
    os.remove(new_doc_file)
    os.rmdir(temp_dir)
