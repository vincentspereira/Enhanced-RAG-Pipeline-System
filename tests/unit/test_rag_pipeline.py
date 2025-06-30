import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import pytest
from pathlib import Path

from Scripts.config.manager import SystemConfig, ModelConfig, VectorStoreConfig, ProcessingConfig, APIConfig, PathsConfig, CacheSettingsConfig, FeatureFlagsConfig, ElasticsearchConfig
from Scripts.rag_pipeline import RAGPipeline

# Mock external dependencies
MockDocumentProcessor = MagicMock()
MockEmbeddingGenerator = MagicMock()
MockElasticsearchFallback = AsyncMock() # Use AsyncMock for async methods

@pytest.fixture
def mock_app_config_es_enabled():
    return SystemConfig(
        model=ModelConfig(embedding_model="test-sentence-transformer", device="cpu", batch_size=4, max_length=128),
        vector_store=VectorStoreConfig(host="localhost", port=6333, collection_name="test_collection", vector_size=384),
        processing=ProcessingConfig(chunk_size=100, chunk_overlap=10, batch_size=4, max_workers=1),
        api=APIConfig(),
        paths=PathsConfig(cache_dir="test_cache_dir"),
        cache_settings=CacheSettingsConfig(ttl=60),
        feature_flags=FeatureFlagsConfig(enable_elasticsearch_fallback=True),
        elasticsearch=ElasticsearchConfig(hosts=["http://localhost:9200"], index_name="test_es_index", enable_hybrid_search_in_es=False)
    )

@pytest.fixture
def mock_app_config_es_disabled():
    return SystemConfig(
        model=ModelConfig(embedding_model="test-sentence-transformer", device="cpu"),
        vector_store=VectorStoreConfig(host="localhost", port=6333, collection_name="test_collection", vector_size=384),
        processing=ProcessingConfig(),
        api=APIConfig(),
        paths=PathsConfig(cache_dir="test_cache_dir"),
        cache_settings=CacheSettingsConfig(),
        feature_flags=FeatureFlagsConfig(enable_elasticsearch_fallback=False),
        elasticsearch=ElasticsearchConfig()
    )

@patch('Scripts.rag_pipeline.SentenceTransformer', MagicMock())
@patch('Scripts.rag_pipeline.QdrantClient', MagicMock())
@patch('Scripts.rag_pipeline.ElasticsearchFallback', MockElasticsearchFallback)
@patch('Scripts.rag_pipeline.DocumentProcessor', MockDocumentProcessor)
@patch('Scripts.rag_pipeline.EmbeddingGenerator', MockEmbeddingGenerator)
@pytest.mark.asyncio
class TestRAGPipelineAsyncInteractions(unittest.TestCase): # Inherit from unittest.TestCase for structure if preferred

    def setUp(self):
        MockElasticsearchFallback.reset_mock()
        MockElasticsearchFallback.return_value.initialize = AsyncMock() # Ensure initialize is AsyncMock
        MockElasticsearchFallback.return_value.index_documents = AsyncMock()
        MockElasticsearchFallback.return_value.search = AsyncMock()
        MockDocumentProcessor.reset_mock()
        MockEmbeddingGenerator.reset_mock()

    async def test_async_initialize_components_es_enabled(self, mock_app_config_es_enabled):
        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()

        pipeline = RAGPipeline(
            app_config=mock_app_config_es_enabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )
        self.assertIsNotNone(pipeline.es_fallback)

        await pipeline.async_initialize_components()
        pipeline.es_fallback.initialize.assert_awaited_once()

    async def test_async_initialize_components_es_disabled(self, mock_app_config_es_disabled):
        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()

        pipeline = RAGPipeline(
            app_config=mock_app_config_es_disabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )
        self.assertIsNone(pipeline.es_fallback)
        await pipeline.async_initialize_components() # Should do nothing for es_fallback
        MockElasticsearchFallback.return_value.initialize.assert_not_called()


    @patch('Scripts.rag_pipeline.tqdm', lambda x, **kwargs: x)
    @patch('asyncio.get_event_loop') # To mock loop.run_in_executor
    async def test_process_documents_indexes_to_es_async(self, mock_get_loop, mock_app_config_es_enabled):
        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(side_effect=lambda _, func, *args: asyncio.coroutine(func)(*args) if asyncio.iscoroutinefunction(func) else func(*args)) # Simplified mock
        mock_get_loop.return_value = mock_loop

        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()

        sample_chunks = [
            {"text": "chunk1 text", "embedding": [0.1]*384, "metadata": {"source": "doc1", "chunk_index": 0}},
            {"text": "chunk2 text", "embedding": [0.2]*384, "metadata": {"source": "doc1", "chunk_index": 1}},
        ]
        # Simulate that process_directory and process_chunks are sync and will be run in executor
        mock_doc_processor_instance.process_directory.return_value = sample_chunks
        mock_emb_gen_instance.process_chunks.return_value = sample_chunks

        mock_es_instance = MockElasticsearchFallback.return_value # Get the instance from the class mock
        mock_es_instance.index_documents = AsyncMock() # Ensure it's an AsyncMock for awaiting

        pipeline = RAGPipeline(
            app_config=mock_app_config_es_enabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )
        # pipeline.es_fallback is already the MockElasticsearchFallback.return_value

        await pipeline.process_documents(input_dir=Path("dummy_dir"), batch_size=32)

        mock_es_instance.index_documents.assert_awaited_once()
        args, kwargs = mock_es_instance.index_documents.call_args
        indexed_docs = args[0]
        passed_embeddings = kwargs.get("embeddings")
        self.assertEqual(len(indexed_docs), 2)
        self.assertEqual(indexed_docs[0]['content'], "chunk1 text")
        self.assertIsNone(passed_embeddings) # Because enable_hybrid_search_in_es is False in fixture

    async def test_keyword_search_calls_es_fallback_async(self, mock_app_config_es_enabled):
        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()

        mock_es_instance = MockElasticsearchFallback.return_value
        mock_es_instance.search = AsyncMock(return_value=[
            {"content": "es result 1", "metadata": {"src": "es_doc1"}, "score": 0.9, "doc_id": "es1"},
        ])

        pipeline = RAGPipeline(
            app_config=mock_app_config_es_enabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )

        results = await pipeline._keyword_search(query="test query", limit=1) # Now directly awaited

        mock_es_instance.search.assert_awaited_once_with(query="test query", query_vector=None, size=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "es result 1")
        self.assertEqual(results[0]['keyword_score'], 0.9)

    async def test_keyword_search_es_disabled_async(self, mock_app_config_es_disabled):
        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()
        pipeline = RAGPipeline(
            app_config=mock_app_config_es_disabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )
        results = await pipeline._keyword_search(query="test query", limit=2)
        self.assertEqual(results, [])

    @patch('asyncio.get_event_loop') # For _semantic_search if it uses run_in_executor
    async def test_search_calls_async_keyword_search(self, mock_get_loop_search, mock_app_config_es_enabled):
        """Test that the main search method calls the async _keyword_search."""
        mock_loop_search = MagicMock()
        # Make run_in_executor pass through the function call for _semantic_search (sync)
        mock_loop_search.run_in_executor = MagicMock(side_effect=lambda _, func, *args: func(*args))
        mock_get_loop_search.return_value = mock_loop_search

        mock_doc_processor_instance = MockDocumentProcessor()
        mock_emb_gen_instance = MockEmbeddingGenerator()

        pipeline = RAGPipeline(
            app_config=mock_app_config_es_enabled,
            doc_processor=mock_doc_processor_instance,
            embedding_generator=mock_emb_gen_instance
        )

        # Mock the sub-search methods
        pipeline._semantic_search = MagicMock(return_value=[{"text": "semantic_res", "metadata": {}, "semantic_score": 0.85}])
        # _keyword_search is now async, so use AsyncMock for it
        pipeline._keyword_search = AsyncMock(return_value=[{"text": "keyword_res", "metadata": {}, "keyword_score": 0.75}])
        pipeline.cache.get = MagicMock(return_value=None) # Cache miss
        pipeline.cache.put = MagicMock()

        await pipeline.search("test query")

        pipeline._semantic_search.assert_called_once()
        pipeline._keyword_search.assert_awaited_once() # Check it was awaited
        pipeline.cache.put.assert_called_once()

if __name__ == '__main__':
    pytest.main()
