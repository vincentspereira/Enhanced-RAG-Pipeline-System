import pytest
import asyncio
import numpy as np
from Scripts.rag_pipeline import EnhancedRAGPipeline
from Scripts.models.embeddings import HuggingFaceEmbedding
from Scripts.vector_stores.base import QdrantVectorStore
from Scripts.active_learning.learner import RelevanceFeedback

@pytest.fixture(scope="module")
def config_path():
    return "tests/test_config.yaml"

@pytest.fixture(scope="module")
def pipeline(config_path):
    return EnhancedRAGPipeline(config_path)

@pytest.fixture(scope="module")
def sample_document():
    return {
        "id": "test-doc-1",
        "content": """
        This is a test document about artificial intelligence and machine learning.
        It contains multiple paragraphs and technical terms.
        
        Neural networks and deep learning are fundamental concepts in AI.
        They enable machines to learn from examples and improve over time.
        """,
        "metadata": {
            "source": "test",
            "author": "test_user",
            "date": "2025-05-23"
        }
    }

@pytest.mark.asyncio
class TestPipelineIntegration:
    async def test_full_pipeline_flow(self, pipeline, sample_document):
        """Test the complete pipeline flow from document ingestion to search"""
        # 1. Process document
        result = await pipeline.process_document(
            sample_document["id"],
            sample_document["content"],
            sample_document["metadata"]
        )
        assert isinstance(result, str)
        
        # Wait for processing to complete
        await asyncio.sleep(2)
        
        # 2. Search for relevant content
        search_results = await pipeline.search(
            "neural networks and deep learning",
            top_k=3,
            rerank=True
        )
        
        assert len(search_results) > 0
        assert all(isinstance(r, dict) for r in search_results)
        assert all("score" in r for r in search_results)
        
        # 3. Add relevance feedback
        pipeline.add_relevance_feedback(
            query_id="test-query-1",
            document_id=search_results[0]["id"],
            is_relevant=True
        )
        
        # 4. Search again with feedback incorporated
        new_results = await pipeline.search(
            "neural networks and deep learning",
            top_k=3,
            rerank=True
        )
        
        assert len(new_results) > 0
        assert "relevance_score" in new_results[0]
    
    async def test_batch_processing(self, pipeline):
        """Test batch document processing"""
        documents = [
            {
                "id": f"batch-doc-{i}",
                "content": f"Test document {i} with some technical content about {topic}",
                "metadata": {"batch": "test", "index": i}
            }
            for i, topic in enumerate([
                "machine learning",
                "neural networks",
                "natural language processing",
                "computer vision",
                "reinforcement learning"
            ])
        ]
        
        # Process documents in batch
        tasks = [
            pipeline.process_document(
                doc["id"],
                doc["content"],
                doc["metadata"]
            )
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == len(documents)
        
        # Wait for processing to complete
        await asyncio.sleep(2)
        
        # Search across all documents
        search_results = await pipeline.search(
            "machine learning and neural networks",
            top_k=5
        )
        
        assert len(search_results) > 0
        assert any("machine learning" in r["metadata"]["content"] for r in search_results)
        assert any("neural networks" in r["metadata"]["content"] for r in search_results)
    
    async def test_error_handling(self, pipeline):
        """Test error handling in the pipeline"""
        # Test with invalid document
        with pytest.raises(Exception):
            await pipeline.process_document(
                "invalid-doc",
                None,
                {}
            )
        
        # Test with empty query
        with pytest.raises(Exception):
            await pipeline.search("")
        
        # Test with invalid feedback
        with pytest.raises(Exception):
            pipeline.add_relevance_feedback(
                query_id="test-query",
                document_id="nonexistent-doc",
                is_relevant=True
            )
    
    async def test_component_integration(self, pipeline):
        """Test integration between different components"""
        # 1. Test embedding model integration
        embedding_model = pipeline.model_registry.get_model("default")
        assert isinstance(embedding_model, HuggingFaceEmbedding)
        
        # 2. Test vector store integration
        vector_store = pipeline.store_registry.get_store("default")
        assert isinstance(vector_store, QdrantVectorStore)
        
        # 3. Test active learning integration
        if pipeline.config.enable_active_learning:
            # Add some feedback
            feedback = RelevanceFeedback(
                query_id="test-query",
                document_id="test-doc",
                is_relevant=True,
                feedback_source="user",
                confidence=1.0
            )
            
            # Generate a test embedding
            test_embedding = embedding_model.encode("test content")
            pipeline.active_learner.add_feedback(feedback, test_embedding)
            
            # Verify feedback was recorded
            assert len(pipeline.active_learner.relevance_history) > 0
    
    async def test_concurrent_access(self, pipeline):
        """Test concurrent access to the pipeline"""
        # Create multiple concurrent searches
        queries = [
            "machine learning",
            "neural networks",
            "artificial intelligence",
            "deep learning",
            "natural language processing"
        ]
        
        async def search_task(query):
            return await pipeline.search(query, top_k=3)
        
        # Execute searches concurrently
        tasks = [search_task(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) > 0 for r in results)
