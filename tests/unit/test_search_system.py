"""
Unit tests for the AdvancedSearchSystem and its components.
"""
import unittest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from Scripts.search_system import AdvancedSearchSystem, SearchResult, CrossEncoderReRanker, KnowledgeGraphSearch
from Scripts.embeddings import EmbeddingProvider # For mocking
from Scripts.knowledge_graph.graph import KnowledgeGraph # For mocking KG
from Scripts.embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta

# Since AdvancedSearchSystem methods are async and involve other async components,
# we use IsolatedAsyncioTestCase.

class TestAdvancedSearchSystem(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Mock EmbeddingProvider
        self.mock_embedding_provider = AsyncMock(spec=EmbeddingProvider)
        self.mock_embedding_provider.get_embedding_dim.return_value = 384 # Example dim
        async def mock_generate_embeddings(texts, **kwargs):
            if isinstance(texts, str): texts = [texts]
            return [[0.1] * 384 for _ in texts] # Return dummy embeddings of correct dim
        self.mock_embedding_provider.generate_embeddings = mock_generate_embeddings

        # Mock EmbeddingStrategyManager
        self.mock_strategy_manager = MagicMock(spec=EmbeddingStrategyManager)
        self.mock_strategy_manager.get_embedding_provider.return_value = self.mock_embedding_provider

        # Sample documents for keyword and semantic indexing
        self.sample_docs = [
            {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
            {"id": "doc2", "text": "A powerful new language model was released today for all languages."},
            {"id": "doc3", "text": "Exploring the future of artificial intelligence and machine learning."},
            {"id": "doc4", "text": "Another document about dogs and foxes, quick and lazy references."},
        ]

        # Mock KnowledgeGraph and KnowledgeGraphSearch
        self.mock_kg = MagicMock(spec=KnowledgeGraph)
        self.mock_kg_search = AsyncMock(spec=KnowledgeGraphSearch)
        async def mock_kg_search_method(query, top_k):
            if "fox" in query:
                return [SearchResult(doc_id="doc1_kg", score=0.8, source_type="graph", content="KG data about fox")]
            return []
        self.mock_kg_search.search = mock_kg_search_method

        self.search_system = AdvancedSearchSystem(
            embedding_strategy_manager=self.mock_strategy_manager,
            knowledge_graph_instance=self.mock_kg, # Pass the mock KG instance
            initial_documents=self.sample_docs
        )
        # Override the KG search component with our mock if it was created
        if self.search_system.knowledge_graph_search:
             self.search_system.knowledge_graph_search.search = mock_kg_search_method


        # Build semantic index (FAISS) with dummy embeddings
        # Need to ensure faiss is available or mock it too for pure unit tests not requiring faiss binary
        try:
            import faiss
            await self.search_system.build_semantic_index(self.sample_docs)
        except ImportError:
            # If faiss is not installed, we can't fully test semantic search.
            # We can mock self.search_system._perform_semantic_search
            self.search_system._perform_semantic_search = AsyncMock(return_value=[
                SearchResult(doc_id="doc1", score=0.9, source_type="semantic", content=self.sample_docs[0]["text"])
            ])
            print("FAISS not installed, _perform_semantic_search mocked.")


    def test_initialization(self):
        self.assertIsNotNone(self.search_system.keyword_search_component)
        self.assertTrue(len(self.search_system.documents_for_keyword_search) == len(self.sample_docs))
        if self.search_system.knowledge_graph_search: # Check if KG search was initialized
             self.assertIsNotNone(self.search_system.knowledge_graph_search.kg)


    async def test_keyword_search_component(self):
        """ Test that keyword search returns something plausible """
        query = "fox dog"
        # The keyword search is via ExistingHybridSearch, which is tested separately if needed.
        # Here we test its integration.
        keyword_results_raw = self.search_system.keyword_search_component.search(
            query, self.search_system.documents_for_keyword_search
        )
        self.assertTrue(len(keyword_results_raw) > 0)
        # Check if doc1 or doc4 (containing fox and dog) are in results
        found_relevant_doc = False
        for res in keyword_results_raw:
            doc_id = self.search_system.doc_ids_for_keyword_search[res['index']]
            if doc_id in ["doc1", "doc4"]:
                found_relevant_doc = True
                break
        self.assertTrue(found_relevant_doc, "Keyword search did not find relevant docs.")

    async def test_semantic_search_via_main_search_method(self):
        """ Test semantic search part of the main search method """
        query = "lazy canine" # Should match "lazy dog"

        # Force only semantic search
        results = await self.search_system.search(query, top_k=1, search_type_weights={"semantic": 1.0, "keyword": 0.0, "graph": 0.0})

        self.assertTrue(len(results) > 0)
        # This assertion depends on the dummy embeddings and FAISS behavior or the mock
        # If _perform_semantic_search was mocked:
        if isinstance(self.search_system._perform_semantic_search, AsyncMock):
             self.assertEqual(results[0].doc_id, "doc1") # From the mock
             self.assertEqual(results[0].source_type, "semantic")
        else:
            # If real FAISS is used with dummy embeddings, it's harder to predict exact match,
            # but we expect some result.
            self.assertIsInstance(results[0], SearchResult)
            self.assertEqual(results[0].source_type, "semantic")


    def test_reciprocal_rank_fusion(self):
        list1 = [SearchResult("docA", 0.9, "textA", source_type="s1"), SearchResult("docB", 0.8, "textB", source_type="s1")]
        list2 = [SearchResult("docB", 0.85, "textB", source_type="s2"), SearchResult("docA", 0.75, "textA", source_type="s2")]
        list3 = [SearchResult("docC", 0.95, "textC", source_type="s3"), SearchResult("docA", 0.85, "textA", source_type="s3")]

        fused_results = self.search_system.reciprocal_rank_fusion([list1, list2, list3], k=1) # Using k=1 for simpler score check

        # Expected RRF scores with k=1:
        # docA: 1/(1+0) + 1/(1+1) + 1/(1+1) = 1 + 0.5 + 0.5 = 2.0
        # docB: 1/(1+1) + 1/(1+0)           = 0.5 + 1     = 1.5
        # docC: 1/(1+0)                     = 1

        self.assertTrue(len(fused_results) == 3)
        self.assertEqual(fused_results[0].doc_id, "docA")
        self.assertAlmostEqual(fused_results[0].score, 1/(1+0) + 1/(1+1) + 1/(1+1)) # Rank 0, 1, 1
        self.assertEqual(fused_results[1].doc_id, "docB")
        self.assertAlmostEqual(fused_results[1].score, 1/(1+1) + 1/(1+0)) # Rank 1, 0
        self.assertEqual(fused_results[2].doc_id, "docC")
        self.assertAlmostEqual(fused_results[2].score, 1/(1+0)) # Rank 0

    @patch('Scripts.search_system.CrossEncoderReRanker.rerank', autospec=True)
    async def test_search_with_reranker_mocked(self, mock_rerank_method):
        # Mock the reranker's rerank method to return its input sorted by a dummy new score
        def side_effect_rerank(self_reranker_instance, query, results):
            for i, r in enumerate(results):
                r.score = 1.0 - (i * 0.1) # Assign new dummy scores
            return sorted(results, key=lambda x: x.score, reverse=True)
        mock_rerank_method.side_effect = side_effect_rerank

        query = "fox"
        # Run search with reranking enabled
        # Ensure initial_documents provide content for reranking
        self.search_system.doc_content_map = {doc['id']: doc['text'] for doc in self.sample_docs}

        _ = await self.search_system.search(query, top_k=2, use_reranker=True)

        mock_rerank_method.assert_called_once()
        # Further assertions could check if the results passed to rerank were the RRF fused ones
        # and if the final results are indeed what the mocked rerank method returned.

    async def test_graph_search_integration(self):
        query = "information about fox"
        # Only enable graph search
        results = await self.search_system.search(query, top_k=1, search_type_weights={"semantic":0,"keyword":0,"graph":1.0})
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].doc_id, "doc1_kg")
        self.assertEqual(results[0].source_type, "graph")


class TestCrossEncoderReRanker(unittest.TestCase):
    @patch('sentence_transformers.CrossEncoder') # Mock the actual CrossEncoder model loading
    def test_reranker_initialization_and_call(self, MockCrossEncoder):
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [0.9, 0.1] # Dummy scores
        MockCrossEncoder.return_value = mock_model_instance

        reranker = CrossEncoderReRanker(model_name="test-model")
        self.assertIsNotNone(reranker.model)

        query = "test query"
        results = [
            SearchResult(doc_id="d1", score=0.5, content="content1", source_type="s1"),
            SearchResult(doc_id="d2", score=0.4, content="content2", source_type="s1"),
        ]
        reranked = reranker.rerank(query, results)

        mock_model_instance.predict.assert_called_once_with([[query, "content1"], [query, "content2"]], show_progress_bar=False)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0].doc_id, "d1") # Based on predict order and scores
        self.assertEqual(reranked[0].score, 0.9)
        self.assertEqual(reranked[1].doc_id, "d2")
        self.assertEqual(reranked[1].score, 0.1)
        self.assertIn('original_fused_score', reranked[0].metadata)


if __name__ == '__main__':
    unittest.main()

```
