"""
Unit tests for the AdvancedSearchSystem and its components.
"""
import unittest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

from Scripts.search_system import AdvancedSearchSystem, SearchResult, CrossEncoderReRanker, KnowledgeGraphSearch
from Scripts.embeddings import EmbeddingProvider # Used by AdvancedSearchSystem for type hints
from Scripts.knowledge_graph.graph import KnowledgeGraph # For mocking KG
from Scripts.embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta
# from dataclasses import dataclass # Not needed if SearchResult imported
from Scripts.federated_search.connectors import FederatedDataSource # Added for federated search tests

from Scripts.embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta
# dataclasses may not be needed if SearchResult is imported directly
# from dataclasses import dataclass

# Since AdvancedSearchSystem methods are async and involve other async components,
# we use IsolatedAsyncioTestCase.

# Ensure Scripts directory is in path for imports if running tests from root
import sys
import os # Ensure os is imported for abspath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Scripts')))

# Import SearchResult directly from the source module
from Scripts.search_system import SearchResult

class TestAdvancedSearchSystem(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Mock EmbeddingProvider (from the old Scripts.embeddings)
        self.mock_embedding_provider_instance = AsyncMock(spec=EmbeddingProvider)
        self.mock_embedding_provider_instance.get_embedding_dim.return_value = 384 # Example dim
        async def mock_generate_embeddings(texts, **kwargs):
            if isinstance(texts, str): texts = [texts]
            # Simulate embeddings that might lead to predictable FAISS results if not mocking FAISS itself
            # For "dog", make it distinct
            if "dog" in texts[0].lower():
                return [[0.1 + i*0.01] * 384 for i, _ in enumerate(texts)]
            return [[0.5 + i*0.01] * 384 for i, _ in enumerate(texts)]
        self.mock_embedding_provider_instance.generate_embeddings = mock_generate_embeddings

        # Mock EmbeddingStrategyManager
        self.mock_strategy_manager = MagicMock(spec=EmbeddingStrategyManager)
        self.mock_strategy_manager.get_embedding_provider.return_value = self.mock_embedding_provider_instance
        self.mock_strategy_manager.default_embedding_strategy_params = {"language": "en", "modality": "text"}


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
        # Mocking FAISS for unit tests to avoid binary dependency issues in all environments
        self.mock_faiss_index = MagicMock()
        async def mock_perform_semantic_search(query_embedding, top_k):
            # Simulate some results based on query_embedding characteristics if needed for more complex tests
            # For now, return a fixed list or a list derived from sample_docs
            # This mock should align with what build_semantic_index would have stored.
            # Let's assume "dog" query matches doc1 and doc4 semantically.
            # A real test with FAISS would be an integration test.
            if np.allclose(query_embedding, np.array([[0.1]*384])): # Matches "dog" based on our mock_generate_embeddings
                 return [
                    SearchResult(doc_id="doc1", score=0.95, source_type="semantic", content=self.sample_docs[0]["text"]),
                    SearchResult(doc_id="doc4", score=0.85, source_type="semantic", content=self.sample_docs[3]["text"]),
                 ]
            return [SearchResult(doc_id="doc3", score=0.9, source_type="semantic", content=self.sample_docs[2]["text"])] # Default mock

        with patch('faiss.IndexFlatIP', return_value=self.mock_faiss_index) as mock_faiss_constructor, \
             patch('faiss.normalize_L2') as mock_faiss_normalize:

            # Ensure that AdvancedSearchSystem's faiss_index is set to our mock after build
            # We need to ensure build_semantic_index uses the mocked faiss.IndexFlatIP
            # One way is to ensure that when build_semantic_index is called, the faiss module's
            # IndexFlatIP is already the mock.
            # The patch context manager should handle this if `import faiss` is inside build_semantic_index.
            # If `import faiss` is at module level, this patching strategy is more complex.
            # Assuming `import faiss` is local to methods using it, or that this patch works.

            # Temporarily assign the mock index to the system instance for testing search method
            # More robustly, build_semantic_index should use the patched faiss and set this.
            # For this setup, we'll also directly mock _perform_semantic_search for simplicity,
            # as testing the FAISS build process itself is an integration concern.

            await self.search_system.build_semantic_index(self.sample_docs) # This will now use mocked faiss if import is local
            # If build_semantic_index internally imports faiss, the patch should work.
            # If faiss is imported at module level in search_system.py, the patch needs to be at module level of search_system.
            # For now, let's confirm the state of self.search_system.faiss_index
            if self.search_system.faiss_index is not self.mock_faiss_index and self.search_system.faiss_index is not None:
                 print(f"Warning: search_system.faiss_index was not the mock. Type: {type(self.search_system.faiss_index)}")
                 # This means the actual faiss was likely used. For CI, this might be an issue.
                 # Fallback to mocking _perform_semantic_search directly for unit test reliability
                 self.search_system._perform_semantic_search = AsyncMock(side_effect=mock_perform_semantic_search)
                 print("FAISS interaction fully mocked by replacing _perform_semantic_search.")
            elif self.search_system.faiss_index is self.mock_faiss_index:
                 print("FAISS IndexFlatIP successfully mocked for build_semantic_index.")
                 # If FAISS was mocked for build, then _perform_semantic_search would use the mock index.
                 # We'd need to configure mock_faiss_index.search to return sensible values.
                 # For simplicity in this unit test, we'll still mock _perform_semantic_search.
                 self.search_system._perform_semantic_search = AsyncMock(side_effect=mock_perform_semantic_search)
                 print("FAISS interaction mocked by replacing _perform_semantic_search (even if build used mock index).")

            else: # faiss_index is None (build failed or was skipped)
                self.search_system._perform_semantic_search = AsyncMock(side_effect=mock_perform_semantic_search)
                print("FAISS index is None, _perform_semantic_search directly mocked.")


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

    def test_personalize_results_mocked_data(self):
        """ Test the _personalize_results method with mocked user data """
        # Sample results to be personalized
        results_to_personalize = [
            SearchResult(doc_id="doc1", score=0.9, content="Content about foxes and nature", metadata={"topic": "nature", "source_name": "doc_source_blog"}),
            SearchResult(doc_id="doc2", score=0.8, content="Content about AI models", metadata={"topic": "AI", "source_name": "doc_source_generic"}),
            SearchResult(doc_id="doc3", score=0.7, content="Content about machine learning", metadata={"topic": "machine learning", "source_name": "doc_source_academic"}),
            SearchResult(doc_id="doc_unrelated", score=0.6, content="Unrelated content", metadata={"topic": "sports"})
        ]

        # Test for user123: prefers AI/ML, liked doc3
        personalized_for_user123 = self.search_system._personalize_results(results_to_personalize.copy(), "user123", "test query for AI")

        self.assertTrue(len(personalized_for_user123) == len(results_to_personalize))
        # Expect doc3 (liked and topic match) to be boosted significantly
        # Expect doc2 (topic match) to be boosted
        # Scores should be different from original for relevant docs

        original_doc3_score = next(r.score for r in results_to_personalize if r.doc_id == "doc3")
        personalized_doc3_score = next(r.score for r in personalized_for_user123 if r.doc_id == "doc3")
        self.assertGreater(personalized_doc3_score, original_doc3_score)
        # Check if doc3 is now ranked higher (it should be due to like and topic)
        self.assertTrue(personalized_for_user123[0].doc_id == "doc3" or personalized_for_user123[0].doc_id == "doc2")


        # Test for user456: prefers foxes/nature, liked doc1, doc4
        personalized_for_user456 = self.search_system._personalize_results(results_to_personalize.copy(), "user456", "test query for nature")
        original_doc1_score = next(r.score for r in results_to_personalize if r.doc_id == "doc1")
        personalized_doc1_score = next(r.score for r in personalized_for_user456 if r.doc_id == "doc1")
        self.assertGreater(personalized_doc1_score, original_doc1_score)
        # Expect doc1 to be boosted significantly and likely be the top result
        self.assertEqual(personalized_for_user456[0].doc_id, "doc1")
        self.assertIn("personalization_factors_applied", personalized_for_user456[0].metadata)


    def test_personalize_results_unknown_user(self):
        """ Test personalization with an unknown user ID, should return original results """
        results_to_personalize = [
            SearchResult(doc_id="doc1", score=0.9, content="Content1"),
            SearchResult(doc_id="doc2", score=0.8, content="Content2"),
        ]
        original_scores = [r.score for r in results_to_personalize]

        personalized_results = self.search_system._personalize_results(results_to_personalize.copy(), "unknown_user_id", "test query")

        self.assertEqual(len(personalized_results), len(results_to_personalize))
        for i, res in enumerate(personalized_results):
            self.assertEqual(res.score, original_scores[i]) # Scores should be unchanged


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

    # Test for Federated Search
    async def test_federated_search_logic(self):
        """Tests the _federated_search_external method and its integration."""
        # Mock FederatedDataSource and its search method
        mock_connector = AsyncMock() # Mocking the FederatedDataSource abstract class directly for simplicity
        # If FederatedDataSource was a concrete class, you'd mock that.
        # If it's an ABC, you might need to create a mock concrete subclass or mock its abstract methods.

        # Let's assume FederatedDataSource is an ABC and we create a runtime mock implementing 'search'
        mock_connector.source_name = "test_fed_source"
        mock_connector.is_available = MagicMock(return_value=True)
        async def mock_search_method(query, top_k):
            return [SearchResult(doc_id="fed_doc1", score=0.7, content=f"Federated result for {query}", source_type="test_fed_source")]
        mock_connector.search = mock_search_method # Assign the async def to the mock

        self.search_system.federated_sources = [mock_connector]

        # Test _federated_search_external directly
        results = await self.search_system._federated_search_external("test query", sources=["test_fed_source"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "fed_doc1")
        self.assertEqual(results[0].source_type, "test_fed_source")

        # Test integration into main search method (mock other sources to return nothing)
        self.search_system._perform_semantic_search = AsyncMock(return_value=[])
        self.search_system.keyword_search_component.search = MagicMock(return_value=[])
        if self.search_system.knowledge_graph_search:
            self.search_system.knowledge_graph_search.search = AsyncMock(return_value=[])

        final_results = await self.search_system.search(
            "another federated test",
            top_k=1,
            search_type_weights={"semantic":0, "keyword":0, "graph":0, "federated": 1.0},
            use_reranker=False
        )
        self.assertEqual(len(final_results), 1)
        self.assertEqual(final_results[0].doc_id, "fed_doc1") # Assuming mock_search_method is general enough
        self.assertEqual(final_results[0].source_type, "test_fed_source") # RRF should preserve this if it's the only source

if __name__ == '__main__':
    unittest.main()

```
