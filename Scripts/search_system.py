"""
Advanced Search System for RAG.

This module implements the core hybrid search functionality, including:
- Semantic search (vector-based)
- Keyword search (BM25, TF-IDF)
- Graph-based search (future integration)
- Advanced ranking and re-ranking mechanisms
- Query enhancement
- Management of various embedding models
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np

# Existing components to integrate/leverage
from Scripts.enhancers.hybrid_search import HybridSearch as ExistingHybridSearch, SearchConfig as ExistingSearchConfig, SearchMode as ExistingSearchMode
from Scripts.knowledge_graph.graph import KnowledgeGraph
from Scripts.embeddings import EmbeddingProvider # Only EmbeddingProvider ABC needed here for type hint
from Scripts.embedding_strategy import EmbeddingStrategyManager, EmbeddingModelMeta # Import new strategy manager

logger = logging.getLogger(__name__)

# Define a common search result format
@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: Optional[str] = None # Full content, might be too large for all results
    chunk_id: Optional[str] = None # If search is over chunks
    metadata: Optional[Dict[str, Any]] = None
    source_type: str # e.g., "semantic", "keyword", "graph"

class AdvancedSearchSystem:
    def __init__(
        self,
        embedding_strategy_manager: EmbeddingStrategyManager, # Use manager instead of single provider
        knowledge_graph_instance: Optional[KnowledgeGraph] = None,
        initial_documents: Optional[List[Dict[str, Any]]] = None,
        default_embedding_strategy_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.embedding_strategy_manager = embedding_strategy_manager
        self.default_embedding_strategy_params = default_embedding_strategy_params or {"language": "en", "modality": "text"}

        # Get a default embedding provider to ascertain default dimension, etc.
        # This provider instance might change per query based on strategy.
        default_provider = self.embedding_strategy_manager.get_embedding_provider(self.default_embedding_strategy_params)
        self.embedding_dim = default_provider.get_embedding_dim()

        # Keyword Search Component (using ExistingHybridSearch for its BM25)
        self.keyword_search_config = ExistingSearchConfig(
            mode=ExistingSearchMode.KEYWORD,
            use_bm25=True,
            enable_spellcheck=False, # Spellcheck will be handled by AdvancedSearchSystem
            enable_query_expansion=False # Query expansion handled by AdvancedSearchSystem
        )
        self.keyword_search_component = ExistingHybridSearch(config=self.keyword_search_config)

        self.documents_for_keyword_search: List[str] = []
        self.doc_ids_for_keyword_search: List[str] = []
        self.doc_content_map: Dict[str, str] = {} # Store original content by id

        # Semantic Search Index (FAISS)
        self.faiss_index: Optional[Any] = None # faiss.Index
        self.faiss_doc_ids: List[str] = [] # To map FAISS indices back to document IDs

        # Query Enhancement (using ExistingHybridSearch for its methods)
        # We create a separate instance for query processing to avoid config clashes if any
        query_enhancer_config = ExistingSearchConfig(enable_spellcheck=True, enable_query_expansion=True)
        self.query_enhancer = ExistingHybridSearch(config=query_enhancer_config)


        if initial_documents:
            # This assumes initial_documents contains text and unique IDs
            # For semantic search, we'd need to embed these and build FAISS index
            # For keyword search, we pass them to build_keyword_index
            self.build_keyword_index(initial_documents)
            # Note: Building FAISS index requires embeddings, so it's a separate step.
            # async def build_semantic(): await self.build_semantic_index(initial_documents)
            # asyncio.run(build_semantic()) # Or handle async setup outside

        # Placeholder for graph search component
        # self.knowledge_graph_search: Optional[KnowledgeGraphSearch] = None
        # KG component will be initialized if a KG is provided
        if knowledge_graph_instance:
            # KG search might also need adaptive embeddings for its internal ops (e.g. matching query entities)
            # For now, pass the default provider, or it could fetch its own using the manager.
            kg_embedding_provider = self.embedding_strategy_manager.get_embedding_provider(
                kwargs.get("kg_embedding_strategy_params", self.default_embedding_strategy_params)
            )
            self.knowledge_graph_search = KnowledgeGraphSearch(
                kg=knowledge_graph_instance,
                embedding_provider=kg_embedding_provider
            )
            logger.info("KnowledgeGraphSearch component initialized.")
        else:
            self.knowledge_graph_search = None
            logger.info("KnowledgeGraph instance not provided, KnowledgeGraphSearch component not initialized.")

        # Re-ranking component
        reranker_model_name = kwargs.get("reranker_model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker: Optional[CrossEncoderReRanker] = CrossEncoderReRanker(model_name=reranker_model_name)


    def build_keyword_index(self, documents: List[Dict[str, Any]]):
        """
        Builds or updates the in-memory keyword search index (e.g., BM25)
        and stores document content.
        `documents` should be a list of dicts, each with at least 'id' and 'text' keys.
        """
        if not documents:
            self.documents_for_keyword_search = []
            self.doc_ids_for_keyword_search = []
            self.doc_content_map = {}
            return

        self.documents_for_keyword_search = [doc['text'] for doc in documents]
        self.doc_ids_for_keyword_search = [doc['id'] for doc in documents]
        self.doc_content_map = {doc['id']: doc['text'] for doc in documents}
        logger.info(f"Keyword index updated with {len(self.documents_for_keyword_search)} documents. Content stored.")


    async def build_semantic_index(self,
                                 documents: List[Dict[str, Any]],
                                 batch_size: int = 32,
                                 embedding_strategy_params: Optional[Dict[str, Any]] = None):
        """
        Builds or updates the FAISS index for semantic search using a specified or default strategy.
        `documents` should be a list of dicts, each with at least 'id' and 'text' keys.
        """
        import faiss # Import faiss dynamically

        if not documents:
            logger.warning("No documents provided to build semantic index.")
            return

        doc_texts = [doc['text'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]

        # Determine embedding strategy for indexing (could be different from query time)
        current_indexing_strategy_params = embedding_strategy_params or self.default_embedding_strategy_params
        indexing_provider = self.embedding_strategy_manager.get_embedding_provider(current_indexing_strategy_params)

        logger.info(f"Generating embeddings for {len(doc_texts)} documents to build semantic index using {indexing_provider.__class__.__name__}...")
        embeddings = await indexing_provider.generate_embeddings(doc_texts, batch_size=batch_size) # Use selected provider

        if not embeddings:
            logger.error("Failed to generate embeddings for semantic index.")
            return

        embeddings_np = np.array(embeddings).astype('float32')

        if self.embedding_dim is None:
            self.embedding_dim = embeddings_np.shape[1]
        elif self.embedding_dim != embeddings_np.shape[1]:
            logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings_np.shape[1]}")
            return

        logger.info(f"Building FAISS index with {embeddings_np.shape[0]} vectors of dimension {self.embedding_dim}.")

        # For cosine similarity, normalize vectors and use IndexFlatIP
        faiss.normalize_L2(embeddings_np)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings_np)
        self.faiss_doc_ids = doc_ids # Store corresponding IDs
        logger.info("FAISS index built successfully.")

    async def _perform_semantic_search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """ Helper for actual FAISS search """
        import faiss # Ensure faiss is available

        if not self.faiss_index or not self.faiss_doc_ids:
            logger.warning("FAISS index not built. Cannot perform semantic search.")
            return []

        query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding_np) # Normalize query vector

        scores, indices = self.faiss_index.search(query_embedding_np, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: # Should not happen with IndexFlatIP unless k > index size
                continue
            doc_id = self.faiss_doc_ids[idx]
            score = float(scores[0][i])
            results.append(SearchResult(doc_id=doc_id, score=score, content=self.doc_content_map.get(doc_id), source_type="semantic"))
        return results

    async def search(
        self,
        query: str,
        top_k: int = 10,
        search_type_weights: Optional[Dict[str, float]] = None,
        use_reranker: bool = False,
        query_embedding_strategy_params: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Performs a hybrid search with optional re-ranking and adaptive query embedding.
        """
        search_type_weights = search_type_weights or {"semantic": 0.6, "keyword": 0.4, "graph": 0.0}
        current_query_embedding_strategy_params = query_embedding_strategy_params or self.default_embedding_strategy_params

        # 0. Query Enhancement
        processed_query = self.query_enhancer.spell_check(query) # Uses its own SBERT, independent of strategy
        processed_query = self.query_enhancer.expand_query(processed_query) # Uses its own SBERT/spaCy
        logger.info(f"Original query: '{query}', Processed query: '{processed_query}'")

        semantic_results_list: List[SearchResult] = []
        keyword_results_list: List[SearchResult] = []
        graph_results_list: List[SearchResult] = []

        # 1. Semantic Search (Vector Search)
        if search_type_weights.get("semantic", 0.0) > 0 and self.faiss_index:
            query_embedding_provider = self.embedding_strategy_manager.get_embedding_provider(current_query_embedding_strategy_params)
            logger.info(f"Performing semantic search for: '{processed_query}' using {query_embedding_provider.__class__.__name__}")

            # Ensure embedding_dim is consistent with the chosen provider for the query
            # self.embedding_dim refers to the FAISS index dimension. Query embedding must match this.
            # This implies the FAISS index should be built with a "primary" or known dimension,
            # and query embeddings must be generated to match it.
            # For true adaptive *query* embeddings that might change dimension/model per query against a *single* index,
            # that index would need to support multi-vector segments or we'd need multiple indices.
            # For now, assume query embedding model matches FAISS index's model type/dimension.
            # If query_embedding_provider.get_embedding_dim() != self.embedding_dim:
            #     logger.warning("Query embedding model dim differs from FAISS index dim. This may lead to issues.")
            #     # Handle this: either re-embed query with index's model type, or error, or have separate indices.
            #     # For now, we assume the strategy manager provides a compatible model for the query
            #     # that matches the self.embedding_dim of the current FAISS index.

            query_embedding_list = await query_embedding_provider.generate_embeddings(processed_query)
            if query_embedding_list:
                query_embedding = query_embedding_list[0]
                # Ensure query_embedding matches self.embedding_dim (dimension of FAISS index)
                if len(query_embedding) != self.embedding_dim:
                    logger.error(f"Critical: Query embedding dimension ({len(query_embedding)}) from {query_embedding_provider.__class__.__name__} "
                                 f"does not match FAISS index dimension ({self.embedding_dim}). Skipping semantic search.")
                else:
                    semantic_results_list = await self._perform_semantic_search(np.array(query_embedding), top_k * 2) # Fetch more for RRF
            else:
                logger.error(f"Failed to generate query embedding for '{processed_query}'. Skipping semantic search.")


        # 2. Keyword Search (BM25/TF-IDF)
        if search_type_weights.get("keyword", 0.0) > 0 and self.documents_for_keyword_search:
            logger.info(f"Performing keyword search for: '{processed_query}'")
            # ExistingHybridSearch.search returns List[Dict[str, Any]] with "index", "score", "content"
            keyword_raw_results = self.keyword_search_component.search(processed_query, self.documents_for_keyword_search)

            # Convert to SearchResult objects
            # Normalization of BM25 scores for RRF might be tricky as they aren't bounded like cosine similarity.
            # RRF is typically used with ranks, so raw scores might be less of an issue if we just use their order.
            # However, if scores are directly used in a weighted sum before RRF, normalization is key.
            # For RRF, we primarily care about the rank.
            for res in keyword_raw_results:
                doc_id = self.doc_ids_for_keyword_search[res['index']]
                keyword_results_list.append(SearchResult(
                    doc_id=doc_id,
                    score=res['score'], # Using raw BM25 score here
                    content=self.doc_content_map.get(doc_id),
                    source_type="keyword"
                ))
            # Sort by score to ensure it's a ranked list for RRF
            keyword_results_list.sort(key=lambda x: x.score, reverse=True)


        # 3. Graph Search (Placeholder)
        if search_type_weights.get("graph", 0.0) > 0 and self.knowledge_graph_search:
            logger.info(f"Performing graph search for: '{processed_query}' (Placeholder)")
            # graph_results_list = self.knowledge_graph_search.search(processed_query, top_k=top_k * 2)
            pass

        # 4. Combine results using Reciprocal Rank Fusion
        all_ranked_lists = []
        if semantic_results_list:
            all_ranked_lists.append(semantic_results_list)
        if keyword_results_list:
            all_ranked_lists.append(keyword_results_list)
        if graph_results_list: # If/when implemented
            all_ranked_lists.append(graph_results_list)

        if not all_ranked_lists:
            return []

        fused_results = self.reciprocal_rank_fusion(all_ranked_lists)

        # 5. Contextual Re-ranking (Optional)
        if use_reranker and self.reranker:
            logger.info(f"Re-ranking top {len(fused_results)} results with cross-encoder...")
            # Ensure content is populated for reranker
            for res_item in fused_results:
                if not res_item.content and res_item.doc_id in self.doc_content_map:
                    res_item.content = self.doc_content_map[res_item.doc_id]

            final_results_list = self.reranker.rerank(processed_query, fused_results)
        else:
            final_results_list = fused_results

        return final_results_list[:top_k]

    def reciprocal_rank_fusion(self, ranked_lists: List[List[SearchResult]], k: int = 60) -> List[SearchResult]:
        """
        Performs Reciprocal Rank Fusion on multiple ranked lists of search results.
        k is a constant used in the RRF formula, typically 60.
        """
        fused_scores: Dict[str, float] = {} # doc_id -> RRF score

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list):
                doc_id = result.doc_id
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                fused_scores[doc_id] += 1.0 / (k + rank + 1) # Rank is 0-indexed

        # Sort documents by their fused RRF score
        sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

        # Create final list of SearchResult objects
        # Need to fetch content/metadata for these doc_ids from a central store or from original lists
        final_results = []
        # This requires a way to get the full SearchResult object from just doc_id
        # For now, this is a conceptual implementation of RRF score calculation.
        # We'd need to map these doc_ids back to their full SearchResult objects.
        logger.info("RRF score calculation complete. Mapping back to full results is needed.")

        # This part is incomplete as it needs to reconstruct SearchResult objects
        # from doc_ids and potentially merge metadata.
        # For a proper implementation, one might store all unique results from all lists
        # in a dictionary by doc_id, then use fused_scores to sort them.

        # Example (simplified, assumes we can retrieve content by doc_id):
        all_retrieved_docs_map = {}
        for r_list in ranked_lists:
            for res in r_list:
                if res.doc_id not in all_retrieved_docs_map:
                    all_retrieved_docs_map[res.doc_id] = res

        final_results = []
        for doc_id, score in sorted_fused_results:
            if doc_id in all_retrieved_docs_map:
                # Create a new SearchResult to avoid modifying originals, assign RRF score
                original_sr = all_retrieved_docs_map[doc_id]
                fused_sr = SearchResult(
                    doc_id=original_sr.doc_id,
                    score=score, # This is the RRF score
                    content=original_sr.content,
                    chunk_id=original_sr.chunk_id,
                    metadata=original_sr.metadata, # Could merge metadata if sources differ
                    source_type="fused" # Indicate this result is from fusion
                )
                final_results.append(fused_sr)

        return final_results

# CrossEncoderReRanker
@dataclass
class CrossEncoderReRanker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # A common default
    model: Optional[Any] = None # Stores the SentenceTransformer model

    def __post_init__(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"CrossEncoderReRanker initialized with model: {self.model_name}")
        except ImportError:
            logger.error("SentenceTransformers library not found. CrossEncoderReRanker will not function.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {e}")
            self.model = None

    def rerank(self, query: str, results: List[SearchResult], top_n: Optional[int] = None) -> List[SearchResult]:
        if not self.model or not results:
            logger.debug("CrossEncoder model not loaded or no results to rerank.")
            return results

        # Consider only top_n results for reranking if specified, to save computation
        results_to_rerank = results[:top_n] if top_n is not None else results

        sentence_pairs = [[query, res.content] for res in results_to_rerank if res.content]

        if not sentence_pairs:
            logger.debug("No content found in results for reranking.")
            return results_to_rerank # Or original results if top_n was used

        try:
            logger.info(f"Reranking {len(sentence_pairs)} pairs with cross-encoder...")
            scores = self.model.predict(sentence_pairs, show_progress_bar=False)

            for res, score in zip(results_to_rerank, scores):
                res.metadata = res.metadata or {}
                res.metadata['original_fused_score'] = res.score # Store pre-rerank score
                res.score = float(score) # Update score with cross-encoder score

            # Sort only the reranked portion
            reranked_results = sorted(results_to_rerank, key=lambda x: x.score, reverse=True)

            # If top_n was used, append the rest of the original results
            if top_n is not None and len(results) > top_n:
                reranked_results.extend(results[top_n:])

            return reranked_results

        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {e}")
            return results # Return original results on error


# KnowledgeGraphSearch component
@dataclass
class KnowledgeGraphSearch:
    kg: KnowledgeGraph # The actual knowledge graph
    embedding_provider: EmbeddingProvider # To embed query & entity names for matching
    # We might need a mapping from KG node IDs to original document IDs if not directly stored in KG
    # For now, assume KG nodes might have a 'doc_id' property or similar.

    def __post_init__(self):
       logger.info("KnowledgeGraphSearch initialized.")

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        logger.info(f"Performing knowledge graph search for query: '{query}'")

        # 1. Extract entities from the query using KG's NLP capabilities
        query_entities = self.kg.extract_entities_from_text(query)
        if not query_entities:
            logger.info("No entities extracted from query for graph search.")
            return []

        logger.debug(f"Query entities for KG search: {[e.name for e in query_entities]}")

        # 2. Find corresponding nodes in the KG
        # This is a simplification. Robust matching would involve:
        #   - Embedding entity names and comparing with embeddings of KG node names.
        #   - Fuzzy string matching.
        #   - Disambiguation if multiple KG nodes match an entity.

        candidate_kg_nodes = set()
        for q_entity in query_entities:
            # Simple name matching for now (case-insensitive)
            for node_id, data in self.kg.graph.nodes(data=True):
                if data.get('name', '').lower() == q_entity.name.lower():
                    candidate_kg_nodes.add(node_id)

        if not candidate_kg_nodes:
            logger.info(f"No matching KG nodes found for entities: {[e.name for e in query_entities]}")
            return []

        logger.debug(f"Candidate KG nodes: {candidate_kg_nodes}")

        # 3. Traverse graph from these seed nodes to find related information/documents
        #    For simplicity, let's find 1-hop neighbors and check if they have associated 'doc_id'.
        #    A more complex approach would involve pathfinding, community detection, centrality, etc.

        related_doc_scores: Dict[str, float] = {} # doc_id -> score

        for node_id in candidate_kg_nodes:
            # Check the node itself
            node_data = self.kg.graph.nodes[node_id]
            if 'doc_id' in node_data: # Assuming nodes can link to documents
                doc_id = node_data['doc_id']
                related_doc_scores[doc_id] = related_doc_scores.get(doc_id, 0.0) + 1.0 # Simple count/score

            # Check neighbors (1-hop)
            for neighbor_id in list(self.kg.graph.successors(node_id)) + list(self.kg.graph.predecessors(node_id)):
                neighbor_data = self.kg.graph.nodes[neighbor_id]
                if 'doc_id' in neighbor_data:
                    doc_id = neighbor_data['doc_id']
                    related_doc_scores[doc_id] = related_doc_scores.get(doc_id, 0.0) + 0.5 # Score neighbors less

        if not related_doc_scores:
            logger.info("No documents found related to KG nodes.")
            return []

        # 4. Convert to SearchResult format
        graph_search_results = []
        for doc_id, score in related_doc_scores.items():
            # We need content for SearchResult; assume it can be fetched if needed, or leave None
            # Here, self.doc_content_map from AdvancedSearchSystem might be useful if KG nodes map to its doc IDs
            graph_search_results.append(SearchResult(
                doc_id=doc_id,
                score=score, # This scoring is very basic
                content=None, # Or fetch from a content store using doc_id
                source_type="graph",
                metadata={"kg_related_nodes": list(candidate_kg_nodes)} # Example metadata
            ))

        # Sort by the simple score
        graph_search_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(graph_search_results)} potential results from graph search.")
        return graph_search_results[:top_k]


# Example Usage (Conceptual)
# async def main():
#     # Initialize embedding provider (e.g., Ollama)
#     ollama_provider = create_embedding_provider(provider="ollama", model="snowflake-arctic-embed2:latest")

#     # Sample documents for keyword search
#     docs_for_keyword = [
#         {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog."},
#         {"id": "doc2", "text": "A powerful new language model was released today."},
#         {"id": "doc3", "text": "Exploring the future of artificial intelligence and machine learning."},
#     ]

#     search_sys = AdvancedSearchSystem(embedding_provider=ollama_provider, initial_documents=docs_for_keyword)

#     query = "machine learning advancements"
#     results = await search_sys.search(query, top_k=5)

#     for res in results:
#         print(f"ID: {res.doc_id}, Score: {res.score:.4f}, Type: {res.source_type}")
#         # print(f"Content: {res.content[:100]}...") # If content is populated

# if __name__ == "__main__":
#    # asyncio.run(main())
#    pass
```
