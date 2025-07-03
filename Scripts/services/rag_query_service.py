import os
import logging
import httpx # For Ollama client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import re # For keyword extraction
import uuid # For Qdrant IDs if not provided in metadata
import hashlib # For generating cache keys
import json # For serializing cache data
import time # For latency timing
import numpy as np

# Vector Store and Embedding Model Imports (similar to doc_processing_service)
from sentence_transformers import SentenceTransformer
import torch # For device selection
from Scripts.storage.vector_store_base import VectorStoreBase, SearchResult as VectorStoreSearchResult # Use the SearchResult from base
from Scripts.storage.qdrant_vector_store import QdrantVectorStore
from Scripts.storage.weaviate_vector_store import WeaviateVectorStore
from Scripts.storage.milvus_vector_store import MilvusVectorStore
from Scripts.storage.chroma_vector_store import ChromaVectorStore
from Scripts.storage.faiss_vector_store import FaissVectorStore

from prometheus_fastapi_instrumentator import Instrumentator, Counter, Histogram
from rank_bm25 import BM25Okapi # For BM25 scoring

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config loader and RedisCacheManager
try:
    from Scripts.utils.config_loader import get_config_value
    from Scripts.utils.redis_cache_manager import RedisCacheManager
except ImportError:
    # ... (fallback import logic as before) ...
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(current_dir)
    if scripts_dir not in sys.path: sys.path.append(scripts_dir)
    try:
        from utils.config_loader import get_config_value
        from utils.redis_cache_manager import RedisCacheManager
    except ImportError as e:
        logger.error(f"Critical import error: {e}", exc_info=True)
        def get_config_value(env_var_name, yaml_path=None, default=None): return os.getenv(env_var_name, default)
        class RedisCacheManager:
            def __init__(self, *args, **kwargs): logger.error("DUMMY RedisCacheManager.")
            def is_available(self): return False
            def get_json(self, key): return None
            def set_json(self, key, data, ttl_seconds=None): pass
            def close(self): pass
        class DummyCounter: def labels(self, *args, **kwargs): return self; def inc(self): pass
        class DummyHistogram: def labels(self, *args, **kwargs): return self; def observe(self, val): pass
        CACHE_HITS_COUNTER, CACHE_MISSES_COUNTER = DummyCounter(), DummyCounter()
        QDRANT_QUERY_LATENCY, OLLAMA_LLM_LATENCY = DummyHistogram(), DummyHistogram()


# --- Configuration ---
# ... (Qdrant, Embedding, Ollama, Redis configs as before) ...
QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
QDRANT_PORT = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
QDRANT_COLLECTION_NAME = get_config_value("QDRANT_COLLECTION_NAME", yaml_path="vector_store.qdrant.collection_name", default="documents")
QDRANT_API_KEY = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")
EMBEDDING_MODEL_NAME = get_config_value("EMBEDDING_MODEL_NAME", yaml_path="model.embedding_model", default="all-MiniLM-L6-v2")
MODEL_DEVICE = get_config_value("MODEL_DEVICE", yaml_path="model.device", default="cpu")
OLLAMA_API_URL = get_config_value("OLLAMA_API_URL", yaml_path="ollama.api_url", default="http://localhost:11434")
LLM_MODEL_NAME = get_config_value("LLM_MODEL_NAME", yaml_path="ollama.llm_model", default="llama2")
REDIS_HOST = get_config_value("REDIS_HOST", yaml_path="caching.redis.host", default="localhost")
REDIS_PORT = int(get_config_value("REDIS_PORT", yaml_path="caching.redis.port", default=6379))
REDIS_DB_RAG = int(get_config_value("REDIS_DB_RAG_CACHE", yaml_path="caching.redis.db_rag_cache", default=1))
SEARCH_RESULTS_CACHE_TTL = int(get_config_value("SEARCH_RESULTS_CACHE_TTL_SECONDS", default=3600))
LLM_ANSWER_CACHE_TTL = int(get_config_value("LLM_ANSWER_CACHE_TTL_SECONDS", default=86400))

# Hybrid Search Configuration
SEMANTIC_SEARCH_TOP_K_CANDIDATES = int(get_config_value("SEMANTIC_SEARCH_TOP_K_CANDIDATES", default=50)) # Fetch more for BM25
RRF_K_CONSTANT = int(get_config_value("RRF_K_CONSTANT", default=60)) # RRF k constant


# --- Global Variables ---
app = FastAPI(title="RAG Query Service")
vector_store: Optional[VectorStoreBase] = None # Unified vector store instance
embedding_model: Optional[SentenceTransformer] = None
llm_client: Optional[httpx.AsyncClient] = None
redis_cache: Optional[RedisCacheManager] = None

# --- Prometheus Custom Metrics ---
# ... (metrics definitions as before) ...
try:
    CACHE_HITS_COUNTER = Counter("rag_cache_hits_total", "Number of cache hits.", labelnames=("cache_type",))
    CACHE_MISSES_COUNTER = Counter("rag_cache_misses_total", "Number of cache misses.", labelnames=("cache_type",))
    QDRANT_QUERY_LATENCY = Histogram("rag_qdrant_query_latency_seconds", "Latency of Qdrant queries.", labelnames=("query_type",))
    OLLAMA_LLM_LATENCY = Histogram("rag_ollama_llm_latency_seconds", "Latency of Ollama LLM calls.", labelnames=("model_name",))
except NameError:
    logger.error("Prometheus metrics could not be defined.")
    class DummyCounter: def labels(self, *args, **kwargs): return self; def inc(self): pass
    class DummyHistogram: def labels(self, *args, **kwargs): return self; def observe(self, val): pass
    CACHE_HITS_COUNTER, CACHE_MISSES_COUNTER = DummyCounter(), DummyCounter()
    QDRANT_QUERY_LATENCY, OLLAMA_LLM_LATENCY = DummyHistogram(), DummyHistogram()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query text.")
    top_k: int = Field(5, gt=0, le=20, description="Number of final results to return after re-ranking.") # Max 20 final results
    # search_type: str = Field("hybrid", description="Type of search: 'semantic', 'keyword', or 'hybrid'.") # Hybrid is now default
    generate_answer: bool = Field(True, description="Whether to generate a natural language answer using an LLM.")
    force_no_cache: bool = Field(False, description="Set to true to bypass cache for this request.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for vector search.")


# Using SearchResult from vector_store_base
class QueryResponse(BaseModel):
    query: str
    search_results: List[VectorStoreSearchResult] # Changed from local SearchResult
    answer: Optional[str] = None
    llm_model_used: Optional[str] = None
    cached_response: bool = Field(False, description="Indicates if the main part of the response (search results or full answer) was served from cache.")


# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    global vector_store, embedding_model, llm_client, redis_cache
    logger.info("RAG Query Service starting up...")

    # Initialize Embedding Model First
    try:
        device_to_use = MODEL_DEVICE if MODEL_DEVICE == "cpu" or torch.cuda.is_available() else "cpu"
        if MODEL_DEVICE == "cuda" and device_to_use == "cpu":
            logger.warning("CUDA specified but not available. Falling back to CPU for embedding model.")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL_NAME} on {device_to_use}. Dim: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        embedding_model = None

    # Initialize Vector Store
    if embedding_model:
        provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="qdrant")
        collection_name = get_config_value(f"VECTOR_STORE_{provider.upper()}_COLLECTION_NAME",
                                           yaml_path=f"vector_store.{provider}.collection_name",
                                           default="documents")
        vector_size = embedding_model.get_sentence_embedding_dimension()
        distance_metric = get_config_value(f"VECTOR_STORE_{provider.upper()}_DISTANCE",
                                           yaml_path=f"vector_store.{provider}.distance_metric",
                                           default="Cosine")

        logger.info(f"Initializing vector store provider: {provider} for collection '{collection_name}'")
        global vector_store # Ensure we're assigning to the global
        try:
            if provider == "qdrant":
                host = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
                port = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
                api_key = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")
                vector_store = QdrantVectorStore(host=host, port=port, api_key=api_key)
            elif provider == "weaviate":
                url = get_config_value("WEAVIATE_URL", yaml_path="vector_store.weaviate.url", default="http://localhost:8080")
                api_key = get_config_value("WEAVIATE_API_KEY", yaml_path="vector_store.weaviate.api_key")
                vector_store = WeaviateVectorStore(url=url, api_key=api_key)
                collection_name = get_config_value(f"WEAVIATE_CLASS_NAME", yaml_path=f"vector_store.weaviate.class_name", default="Document")
            elif provider == "milvus":
                host = get_config_value("MILVUS_HOST", yaml_path="vector_store.milvus.host", default="localhost")
                port = get_config_value("MILVUS_PORT", yaml_path="vector_store.milvus.port", default="19530")
                vector_store = MilvusVectorStore(host=host, port=port)
            elif provider == "chroma":
                chroma_path = get_config_value("CHROMA_PATH", yaml_path="vector_store.chroma.path")
                chroma_host = get_config_value("CHROMA_HOST", yaml_path="vector_store.chroma.host")
                chroma_port = get_config_value("CHROMA_PORT", yaml_path="vector_store.chroma.port")
                if chroma_host and chroma_port:
                    vector_store = ChromaVectorStore(host=chroma_host, port=int(chroma_port))
                elif chroma_path:
                    vector_store = ChromaVectorStore(path=chroma_path)
                else:
                    vector_store = ChromaVectorStore()
            elif provider == "faiss":
                index_path = get_config_value("FAISS_INDEX_PATH", yaml_path="vector_store.faiss.index_file_path", default="data/faiss_index.bin")
                metadata_path = get_config_value("FAISS_METADATA_PATH", yaml_path="vector_store.faiss.metadata_file_path", default="data/faiss_metadata.pkl")
                vector_store = FaissVectorStore(index_file_path=index_path, metadata_file_path=metadata_path)
            else:
                logger.error(f"Unsupported vector store provider: {provider}")
                raise ValueError(f"Unsupported vector store provider: {provider}")

            if vector_store:
                # For RAG Query service, we assume collection is already created and initialized by Doc Processing.
                # We might just need to ensure it's loaded or accessible.
                # A light check like get_collection_info or health_check is good.
                if not await vector_store.health_check():
                     logger.error(f"Vector store provider '{provider}' is not healthy.")
                     vector_store = None # Mark as unusable
                else:
                    logger.info(f"Vector store '{provider}' initialized and healthy for collection '{collection_name}'.")
            else:
                 logger.error(f"Vector store provider '{provider}' could not be instantiated.")
        except Exception as e:
            logger.error(f"Failed to initialize vector store provider '{provider}': {e}", exc_info=True)
            vector_store = None
    else:
        logger.error("Embedding model failed to load. Vector store initialization skipped.")
        vector_store = None

    llm_client = httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=httpx.Timeout(120.0))
    logger.info(f"LLM client for Ollama at {OLLAMA_API_URL} initialized.")

    redis_cache = RedisCacheManager(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_RAG)
    if redis_cache.is_available():
        logger.info(f"Redis cache connected at {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB_RAG}")
    else:
        logger.warning(f"Redis cache NOT available at {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB_RAG}. Service will run without caching.")

    if not vector_store or not embedding_model:
        logger.error("CRITICAL: Vector Store or Embedding Model failed initialization.")
    else:
        logger.info("RAG Query Service core components (Vector Store, Embedding Model) initialized.")

    instrumentator = Instrumentator(should_group_status_codes=True, should_instrument_requests_inprogress=True, excluded_handlers=["/health", "/metrics"], inprogress_name="rag_inprogress_requests",inprogress_labels=True,)
    instrumentator.instrument(app).expose(app)
    if not isinstance(CACHE_HITS_COUNTER, DummyCounter):
        instrumentator.add(CACHE_HITS_COUNTER); instrumentator.add(CACHE_MISSES_COUNTER)
        instrumentator.add(QDRANT_QUERY_LATENCY); instrumentator.add(OLLAMA_LLM_LATENCY)
    logger.info("Prometheus instrumentation complete.")

@app.on_event("shutdown")
async def shutdown_event():
    global vector_store, llm_client, redis_cache # Added vector_store
    logger.info("RAG Query Service shutting down...")
    if vector_store:
        try:
            await vector_store.close()
            logger.info("Vector store connection closed.")
        except Exception as e:
            logger.error(f"Error closing vector store: {e}", exc_info=True)
    if llm_client:
        await llm_client.aclose()
        logger.info("LLM client closed.")
    if redis_cache:
        redis_cache.close()
        logger.info("Redis cache client closed.")
    logger.info("RAG Query Service shutdown complete.")

# --- Helper Functions ---
def _generate_cache_key(prefix: str, query: str, top_k: int, search_type_or_context: str) -> str:
    # ... (as before)
    key_string = f"{prefix}:{query}:{top_k}:{search_type_or_context}" # search_type_or_context can be context for LLM cache
    return hashlib.md5(key_string.encode()).hexdigest()

def _format_context_for_llm(search_results: List[VectorStoreSearchResult], max_context_results: int = 5) -> str: # Type hint updated
    context_parts = []
    # Assuming higher score is better for sorting. This might need adjustment based on actual scores from stores (distance vs similarity).
    # For now, let's assume vector_store_base.SearchResult score is normalized (higher is better).
    sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
    for i, result in enumerate(sorted_results[:max_context_results]):
        text_content = result.payload.get("text") if result.payload else None
        # search_method is not part of VectorStoreSearchResult.payload by default.
        # It was part of the local SearchResult. For now, omitting it from context string.
        # If needed, individual vector store impls could add it to payload.metadata.
        if text_content:
            context_parts.append(f"Source {i+1} (ID: {result.id}, Score: {result.score:.4f}):\n{text_content}\n")
    if not context_parts: return "No relevant information found in the knowledge base to answer the question."
    return "\n---\n".join(context_parts)

def _tokenize_text_for_bm25(text: str) -> List[str]:
    """Basic tokenizer for BM25: lowercase and split by non-alphanumeric."""
    if not text: return []
    words = re.findall(r'\b\w+\b', text.lower())
    # Optional: Add stop word removal here if desired for BM25 corpus
    return words

async def _semantic_search_with_vector_store(
    query_vector: List[float],
    top_k: int,
    filters: Optional[Dict[str, Any]] = None
) -> List[VectorStoreSearchResult]: # Return type updated
    if not vector_store:
        logger.error("Vector store not initialized for semantic search.")
        return []

    start_time = time.time()
    provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="qdrant")
    collection_name_cfg_key = f"vector_store.{provider}.collection_name"
    if provider == "weaviate":
        collection_name_cfg_key = f"vector_store.weaviate.class_name"
    target_collection_name = get_config_value(f"VECTOR_STORE_{provider.upper()}_COLLECTION_NAME",
                                           yaml_path=collection_name_cfg_key,
                                           default="documents")
    try:
        # The search method from VectorStoreBase is used here
        # It's assumed that the SearchResult from base includes text and metadata in its payload if available
        results = await vector_store.search(
            collection_name=target_collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            with_vectors=False # Usually not needed for RAG context
        )
        # Post-process to add search_method if needed, or ensure base SearchResult can hold it
        # For now, the base SearchResult doesn't have 'search_method'.
        # We can create new local SearchResult objects if we need that field specifically for hybrid logic.
        # However, the QueryResponse is already updated to List[VectorStoreSearchResult].
        # Let's assume for now that the distinction for 'semantic' is implicit or handled by RRF logic.
        return results
    except Exception as e:
        logger.error(f"Semantic search error with provider {provider}: {e}", exc_info=True)
        return []
    finally:
        latency = time.time() - start_time
        # Update Prometheus metric label to be generic or provider-specific
        QDRANT_QUERY_LATENCY.labels(query_type=f"semantic_{provider}").observe(latency)


async def _generate_llm_answer(query: str, context: str, model_name: str, query_request_details: QueryRequest) -> Optional[str]:
    if not llm_client: return "LLM client not available for answer generation."
    llm_answer_cache_key = _generate_cache_key(f"llm_answer:{LLM_MODEL_NAME}", query_request_details.query, query_request_details.top_k, context)
    if not query_request_details.force_no_cache and redis_cache and redis_cache.is_available():
        cached_llm_answer = redis_cache.get_string(llm_answer_cache_key)
        if cached_llm_answer: CACHE_HITS_COUNTER.labels(cache_type="llm_answer").inc(); return cached_llm_answer
        CACHE_MISSES_COUNTER.labels(cache_type="llm_answer").inc()
    prompt = f"Based ONLY on the following context, please answer the question. If the context does not provide an answer, state that the information is not available in the provided context. Do not use any external knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    start_time = time.time()
    try:
        response = await llm_client.post("/api/generate", json=payload)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if answer and redis_cache and redis_cache.is_available() and not query_request_details.force_no_cache:
            redis_cache.set_string(llm_answer_cache_key, answer, ttl_seconds=LLM_ANSWER_CACHE_TTL)
        return answer
    except Exception as e: logger.error(f"LLM answer generation error: {e}", exc_info=True); return "Failed to generate an answer due to an internal error with the LLM service."
    finally:
        latency = time.time() - start_time
        OLLAMA_LLM_LATENCY.labels(model_name=model_name).observe(latency)

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def perform_query(request: QueryRequest):
    # Use unified vector_store now
    if not vector_store or not embedding_model:
        raise HTTPException(status_code=503, detail="Search components (vector store or embedding model) not ready.")
    if request.generate_answer and not llm_client:
        raise HTTPException(status_code=503, detail="LLM component not ready.")

    logger.info(f"Received query: '{request.query}', top_k: {request.top_k}, generate_answer: {request.generate_answer}, filters: {request.filters}, force_no_cache: {request.force_no_cache}")

    final_search_results: List[VectorStoreSearchResult] = [] # Use VectorStoreSearchResult
    cached_search_results_flag = False

    # Cache key should ideally include filters if they affect the semantic search part significantly
    # For simplicity, current key does not include request.filters. This could be an enhancement.
    cache_context_for_key = f"semantic_bm25_rrf_filters_{json.dumps(request.filters, sort_keys=True) if request.filters else 'None'}"
    search_results_cache_key = _generate_cache_key(
        "hybrid_search_results_v3", # incremented version due to filter addition
        request.query,
        SEMANTIC_SEARCH_TOP_K_CANDIDATES,
        cache_context_for_key
    )

    if not request.force_no_cache and redis_cache and redis_cache.is_available():
        cached_search_data = redis_cache.get_json(search_results_cache_key)
        if cached_search_data:
            # Ensure items are parsed into VectorStoreSearchResult
            final_search_results = [VectorStoreSearchResult(**item) for item in cached_search_data]
            logger.info(f"Cache HIT for combined search results: key='{search_results_cache_key}'")
            CACHE_HITS_COUNTER.labels(cache_type="search_results_combined").inc()
            cached_search_results_flag = True
        else:
            logger.info(f"Cache MISS for combined search results: key='{search_results_cache_key}'")
            CACHE_MISSES_COUNTER.labels(cache_type="search_results_combined").inc()

    if not final_search_results: # Not in cache or cache unavailable/bypassed
        query_vector = embedding_model.encode(request.query, convert_to_tensor=False).tolist()

        # 1. Get semantic candidates using the new abstracted search
        semantic_candidates = await _semantic_search_with_vector_store(
            query_vector,
            SEMANTIC_SEARCH_TOP_K_CANDIDATES,
            filters=request.filters # Pass filters to semantic search
        )

        if not semantic_candidates:
            logger.info("No semantic candidates found from vector store.")
        else:
            logger.info(f"Retrieved {len(semantic_candidates)} semantic candidates for BM25 re-ranking.")

            # Adapt to VectorStoreSearchResult which has payload.text and payload.metadata
            candidate_texts = [res.payload.get("text") for res in semantic_candidates if res.payload and res.payload.get("text")]
            # Ensure IDs used for mapping are consistent (original doc IDs)
            candidate_ids_map = {idx: res.id for idx, res in enumerate(semantic_candidates) if res.payload and res.payload.get("text")}

            if candidate_texts:
                tokenized_corpus = [_tokenize_text_for_bm25(text) for text in candidate_texts]
                tokenized_query = _tokenize_text_for_bm25(request.query)

                bm25 = BM25Okapi(tokenized_corpus)
                bm25_scores = bm25.get_scores(tokenized_query)

                # Normalize BM25 scores (e.g., simple min-max or just use raw if RRF handles scales)
                # For RRF, raw ranks are used, so direct scores might not need complex normalization.
                # Let's assume BM25 scores are positive. If they can be negative, handle accordingly.
                # max_bm25_score = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
                # normalized_bm25_scores = [s / max_bm25_score if max_bm25_score > 0 else 0 for s in bm25_scores]

                # Combine with semantic scores using RRF
                # Create lists of (doc_id, score) for each method
                semantic_ranked_list = {res.id: (idx + 1, res.score) for idx, res in enumerate(semantic_candidates)} # id -> (rank, score)
                bm25_ranked_list_tuples = sorted([(candidate_ids_map[i], bm25_scores[i]) for i in range(len(bm25_scores))], key=lambda x: x[1], reverse=True)
                bm25_ranked_list = {doc_id: (idx + 1, score) for idx, (doc_id, score) in enumerate(bm25_ranked_list_tuples)}

                # RRF calculation
                rrf_scores: Dict[Union[int,str,uuid.UUID], float] = {}
                all_doc_ids = set(semantic_ranked_list.keys()).union(set(bm25_ranked_list.keys()))

                for doc_id in all_doc_ids:
                    s = 0.0
                    if doc_id in semantic_ranked_list:
                        s += 1.0 / (RRF_K_CONSTANT + semantic_ranked_list[doc_id][0])
                    if doc_id in bm25_ranked_list: # Only add if BM25 score is positive
                        if bm25_ranked_list[doc_id][1] > 0: # Add only if BM25 score is positive
                             s += 1.0 / (RRF_K_CONSTANT + bm25_ranked_list[doc_id][0])
                    if s > 0:
                        rrf_scores[doc_id] = s

                # Map RRF scores back to SearchResult objects
                temp_results_map = {res.id: res for res in semantic_candidates} # Get original objects
                for kres_id, (bm25_rank, bm25_s) in bm25_ranked_list.items(): # Add any unique keyword results if needed (not done here)
                    if kres_id not in temp_results_map and bm25_s > 0: # Should not happen if BM25 is on semantic_candidates
                        # This case implies keyword search was broader than semantic candidates
                        # For now, we only re-rank semantic_candidates.
                        pass


                final_search_results = []
                for doc_id, rrf_score in rrf_scores.items():
                    original_result = temp_results_map.get(doc_id) # original_result is a VectorStoreSearchResult
                    if original_result:
                        # Create new VectorStoreSearchResult for the final list
                        # The 'payload' will contain text and metadata from the original result.
                        final_search_results.append(VectorStoreSearchResult(
                            id=original_result.id,
                            score=rrf_score, # Use RRF score
                            payload=original_result.payload, # Keep original payload (text, metadata)
                            vector=original_result.vector # Keep original vector if present
                            # 'search_method' is not part of VectorStoreSearchResult. If needed, add to payload.metadata.
                        ))

                final_search_results.sort(key=lambda x: x.score, reverse=True)

            else: # No text in semantic candidates for BM25
                final_search_results = semantic_candidates # Fallback to pure semantic

        final_search_results = final_search_results[:request.top_k] # Apply final top_k

        if redis_cache and redis_cache.is_available() and not request.force_no_cache and final_search_results:
            redis_cache.set_json(search_results_cache_key, [res.model_dump() for res in final_search_results], ttl_seconds=SEARCH_RESULTS_CACHE_TTL)

    llm_answer: Optional[str] = None
    llm_model_used: Optional[str] = LLM_MODEL_NAME if request.generate_answer else None

    if request.generate_answer:
        context_for_llm = _format_context_for_llm(final_search_results, max_context_results=request.top_k)
        llm_answer = await _generate_llm_answer(request.query, context_for_llm, LLM_MODEL_NAME, request)

    return QueryResponse(
        query=request.query,
        search_results=final_search_results,
        answer=llm_answer,
        llm_model_used=llm_model_used,
        cached_response=cached_search_results_flag
    )

@app.get("/health", tags=["Health"])
async def health_check():
    em_ready = bool(embedding_model)
    llm_cli_ready = bool(llm_client)
    redis_ready = bool(redis_cache and redis_cache.is_available())

    vs_ready = False
    vs_provider = "N/A"
    if vector_store:
        vs_ready = await vector_store.health_check()
        vs_provider = get_config_value("VECTOR_STORE_PROVIDER", yaml_path="vector_store.provider", default="unknown")

    ollama_service_healthy = False
    if llm_cli_ready:
        try:
            response = await llm_client.get("/") # Ollama root path usually returns "Ollama is running"
            ollama_service_healthy = response.status_code == 200
        except Exception:
            ollama_service_healthy = False

    overall_status = "healthy"
    component_statuses = {
        "embedding_model_loaded": em_ready,
        "vector_store_provider": vs_provider,
        "vector_store_healthy": vs_ready,
        "llm_client_initialized": llm_cli_ready,
        "ollama_service_accessible": ollama_service_healthy,
        "redis_cache_connected": redis_ready
    }
    if not all(component_statuses.values()): # Simplified check, some components might be optional based on config
         if not em_ready : logger.warning("Health Check: Embedding model not ready.")
         if not vs_ready : logger.warning(f"Health Check: Vector store '{vs_provider}' not ready.")
         if not llm_cli_ready : logger.warning("Health Check: LLM Client not initialized (Ollama).")
         if not ollama_service_healthy : logger.warning("Health Check: Ollama service not accessible.")
         # Redis not being ready might be acceptable if caching is optional.
         overall_status = "degraded"


    return {"status": overall_status, "components": component_statuses}

if __name__ == "__main__":
    # ... (uvicorn startup as before) ...
    import uvicorn
    SERVICE_PORT = int(get_config_value("RAG_SERVICE_PORT", default=8001))
    SERVICE_HOST = get_config_value("RAG_SERVICE_HOST", default="0.0.0.0")
    logger.info(f"Starting RAG Query Service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
