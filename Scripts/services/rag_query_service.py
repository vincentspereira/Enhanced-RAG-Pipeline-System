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

from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import torch # For device selection

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config loader and RedisCacheManager
try:
    from Scripts.utils.config_loader import get_config_value
    from Scripts.utils.redis_cache_manager import RedisCacheManager
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(current_dir)
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    try:
        from utils.config_loader import get_config_value
        from utils.redis_cache_manager import RedisCacheManager
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value or RedisCacheManager. Error: {e}", exc_info=True)
        def get_config_value(env_var_name, yaml_path=None, default=None): return os.getenv(env_var_name, default)
        class RedisCacheManager: # Dummy for fallback
            def __init__(self, *args, **kwargs): logger.error("Using DUMMY RedisCacheManager due to import error.")
            def is_available(self): return False
            def get_json(self, key): return None
            def set_json(self, key, data, ttl_seconds=None): pass
            def close(self): pass

# --- Configuration ---
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
REDIS_DB_RAG = int(get_config_value("REDIS_DB_RAG_CACHE", yaml_path="caching.redis.db_rag_cache", default=1)) # Specific DB for this service
SEARCH_RESULTS_CACHE_TTL = int(get_config_value("SEARCH_RESULTS_CACHE_TTL_SECONDS", default=3600)) # 1 hour
LLM_ANSWER_CACHE_TTL = int(get_config_value("LLM_ANSWER_CACHE_TTL_SECONDS", default=86400)) # 24 hours

# --- Global Variables ---
app = FastAPI(title="RAG Query Service")
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None
llm_client: Optional[httpx.AsyncClient] = None
redis_cache: Optional[RedisCacheManager] = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query text.")
    top_k: int = Field(5, gt=0, le=100, description="Number of primary search results to retrieve for context.")
    search_type: str = Field("hybrid", description="Type of search: 'semantic', 'keyword', or 'hybrid'.")
    generate_answer: bool = Field(True, description="Whether to generate a natural language answer using an LLM.")
    force_no_cache: bool = Field(False, description="Set to true to bypass cache for this request.")

class SearchResult(BaseModel):
    id: Union[int, str, uuid.UUID]
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    search_method: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    search_results: List[SearchResult]
    answer: Optional[str] = None
    llm_model_used: Optional[str] = None
    cached_response: bool = Field(False, description="Indicates if the main part of the response (search results or full answer) was served from cache.")

# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    global qdrant_client, embedding_model, llm_client, redis_cache
    logger.info("RAG Query Service starting up...")

    try: # Qdrant
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Qdrant connected: {QDRANT_COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Qdrant connection error: {e}", exc_info=True); qdrant_client = None

    try: # Embedding Model
        device_to_use = MODEL_DEVICE if MODEL_DEVICE == "cpu" or torch.cuda.is_available() else "cpu"
        if MODEL_DEVICE == "cuda" and device_to_use == "cpu": logger.warning("CUDA specified but not available. Falling back to CPU for embedding model.")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info(f"Embedding model loaded: {EMBEDDING_MODEL_NAME} on {device_to_use}. Dim: {embedding_model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True); embedding_model = None

    # LLM Client
    llm_client = httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=httpx.Timeout(120.0))
    logger.info(f"LLM client for Ollama at {OLLAMA_API_URL} initialized.")

    # Redis Cache
    redis_cache = RedisCacheManager(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_RAG)
    if redis_cache.is_available(): logger.info(f"Redis cache connected at {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB_RAG}")
    else: logger.warning(f"Redis cache NOT available at {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB_RAG}. Service will run without caching.")

    if not qdrant_client or not embedding_model: logger.error("CRITICAL: Qdrant or Embedding Model failed initialization.")
    else: logger.info("RAG Query Service core components (Qdrant, Embedding Model) initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG Query Service shutting down...")
    if llm_client: await llm_client.aclose(); logger.info("LLM client closed.")
    if redis_cache: redis_cache.close(); logger.info("Redis cache client closed.")
    logger.info("RAG Query Service shutdown complete.")

# --- Helper Functions ---
def _generate_cache_key(prefix: str, query: str, top_k: int, search_type: str) -> str:
    key_string = f"{prefix}:{query}:{top_k}:{search_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def _format_context_for_llm(search_results: List[SearchResult], max_context_results: int = 5) -> str:
    context_parts = []
    sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
    for i, result in enumerate(sorted_results[:max_context_results]):
        if result.text:
            context_parts.append(f"Source {i+1} (ID: {result.id}, Score: {result.score:.4f}, Method: {result.search_method}):\n{result.text}\n")
    if not context_parts: return "No relevant information found in the knowledge base to answer the question."
    return "\n---\n".join(context_parts)

def _extract_keywords_from_query(query: str, min_len: int = 3) -> List[str]:
    words = re.findall(r'\b\w+\b', query.lower())
    stop_words = set(["the", "a", "is", "in", "it", "to", "of", "and", "for", "on", "with", "this", "that", "an", "by", "as", "at", "or", "if", "not", "be", "was", "were", "am", "are", "has", "had", "do", "does", "did", "will", "would", "should", "can", "could", "may", "might", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs", "from", "what", "who", "when", "where", "why", "how", "which", "what", "about", "what's", "tell", "give", "explain"])
    return list(set(word for word in words if word not in stop_words and len(word) >= min_len and not word.isdigit()))

async def _semantic_search(query_vector: List[float], top_k: int) -> List[SearchResult]:
    if not qdrant_client: return []
    try:
        hits = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=query_vector, limit=top_k, with_payload=True)
        return [SearchResult(id=hit.id, score=hit.score, text=hit.payload.get("text") if hit.payload else None, metadata=hit.payload.get("metadata") if hit.payload else None, search_method="semantic") for hit in hits]
    except Exception as e: logger.error(f"Semantic search error: {e}", exc_info=True); return []

async def _keyword_filter_search(query_text: str, top_k: int) -> List[SearchResult]:
    if not qdrant_client or not embedding_model : return []
    query_keywords = _extract_keywords_from_query(query_text)
    if not query_keywords: return []
    logger.info(f"Keyword filter search with keywords: {query_keywords}")

    keyword_conditions = [qdrant_models.FieldCondition(key="metadata.keywords", match=qdrant_models.MatchValue(value=kw)) for kw in query_keywords]
    query_filter = qdrant_models.Filter(should=keyword_conditions)
    try:
        hits, _ = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, scroll_filter=query_filter, limit=top_k * 5, with_payload=True, with_vectors=False)
        results = []
        for hit in hits:
            num_matched_keywords = len(set(query_keywords).intersection(set(hit.payload.get("metadata", {}).get("keywords", [])))) if hit.payload else 0
            keyword_score = (float(num_matched_keywords) / len(query_keywords)) if query_keywords and num_matched_keywords > 0 else 0.0
            if keyword_score > 0: results.append(SearchResult(id=hit.id, score=keyword_score, text=hit.payload.get("text") if hit.payload else None, metadata=hit.payload.get("metadata") if hit.payload else None, search_method="keyword"))
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Keyword filter search found {len(hits)} raw matches, returning top {top_k} after scoring.")
        return results[:top_k]
    except Exception as e: logger.error(f"Keyword filter search error: {e}", exc_info=True); return []

async def _generate_llm_answer(query: str, context: str, model_name: str, query_request_details: QueryRequest) -> Optional[str]:
    if not llm_client: return "LLM client not available for answer generation."

    llm_answer_cache_key = _generate_cache_key(f"llm_answer:{LLM_MODEL_NAME}", query_request_details.query, query_request_details.top_k, context) # Cache key includes context hash
    if not query_request_details.force_no_cache and redis_cache and redis_cache.is_available():
        cached_llm_answer = redis_cache.get_string(llm_answer_cache_key)
        if cached_llm_answer:
            logger.info(f"Cache HIT for LLM answer: key='{llm_answer_cache_key}'")
            return cached_llm_answer
        logger.info(f"Cache MISS for LLM answer: key='{llm_answer_cache_key}'")

    prompt = f"Based ONLY on the following context, please answer the question. If the context does not provide an answer, state that the information is not available in the provided context. Do not use any external knowledge.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = await llm_client.post("/api/generate", json=payload)
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        if answer and redis_cache and redis_cache.is_available() and not query_request_details.force_no_cache:
            redis_cache.set_string(llm_answer_cache_key, answer, ttl_seconds=LLM_ANSWER_CACHE_TTL)
        return answer
    except Exception as e:
        logger.error(f"LLM answer generation error: {e}", exc_info=True)
        return "Failed to generate an answer due to an internal error with the LLM service."

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def perform_query(request: QueryRequest):
    if not qdrant_client or not embedding_model: raise HTTPException(status_code=503, detail="Search components not ready.")
    if request.generate_answer and not llm_client: raise HTTPException(status_code=503, detail="LLM component not ready.")

    logger.info(f"Received query: '{request.query}', top_k: {request.top_k}, search_type: {request.search_type}, generate_answer: {request.generate_answer}, force_no_cache: {request.force_no_cache}")

    final_search_results: List[SearchResult] = []
    cached_search_results_flag = False

    search_results_cache_key = _generate_cache_key("search_results", request.query, request.top_k, request.search_type)
    if not request.force_no_cache and redis_cache and redis_cache.is_available():
        cached_search_data = redis_cache.get_json(search_results_cache_key)
        if cached_search_data:
            final_search_results = [SearchResult(**item) for item in cached_search_data]
            logger.info(f"Cache HIT for search results: key='{search_results_cache_key}'")
            cached_search_results_flag = True

    if not final_search_results: # Not in cache or cache unavailable/bypassed
        logger.info(f"Cache MISS for search results or cache bypassed: key='{search_results_cache_key}'")
        semantic_results: List[SearchResult] = []
        keyword_search_results: List[SearchResult] = []

        if request.search_type in ["semantic", "hybrid"]:
            query_vector = embedding_model.encode(request.query, convert_to_tensor=False).tolist()
            semantic_results = await _semantic_search(query_vector, request.top_k)
        if request.search_type in ["keyword", "hybrid"]:
            keyword_search_results = await _keyword_filter_search(request.query, request.top_k)

        combined_results_dict: Dict[Union[int, str, uuid.UUID], SearchResult] = {}
        if request.search_type == "semantic":
            for res in semantic_results: combined_results_dict[res.id] = res
        elif request.search_type == "keyword":
            for res in keyword_search_results: combined_results_dict[res.id] = res
        elif request.search_type == "hybrid":
            for res in semantic_results: combined_results_dict[res.id] = res
            for kres in keyword_search_results:
                if kres.id in combined_results_dict:
                    combined_results_dict[kres.id].score = max(combined_results_dict[kres.id].score, kres.score) + 0.1
                    combined_results_dict[kres.id].search_method = "hybrid_boosted"
                else: combined_results_dict[kres.id] = kres

        final_search_results = sorted(list(combined_results_dict.values()), key=lambda x: x.score, reverse=True)[:request.top_k]

        if redis_cache and redis_cache.is_available() and not request.force_no_cache and final_search_results:
            redis_cache.set_json(search_results_cache_key, [res.model_dump() for res in final_search_results], ttl_seconds=SEARCH_RESULTS_CACHE_TTL)

    llm_answer: Optional[str] = None
    llm_model_used: Optional[str] = LLM_MODEL_NAME if request.generate_answer else None
    cached_llm_answer_flag = False

    if request.generate_answer:
        context_for_llm = _format_context_for_llm(final_search_results, max_context_results=request.top_k)
        # Note: _generate_llm_answer now handles its own caching logic internally
        llm_answer = await _generate_llm_answer(request.query, context_for_llm, LLM_MODEL_NAME, request)
        # Check if the answer came from cache by seeing if it's different from a cache miss scenario's default error message
        if redis_cache and redis_cache.is_available() and not request.force_no_cache:
             # Re-check cache for LLM answer to determine if this specific call resulted in a cache hit for the answer
            llm_answer_cache_key = _generate_cache_key(f"llm_answer:{LLM_MODEL_NAME}", request.query, request.top_k, context_for_llm)
            if redis_cache.get_string(llm_answer_cache_key) == llm_answer : # Check if what we have is from cache
                 # This logic is slightly complex; simpler if _generate_llm_answer returned a tuple (answer, was_cached)
                 # For now, assume if search results were cached, and answer exists, it might have been part of a fully cached response
                 # The QueryResponse.cached_response will primarily reflect search_results caching.
                 pass # The internal caching in _generate_llm_answer handles it.

    return QueryResponse(
        query=request.query,
        search_results=final_search_results,
        answer=llm_answer,
        llm_model_used=llm_model_used,
        cached_response=cached_search_results_flag # True if search_results came from cache
    )

@app.get("/health", tags=["Health"])
async def health_check():
    q_ready = False
    if qdrant_client:
        try:
            # A more reliable check for Qdrant readiness might involve trying to get collection info
            # This is a placeholder, actual readiness might need a lightweight API call to qdrant
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME) # Re-check collection
            q_ready = True
        except Exception:
            q_ready = False # Could not get collection, assume not fully ready

    em_ready = bool(embedding_model)
    llm_cli_ready = bool(llm_client)
    redis_ready = bool(redis_cache and redis_cache.is_available())
    ollama_service_healthy = False
    if llm_cli_ready:
        try:
            response = await llm_client.get("/")
            ollama_service_healthy = response.status_code == 200
        except Exception: ollama_service_healthy = False

    status = "healthy" if q_ready and em_ready and llm_cli_ready and ollama_service_healthy and redis_ready else "degraded"
    return {"status": status, "components": {"qdrant_accessible": q_ready, "embedding_model_loaded": em_ready, "llm_client_initialized": llm_cli_ready, "ollama_service_accessible": ollama_service_healthy, "redis_cache_connected": redis_ready}}

if __name__ == "__main__":
    import uvicorn
    SERVICE_PORT = int(get_config_value("RAG_SERVICE_PORT", default=8001))
    SERVICE_HOST = get_config_value("RAG_SERVICE_HOST", default="0.0.0.0")
    logger.info(f"Starting RAG Query Service on {SERVICE_HOST}:{SERVICE_PORT}")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
