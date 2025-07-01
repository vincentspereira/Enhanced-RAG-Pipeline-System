import os
import logging
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer
import torch # For device selection

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import config loader
try:
    from Scripts.utils.config_loader import get_config_value
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from utils.config_loader import get_config_value
    except ImportError as e:
        logger.error(f"Critical: Failed to import get_config_value for RAG Query Service. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None): # Basic fallback
            return os.getenv(env_var_name, default)

# --- Configuration ---
QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
QDRANT_PORT = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
QDRANT_COLLECTION_NAME = get_config_value("QDRANT_COLLECTION_NAME", yaml_path="vector_store.qdrant.collection_name", default="documents")
QDRANT_API_KEY = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")

EMBEDDING_MODEL_NAME = get_config_value("EMBEDDING_MODEL_NAME", yaml_path="model.embedding_model", default="all-MiniLM-L6-v2")
MODEL_DEVICE = get_config_value("MODEL_DEVICE", yaml_path="model.device", default="cpu")

# --- Configuration ---
QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")
QDRANT_PORT = int(get_config_value("QDRANT_PORT", yaml_path="vector_store.qdrant.port", default=6333))
QDRANT_COLLECTION_NAME = get_config_value("QDRANT_COLLECTION_NAME", yaml_path="vector_store.qdrant.collection_name", default="documents")
QDRANT_API_KEY = get_config_value("QDRANT_API_KEY", yaml_path="vector_store.qdrant.api_key")

EMBEDDING_MODEL_NAME = get_config_value("EMBEDDING_MODEL_NAME", yaml_path="model.embedding_model", default="all-MiniLM-L6-v2")
MODEL_DEVICE = get_config_value("MODEL_DEVICE", yaml_path="model.device", default="cpu")

OLLAMA_API_URL = get_config_value("OLLAMA_API_URL", yaml_path="ollama.api_url", default="http://localhost:11434")
LLM_MODEL_NAME = get_config_value("LLM_MODEL_NAME", yaml_path="ollama.llm_model", default="llama2") # e.g., llama2, mistral

# --- Global Variables ---
app = FastAPI(title="RAG Query Service")
qdrant_client: Optional[QdrantClient] = None
embedding_model: Optional[SentenceTransformer] = None
llm_client: Optional[httpx.AsyncClient] = None # For Ollama

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query text.")
    top_k: int = Field(5, gt=0, le=100, description="Number of search results to retrieve for context.")
    # Add filters later if needed: filters: Optional[Dict[str, Any]] = None
    generate_answer: bool = Field(True, description="Whether to generate a natural language answer using an LLM.")

class SearchResult(BaseModel):
    id: Union[int, str] # Qdrant point ID can be int or UUID string
    score: float
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    search_results: List[SearchResult]
    answer: Optional[str] = None
    llm_model_used: Optional[str] = None

# --- Service Initialization and Shutdown ---
@app.on_event("startup")
async def startup_event():
    global qdrant_client, embedding_model, llm_client
    logger.info("RAG Query Service starting up...")

    # Initialize Qdrant Client
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            # prefer_grpc=True, # Consider enabling for performance if Qdrant server supports it well
        )
        # Test connection / check collection
        try:
            qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.info(f"Successfully connected to Qdrant and collection '{QDRANT_COLLECTION_NAME}' exists.")
        except Exception as e: # Catching generic Exception as specific Qdrant exceptions can vary
            logger.error(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found or connection error: {e}. Please ensure it's created with appropriate vector params.")
            # For Iteration 1, we assume collection exists. Creation logic could be added.
            # Example:
            # vector_size = 384 # For all-MiniLM-L6-v2
            # self.client.create_collection(
            #     collection_name=self.collection_name,
            #     vectors_config=qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE)
            # )

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}")
        qdrant_client = None # Ensure it's None if init fails

    # Initialize Embedding Model
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {MODEL_DEVICE}")
        device_to_use = MODEL_DEVICE
        if device_to_use == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back to CPU.")
            device_to_use = "cpu"

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_to_use)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        embedding_model = None

    if not qdrant_client or not embedding_model:
        logger.error("RAG Query Service startup failed due to component initialization errors.")
        # Optionally, could raise an exception here to prevent FastAPI from starting if critical components fail
    else:
        logger.info("RAG Query Service startup complete.")

    # Initialize HTTP client for Ollama
    global llm_client
    llm_client = httpx.AsyncClient(base_url=OLLAMA_API_URL, timeout=httpx.Timeout(120.0)) # Longer timeout for LLM
    logger.info(f"LLM client initialized for Ollama at {OLLAMA_API_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    global qdrant_client, llm_client
    logger.info("RAG Query Service shutting down...")
    if qdrant_client:
        try:
            # QdrantClient resource cleanup if any (usually not needed for client object itself)
            pass
        except Exception as e:
            logger.error(f"Error during Qdrant client cleanup: {e}")
    if llm_client:
        try:
            await llm_client.aclose()
            logger.info("LLM client closed.")
        except Exception as e:
            logger.error(f"Error closing LLM client: {e}")
    logger.info("RAG Query Service shutdown complete.")

# --- Helper Functions ---
def _format_context_for_llm(search_results: List[SearchResult]) -> str:
    context_parts = []
    for i, result in enumerate(search_results):
        if result.text: # Only include results with text
            context_parts.append(f"Source {i+1} (ID: {result.id}, Score: {result.score:.4f}):\n{result.text}\n")
    if not context_parts:
        return "No relevant information found in the knowledge base."
    return "\n---\n".join(context_parts)

async def _generate_llm_answer(query: str, context: str, model_name: str) -> Optional[str]:
    if not llm_client:
        logger.error("LLM client not initialized.")
        return None

    prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False # For simplicity, get full response. True for streaming.
        # "options": {"temperature": 0.7} # Add other Ollama options if needed
    }

    try:
        logger.info(f"Sending request to LLM (model: {model_name}). Prompt length: {len(prompt)}")
        response = await llm_client.post("/api/generate", json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        response_data = response.json()
        answer = response_data.get("response", "").strip()
        logger.info(f"LLM (model: {model_name}) generated answer. Answer length: {len(answer)}")
        return answer
    except httpx.RequestError as e:
        logger.error(f"LLM request error to {e.request.url!r}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"LLM HTTP error {e.response.status_code} while requesting {e.request.url!r}. Response: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error during LLM answer generation: {e}", exc_info=True)

    return None


# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def perform_query(request: QueryRequest):
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Service not ready. Search components failed to initialize.")
    if request.generate_answer and not llm_client:
        raise HTTPException(status_code=503, detail="Service not ready. LLM component failed to initialize.")


    logger.info(f"Received query: '{request.query}', top_k: {request.top_k}, generate_answer: {request.generate_answer}")

    try:
        # 1. Encode the query
        query_vector = embedding_model.encode(request.query, convert_to_tensor=False).tolist()

        # 2. Search Qdrant
        search_results_qdrant = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=request.top_k,
            with_payload=True,
        )

        # 3. Format search results
        formatted_search_results: List[SearchResult] = []
        for hit in search_results_qdrant:
            text_content = hit.payload.get("text", hit.payload.get("content")) if hit.payload else None
            formatted_search_results.append(SearchResult(
                id=hit.id,
                score=hit.score,
                text=text_content,
                metadata=hit.payload.get("metadata") if hit.payload else None
            ))

        logger.info(f"Found {len(formatted_search_results)} search results for query '{request.query}'")

        # 4. Generate answer using LLM if requested
        llm_answer = None
        llm_model_used = None
        if request.generate_answer:
            if not formatted_search_results:
                logger.info("No search results found, cannot generate contextual answer.")
                llm_answer = "I couldn't find any relevant information to answer your question."
            else:
                context_for_llm = _format_context_for_llm(formatted_search_results)
                llm_answer = await _generate_llm_answer(request.query, context_for_llm, LLM_MODEL_NAME)
                llm_model_used = LLM_MODEL_NAME
                if llm_answer is None: # If LLM call failed
                    llm_answer = "There was an issue generating an answer. Please try again later."

        return QueryResponse(
            query=request.query,
            search_results=formatted_search_results,
            answer=llm_answer,
            llm_model_used=llm_model_used
        )

    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")


@app.get("/health", tags=["Health"])
async def health_check():
    # Basic health check, can be expanded to check component status
    q_ready = bool(qdrant_client)
    em_ready = bool(embedding_model)
    llm_ready = bool(llm_client) # Check if llm_client is initialized

    # Optionally, try a lightweight check to Ollama if llm_client is initialized
    ollama_service_healthy = False
    if llm_ready:
        try:
            # A lightweight request to Ollama's root or /api/tags
            response = await llm_client.get("/")
            ollama_service_healthy = response.status_code == 200
        except Exception:
            ollama_service_healthy = False # Could not connect or other error

    if q_ready and em_ready and llm_ready and ollama_service_healthy:
        status = "healthy"
    else:
        status = "degraded"

    return {
        "status": status,
        "components": {
            "qdrant_initialized": q_ready,
            "embedding_model_initialized": em_ready,
            "llm_client_initialized": llm_ready,
            "ollama_service_accessible": ollama_service_healthy
        }
    }


if __name__ == "__main__":
    import uvicorn
    SERVICE_PORT = int(get_config_value("RAG_SERVICE_PORT", default=8001))
    SERVICE_HOST = get_config_value("RAG_SERVICE_HOST", default="0.0.0.0") # Host for the service itself

    # Example: Set Qdrant host for local testing if not done via global env vars
    # os.environ["QDRANT_HOST"] = "localhost"
    # Re-initialize constants if you set env vars here for __main__
    # QDRANT_HOST = get_config_value("QDRANT_HOST", yaml_path="vector_store.qdrant.host", default="localhost")

    logger.info(f"Starting RAG Query Service on {SERVICE_HOST}:{SERVICE_PORT}")
    logger.info(f"Configured Qdrant: host={QDRANT_HOST}, port={QDRANT_PORT}, collection={QDRANT_COLLECTION_NAME}")
    logger.info(f"Configured Embedding Model: name={EMBEDDING_MODEL_NAME}, device={MODEL_DEVICE}")

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)

# To run this:
# 1. Ensure Qdrant is running and the collection exists (or add creation logic).
#    Example Qdrant Docker: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
# 2. Set environment variables if defaults are not suitable:
#    export QDRANT_HOST="your_qdrant_host"
#    export QDRANT_COLLECTION_NAME="your_collection"
#    export EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2" # if different
#    export RAG_SERVICE_PORT=8001
# 3. python Scripts/services/rag_query_service.py

# Example curl to test (after documents are indexed in Qdrant):
# curl -X POST http://localhost:8001/query \
# -H "Content-Type: application/json" \
# -d '{"query": "What is FastAPI?", "top_k": 3}'
