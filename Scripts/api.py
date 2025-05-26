from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from qdrant_client import QdrantClient
from typing import List, Optional, Dict, Any, Literal, Callable
import uvicorn
import logging
import os
import time
from functools import lru_cache
import json
from datetime import datetime
from dotenv import load_dotenv

from embeddings import create_embedding_provider, EmbeddingProvider
from integrations import CopilotAgent, CopilotRequest, CopilotResponse
from api_v1.router import router as v1_router
from api_v1.webhook import webhook_manager
from api_v1.models import WebhookEvent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="Retrieval Augmented Generation API with versioning, webhooks, and bulk operations support"
)

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
class RateLimiter:
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.tokens = {}
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self.tokens = {k: v for k, v in self.tokens.items() if now - v["timestamp"] < self.period}
        
        if key not in self.tokens:
            self.tokens[key] = {"count": 1, "timestamp": now}
            return True
        
        if self.tokens[key]["count"] < self.calls:
            self.tokens[key]["count"] += 1
            return True
        
        return False

rate_limiter = RateLimiter(calls=100, period=60)  # 100 calls per minute

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Callable):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests"}
        )
    
    response = await call_next(request)
    return response

# Mount v1 router
app.include_router(v1_router)

# Global settings
class Settings:
    def __init__(self):
        self.embedding_provider = "ollama"
        self.embedding_model = "snowflake-arctic-embed2:latest"
        self.cache_dir = "embeddings_cache"
        self.batch_size = 32
        self.openai_api_key = None
        self.openai_organization = None
        self.copilot_api_key = os.getenv("GITHUB_COPILOT_API_KEY")
        self.copilot_endpoint = os.getenv("GITHUB_COPILOT_ENDPOINT", 
                                        "https://api.githubcopilot.com/chat/completions")

settings = Settings()

# Dependency for embedding provider
@lru_cache()
def get_embedding_provider():
    return create_embedding_provider(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        cache_dir=settings.cache_dir,
        batch_size=settings.batch_size,
        api_key=settings.openai_api_key,
        organization=settings.openai_organization
    )

# Dependency for Copilot agent
async def get_copilot_agent():
    if not settings.copilot_api_key:
        raise HTTPException(status_code=500, detail="GitHub Copilot API key not configured")
    
    agent = CopilotAgent(
        api_key=settings.copilot_api_key,
        endpoint=settings.copilot_endpoint
    )
    async with agent as session:
        yield session

# Request/Response models
class Query(BaseModel):
    text: str
    limit: Optional[int] = 5
    collection_name: Optional[str] = "documents"

class SearchResponse(BaseModel):
    matches: List[dict]
    query_vector: List[float]

async def get_embeddings(text: str, provider: EmbeddingProvider = Depends(get_embedding_provider)) -> List[float]:
    """Get embeddings using the configured provider."""
    embeddings = await provider.generate_embeddings(text)
    return embeddings[0]  # First embedding since we only have one text

@app.post("/search", response_model=SearchResponse)
async def search(query: Query):
    """Search for similar documents"""
    try:
        # Get query embeddings
        query_vector = await get_embeddings(query.text)
        
        # Search Qdrant
        search_result = qdrant_client.search(
            collection_name=query.collection_name,
            query_vector=query_vector,
            limit=query.limit
        )
        
        # Format results
        matches = []
        for scored_point in search_result:
            matches.append({
                "score": scored_point.score,
                "text": scored_point.payload.get("text"),
                "metadata": {
                    k: v for k, v in scored_point.payload.items()
                    if k != "text"
                }
            })
        
        return SearchResponse(
            matches=matches,
            query_vector=query_vector
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections"""
    try:
        collections = qdrant_client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ConfigUpdateRequest(BaseModel):
    """Request model for updating API configuration."""
    embedding_provider: Optional[Literal["ollama", "openai"]] = None
    embedding_model: Optional[str] = None
    cache_dir: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    batch_size: Optional[int] = None

@app.get("/config")
async def get_config():
    """Get current API configuration."""
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "cache_dir": settings.cache_dir,
        "batch_size": settings.batch_size,
        # Don't return sensitive values
        "openai_api_key": "..." if settings.openai_api_key else None,
        "openai_organization": "..." if settings.openai_organization else None
    }

@app.post("/config")
async def update_config(config: ConfigUpdateRequest):
    """Update API configuration."""
    global _embedding_provider
    
    # Update settings
    for field, value in config.dict(exclude_unset=True).items():
        if value is not None:
            setattr(settings, field, value)
    
    # Force recreation of embedding provider with new settings
    _embedding_provider = None
    
    return await get_config()

@app.get("/health")
async def health_check(provider: EmbeddingProvider = Depends(get_embedding_provider)):
    """Health check endpoint."""
    try:
        # Check Qdrant connection
        qdrant_client.get_collections()
        
        # Check embeddings provider
        test_embedding = await provider.generate_embeddings("health check")
        if not test_embedding or len(test_embedding[0]) != provider.get_embedding_dim():
            raise Exception("Embedding provider test failed")
        
        return {
            "status": "healthy",
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dim": provider.get_embedding_dim()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/copilot/chat", response_model=CopilotResponse)
async def chat_with_copilot(
    request: CopilotRequest,
    agent: CopilotAgent = Depends(get_copilot_agent)
):
    """Chat with GitHub Copilot Agent with RAG context."""
    try:
        # If no context provided, get relevant documents from Qdrant
        if not request.context:
            # Get query embeddings
            provider = get_embedding_provider()
            query_vector = (await provider.generate_embeddings(request.query))[0]
            
            # Search Qdrant
            search_result = qdrant_client.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=5  # Get top 5 relevant documents
            )
            
            # Add search results to context
            request.context = [{
                "text": result.payload.get("text"),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            } for result in search_result]
        
        return await agent.get_completion(request)
        
    except Exception as e:
        logger.error(f"Error in Copilot chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/copilot/chat/stream")
async def stream_chat_with_copilot(
    request: CopilotRequest,
    agent: CopilotAgent = Depends(get_copilot_agent)
):
    """Stream chat with GitHub Copilot Agent with RAG context."""
    try:
        # If no context provided, get relevant documents from Qdrant
        if not request.context:
            # Get query embeddings
            provider = get_embedding_provider()
            query_vector = (await provider.generate_embeddings(request.query))[0]
            
            # Search Qdrant
            search_result = qdrant_client.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=5  # Get top 5 relevant documents
            )
            
            # Add search results to context
            request.context = [{
                "text": result.payload.get("text"),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            } for result in search_result]

        # Set streaming flag
        request.stream = True
        
        # Create async generator for streaming response
        async def generate():
            try:
                async for token in agent.stream_completion(request):
                    if token:
                        yield f"data: {json.dumps({'content': token})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in Copilot stream chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Event handlers
@app.on_event("startup")
async def startup_event():
    # Trigger webhook event for API startup
    await webhook_manager.trigger_event(
        WebhookEvent(
            event_type="api.startup",
            timestamp=datetime.utcnow(),
            data={"status": "API started successfully"}
        )
    )

@app.on_event("shutdown")
async def shutdown_event():
    # Trigger webhook event for API shutdown
    await webhook_manager.trigger_event(
        WebhookEvent(
            event_type="api.shutdown",
            timestamp=datetime.utcnow(),
            data={"status": "API shutting down"}
        )
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)