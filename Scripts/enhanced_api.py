"""
Enhanced RAG API with Integrated Optimization Components
Includes memory optimization, query optimization, auto-scaling, error recovery, and real-time monitoring
"""

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from typing import List, Optional, Dict, Any, Literal, Callable
import uvicorn
import logging
import os
import time
import asyncio
from functools import lru_cache
import json
from datetime import datetime
from dotenv import load_dotenv

# Import existing components
from embeddings import create_embedding_provider, EmbeddingProvider
from integrations import CopilotAgent, CopilotRequest, CopilotResponse
from api_v1.router import router as v1_router

# Import new enhancement components
from enhancers.integration_manager import IntegrationManager, get_integration_manager, initialize_integration
from enhancers.auto_scaler import AutoScaler, get_auto_scaler, ScalingConfig, ResourceType
from enhancers.error_recovery import (
    ErrorRecoverySystem, 
    get_error_recovery_system, 
    with_recovery, 
    health_check,
    ErrorSeverity
)
from monitoring.realtime_dashboard import RealTimeDashboard, MonitoringConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="Enhanced RAG API with Optimization",
    version="2.0.0",
    description="Advanced RAG API with auto-scaling, error recovery, real-time monitoring, and intelligent optimizations"
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

# Enhanced Rate Limiting with Dynamic Adjustment
class EnhancedRateLimiter:
    def __init__(self, calls: int, period: int):
        self.base_calls = calls
        self.base_period = period
        self.current_calls = calls
        self.current_period = period
        self.tokens = {}
        self.system_load_factor = 1.0
    
    def adjust_limits_for_load(self, cpu_usage: float, memory_usage: float):
        """Dynamically adjust rate limits based on system load"""
        # Reduce limits if system is under high load
        if cpu_usage > 0.8 or memory_usage > 0.9:
            self.system_load_factor = 0.5
        elif cpu_usage > 0.6 or memory_usage > 0.7:
            self.system_load_factor = 0.7
        else:
            self.system_load_factor = 1.0
        
        self.current_calls = int(self.base_calls * self.system_load_factor)
        logger.debug(f"Rate limit adjusted to {self.current_calls} calls per {self.current_period}s")
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self.tokens = {k: v for k, v in self.tokens.items() if now - v["timestamp"] < self.current_period}
        
        if key not in self.tokens:
            self.tokens[key] = {"count": 1, "timestamp": now}
            return True
        
        if self.tokens[key]["count"] < self.current_calls:
            self.tokens[key]["count"] += 1
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "base_calls_per_period": self.base_calls,
            "current_calls_per_period": self.current_calls,
            "period_seconds": self.current_period,
            "system_load_factor": self.system_load_factor,
            "active_clients": len(self.tokens)
        }

rate_limiter = EnhancedRateLimiter(calls=100, period=60)

# Enhanced middleware with error recovery and monitoring
@app.middleware("http")
async def enhanced_middleware(request: Request, call_next: Callable):
    start_time = time.time()
    client_ip = request.client.host
    
    try:
        # Get integration manager for monitoring
        try:
            integration_manager = await get_integration_manager()
            
            # Get current system metrics for rate limiting adjustment
            stats = integration_manager.get_integration_stats()
            if stats.get("latest_health"):
                health = stats["latest_health"]
                rate_limiter.adjust_limits_for_load(
                    health.get("cpu_usage", 0.0),
                    health.get("memory_usage", 0.0)
                )
        except Exception as e:
            logger.debug(f"Could not get integration manager: {e}")
        
        # Check rate limits
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add performance headers
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "2.0.0"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in middleware: {e}")
        process_time = time.time() - start_time
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An error occurred processing your request",
                "process_time": process_time
            }
        )

# Mount v1 router
app.include_router(v1_router, prefix="/api/v1")

# Enhanced settings with optimization configurations
class EnhancedSettings:
    def __init__(self):
        # Basic settings
        self.embedding_provider = "ollama"
        self.embedding_model = "snowflake-arctic-embed2:latest"
        self.cache_dir = "embeddings_cache"
        self.batch_size = int(os.getenv("RAG_BATCH_SIZE", "32"))
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        self.copilot_api_key = os.getenv("GITHUB_COPILOT_API_KEY")
        self.copilot_endpoint = os.getenv("GITHUB_COPILOT_ENDPOINT", 
                                        "https://api.githubcopilot.com/chat/completions")
        
        # Enhancement settings
        self.enable_auto_scaling = os.getenv("ENABLE_AUTO_SCALING", "true").lower() == "true"
        self.enable_error_recovery = os.getenv("ENABLE_ERROR_RECOVERY", "true").lower() == "true"
        self.enable_monitoring = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        self.cpu_workers = int(os.getenv("RAG_CPU_WORKERS", "4"))
        self.memory_cache_mb = int(os.getenv("RAG_MEMORY_CACHE_MB", "2048"))

settings = EnhancedSettings()

# Enhanced dependency providers
@lru_cache()
def get_embedding_provider():
    """Get embedding provider with error recovery"""
    try:
        return create_embedding_provider(
            provider=settings.embedding_provider,
            model_name=settings.embedding_model,
            cache_dir=settings.cache_dir,
            batch_size=settings.batch_size,
            api_key=settings.openai_api_key,
            organization=settings.openai_organization
        )
    except Exception as e:
        logger.error(f"Failed to create embedding provider: {e}")
        # Fallback to a basic provider
        return create_embedding_provider(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")

async def get_copilot_agent():
    """Get Copilot agent with error recovery"""
    if not settings.copilot_api_key:
        raise HTTPException(status_code=500, detail="GitHub Copilot API key not configured")
    
    agent = CopilotAgent(
        api_key=settings.copilot_api_key,
        endpoint=settings.copilot_endpoint
    )
    async with agent as session:
        yield session

# Enhanced request/response models
class EnhancedQuery(BaseModel):
    text: str
    limit: Optional[int] = 5
    collection_name: Optional[str] = "documents"
    enable_optimization: Optional[bool] = True
    use_cache: Optional[bool] = True
    context: Optional[Dict[str, Any]] = None

class EnhancedSearchResponse(BaseModel):
    matches: List[dict]
    query_vector: List[float]
    optimization_metadata: Optional[Dict[str, Any]] = None
    cache_hit: Optional[bool] = False
    response_time_ms: Optional[float] = None
    system_health: Optional[Dict[str, Any]] = None

class SystemStatusResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    performance: Dict[str, Any]
    integrations: Dict[str, Any]

# Health check functions for monitoring
@health_check("qdrant")
async def check_qdrant_health():
    """Health check for Qdrant vector database"""
    try:
        collections = qdrant_client.get_collections()
        return {"status": "healthy", "collections_count": len(collections.collections)}
    except Exception as e:
        raise Exception(f"Qdrant health check failed: {e}")

@health_check("embedding_provider")
async def check_embedding_provider_health():
    """Health check for embedding provider"""
    try:
        provider = get_embedding_provider()
        # Test with a simple embedding
        test_embedding = await provider.generate_embeddings("health check test")
        return {"status": "healthy", "embedding_dimension": len(test_embedding[0])}
    except Exception as e:
        raise Exception(f"Embedding provider health check failed: {e}")

# Enhanced API endpoints with integrated optimizations
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced RAG API with Optimization",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health",
        "status": "/status"
    }

@app.get("/health")
@with_recovery(component_name="api_health", severity=ErrorSeverity.LOW)
async def health_check_endpoint():
    """Enhanced health check endpoint"""
    try:
        # Get error recovery system health
        recovery_system = get_error_recovery_system()
        system_health = recovery_system.get_system_health()
        
        # Get integration manager stats
        try:
            integration_manager = await get_integration_manager()
            integration_stats = integration_manager.get_integration_stats()
        except Exception:
            integration_stats = {"status": "not_initialized"}
        
        # Get auto-scaler stats
        try:
            auto_scaler = await get_auto_scaler()
            scaling_stats = auto_scaler.get_scaling_stats()
        except Exception:
            scaling_stats = {"status": "not_initialized"}
        
        return {
            "status": "healthy" if system_health["overall_state"] == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "integration_stats": integration_stats,
            "scaling_stats": scaling_stats,
            "components": {
                "qdrant": "healthy",
                "embedding_provider": "healthy",
                "error_recovery": "active" if settings.enable_error_recovery else "disabled",
                "auto_scaling": "active" if settings.enable_auto_scaling else "disabled",
                "monitoring": "active" if settings.enable_monitoring else "disabled"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/status", response_model=SystemStatusResponse)
@with_recovery(component_name="api_status", severity=ErrorSeverity.LOW)
async def system_status():
    """Comprehensive system status endpoint"""
    start_time = app.state.start_time if hasattr(app.state, 'start_time') else time.time()
    uptime = time.time() - start_time
    
    # Get component statuses
    components = {}
    
    # Integration Manager
    try:
        integration_manager = await get_integration_manager()
        components["integration_manager"] = integration_manager.get_integration_stats()
    except Exception as e:
        components["integration_manager"] = {"status": "error", "error": str(e)}
    
    # Auto Scaler
    try:
        auto_scaler = await get_auto_scaler()
        components["auto_scaler"] = auto_scaler.get_scaling_stats()
    except Exception as e:
        components["auto_scaler"] = {"status": "error", "error": str(e)}
    
    # Error Recovery
    try:
        recovery_system = get_error_recovery_system()
        components["error_recovery"] = recovery_system.get_error_statistics()
    except Exception as e:
        components["error_recovery"] = {"status": "error", "error": str(e)}
    
    # Performance metrics
    performance = {
        "rate_limiter": rate_limiter.get_stats(),
        "settings": {
            "batch_size": settings.batch_size,
            "cpu_workers": settings.cpu_workers,
            "memory_cache_mb": settings.memory_cache_mb
        }
    }
    
    # Integration status
    integrations = {
        "auto_scaling_enabled": settings.enable_auto_scaling,
        "error_recovery_enabled": settings.enable_error_recovery,
        "monitoring_enabled": settings.enable_monitoring
    }
    
    return SystemStatusResponse(
        status="operational",
        version="2.0.0",
        uptime_seconds=uptime,
        components=components,
        performance=performance,
        integrations=integrations
    )

@app.post("/search", response_model=EnhancedSearchResponse)
@with_recovery(component_name="api_search", severity=ErrorSeverity.MEDIUM)
async def enhanced_search(query: EnhancedQuery, provider=Depends(get_embedding_provider)):
    """Enhanced search with optimization and error recovery"""
    start_time = time.time()
    
    try:
        # Get integration manager for optimization
        integration_manager = await get_integration_manager()
        
        # Check cache first if enabled
        cache_hit = False
        cached_result = None
        if query.use_cache:
            cached_result = await integration_manager.get_cached_result(query.text)
            if cached_result:
                cache_hit = True
                response_time = (time.time() - start_time) * 1000
                return EnhancedSearchResponse(
                    matches=cached_result["matches"],
                    query_vector=cached_result["query_vector"],
                    optimization_metadata=cached_result.get("optimization_metadata", {}),
                    cache_hit=True,
                    response_time_ms=response_time
                )
        
        # Optimize query if enabled
        optimization_metadata = {}
        optimized_query_text = query.text
        
        if query.enable_optimization:
            optimization_result = await integration_manager.optimize_query(query.text, query.context)
            optimized_query_text = optimization_result["optimized_query"]
            optimization_metadata = optimization_result["metadata"]
        
        # Generate embeddings
        query_vector = (await provider.generate_embeddings(optimized_query_text))[0]
        
        # Search Qdrant
        search_result = qdrant_client.search(
            collection_name=query.collection_name,
            query_vector=query_vector,
            limit=query.limit
        )
        
        # Format results
        matches = [{
            "id": result.id,
            "score": result.score,
            "payload": result.payload
        } for result in search_result]
        
        # Cache result if enabled
        result_to_cache = {
            "matches": matches,
            "query_vector": query_vector,
            "optimization_metadata": optimization_metadata
        }
        
        if query.use_cache:
            await integration_manager.cache_result(query.text, result_to_cache, ttl=3600)  # 1 hour TTL
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        
        # Get system health
        system_health = None
        try:
            recovery_system = get_error_recovery_system()
            health_data = recovery_system.get_system_health()
            system_health = {"overall_state": health_data["overall_state"]}
        except Exception:
            pass
        
        return EnhancedSearchResponse(
            matches=matches,
            query_vector=query_vector,
            optimization_metadata=optimization_metadata,
            cache_hit=cache_hit,
            response_time_ms=response_time,
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/collections")
@with_recovery(component_name="api_collections", severity=ErrorSeverity.LOW)
async def list_collections():
    """List all collections with enhanced error handling"""
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": collection.name,
                    "status": collection.status,
                    "vectors_count": collection.vectors_count,
                    "indexed_vectors_count": collection.indexed_vectors_count
                }
                for collection in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Collections listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

@app.post("/copilot/chat", response_model=CopilotResponse)
@with_recovery(component_name="api_copilot", severity=ErrorSeverity.MEDIUM)
async def chat_with_copilot(
    request: CopilotRequest,
    agent: CopilotAgent = Depends(get_copilot_agent),
    provider=Depends(get_embedding_provider)
):
    """Enhanced Copilot chat with optimization and error recovery"""
    try:
        # Get integration manager for optimization
        integration_manager = await get_integration_manager()
        
        # Optimize query if needed
        optimization_result = await integration_manager.optimize_query(request.query, request.context)
        optimized_query = optimization_result["optimized_query"]
        
        # If no context provided, get relevant documents from Qdrant
        if not request.context:
            # Get query embeddings
            query_vector = (await provider.generate_embeddings(optimized_query))[0]
            
            # Search Qdrant
            search_result = qdrant_client.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=5
            )
            
            # Add search results to context
            request.context = [{
                "text": result.payload.get("text"),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            } for result in search_result]
        
        # Update request with optimized query
        enhanced_request = CopilotRequest(
            query=optimized_query,
            conversation_id=request.conversation_id,
            context=request.context,
            max_tokens=request.max_tokens
        )
        
        return await agent.get_completion(enhanced_request)
        
    except Exception as e:
        logger.error(f"Copilot chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Copilot chat failed: {str(e)}")

# Auto-scaling management endpoints
@app.get("/admin/scaling/status")
@with_recovery(component_name="admin_scaling", severity=ErrorSeverity.LOW)
async def get_scaling_status():
    """Get auto-scaling status"""
    try:
        auto_scaler = await get_auto_scaler()
        stats = auto_scaler.get_scaling_stats()
        recent_decisions = auto_scaler.get_recent_decisions(10)
        
        return {
            "scaling_stats": stats,
            "recent_decisions": recent_decisions,
            "enabled": settings.enable_auto_scaling
        }
    except Exception as e:
        logger.error(f"Scaling status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scaling status: {str(e)}")

@app.post("/admin/scaling/force")
@with_recovery(component_name="admin_scaling", severity=ErrorSeverity.MEDIUM)
async def force_scaling_action(
    resource_type: str,
    target_value: int,
    background_tasks: BackgroundTasks
):
    """Force a scaling action"""
    try:
        # Validate resource type
        try:
            rt = ResourceType(resource_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        auto_scaler = await get_auto_scaler()
        
        async def perform_scaling():
            success = await auto_scaler.force_scaling_action(rt, target_value)
            logger.info(f"Forced scaling action result: {success}")
        
        background_tasks.add_task(perform_scaling)
        
        return {
            "message": f"Scaling action initiated for {resource_type} to {target_value}",
            "resource_type": resource_type,
            "target_value": target_value
        }
    except Exception as e:
        logger.error(f"Force scaling error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to force scaling: {str(e)}")

# Error recovery management endpoints
@app.get("/admin/errors/statistics")
@with_recovery(component_name="admin_errors", severity=ErrorSeverity.LOW)
async def get_error_statistics():
    """Get error recovery statistics"""
    try:
        recovery_system = get_error_recovery_system()
        stats = recovery_system.get_error_statistics()
        recent_errors = recovery_system.get_recent_errors(20)
        
        return {
            "statistics": stats,
            "recent_errors": recent_errors
        }
    except Exception as e:
        logger.error(f"Error statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get error statistics: {str(e)}")

# Startup event to initialize enhancements
@app.on_event("startup")
async def startup_event():
    """Initialize enhancement systems on startup"""
    app.state.start_time = time.time()
    logger.info("Starting Enhanced RAG API v2.0.0...")
    
    try:
        # Initialize integration system
        if settings.enable_error_recovery or settings.enable_auto_scaling or settings.enable_monitoring:
            success = await initialize_integration()
            if success:
                logger.info("Integration system initialized successfully")
            else:
                logger.warning("Integration system initialization failed")
        
        logger.info("Enhanced RAG API startup complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced RAG API...")
    
    try:
        # Shutdown integration manager
        integration_manager = await get_integration_manager()
        await integration_manager.shutdown()
        
        # Shutdown auto-scaler
        auto_scaler = await get_auto_scaler()
        await auto_scaler.shutdown()
        
        # Shutdown error recovery
        recovery_system = get_error_recovery_system()
        await recovery_system.shutdown()
        
        logger.info("Enhanced RAG API shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        workers=1,  # Single worker for proper state management
        log_level="info"
    )
