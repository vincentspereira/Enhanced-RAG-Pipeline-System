"""
Enhanced RAG API with Integrated Optimization Components
Includes memory optimization, query optimization, auto-scaling, error recovery, and real-time monitoring
"""

from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks, Header, status as fastapi_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # Added OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from typing import List, Optional, Dict, Any, Literal, Callable, Union
import uvicorn
import logging
import os
import time
import asyncio
from functools import lru_cache
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

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

# Import ConfigManager and specific config dataclasses
from config.manager import (
    ConfigManager,
    SystemConfig as AppSystemConfig,
    ModelConfig as AppModelConfig,
    VectorStoreConfig as AppVectorStoreConfig,
    ProcessingConfig as AppProcessingConfig,
    APIConfig as AppAPIConfig,
    PathsConfig as AppPathsConfig,
    CacheSettingsConfig as AppCacheSettingsConfig,
    FeatureFlagsConfig as AppFeatureFlagsConfig,
    ElasticsearchConfig as AppElasticsearchConfig,
    AuthConfigData # Imported AuthConfigData
)

# Import RAGPipeline and its dependencies
from rag_pipeline import RAGPipeline
from document_processor import DocumentProcessor as ActualDocumentProcessor
from embedding_generator import EmbeddingGenerator as ActualEmbeddingGenerator

# Import AuthManager and related items
from auth.auth_manager import AuthManager, AuthConfig as AppAuthConfig, User as AuthUser, Permission

# Import Notification Service
from notification_service import initialize_notification_service, get_notification_service, Notifier

# Import Audit Logger
from security.audit_logger import AuditLogger, AuditEvent

# Import QueryCache
from caching.query_cache import QueryCache, CacheConfig as QueryCacheModuleConfig

# Import DB Connectors
from integrations.postgres_connector import PostgresConnector
from integrations.mongodb_connector import MongoDBConnector
from integrations.mysql_connector import MySQLConnector
from integrations.sqlite_connector import SQLiteConnector
from integrations.snowflake_connector import SnowflakeConnector
from integrations.bigquery_connector import BigQueryConnector
from vector_stores.chroma_vector_store import ChromaVectorStore # Assuming Chroma client is stateful for app.state


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enhanced RAG API with Optimization",
    version="2.0.0",
    description="Advanced RAG API with auto-scaling, error recovery, real-time monitoring, and intelligent optimizations"
)

# Initialize ConfigManager
config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if not os.path.exists(config_file_path):
    ConfigManager(config_path=config_file_path).save_config()
config_manager = ConfigManager(config_path=config_file_path)
app_config: AppSystemConfig = config_manager.config

# Global Qdrant client for health checks
qdrant_client = QdrantClient(
    host=app_config.vector_store.host,
    port=app_config.vector_store.port
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Rate Limiter (definition as before)
class EnhancedRateLimiter:
    def __init__(self, calls: int, period: int):
        self.base_calls = calls
        self.base_period = period
        self.current_calls = calls
        self.current_period = period
        self.tokens = {}
        self.system_load_factor = 1.0
    def adjust_limits_for_load(self, cpu_usage: float, memory_usage: float):
        if cpu_usage > 0.8 or memory_usage > 0.9: self.system_load_factor = 0.5
        elif cpu_usage > 0.6 or memory_usage > 0.7: self.system_load_factor = 0.7
        else: self.system_load_factor = 1.0
        self.current_calls = int(self.base_calls * self.system_load_factor)
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
        return {"base_calls_per_period": self.base_calls, "current_calls_per_period": self.current_calls,
                "period_seconds": self.current_period, "system_load_factor": self.system_load_factor,
                "active_clients": len(self.tokens)}
rate_limiter = EnhancedRateLimiter(calls=app_config.api.max_concurrent_requests * 60, period=60)

@app.middleware("http")
async def enhanced_middleware(request: Request, call_next: Callable):
    start_time = time.time()
    client_ip = request.client.host
    try:
        if hasattr(app.state, 'integration_manager_instance') and app.state.integration_manager_instance:
            integration_manager = app.state.integration_manager_instance
            stats = integration_manager.get_integration_stats()
            if stats.get("latest_health"):
                health = stats["latest_health"]
                rate_limiter.adjust_limits_for_load(health.get("cpu_usage",0.0), health.get("memory_usage",0.0))
        if not rate_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"error": "Too many requests"})
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = app.version
        return response
    except Exception as e:
        logger.error(f"Error in middleware: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

# Mount v1 router (which now includes all sub-routers)
app.include_router(v1_router, prefix="/api/v1")


# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event_main():
    app.state.start_time = time.time()
    logger.info(f"Starting {app.title} v{app.version}...")
    app.state.app_config = app_config
    app.state.config_manager = config_manager

    # Initialize AuthManager
    auth_config_data_from_app = app_config.auth # This is AuthConfigData from config.manager
    secret_key_from_env = os.getenv("API_SECRET_KEY", auth_config_data_from_app.secret_key)

    # Map AuthConfigData to AppAuthConfig (from auth_manager)
    auth_manager_config = AppAuthConfig(
        secret_key=secret_key_from_env,
        token_expire_minutes=auth_config_data_from_app.token_expire_minutes,
        refresh_token_expire_days=auth_config_data_from_app.refresh_token_expire_days,
        password_min_length=auth_config_data_from_app.password_min_length,
        max_failed_attempts=auth_config_data_from_app.max_failed_attempts,
        lockout_duration_minutes=auth_config_data_from_app.lockout_duration_minutes,
        api_key_prefix=auth_config_data_from_app.api_key_prefix,
        redis_url=auth_config_data_from_app.redis_url
    )
    auth_manager_instance = AuthManager(config=auth_manager_config)
    app.state.auth_manager = auth_manager_instance
    logger.info("AuthManager initialized.")

    doc_processor = ActualDocumentProcessor(config=app_config.processing)
    emb_generator = ActualEmbeddingGenerator(
        model_name=app_config.model.embedding_model,
        device=app_config.model.device,
        cpu_thread_count=app_config.model.cpu_thread_count # Pass cpu_thread_count
    )
    rag_pipeline_instance = RAGPipeline(
        app_config=app_config,
        doc_processor=doc_processor,
        embedding_generator=emb_generator
    )
    await rag_pipeline_instance.async_initialize_components()
    app.state.rag_pipeline = rag_pipeline_instance
    logger.info("RAGPipeline initialized.")

    # Initialize other enhancement systems
    if app_config.feature_flags.enable_error_recovery or \
       app_config.feature_flags.enable_auto_scaling or \
       app_config.feature_flags.enable_monitoring: # Placeholder for actual monitoring flag
        try:
            app.state.integration_manager_instance = await initialize_integration()
            if app.state.integration_manager_instance: logger.info("Integration system initialized.")
            else: logger.warning("Integration system initialization failed or returned None.")
        except Exception as e:
            logger.error(f"Failed to initialize integration system: {e}")
            app.state.integration_manager_instance = None
    else:
        app.state.integration_manager_instance = None
        logger.info("Enhancement systems disabled by feature flags.")

    # Initialize Notification Service
    try:
        initialize_notification_service(app_config) # app_config is SystemConfig which now includes NotificationConfig
        app.state.notification_service = get_notification_service() # Store instance if needed by other parts via app.state
        logger.info("Notification service initialized.")
        if app.state.notification_service:
            await app.state.notification_service.send_notification(
                subject=f"{app.title} Startup",
                message=f"{app.title} v{app.version} has started successfully.",
                metadata={"startup_time": app.state.start_time}
            )
    except Exception as e:
        logger.error(f"Failed to initialize or use notification service during startup: {e}")
        app.state.notification_service = None # Ensure it's None if init failed

    # Initialize Audit Logger
    try:
        # Assuming app_config.audit is AuditLoggerConfig from config.manager
        audit_logger_config_dict = {
            "log_dir": str(app_config.audit.log_dir), # Convert Path if it's Path object
            "use_elasticsearch": app_config.audit.use_elasticsearch,
            "elasticsearch_url": app_config.audit.elasticsearch_url
            # elasticsearch_index_prefix is handled by AuditLogger default if not in dict
        }
        audit_logger_instance = AuditLogger(config=audit_logger_config_dict)
        app.state.audit_logger = audit_logger_instance
        logger.info("AuditLogger initialized.")
        # Log successful startup
        app.state.audit_logger.log_event(AuditEvent(
            event_type="system_startup",
            user_id="system",
            action="Application Started",
            resource_type="application",
            resource_id=app.title,
            status="success",
            metadata={"version": app.version}
        ))
    except Exception as e:
        logger.error(f"Failed to initialize AuditLogger: {e}", exc_info=True)
        app.state.audit_logger = None

    # Initialize QueryCache
    try:
        qc_settings = app_config.query_cache # This is QueryCacheSettings from config.manager
        query_cache_module_config = QueryCacheModuleConfig(
            cache_type=qc_settings.cache_type,
            redis_url=qc_settings.redis_url,
            disk_cache_dir=str(app_config.paths.cache_dir / "query_cache_disk"), # Ensure path is string
            default_ttl=qc_settings.default_ttl
            # Add other mappings from QueryCacheSettings to QueryCacheModuleConfig if they expand
        )
        query_cache_instance = QueryCache(config=query_cache_module_config)
        app.state.query_cache = query_cache_instance
        logger.info(f"QueryCache initialized with type: {query_cache_module_config.cache_type}")
        # Example: Test query cache with a dummy operation
        # await app.state.query_cache.cache_result("startup_test_key", {"status": "ok"})
        # cached_val = await app.state.query_cache.get_cached_result("startup_test_key")
        # logger.info(f"QueryCache test: {cached_val}")

    except Exception as e:
        logger.error(f"Failed to initialize QueryCache: {e}", exc_info=True)
        app.state.query_cache = None

    # Initialize PostgresConnector if configured
    if app_config.postgres:
        try:
            pg_connector_instance = PostgresConnector(config=app_config.postgres)
            app.state.postgres_connector = pg_connector_instance
            logger.info("PostgresConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgresConnector: {e}", exc_info=True)
            app.state.postgres_connector = None
    else:
        app.state.postgres_connector = None
        logger.info("PostgreSQL not configured, skipping PostgresConnector initialization.")

    # Initialize MongoDBConnector if configured
    if app_config.mongodb:
        try:
            mongo_connector_instance = MongoDBConnector(config=app_config.mongodb)
            app.state.mongodb_connector = mongo_connector_instance
            logger.info("MongoDBConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDBConnector: {e}", exc_info=True)
            app.state.mongodb_connector = None
    else:
        app.state.mongodb_connector = None
        logger.info("MongoDB not configured, skipping MongoDBConnector initialization.")

    # Initialize MySQLConnector if configured
    if app_config.mysql:
        try:
            mysql_connector_instance = MySQLConnector(config=app_config.mysql)
            app.state.mysql_connector = mysql_connector_instance
            logger.info("MySQLConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize MySQLConnector: {e}", exc_info=True)
            app.state.mysql_connector = None
    else:
        app.state.mysql_connector = None
        logger.info("MySQL not configured, skipping MySQLConnector initialization.")

    # Initialize SQLiteConnector if configured
    if app_config.sqlite:
        try:
            sqlite_connector_instance = SQLiteConnector(config=app_config.sqlite)
            app.state.sqlite_connector = sqlite_connector_instance
            logger.info("SQLiteConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SQLiteConnector: {e}", exc_info=True)
            app.state.sqlite_connector = None
    else:
        app.state.sqlite_connector = None
        logger.info("SQLite not configured, skipping SQLiteConnector initialization.")

    # Initialize ChromaDBVectorStore if configured
    if app_config.chromadb:
        try:
            # Assuming ChromaDBConfig aligns with ChromaVectorStore's expected config
            chroma_vs_instance = ChromaVectorStore(config=app_config.chromadb)
            app.state.chroma_vector_store = chroma_vs_instance
            logger.info(f"ChromaVectorStore initialized (mode: {app_config.chromadb.mode}).")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaVectorStore: {e}", exc_info=True)
            app.state.chroma_vector_store = None
    else:
        app.state.chroma_vector_store = None
        logger.info("ChromaDB not configured, skipping ChromaVectorStore initialization.")

    # Initialize SnowflakeConnector if configured
    if app_config.snowflake:
        try:
            snowflake_connector_instance = SnowflakeConnector(config=app_config.snowflake)
            app.state.snowflake_connector = snowflake_connector_instance
            logger.info("SnowflakeConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize SnowflakeConnector: {e}", exc_info=True)
            app.state.snowflake_connector = None
    else:
        app.state.snowflake_connector = None
        logger.info("Snowflake not configured, skipping SnowflakeConnector initialization.")

    # Initialize BigQueryConnector if configured
    if app_config.bigquery:
        try:
            bigquery_connector_instance = BigQueryConnector(config=app_config.bigquery)
            app.state.bigquery_connector = bigquery_connector_instance
            logger.info("BigQueryConnector initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize BigQueryConnector: {e}", exc_info=True)
            app.state.bigquery_connector = None
    else:
        app.state.bigquery_connector = None
        logger.info("BigQuery not configured, skipping BigQueryConnector initialization.")


    logger.info(f"{app.title} startup complete.")

@app.on_event("shutdown")
async def shutdown_event_main():
    logger.info(f"Shutting down {app.title}...")

    # Close Postgres connection pool
    if hasattr(app.state, 'postgres_connector') and app.state.postgres_connector:
        app.state.postgres_connector.close_pool()
        logger.info("PostgresConnector pool closed.")

    # Close MongoDB connection
    if hasattr(app.state, 'mongodb_connector') and app.state.mongodb_connector:
        app.state.mongodb_connector.close_connection()
        logger.info("MongoDBConnector connection closed.")

    # Close MySQL connection
    if hasattr(app.state, 'mysql_connector') and app.state.mysql_connector:
        app.state.mysql_connector.close_connection()
        logger.info("MySQLConnector connection closed.")

    # Close SQLite connection
    if hasattr(app.state, 'sqlite_connector') and app.state.sqlite_connector:
        app.state.sqlite_connector.close_connection()
        logger.info("SQLiteConnector connection closed.")

    # Close Snowflake connection
    if hasattr(app.state, 'snowflake_connector') and app.state.snowflake_connector:
        app.state.snowflake_connector.close_connection()
        logger.info("SnowflakeConnector connection closed.")

    # Close BigQuery client (if it has an explicit close)
    if hasattr(app.state, 'bigquery_connector') and app.state.bigquery_connector:
        app.state.bigquery_connector.close_connection() # Assumes close_connection exists
        logger.info("BigQueryConnector resources released (if applicable).")

    # ChromaDB client might not need explicit close if it's HTTP based or managed internally
    # if hasattr(app.state, 'chroma_vector_store') and app.state.chroma_vector_store:
    #     # app.state.chroma_vector_store.client.close() # If applicable
    #     logger.info("ChromaVectorStore client closed (if applicable).")


    if hasattr(app.state, 'rag_pipeline') and app.state.rag_pipeline and app.state.rag_pipeline.es_fallback:
        await app.state.rag_pipeline.es_fallback.close()
        logger.info("Closed RAGPipeline Elasticsearch connection.")

    if hasattr(app.state, 'integration_manager_instance') and app.state.integration_manager_instance:
        try:
            # Assuming get_integration_manager() retrieves the instance from app.state or re-initializes if needed
            integration_manager = await get_integration_manager()
            if integration_manager: await integration_manager.shutdown()

            if app_config.feature_flags.enable_auto_scaling:
                auto_scaler = await get_auto_scaler()
                if auto_scaler: await auto_scaler.shutdown()
            if app_config.feature_flags.enable_error_recovery:
                recovery_system = get_error_recovery_system()
                if recovery_system: await recovery_system.shutdown()
            logger.info("Enhancement systems shutdown complete.")
        except Exception as e:
            logger.error(f"Shutdown error during enhancer cleanup: {e}")
    logger.info(f"{app.title} shutdown complete.")

# --- Dependency Providers ---
# (get_embedding_provider_dependency, get_copilot_agent_dependency, verify_copilot_token_dependency as previously defined)
@lru_cache()
def get_embedding_provider_dependency():
    try:
        return create_embedding_provider(
            provider=app_config.model.llm_service, model_name=app_config.model.embedding_model,
            cache_dir=str(app_config.paths.cache_dir), batch_size=app_config.model.batch_size,
        )
    except Exception as e:
        logger.error(f"Failed to create embedding provider: {e}")
        return create_embedding_provider(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")

async def get_copilot_agent_dependency():
    copilot_api_key = os.getenv("GITHUB_COPILOT_API_KEY")
    copilot_endpoint = os.getenv("GITHUB_COPILOT_ENDPOINT", "https://api.githubcopilot.com/chat/completions")
    if not copilot_api_key:
        raise HTTPException(status_code=500, detail="GitHub Copilot API key not configured")
    agent = CopilotAgent(api_key=copilot_api_key, endpoint=copilot_endpoint)
    async with agent as session: yield session

async def verify_copilot_token_dependency(x_copilot_token: Optional[str] = Header(None, alias="X-Copilot-Token")):
    if x_copilot_token and not x_copilot_token.startswith("gca_"):
        raise HTTPException(status_code=401, detail="Invalid Copilot Agent token format.")
    return x_copilot_token

# Dependency to get AuthManager instance
def get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth_manager

# --- Pydantic Models (SearchQueryInput, ProcessRequestInput, etc. as previously defined) ---
class SearchQueryInput(BaseModel):
    query: str; limit: Optional[int] = 5; filters: Optional[Dict[str, Any]] = None; categories: Optional[Union[List[str], str]] = None
class ProcessRequestInput(BaseModel):
    directory_path: str; batch_size: Optional[int] = 32
class GenerateRequestInput(BaseModel):
    question: str; template_name: Optional[str] = "qa_prompt"; limit: Optional[int] = 5
class SearchResultItem(BaseModel):
    text: str; metadata: Dict[str, Any]; score: float
class MigratedSearchResponse(BaseModel):
    results: List[SearchResultItem]
class ConfigUpdateRequestModel(BaseModel):
    model: Optional[AppModelConfig] = None; vector_store: Optional[AppVectorStoreConfig] = None
    processing: Optional[AppProcessingConfig] = None; api: Optional[AppAPIConfig] = None
    paths: Optional[AppPathsConfig] = None; cache_settings: Optional[AppCacheSettingsConfig] = None
    feature_flags: Optional[AppFeatureFlagsConfig] = None; elasticsearch: Optional[AppElasticsearchConfig] = None
class EnhancedQuery(BaseModel):
    text: str; limit: Optional[int] = 5; collection_name: Optional[str] = "documents"
    enable_optimization: Optional[bool] = True; use_cache: Optional[bool] = True; context: Optional[Dict[str, Any]] = None
class EnhancedSearchResponse(BaseModel):
    matches: List[dict]; query_vector: List[float]; optimization_metadata: Optional[Dict[str, Any]] = None
    cache_hit: Optional[bool] = False; response_time_ms: Optional[float] = None; system_health: Optional[Dict[str, Any]] = None
class SystemStatusResponse(BaseModel):
    status: str; version: str; uptime_seconds: float; components: Dict[str, Dict[str, Any]]
    performance: Dict[str, Any]; integrations: Dict[str, Any]


# --- Core RAG Endpoints ---
# (search_rag_endpoint, process_documents_endpoint, etc. as previously defined)
@app.post("/search_rag", response_model=MigratedSearchResponse, tags=["RAG Core"])
async def search_rag_endpoint(
    payload: SearchQueryInput,
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.SEARCH_BASIC)
):
    rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
    if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    try:
        results = await rag_pipeline.search(query=payload.query, limit=payload.limit, filters=payload.filters, categories=payload.categories)
        return MigratedSearchResponse(results=[SearchResultItem(**res) for res in results])
    except Exception as e: logger.error(f"Search RAG error: {str(e)}"); raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_docs", tags=["RAG Core"])
async def process_documents_endpoint(
    payload: ProcessRequestInput,
    background_tasks: BackgroundTasks,
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.DOCUMENT_WRITE)
):
    rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
    if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    path = Path(payload.directory_path)
    if not path.exists() or not path.is_dir(): raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    background_tasks.add_task(rag_pipeline.process_documents, path, payload.batch_size)
    return {"message": f"Started processing documents from {path}"}

@app.get("/system_status_rag", tags=["RAG Core"])
async def get_system_status_rag_endpoint(
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.API_READ)
):
    rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
    query_cache: Optional[QueryCache] = fastapi_req.app.state.query_cache
    cache_key_prefix = "system_status_rag" # For QueryCache's _generate_key which takes a query string

    if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    if query_cache:
        # For a parameterless endpoint, the "query" part of the key can be static.
        # Params can be None or an empty dict.
        cached_data = await query_cache.get_cached_result(query=cache_key_prefix, params={})
        if cached_data:
            logger.info(f"Returning cached data for {cache_key_prefix}")
            return cached_data

    try:
        loop = asyncio.get_event_loop()
        collection_info = await loop.run_in_executor(None, rag_pipeline.get_collection_info)

        if query_cache:
            # Cache the result. TTL can be from QueryCache's default_ttl or specified here.
            await query_cache.cache_result(query=cache_key_prefix, params={}, result=collection_info)
            logger.info(f"Cached data for {cache_key_prefix}")

        return collection_info
    except Exception as e:
        logger.error(f"RAG System status error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_response", tags=["RAG Core"])
async def generate_response_endpoint(
    payload: GenerateRequestInput,
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.API_EXECUTE)
):
    rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
    if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
    try:
        search_results = await rag_pipeline.search(payload.question, limit=payload.limit)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, rag_pipeline.generate_response, payload.question, search_results, payload.template_name)
        return response
    except Exception as e: logger.error(f"Generation error: {str(e)}"); raise HTTPException(status_code=500, detail=str(e))

# --- Copilot Endpoints ---
# (consolidated_chat_with_copilot, consolidated_stream_chat_with_copilot as previously defined)
@app.post("/copilot/chat", response_model=CopilotResponse, tags=["Copilot"])
async def consolidated_chat_with_copilot(
    copilot_req_body: CopilotRequest,
    fastapi_req: Request, # Moved fastapi_req up to be before Depends that use it or app.state
    agent: CopilotAgent = Depends(get_copilot_agent_dependency),
    copilot_token: Optional[str] = Depends(verify_copilot_token_dependency),
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.API_EXECUTE)
):
    try:
        integration_manager = await get_integration_manager()
        optimization_result = await integration_manager.optimize_query(copilot_req_body.query, copilot_req_body.context)
        optimized_query = optimization_result["optimized_query"]
        current_context = copilot_req_body.context
        if not current_context:
            rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
            if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized for Copilot context")
            search_results_rag = await rag_pipeline.search(query=optimized_query, limit=5)
            current_context = [{"text": sr.get("text"), "metadata": sr.get("metadata"), "score": sr.get("score")} for sr in search_results_rag]
        agent_request = CopilotRequest(query=optimized_query, conversation_id=copilot_req_body.conversation_id, context=current_context, max_tokens=copilot_req_body.max_tokens)
        return await agent.get_completion(agent_request)
    except Exception as e: logger.error(f"Copilot chat error: {e}"); raise HTTPException(status_code=500, detail=f"Copilot chat failed: {str(e)}")

@app.post("/copilot/chat/stream", tags=["Copilot"])
async def consolidated_stream_chat_with_copilot(
    copilot_req_body: CopilotRequest,
    fastapi_req: Request, # Moved fastapi_req up
    agent: CopilotAgent = Depends(get_copilot_agent_dependency),
    copilot_token: Optional[str] = Depends(verify_copilot_token_dependency),
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.API_EXECUTE)
):
    try:
        integration_manager = await get_integration_manager()
        optimization_result = await integration_manager.optimize_query(copilot_req_body.query, copilot_req_body.context)
        optimized_query = optimization_result["optimized_query"]
        current_context = copilot_req_body.context
        if not current_context:
            rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
            if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized for Copilot stream context")
            search_results_rag = await rag_pipeline.search(query=optimized_query, limit=5)
            current_context = [{"text": sr.get("text"), "metadata": sr.get("metadata"),"score": sr.get("score")} for sr in search_results_rag]
        agent_request = CopilotRequest(query=optimized_query, conversation_id=copilot_req_body.conversation_id, context=current_context, max_tokens=copilot_req_body.max_tokens, stream=True)
        async def event_generator():
            try:
                async for token_chunk in agent.stream_completion(agent_request):
                    if token_chunk: yield f"data: {json.dumps({'content': token_chunk})}\n\n"
            except Exception as e: logger.error(f"Error during Copilot stream: {str(e)}"); yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally: yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e: logger.error(f"Error in Copilot stream chat endpoint: {e}"); raise HTTPException(status_code=500, detail=f"Copilot stream chat failed: {str(e)}")


# --- Token Endpoint ---
@app.post("/token", tags=["Authentication"])
async def login_for_access_token(
    fastapi_req: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    audit_logger: Optional[AuditLogger] = fastapi_req.app.state.audit_logger
    ip_address = fastapi_req.client.host if fastapi_req.client else "N/A"

    try:
        user = await auth_manager.authenticate_user(
            username=form_data.username,
            password=form_data.password,
            audit_logger=audit_logger, # Pass logger
            ip_address=ip_address      # Pass IP
        )
    except HTTPException as e: # Catch auth-specific HTTPExceptions to ensure they are re-raised
        # Audit log for failure is already handled inside authenticate_user/_record_failed_attempt
        raise e
    except Exception as e: # Catch other unexpected errors during auth
        if audit_logger:
            audit_logger.log_event(AuditEvent(
                event_type="user_login_error", user_id=form_data.username, action="User Login Error",
                resource_type="user_session", resource_id=form_data.username, status="failure",
                ip_address=ip_address, metadata={"error": str(e)}
            ))
        raise HTTPException(status_code=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during authentication.")


    if not user: # Should be handled by exceptions from authenticate_user, but as a safeguard
        # This path is less likely if authenticate_user raises HTTPException on failure
        raise HTTPException(
            status_code=fastapi_status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )

    access_token = auth_manager.create_access_token(user=user)
    return {"access_token": access_token, "token_type": "bearer"}


# --- Admin Config Management Endpoints ---
@app.get("/admin/config", response_model=AppSystemConfig, tags=["Admin"])
async def get_admin_config_endpoint( # Renamed
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.ADMIN_READ) # Protect with permission
):
    config_manager_instance: ConfigManager = fastapi_req.app.state.config_manager
    return config_manager_instance.config

@app.post("/admin/config/update", response_model=AppSystemConfig, tags=["Admin"])
async def update_admin_config_endpoint( # Renamed
    config_update: ConfigUpdateRequestModel,
    fastapi_req: Request,
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.ADMIN_WRITE) # Protect with permission
):
    config_manager_instance: ConfigManager = fastapi_req.app.state.config_manager
    audit_logger: Optional[AuditLogger] = fastapi_req.app.state.audit_logger

    try:
        update_dict = config_update.dict(exclude_unset=True)
        # For audit, log what is being attempted to change, avoid logging sensitive values if any in full config
        # A summary of changed keys might be better than logging the full update_dict if it can contain secrets.
        # For now, logging the keys that are being updated.
        changed_sections = list(update_dict.keys())

        current_config: AppSystemConfig = config_manager_instance.config
        for section_name, section_updates in update_dict.items():
            if hasattr(current_config, section_name) and isinstance(section_updates, dict):
                section_obj = getattr(current_config, section_name)
                for key, value in section_updates.items():
                    if hasattr(section_obj, key): setattr(section_obj, key, value)
            elif hasattr(current_config, section_name): setattr(current_config, section_name, section_updates)
        config_manager_instance.save_config()
        app.state.app_config = config_manager_instance.config
        logger.info(f"System configuration updated by user {current_user.username} with changes to sections: {changed_sections}")

        if audit_logger:
            audit_logger.log_event(AuditEvent(
                event_type="config_update",
                user_id=current_user.id,
                action="Admin Configuration Updated",
                resource_type="system_configuration",
                resource_id="app_config",
                status="success",
                ip_address=fastapi_req.client.host if fastapi_req.client else "N/A",
                metadata={"updated_sections": changed_sections}
            ))
        return app.state.app_config
    except Exception as e:
        logger.error(f"Failed to update configuration by user {current_user.username}: {e}", exc_info=True)
        if audit_logger:
            audit_logger.log_event(AuditEvent(
                event_type="config_update_failed",
                user_id=current_user.id,
                action="Admin Configuration Update Failed",
                resource_type="system_configuration",
                resource_id="app_config",
                status="failure",
                ip_address=fastapi_req.client.host if fastapi_req.client else "N/A",
                metadata={"error": str(e), "attempted_updates_to_sections": changed_sections}
            ))
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

# --- Adding Audit Log to /process_docs ---
@app.post("/process_docs", tags=["RAG Core"])
async def process_documents_endpoint(
    payload: ProcessRequestInput,
    background_tasks: BackgroundTasks,
    fastapi_req: Request, # fastapi_req must be before current_user if current_user depends on it implicitly via get_auth_manager
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.DOCUMENT_WRITE)
):
    rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
    audit_logger: Optional[AuditLogger] = fastapi_req.app.state.audit_logger

    if not rag_pipeline: raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")

    path = Path(payload.directory_path)
    if not path.exists() or not path.is_dir():
        if audit_logger:
            audit_logger.log_event(AuditEvent(
                event_type="document_processing_trigger_failed",
                user_id=current_user.id,
                action="Document Processing Triggered - Path Not Found",
                resource_type="directory",
                resource_id=str(path),
                status="failure",
                ip_address=fastapi_req.client.host if fastapi_req.client else "N/A",
                metadata={"directory_path": payload.directory_path, "error": "Path not found or not a directory"}
            ))
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    background_tasks.add_task(rag_pipeline.process_documents, path, payload.batch_size)

    if audit_logger:
        audit_logger.log_event(AuditEvent(
            event_type="document_processing_triggered",
            user_id=current_user.id,
            action="Document Processing Triggered",
            resource_type="directory",
            resource_id=str(path),
            status="success", # Indicates triggering was successful, not completion of processing
            ip_address=fastapi_req.client.host if fastapi_req.client else "N/A",
            metadata={"directory_path": payload.directory_path, "batch_size": payload.batch_size}
        ))

    return {"message": f"Started processing documents from {path}"}


# --- Original Endpoints from enhanced_api.py (Health, Status, Search_Direct_Qdrant, Admin) ---
@health_check("qdrant")
async def check_qdrant_health():
    try:
        rag_pipeline_instance = app.state.rag_pipeline if hasattr(app.state, 'rag_pipeline') else None
        client_to_use = rag_pipeline_instance.client if rag_pipeline_instance and hasattr(rag_pipeline_instance, 'client') else qdrant_client
        collections = client_to_use.get_collections()
        return {"status": "healthy", "collections_count": len(collections.collections)}
    except Exception as e: raise Exception(f"Qdrant health check failed: {e}")

@health_check("embedding_provider")
async def check_embedding_provider_health():
    try:
        provider = get_embedding_provider_dependency()
        test_embedding = await provider.generate_embeddings("health check test")
        return {"status": "healthy", "embedding_dimension": len(test_embedding[0])}
    except Exception as e: raise Exception(f"Embedding provider health check failed: {e}")

@app.get("/", response_model=Dict[str, str], include_in_schema=False)
async def root_redirect(): return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
@with_recovery(component_name="api_health", severity=ErrorSeverity.LOW)
async def health_check_endpoint_main():
    recovery_system = get_error_recovery_system()
    system_health_val = recovery_system.get_system_health()
    integration_stats = {}
    if hasattr(app.state, 'integration_manager_instance') and app.state.integration_manager_instance:
        integration_stats = app.state.integration_manager_instance.get_integration_stats()
    scaling_stats = {}
    if app_config.feature_flags.enable_auto_scaling: # Check flag before getting scaler
        auto_scaler_instance = await get_auto_scaler()
        if auto_scaler_instance: scaling_stats = auto_scaler_instance.get_scaling_stats()
    return {"status": "healthy" if system_health_val["overall_state"] == "healthy" else "degraded",
            "timestamp": datetime.now().isoformat(), "system_health": system_health_val,
            "integration_stats": integration_stats, "scaling_stats": scaling_stats,
            "components": {"qdrant": (await check_qdrant_health())["status"],
                           "embedding_provider": (await check_embedding_provider_health())["status"],
                           "error_recovery": "active" if app_config.feature_flags.enable_error_recovery else "disabled",
                           "auto_scaling": "active" if app_config.feature_flags.enable_auto_scaling else "disabled",
                           "monitoring": "active" if app_config.feature_flags.enable_monitoring else "disabled"}}

@app.get("/status", response_model=SystemStatusResponse, tags=["System"])
@with_recovery(component_name="api_status", severity=ErrorSeverity.LOW)
async def system_status_main():
    start_time_val = app.state.start_time if hasattr(app.state, 'start_time') else time.time()
    uptime = time.time() - start_time_val
    components_val = {}
    if hasattr(app.state, 'integration_manager_instance') and app.state.integration_manager_instance:
        components_val["integration_manager"] = app.state.integration_manager_instance.get_integration_stats()
    if app_config.feature_flags.enable_auto_scaling:
        auto_scaler_instance = await get_auto_scaler()
        if auto_scaler_instance: components_val["auto_scaler"] = auto_scaler_instance.get_scaling_stats()
    if app_config.feature_flags.enable_error_recovery:
        recovery_system_instance = get_error_recovery_system()
        if recovery_system_instance: components_val["error_recovery"] = recovery_system_instance.get_error_statistics()
    performance_val = {"rate_limiter": rate_limiter.get_stats(),
                       "settings": {"batch_size": app_config.model.batch_size,
                                    "cpu_workers": app_config.processing.max_workers,}}
    integrations_val = {"auto_scaling_enabled": app_config.feature_flags.enable_auto_scaling,
                        "error_recovery_enabled": app_config.feature_flags.enable_error_recovery,
                        "monitoring_enabled": app_config.feature_flags.enable_monitoring}
    return SystemStatusResponse(status="operational", version=app.version, uptime_seconds=uptime,
                                components=components_val, performance=performance_val, integrations=integrations_val)

@app.post("/search_direct_qdrant", response_model=EnhancedSearchResponse, tags=["Search"])
@with_recovery(component_name="api_search_direct", severity=ErrorSeverity.MEDIUM)
async def search_direct_qdrant_endpoint(
    query: EnhancedQuery,
    fastapi_req: Request, # Moved fastapi_req up
    provider=Depends(get_embedding_provider_dependency),
    current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.SEARCH_ADVANCED)
):
    start_time = time.time()
    try:
        integration_manager = await get_integration_manager() # Assumes this is fine to call multiple times or is singleton from app.state
        cache_hit = False
        if query.use_cache:
            cached_result = await integration_manager.get_cached_result(query.text)
            if cached_result:
                cache_hit = True; response_time = (time.time() - start_time) * 1000
                return EnhancedSearchResponse(matches=cached_result["matches"], query_vector=cached_result["query_vector"],
                                            optimization_metadata=cached_result.get("optimization_metadata", {}),
                                            cache_hit=True, response_time_ms=response_time)
        optimization_metadata = {}; optimized_query_text = query.text
        if query.enable_optimization:
            optimization_result = await integration_manager.optimize_query(query.text, query.context)
            optimized_query_text = optimization_result["optimized_query"]; optimization_metadata = optimization_result["metadata"]
        query_vector = (await provider.generate_embeddings(optimized_query_text))[0]
        search_result = qdrant_client.search(collection_name=query.collection_name, query_vector=query_vector, limit=query.limit)
        matches = [{"id": result.id, "score": result.score, "payload": result.payload} for result in search_result]
        result_to_cache = {"matches": matches, "query_vector": query_vector, "optimization_metadata": optimization_metadata}
        if query.use_cache: await integration_manager.cache_result(query.text, result_to_cache, ttl=app_config.cache_settings.ttl)
        response_time = (time.time() - start_time) * 1000; system_health = None
        try:
            recovery_system = get_error_recovery_system()
            health_data = recovery_system.get_system_health(); system_health = {"overall_state": health_data["overall_state"]}
        except Exception: pass
        return EnhancedSearchResponse(matches=matches, query_vector=query_vector, optimization_metadata=optimization_metadata,
                                    cache_hit=cache_hit, response_time_ms=response_time, system_health=system_health)
    except Exception as e: logger.error(f"Direct Qdrant Search error: {e}"); raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/admin/scaling/status", tags=["Admin"])
@with_recovery(component_name="admin_scaling_status", severity=ErrorSeverity.LOW)
async def get_admin_scaling_status(fastapi_req: Request, current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.ADMIN_READ)): # type: ignore
    auto_scaler = await get_auto_scaler()
    stats = auto_scaler.get_scaling_stats(); recent_decisions = auto_scaler.get_recent_decisions(10)
    return {"scaling_stats": stats, "recent_decisions": recent_decisions, "enabled": app_config.feature_flags.enable_auto_scaling}

@app.post("/admin/scaling/force", tags=["Admin"])
@with_recovery(component_name="admin_scaling_force", severity=ErrorSeverity.MEDIUM)
async def force_admin_scaling_action(resource_type: str, target_value: int, background_tasks: BackgroundTasks, fastapi_req: Request, current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.ADMIN_WRITE)): # type: ignore
    try: rt = ResourceType(resource_type)
    except ValueError: raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
    auto_scaler = await get_auto_scaler()
    async def perform_scaling(): success = await auto_scaler.force_scaling_action(rt, target_value); logger.info(f"Forced scaling action result: {success}")
    background_tasks.add_task(perform_scaling)
    return {"message": f"Scaling action initiated for {resource_type} to {target_value}"}

@app.get("/admin/errors/statistics", tags=["Admin"])
@with_recovery(component_name="admin_errors_stats", severity=ErrorSeverity.LOW)
async def get_admin_error_statistics(fastapi_req: Request, current_user: AuthUser = Depends(get_auth_manager).require_permission(Permission.ADMIN_READ)): # type: ignore
    recovery_system = get_error_recovery_system()
    stats = recovery_system.get_error_statistics(); recent_errors = recovery_system.get_recent_errors(20)
    return {"statistics": stats, "recent_errors": recent_errors}

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app",
        host=app_config.api.host,
        port=app_config.api.port,
        workers=app_config.api.workers,
        reload=False,
        log_level="info"
    )
