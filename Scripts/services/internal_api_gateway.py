import httpx
import logging
from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.responses import JSONResponse, StreamingResponse
import os
from typing import List, Optional

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
        logger.error(f"Critical: Failed to import get_config_value for Internal API Gateway. Error: {e}")
        def get_config_value(env_var_name, yaml_path=None, default=None):
            return os.getenv(env_var_name, default)

# --- Service URLs ---
RAG_QUERY_SERVICE_URL = get_config_value("RAG_QUERY_SERVICE_URL", default="http://localhost:8001")
DOC_PROCESSING_SERVICE_URL = get_config_value("DOC_PROCESSING_SERVICE_URL", default="http://localhost:8002")

# --- API Key Configuration ---
API_KEY_NAME = "X-API-Key" # Standard header name for API keys
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Load valid API keys from environment variable (comma-separated string)
# For production, these should ideally be managed via Kubernetes Secrets and injected securely.
VALID_API_KEYS_CSV = get_config_value("VALID_API_KEYS_CSV", default="") # e.g., "key1,key2,anothersecretkey"
VALID_API_KEYS: List[str] = [key.strip() for key in VALID_API_KEYS_CSV.split(',') if key.strip()]

if not VALID_API_KEYS:
    logger.warning("No VALID_API_KEYS_CSV configured for the API Gateway. All requests will be allowed (or fail if auth is strictly enforced by a dependency). This is insecure for protected environments.")
else:
    logger.info(f"API Gateway initialized with {len(VALID_API_KEYS)} valid API key(s).")

async def get_api_key_dependency(key: Optional[str] = Security(api_key_header)):
    """
    Dependency to validate the API key.
    """
    if not VALID_API_KEYS: # If no keys are configured, effectively bypass auth (not recommended for sensitive routes)
        logger.warning("API key validation bypassed as no valid keys are configured.")
        return None # Or raise an error if auth is mandatory

    if not key:
        logger.warning("API key missing from request.")
        raise HTTPException(
            status_code=403, # Forbidden
            detail="Not authenticated: API key required."
        )
    if key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key provided: '{key[:10]}...'") # Log only a prefix for security
        raise HTTPException(
            status_code=403,
            detail="Not authorized: Invalid API key."
        )
    logger.debug(f"Valid API key received: '{key[:10]}...'")
    return key


app = FastAPI(title="Internal API Gateway")
# To protect all routes by default, you can add dependencies=[Security(get_api_key_dependency)] to FastAPI instance:
# app = FastAPI(title="Internal API Gateway", dependencies=[Security(get_api_key_dependency)])
# However, for this iteration, we'll apply it per route group or individually if needed.
# For now, let's protect the main routing functions.

# HTTP client that will be used to make requests to other services
# It's good practice to reuse the client instance.
# For production, consider more advanced configurations (timeouts, connection pools).
client = httpx.AsyncClient()

@app.on_event("startup")
async def startup_event():
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(60.0)) # Default 60s timeout
    logger.info("Internal API Gateway started. HTTPX client initialized.")
    logger.info(f"RAG Query Service URL: {RAG_QUERY_SERVICE_URL}")
    logger.info(f"Document Processing Service URL: {DOC_PROCESSING_SERVICE_URL}")


@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    logger.info("Internal API Gateway shutting down. HTTPX client closed.")


async def _forward_request(service_url: str, request: Request):
    """
    Generic function to forward a request to a backend service.
    It streams the response back to the client.
    """
    url_path = request.url.path
    # If the gateway itself has path prefixes for routing, remove them before forwarding
    # For example, if /rag_query/* goes to RAG Query Service, url_path sent to service should be /*
    # Current setup assumes direct path forwarding or specific route handlers.

    # Construct the target URL
    target_url = f"{service_url.rstrip('/')}{url_path}"

    headers = dict(request.headers)
    # Host header should reflect the target service, not the gateway's host
    headers["host"] = httpx.URL(service_url).host
    # Remove problematic headers for forwarding if any (e.g. content-length for GETs with no body)
    headers.pop("content-length", None) if request.method in ["GET", "HEAD", "DELETE"] else None


    try:
        logger.debug(f"Forwarding {request.method} request to {target_url}")

        # Stream request body if present
        req_body_bytes = await request.body()

        rp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=request.query_params,
            content=req_body_bytes if req_body_bytes else None,
        )

        # Stream the response back
        return StreamingResponse(
            rp.aiter_raw(),
            status_code=rp.status_code,
            headers=dict(rp.headers), # Convert Headers object to dict
            media_type=rp.headers.get("content-type")
        )

    except httpx.RequestError as exc:
        logger.error(f"Error forwarding request to {target_url}: {exc}")
        # More specific error handling can be added here (e.g. ConnectError, Timeout)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}") # Service Unavailable
    except Exception as e:
        logger.error(f"Unexpected error during request forwarding to {target_url}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error in gateway.")


# --- RAG Query Service Routes ---
@app.api_route("/rag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def route_to_rag_query_service(request: Request, path: str, api_key: APIKey = Security(get_api_key_dependency)):
    logger.info(f"Routing to RAG Query Service: /rag/{path} (auth valid for key prefix: {str(api_key)[:10]}...)")
    return await _forward_request(RAG_QUERY_SERVICE_URL, request)

# --- Document Processing Service Routes ---
@app.api_route("/document/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def route_to_doc_processing_service(request: Request, path: str, api_key: APIKey = Security(get_api_key_dependency)):
    logger.info(f"Routing to Document Processing Service: /document/{path} (auth valid for key prefix: {str(api_key)[:10]}...)")
    return await _forward_request(DOC_PROCESSING_SERVICE_URL, request)


# --- Health Check for the Gateway itself (typically does not require API key) ---
@app.get("/gateway_health", tags=["Gateway Health"])
async def health_check():
    # Note: If VALID_API_KEYS is empty, get_api_key_dependency returns None and doesn't raise error.
    # If keys are configured, this health check would also need a key if the dependency was app-wide.
    # Since it's route-specific for now, this health check remains open.
    return JSONResponse({"status": "healthy", "service": "Internal API Gateway"})


if __name__ == "__main__":
    import uvicorn
    # This allows running the gateway directly for local testing.
    # For K8s, this will be run by the Docker CMD.

    # Example: Set environment variables for local testing if not already set
    # os.environ["RAG_QUERY_SERVICE_URL"] = "http://localhost:8001"
    # os.environ["DOC_PROCESSING_SERVICE_URL"] = "http://localhost:8002"
    # Re-initialize to pick up env vars if set here for __main__
    # RAG_QUERY_SERVICE_URL = get_config_value("RAG_QUERY_SERVICE_URL", default="http://localhost:8001")
    # DOC_PROCESSING_SERVICE_URL = get_config_value("DOC_PROCESSING_SERVICE_URL", default="http://localhost:8002")

    GATEWAY_PORT = int(get_config_value("API_GATEWAY_PORT", default=8000))
    GATEWAY_HOST = get_config_value("API_GATEWAY_HOST", default="0.0.0.0")

    logger.info(f"Starting Internal API Gateway on {GATEWAY_HOST}:{GATEWAY_PORT}")
    logger.info(f"Configured RAG Query Service URL: {RAG_QUERY_SERVICE_URL}")
    logger.info(f"Configured Document Processing Service URL: {DOC_PROCESSING_SERVICE_URL}")

    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)

# To run this:
# Ensure backend services (RAG Query, Doc Processing) are running on their respective ports.
# Set environment variables if defaults are not suitable:
# export RAG_QUERY_SERVICE_URL="http://actual-rag-host:port"
# export DOC_PROCESSING_SERVICE_URL="http://actual-docproc-host:port"
# export API_GATEWAY_PORT=8000
# python Scripts/services/internal_api_gateway.py

# Then you can send requests like:
# curl http://localhost:8000/rag/some/path
# curl -X POST http://localhost:8000/document/upload -d "{...}"
# These will be forwarded to the respective backend services.
