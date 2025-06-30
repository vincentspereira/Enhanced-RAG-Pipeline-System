"""
API router for external API integrations hub.

This module provides API endpoints for managing and interacting with 
external API services through a unified hub with OAuth2 authentication
and rate limiting.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, Request, Response, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from pydantic import BaseModel, Field, validator, AnyHttpUrl
from datetime import datetime, timedelta
import httpx
import asyncio
import time

from Scripts.api_hub.oauth2 import OAuth2Manager, OAuth2Provider, GoogleOAuth2Provider, GithubOAuth2Provider
from Scripts.api_hub.rate_limiter import RateLimiter, RateLimit
from Scripts.api_hub.analytics import ApiUsageAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API router
router = APIRouter(prefix="/api-hub", tags=["API Integration Hub"])

# Auth imports
from ...auth.dependencies import get_auth_manager_dependency
from ...auth.auth_manager import AuthManager, AuthUser, Permission


# Pydantic models for API
class APIServiceModel(BaseModel):
    """Model for an external API service."""
    id: str = Field(..., description="Unique service ID")
    name: str = Field(..., description="Service name")
    description: Optional[str] = Field(None, description="Service description")
    base_url: AnyHttpUrl = Field(..., description="Base URL for API calls")
    auth_type: str = Field(..., description="Authentication type (oauth2, api_key, none)")
    oauth2_provider: Optional[str] = Field(None, description="OAuth2 provider name if auth_type is oauth2")
    rate_limits: Optional[List[Dict[str, Any]]] = Field(None, description="Rate limit configurations")
    endpoints: List[Dict[str, Any]] = Field(..., description="Available endpoints")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "github-api",
                "name": "GitHub API",
                "description": "GitHub REST API for accessing repositories, issues, and more",
                "base_url": "https://api.github.com",
                "auth_type": "oauth2",
                "oauth2_provider": "github",
                "rate_limits": [
                    {
                        "name": "authenticated",
                        "limit": 5000,
                        "window": 3600,
                        "description": "5000 requests per hour for authenticated users"
                    },
                    {
                        "name": "unauthenticated",
                        "limit": 60,
                        "window": 3600,
                        "description": "60 requests per hour for unauthenticated users"
                    }
                ],
                "endpoints": [
                    {
                        "path": "/user",
                        "method": "GET",
                        "description": "Get authenticated user information",
                        "requires_auth": True
                    },
                    {
                        "path": "/repos/{owner}/{repo}",
                        "method": "GET",
                        "description": "Get repository information",
                        "requires_auth": False,
                        "parameters": [
                            {"name": "owner", "in": "path", "required": True},
                            {"name": "repo", "in": "path", "required": True}
                        ]
                    }
                ]
            }
        }


class APIRequestModel(BaseModel):
    """Model for an API request through the hub."""
    service_id: str = Field(..., description="API service ID")
    endpoint: str = Field(..., description="Endpoint path")
    method: str = Field("GET", description="HTTP method")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    body: Optional[Dict[str, Any]] = Field(None, description="Request body for POST/PUT/PATCH requests")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    
    class Config:
        schema_extra = {
            "example": {
                "service_id": "github-api",
                "endpoint": "/repos/{owner}/{repo}",
                "method": "GET",
                "params": {
                    "owner": "microsoft",
                    "repo": "vscode"
                }
            }
        }


class OAuth2AuthorizeModel(BaseModel):
    """Model for OAuth2 authorization request."""
    provider: str = Field(..., description="OAuth2 provider name")
    redirect_uri: AnyHttpUrl = Field(..., description="Redirect URI for authorization flow")
    scope: Optional[str] = Field(None, description="OAuth2 scopes (space-separated)")
    state: Optional[str] = Field(None, description="OAuth2 state parameter for security")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "github",
                "redirect_uri": "https://example.com/oauth2/callback",
                "scope": "user repo",
                "state": "random-state-string"
            }
        }


class OAuth2TokenExchangeModel(BaseModel):
    """Model for OAuth2 token exchange request."""
    provider: str = Field(..., description="OAuth2 provider name")
    code: str = Field(..., description="Authorization code from OAuth2 provider")
    redirect_uri: AnyHttpUrl = Field(..., description="Redirect URI used for authorization")
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "github",
                "code": "authorization-code-from-provider",
                "redirect_uri": "https://example.com/oauth2/callback"
            }
        }


class ServiceMetricsModel(BaseModel):
    """Model for API service performance metrics."""
    service_id: str = Field(..., description="API service ID")
    uptime_percentage: float = Field(..., description="Service uptime percentage")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")
    success_rate: float = Field(..., description="Success rate (0-1)")
    request_count: int = Field(..., description="Total request count")
    error_count: int = Field(..., description="Total error count")
    last_checked: datetime = Field(..., description="Last time metrics were updated")
    
    class Config:
        schema_extra = {
            "example": {
                "service_id": "github-api",
                "uptime_percentage": 99.9,
                "avg_response_time_ms": 245.6,
                "success_rate": 0.987,
                "request_count": 12450,
                "error_count": 163,
                "last_checked": "2025-05-24T12:34:56"
            }
        }


class ServiceHealthCheckModel(BaseModel):
    """Model for API service health check."""
    service_id: str = Field(..., description="API service ID")
    endpoint: Optional[str] = Field(None, description="Specific endpoint to check")
    
    class Config:
        schema_extra = {
            "example": {
                "service_id": "github-api",
                "endpoint": "/user"
            }
        }


class BatchRequestModel(BaseModel):
    """Model for a batch of API requests."""
    requests: List[APIRequestModel] = Field(..., description="List of API requests to execute")
    parallel: bool = Field(False, description="Whether to execute requests in parallel")
    
    class Config:
        schema_extra = {
            "example": {
                "requests": [
                    {
                        "service_id": "github-api",
                        "endpoint": "/repos/{owner}/{repo}",
                        "method": "GET",
                        "params": {
                            "owner": "microsoft",
                            "repo": "vscode"
                        }
                    },
                    {
                        "service_id": "github-api",
                        "endpoint": "/repos/{owner}/{repo}/issues",
                        "method": "GET",
                        "params": {
                            "owner": "microsoft",
                            "repo": "vscode",
                            "state": "open"
                        }
                    }
                ],
                "parallel": True
            }
        }


class WebhookModel(BaseModel):
    """Model for a webhook configuration."""
    id: Optional[str] = Field(None, description="Webhook ID (generated if not provided)")
    url: AnyHttpUrl = Field(..., description="URL to send webhook events to")
    secret: Optional[str] = Field(None, description="Secret for webhook signature")
    events: List[str] = Field(..., description="Events to subscribe to")
    description: Optional[str] = Field(None, description="Webhook description")
    active: bool = Field(True, description="Whether the webhook is active")
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/webhook",
                "secret": "your-webhook-secret",
                "events": ["service.created", "service.updated", "request.error"],
                "description": "Notify my service about API hub events",
                "active": True
            }
        }


# Initialize managers and providers
oauth2_manager = OAuth2Manager()

# Load configuration from environment or config file
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "your-github-client-id")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "your-github-client-secret")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "your-google-client-id")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "your-google-client-secret")

# Register OAuth2 providers
github_provider = GithubOAuth2Provider(
    client_id=GITHUB_CLIENT_ID,
    client_secret=GITHUB_CLIENT_SECRET
)
google_provider = GoogleOAuth2Provider(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET
)

oauth2_manager.register_provider(github_provider)
oauth2_manager.register_provider(google_provider)

# Initialize rate limiter
rate_limiter = RateLimiter()

# Initialize API usage analytics
analytics = ApiUsageAnalytics()

# Default rate limits
DEFAULT_RATE_LIMITS = [
    RateLimit("default", 100, 60, "Default limit of 100 requests per minute"),
    RateLimit("burst", 10, 1, "Burst limit of 10 requests per second")
]

for limit in DEFAULT_RATE_LIMITS:
    rate_limiter.add_limit(limit)

# API services registry
API_SERVICES_FILE = "data/api_hub/services.json"
WEBHOOKS_FILE = "data/api_hub/webhooks.json"
os.makedirs(os.path.dirname(API_SERVICES_FILE), exist_ok=True)

def load_api_services():
    """Load API services from JSON file."""
    if os.path.exists(API_SERVICES_FILE):
        try:
            with open(API_SERVICES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading API services: {e}")
    
    # Return empty list if file doesn't exist or error
    return []

def save_api_services(services):
    """Save API services to JSON file."""
    try:
        with open(API_SERVICES_FILE, 'w') as f:
            json.dump(services, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving API services: {e}")
        return False

def load_webhooks():
    """Load webhooks from JSON file."""
    if os.path.exists(WEBHOOKS_FILE):
        try:
            with open(WEBHOOKS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading webhooks: {e}")
    
    # Return empty list if file doesn't exist or error
    return []

def save_webhooks(webhooks):
    """Save webhooks to JSON file."""
    try:
        with open(WEBHOOKS_FILE, 'w') as f:
            json.dump(webhooks, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving webhooks: {e}")
        return False

# Load services
api_services = load_api_services()

# Load webhooks
webhooks = load_webhooks()

# Helper dependencies
def get_oauth2_manager():
    return oauth2_manager

def get_rate_limiter():
    return rate_limiter

def get_analytics():
    return analytics

def get_token_from_request(request: Request):
    """Extract token from Authorization header."""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None
    
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    return parts[1]

# API Endpoints
@router.get("/services", summary="List available API services")
async def list_services():
    """List all available external API services."""
    return {
        "services": api_services,
        "count": len(api_services)
    }


@router.get("/services/{service_id}", summary="Get API service details")
async def get_service(service_id: str = Path(..., description="API service ID")):
    """Get details of a specific API service."""
    # Find service
    service = next((s for s in api_services if s["id"] == service_id), None)
    if not service:
        raise HTTPException(status_code=404, detail=f"API service with ID '{service_id}' not found")
    
    return service


@router.post("/services", summary="Register a new API service")
async def register_service(service: APIServiceModel):
    """Register a new external API service."""
    # Check if service already exists
    if any(s["id"] == service.id for s in api_services):
        raise HTTPException(status_code=400, detail=f"API service with ID '{service.id}' already exists")
    
    # Add service
    service_dict = service.dict()
    api_services.append(service_dict)
    
    # Save services
    if not save_api_services(api_services):
        raise HTTPException(status_code=500, detail="Failed to save API service")
    
    # Notify webhooks
    asyncio.create_task(
        notify_webhooks(
            "service.created", 
            {"service": service_dict}
        )
    )
    
    return {
        "message": f"API service '{service.name}' registered successfully",
        "service": service
    }


@router.put("/services/{service_id}", summary="Update API service")
async def update_service(
    service: APIServiceModel,
    service_id: str = Path(..., description="API service ID")
):
    """Update an existing API service."""
    # Check if IDs match
    if service.id != service_id:
        raise HTTPException(status_code=400, detail="Service ID in path and request body must match")
    
    # Find service index
    service_index = next((i for i, s in enumerate(api_services) if s["id"] == service_id), None)
    if service_index is None:
        raise HTTPException(status_code=404, detail=f"API service with ID '{service_id}' not found")
    
    # Store old service for webhook
    old_service = api_services[service_index]
    
    # Update service
    service_dict = service.dict()
    api_services[service_index] = service_dict
    
    # Save services
    if not save_api_services(api_services):
        raise HTTPException(status_code=500, detail="Failed to save API service")
    
    # Notify webhooks
    asyncio.create_task(
        notify_webhooks(
            "service.updated", 
            {
                "service": service_dict,
                "old_service": old_service
            }
        )
    )
    
    return {
        "message": f"API service '{service.name}' updated successfully",
        "service": service
    }


@router.delete("/services/{service_id}", summary="Delete API service")
async def delete_service(service_id: str = Path(..., description="API service ID")):
    """Delete an API service."""
    # Find service index
    service_index = next((i for i, s in enumerate(api_services) if s["id"] == service_id), None)
    if service_index is None:
        raise HTTPException(status_code=404, detail=f"API service with ID '{service_id}' not found")
    
    # Remove service
    service = api_services.pop(service_index)
    
    # Save services
    if not save_api_services(api_services):
        raise HTTPException(status_code=500, detail="Failed to save API service")
    
    # Notify webhooks
    asyncio.create_task(
        notify_webhooks(
            "service.deleted", 
            {"service": service}
        )
    )
    
    return {
        "message": f"API service '{service['name']}' deleted successfully"
    }


@router.post("/oauth2/authorize", summary="Get OAuth2 authorization URL")
async def get_oauth2_authorization_url(
    request: OAuth2AuthorizeModel,
    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    analytics: ApiUsageAnalytics = Depends(get_analytics),
    client_request: Request = None
):
    """Get the OAuth2 authorization URL for a provider."""
    client_id = client_request.client.host if client_request else "unknown"
    success = False
    error_msg = None
    
    try:
        # Get provider
        provider = oauth2_manager.get_provider(request.provider)
        if not provider:
            error_msg = f"OAuth2 provider '{request.provider}' not found"
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Set redirect URI
        provider.redirect_uri = str(request.redirect_uri)
        
        # Set scopes if provided
        if request.scope:
            provider.scopes = request.scope.split()        # Get authorization URL
        auth_url = provider.get_authorization_url(state=request.state)
        
        success = True
        
        # Track OAuth event
        analytics.track_oauth_event(
            event_type="authorize",
            provider=request.provider,
            client_id=client_id,
            success=True,
            error=None
        )
        
        return {
            "authorization_url": auth_url,
            "provider": request.provider,
            "state": request.state
        }
    except Exception as e:
        logger.error(f"Error getting OAuth2 authorization URL: {e}")
        error_msg = str(e)
        
        # Track failed OAuth event
        analytics.track_oauth_event(
            event_type="authorize",
            provider=request.provider if hasattr(request, 'provider') else "unknown",
            client_id=client_id,
            success=False,
            error=error_msg
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/oauth2/token", summary="Exchange OAuth2 code for token")
async def exchange_oauth2_code(
    request: OAuth2TokenExchangeModel,    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    analytics: ApiUsageAnalytics = Depends(get_analytics),
    client_request: Request = None
):
    """Exchange an OAuth2 authorization code for an access token."""
    client_id = client_request.client.host if client_request else "unknown"
    error_msg = None
    
    try:
        # Get provider
        provider = oauth2_manager.get_provider(request.provider)
        if not provider:
            error_msg = f"OAuth2 provider '{request.provider}' not found"
            raise HTTPException(status_code=404, detail=error_msg)
        
        # Set redirect URI
        provider.redirect_uri = str(request.redirect_uri)
        
        # Exchange code for token
        token_data = await provider.exchange_code_for_token(request.code)
        
        # Create JWT
        jwt_token = oauth2_manager.create_jwt_token(
            provider=request.provider,
            token_data=token_data
        )
        
        # Track successful OAuth token exchange
        analytics.track_oauth_event(
            event_type="token",
            provider=request.provider,            client_id=client_id,
            success=True,
            error=None
        )
        
        return {
            "access_token": jwt_token,
            "token_type": "bearer",
            "provider": request.provider,
            "expires_in": 3600  # 1 hour
        }
    except Exception as e:
        logger.error(f"Error exchanging OAuth2 code: {e}")
        error_msg = str(e)
        
        # Track failed OAuth token exchange
        analytics.track_oauth_event(
            event_type="token",
            provider=request.provider if hasattr(request, 'provider') else "unknown",
            client_id=client_id,
            success=False,
            error=error_msg
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/request", summary="Make an API request")
async def make_api_request(
    request_model: APIRequestModel,
    request: Request,
    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Make a request to an external API through the hub."""
    # Get token from Authorization header
    token = get_token_from_request(request)
    
    # Check rate limit
    client_id = token or request.client.host
    if not rate_limiter.check_rate_limit(client_id, "default"):
        # Track rate limit event
        rate_limiter_info = rate_limiter.get_limit_info("default")
        analytics.track_rate_limit_event(
            client_id=client_id,
            limit_name="default",
            limit_value=rate_limiter_info["limit"],
            window_seconds=rate_limiter_info["window"],
            current_count=rate_limiter_info["current"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    start_time = time.time()
    token_type = None
    error_msg = None
    status_code = 500  # Default to error
    
    try:
        # Find service
        service = next((s for s in api_services if s["id"] == request_model.service_id), None)
        if not service:
            raise HTTPException(status_code=404, detail=f"API service with ID '{request_model.service_id}' not found")
        
        # Verify endpoint exists
        endpoint = next((e for e in service["endpoints"] if e["path"] == request_model.endpoint), None)
        if not endpoint:
            raise HTTPException(status_code=404, detail=f"Endpoint '{request_model.endpoint}' not found for service '{service['name']}'")
        
        # Check if authentication is required
        if endpoint.get("requires_auth", False) and service["auth_type"] == "oauth2":
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required for this endpoint"
                )
            
            # Validate token
            token_data = oauth2_manager.validate_jwt_token(token)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Check if token is for the right provider
            if token_data["provider"] != service["oauth2_provider"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Token is for provider '{token_data['provider']}', but '{service['oauth2_provider']}' is required"
                )
            
            token_type = service["oauth2_provider"]
            
            # Get OAuth token for the external service
            oauth_token = token_data.get("external_token", {}).get("access_token")
            if not oauth_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="No OAuth token found for the external service"
                )
        
        # Prepare request URL
        base_url = service["base_url"]
        endpoint_path = request_model.endpoint
        
        # Replace path parameters
        if request_model.params:
            for param_name, param_value in request_model.params.items():
                if "{" + param_name + "}" in endpoint_path:
                    endpoint_path = endpoint_path.replace("{" + param_name + "}", str(param_value))
        
        url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        
        # Prepare headers
        headers = request_model.headers or {}
        
        # Add authentication header if needed
        if endpoint.get("requires_auth", False):
            if service["auth_type"] == "oauth2" and token_type:
                headers["Authorization"] = f"Bearer {oauth_token}"
            elif service["auth_type"] == "api_key" and service.get("api_key_header") and service.get("api_key"):
                headers[service["api_key_header"]] = service["api_key"]
        
        # Prepare query parameters (for those not used in path)
        query_params = {}
        if request_model.params:
            for param_name, param_value in request_model.params.items():
                if "{" + param_name + "}" not in endpoint_path:
                    query_params[param_name] = param_value
        
        # Make the actual API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            if request_model.method.upper() == "GET":
                response = await client.get(url, params=query_params, headers=headers)
            elif request_model.method.upper() == "POST":
                response = await client.post(url, params=query_params, headers=headers, json=request_model.body)
            elif request_model.method.upper() == "PUT":
                response = await client.put(url, params=query_params, headers=headers, json=request_model.body)
            elif request_model.method.upper() == "PATCH":
                response = await client.patch(url, params=query_params, headers=headers, json=request_model.body)
            elif request_model.method.upper() == "DELETE":
                response = await client.delete(url, params=query_params, headers=headers, json=request_model.body)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported HTTP method: {request_model.method}"
                )
            
            status_code = response.status_code
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Try to parse the response as JSON, fall back to text if not JSON
            try:
                result = response.json()
            except:
                result = response.text
            
            # Increment rate limit counter
            rate_limiter.increment_rate_limit(client_id, "default")
            
            response_time = (time.time() - start_time) * 1000  # convert to ms
            
            # Track the request
            analytics.track_request(
                service_id=service["id"],
                endpoint=request_model.endpoint,
                method=request_model.method,
                client_id=client_id,
                status_code=status_code,
                response_time_ms=response_time,
                token_type=token_type,
                error=None
            )
            
            return {
                "status_code": status_code,
                "service": service["name"],
                "endpoint": request_model.endpoint,
                "method": request_model.method,
                "result": result,
                "response_time_ms": response_time
            }
            
    except httpx.HTTPStatusError as e:
        status_code = e.response.status_code
        error_msg = f"External API error: {e.response.text}"
        
        # Notify webhooks about the error
        asyncio.create_task(
            notify_webhooks(
                "request.error",
                {
                    "service_id": request_model.service_id,
                    "endpoint": request_model.endpoint,
                    "method": request_model.method,
                    "status_code": status_code,
                    "error": error_msg
                }
            )
        )
        
        raise HTTPException(status_code=status_code, detail=error_msg)
    except HTTPException as e:
        status_code = e.status_code
        error_msg = str(e.detail)
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error making API request: {e}")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Track failed requests too
        if error_msg:
            response_time = (time.time() - start_time) * 1000  # convert to ms
            analytics.track_request(
                service_id=request_model.service_id,
                endpoint=request_model.endpoint,
                method=request_model.method,
                client_id=client_id,
                status_code=status_code,
                response_time_ms=response_time,
                token_type=token_type,
                error=error_msg
            )


@router.post("/request/batch", summary="Make a batch of API requests")
async def make_batch_api_request(
    batch_request: BatchRequestModel,
    request: Request,
    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Make a batch request to external APIs through the hub."""
    # Get token from Authorization header
    token = get_token_from_request(request)
    
    # Prepare results
    results = []
    errors = []
    start_time = time.time()
    
    # Execute requests
    async def execute_request(request_model: APIRequestModel):
        """Execute a single API request."""
        nonlocal token
        token_type = None
        error_msg = None
        status_code = 500  # Default to error
        
        try:
            # Find service
            service = next((s for s in api_services if s["id"] == request_model.service_id), None)
            if not service:
                raise HTTPException(status_code=404, detail=f"API service with ID '{request_model.service_id}' not found")
            
            # Verify endpoint exists
            endpoint = next((e for e in service["endpoints"] if e["path"] == request_model.endpoint), None)
            if not endpoint:
                raise HTTPException(status_code=404, detail=f"Endpoint '{request_model.endpoint}' not found for service '{service['name']}'")
            
            # Check if authentication is required
            if endpoint.get("requires_auth", False) and service["auth_type"] == "oauth2":
                if not token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required for this endpoint"
                    )
                
                # Validate token
                token_data = oauth2_manager.validate_jwt_token(token)
                if not token_data:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token"
                    )
                
                # Check if token is for the right provider
                if token_data["provider"] != service["oauth2_provider"]:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Token is for provider '{token_data['provider']}', but '{service['oauth2_provider']}' is required"
                    )
                
                token_type = service["oauth2_provider"]
                
                # Get OAuth token for the external service
                oauth_token = token_data.get("external_token", {}).get("access_token")
                if not oauth_token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="No OAuth token found for the external service"
                    )
            
            # Prepare request URL
            base_url = service["base_url"]
            endpoint_path = request_model.endpoint
            
            # Replace path parameters
            if request_model.params:
                for param_name, param_value in request_model.params.items():
                    if "{" + param_name + "}" in endpoint_path:
                        endpoint_path = endpoint_path.replace("{" + param_name + "}", str(param_value))
            
            url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
            
            # Prepare headers
            headers = request_model.headers or {}
            
            # Add authentication header if needed
            if endpoint.get("requires_auth", False):
                if service["auth_type"] == "oauth2" and token_type:
                    headers["Authorization"] = f"Bearer {oauth_token}"
                elif service["auth_type"] == "api_key" and service.get("api_key_header") and service.get("api_key"):
                    headers[service["api_key_header"]] = service["api_key"]
            
            # Prepare query parameters (for those not used in path)
            query_params = {}
            if request_model.params:
                for param_name, param_value in request_model.params.items():
                    if "{" + param_name + "}" not in endpoint_path:
                        query_params[param_name] = param_value
            
            # Make the actual API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                if request_model.method.upper() == "GET":
                    response = await client.get(url, params=query_params, headers=headers)
                elif request_model.method.upper() == "POST":
                    response = await client.post(url, params=query_params, headers=headers, json=request_model.body)
                elif request_model.method.upper() == "PUT":
                    response = await client.put(url, params=query_params, headers=headers, json=request_model.body)
                elif request_model.method.upper() == "PATCH":
                    response = await client.patch(url, params=query_params, headers=headers, json=request_model.body)
                elif request_model.method.upper() == "DELETE":
                    response = await client.delete(url, params=query_params, headers=headers, json=request_model.body)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported HTTP method: {request_model.method}"
                    )
                
                status_code = response.status_code
                
                # Check if the request was successful
                response.raise_for_status()
                
                # Try to parse the response as JSON, fall back to text if not JSON
                try:
                    result = response.json()
                except:
                    result = response.text
                
                return {
                    "status_code": status_code,
                    "service": service["name"],
                    "endpoint": request_model.endpoint,
                    "method": request_model.method,
                    "result": result
                }
        
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_msg = f"External API error: {e.response.text}"
            
            # Notify webhooks about the error
            asyncio.create_task(
                notify_webhooks(
                    "request.error",
                    {
                        "service_id": request_model.service_id,
                        "endpoint": request_model.endpoint,
                        "method": request_model.method,
                        "status_code": status_code,
                        "error": error_msg
                    }
                )
            )
            
            raise HTTPException(status_code=status_code, detail=error_msg)
        except HTTPException as e:
            status_code = e.status_code
            error_msg = str(e.detail)
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error making API request: {e}")
            raise HTTPException(status_code=500, detail=error_msg)
    
    # Execute in parallel or sequentially based on request
    if batch_request.parallel:
        # Execute all requests in parallel
        results = await asyncio.gather(
            *[execute_request(req) for req in batch_request.requests],
            return_exceptions=True
        )
    else:
        # Execute requests sequentially
        for req in batch_request.requests:
            result = await execute_request(req)
            results.append(result)
    
    # Prepare response
    response_data = {
        "status_code": 207,  # Multi-Status
        "results": results
    }
    
    return response_data


@router.post("/batch", summary="Make multiple API requests in a batch")
async def make_batch_requests(
    batch_request: BatchRequestModel,
    request: Request,
    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Make multiple API requests in a single batch."""
    # Get token from Authorization header
    token = get_token_from_request(request)
    client_id = token or request.client.host
    
    # Check if batch is too large
    if len(batch_request.requests) > 20:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 20 requests per batch."
        )
    
    # Check rate limit (count as multiple requests)
    if not rate_limiter.check_rate_limit(client_id, "default", count=len(batch_request.requests)):
        # Track rate limit event
        rate_limiter_info = rate_limiter.get_limit_info("default")
        analytics.track_rate_limit_event(
            client_id=client_id,
            limit_name="default",
            limit_value=rate_limiter_info["limit"],
            window_seconds=rate_limiter_info["window"],
            current_count=rate_limiter_info["current"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )
    
    # Execute requests
    if batch_request.parallel:
        # Execute in parallel
        tasks = []
        
        for req in batch_request.requests:
            # Create a mock request object for each API request
            mock_request = Request(scope={"type": "http", "client": request.client})
            tasks.append(
                make_api_request(
                    request_model=req,
                    request=mock_request,
                    oauth2_manager=oauth2_manager,
                    rate_limiter=rate_limiter,
                    analytics=analytics
                )
            )
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception
                if isinstance(result, HTTPException):
                    processed_results.append({
                        "success": False,
                        "status_code": result.status_code,
                        "error": result.detail,
                        "request_index": i
                    })
                else:
                    processed_results.append({
                        "success": False,
                        "status_code": 500,
                        "error": str(result),
                        "request_index": i
                    })
            else:
                # Success
                processed_results.append({
                    "success": True,
                    "result": result,
                    "request_index": i
                })
    else:
        # Execute sequentially
        processed_results = []
        
        for i, req in enumerate(batch_request.requests):
            try:
                # Create a mock request object for each API request
                mock_request = Request(scope={"type": "http", "client": request.client})
                result = await make_api_request(
                    request_model=req,
                    request=mock_request,
                    oauth2_manager=oauth2_manager,
                    rate_limiter=rate_limiter,
                    analytics=analytics
                )
                
                processed_results.append({
                    "success": True,
                    "result": result,
                    "request_index": i
                })
            except HTTPException as e:
                processed_results.append({
                    "success": False,
                    "status_code": e.status_code,
                    "error": e.detail,
                    "request_index": i
                })
            except Exception as e:
                processed_results.append({
                    "success": False,
                    "status_code": 500,
                    "error": str(e),
                    "request_index": i
                })
    
    # Increment rate limit counter
    rate_limiter.increment_rate_limit(client_id, "default", count=len(batch_request.requests))
    
    return {
        "batch_size": len(batch_request.requests),
        "parallel": batch_request.parallel,
        "results": processed_results,
        "successful_requests": sum(1 for r in processed_results if r.get("success", False)),
        "failed_requests": sum(1 for r in processed_results if not r.get("success", False))
    }


@router.get("/analytics/requests", summary="Get API request analytics")
async def get_request_analytics(
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    service_id: Optional[str] = Query(None, description="Filter by service ID"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Get analytics data for API requests."""
    # Parse date strings to datetime objects
    start_datetime = None
    end_datetime = None
    
    if start_time:
        try:
            start_datetime = datetime.fromisoformat(start_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    if end_time:
        try:
            end_datetime = datetime.fromisoformat(end_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    # Get analytics data
    stats = analytics.get_request_stats(
        start_time=start_datetime,
        end_time=end_datetime,
        service_id=service_id,
        client_id=client_id
    )
    
    return stats


@router.get("/analytics/rate-limits", summary="Get rate limit analytics")
async def get_rate_limit_analytics(
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    client_id: Optional[str] = Query(None, description="Filter by client ID"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Get analytics data for rate limit events."""
    # Parse date strings to datetime objects
    start_datetime = None
    end_datetime = None
    
    if start_time:
        try:
            start_datetime = datetime.fromisoformat(start_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    if end_time:
        try:
            end_datetime = datetime.fromisoformat(end_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    # Get analytics data
    stats = analytics.get_rate_limit_stats(
        start_time=start_datetime,
        end_time=end_datetime,
        client_id=client_id
    )
    
    return stats


@router.get("/analytics/oauth", summary="Get OAuth analytics")
async def get_oauth_analytics(
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    provider: Optional[str] = Query(None, description="Filter by OAuth provider"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Get analytics data for OAuth events."""
    # Parse date strings to datetime objects
    start_datetime = None
    end_datetime = None
    
    if start_time:
        try:
            start_datetime = datetime.fromisoformat(start_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    if end_time:
        try:
            end_datetime = datetime.fromisoformat(end_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS).")
    
    # Get analytics data
    stats = analytics.get_oauth_stats(
        start_time=start_datetime,
        end_time=end_datetime,
        provider=provider,
        event_type=event_type
    )
    
    return stats


@router.get("/analytics/dashboard", summary="Get analytics dashboard data")
async def get_analytics_dashboard(
    days: int = Query(7, description="Number of days of data to include"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Get consolidated analytics dashboard data."""
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Get all analytics data
    request_stats = analytics.get_request_stats(start_time=start_time, end_time=end_time)
    rate_limit_stats = analytics.get_rate_limit_stats(start_time=start_time, end_time=end_time)
    oauth_stats = analytics.get_oauth_stats(start_time=start_time, end_time=end_time)
    
    # Get daily request counts
    daily_stats = analytics.get_daily_request_stats(start_time=start_time, end_time=end_time)
    
    # Combine data
    return {
        "request_stats": request_stats,
        "rate_limit_stats": rate_limit_stats,
        "oauth_stats": oauth_stats,
        "daily_stats": daily_stats,
        "period": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "days": days
        }
    }


@router.get("/analytics/visualizations", summary="Generate and retrieve visualization charts")
async def get_analytics_visualizations(
    days: int = Query(7, description="Number of days of data to include"),
    output_format: str = Query("html", description="Output format (html, json)"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Generate and retrieve visualization charts for API analytics."""
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Create temporary directory for chart output
    import tempfile
    import shutil
    from pathlib import Path
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Generate report
        report_data = analytics.generate_report(
            start_time=start_time,
            end_time=end_time,
            output_dir=temp_dir
        )
        
        if output_format == "json":
            # Return only the data as JSON
            return report_data
        else:
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>API Analytics Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .chart-container {{ margin-bottom: 30px; }}
                    .chart {{ max-width: 100%; }}
                    .stats {{ display: flex; flex-wrap: wrap; }}
                    .stat-card {{ 
                        background-color: #f5f5f5; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 10px;
                        flex: 1;
                        min-width: 200px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .stat-value {{ 
                        font-size: 24px; 
                        font-weight: bold; 
                        margin: 10px 0; 
                        color: #0066cc;
                    }}
                </style>
            </head>
            <body>
                <h1>API Analytics Dashboard</h1>
                <p>Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}</p>
                
                <h2>Summary Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <h3>Total Requests</h3>
                        <div class="stat-value">{report_data.get('request_stats', {}).get('total_requests', 0)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Success Rate</h3>
                        <div class="stat-value">{report_data.get('request_stats', {}).get('success_rate', 0) * 100:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <h3>Avg Response Time</h3>
                        <div class="stat-value">{report_data.get('request_stats', {}).get('avg_response_time_ms', 0):.1f} ms</div>
                    </div>
                    <div class="stat-card">
                        <h3>Rate Limit Events</h3>
                        <div class="stat-value">{report_data.get('rate_limit_stats', {}).get('total_events', 0)}</div>
                    </div>
                </div>
                
                <h2>Request Trends</h2>
                <div class="chart-container">
                    <img class="chart" src="data:image/png;base64,{report_data.get('charts', {}).get('requests_by_day', '')}" alt="Requests by Day">
                </div>
                
                <h2>Response Time Trends</h2>
                <div class="chart-container">
                    <img class="chart" src="data:image/png;base64,{report_data.get('charts', {}).get('response_times', '')}" alt="Response Times">
                </div>
                
                <h2>Service Usage</h2>
                <div class="chart-container">
                    <img class="chart" src="data:image/png;base64,{report_data.get('charts', {}).get('service_usage', '')}" alt="Service Usage">
                </div>
                
                <h2>Status Code Distribution</h2>
                <div class="chart-container">
                    <img class="chart" src="data:image/png;base64,{report_data.get('charts', {}).get('status_codes', '')}" alt="Status Code Distribution">
                </div>
            </body>
            </html>
            """
            
            return HTMLResponse(content=html_content)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/services/{service_id}/health-check", summary="Check API service health")
async def check_service_health(
    service_id: str = Path(..., description="API service ID"),
    check_data: Optional[ServiceHealthCheckModel] = Body(None),
    oauth2_manager: OAuth2Manager = Depends(get_oauth2_manager),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Check if an API service is healthy and responding."""
    # Find service
    service = next((s for s in api_services if s["id"] == service_id), None)
    if not service:
        raise HTTPException(status_code=404, detail=f"API service with ID '{service_id}' not found")
    
    # Determine endpoint to check
    endpoint_to_check = None
    if check_data and check_data.endpoint:
        endpoint_to_check = check_data.endpoint
    else:
        # Find a simple endpoint that doesn't require auth
        for endpoint in service["endpoints"]:
            if not endpoint.get("requires_auth", False):
                endpoint_to_check = endpoint["path"]
                break
        
        # If no simple endpoint found, use the first one
        if not endpoint_to_check and service["endpoints"]:
            endpoint_to_check = service["endpoints"][0]["path"]
    
    if not endpoint_to_check:
        raise HTTPException(status_code=400, detail="No suitable endpoint found for health check")
      # Prepare request
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            base_url = service["base_url"]
            url = f"{base_url.rstrip('/')}/{endpoint_to_check.lstrip('/')}"
            
            # Make request
            response = await client.get(url)
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check response
            is_healthy = 200 <= response.status_code < 400
            
            result = {
                "service_id": service_id,
                "name": service["name"],
                "endpoint_checked": endpoint_to_check,
                "status_code": response.status_code,
                "is_healthy": is_healthy,
                "response_time_ms": response_time_ms,
                "checked_at": datetime.now().isoformat()
            }
            
            # Store metrics
            analytics.track_request(
                service_id=service_id,
                endpoint=endpoint_to_check,
                method="GET",
                client_id="health-check",
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                token_type=None,
                error=None if is_healthy else "Health check failed"
            )
            
            return result
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        error_message = str(e)
        
        # Store failed check
        analytics.track_request(
            service_id=service_id,
            endpoint=endpoint_to_check,
            method="GET",
            client_id="health-check",
            status_code=500,
            response_time_ms=response_time_ms,
            token_type=None,
            error=error_message
        )
        
        return {
            "service_id": service_id,
            "name": service["name"],
            "endpoint_checked": endpoint_to_check,
            "is_healthy": False,
            "error": error_message,
            "response_time_ms": response_time_ms,
            "checked_at": datetime.now().isoformat()
        }


@router.get("/services/{service_id}/metrics", summary="Get API service metrics")
async def get_service_metrics(
    service_id: str = Path(..., description="API service ID"),
    days: int = Query(7, description="Number of days of data to include"),
    analytics: ApiUsageAnalytics = Depends(get_analytics)
):
    """Get performance metrics for an API service."""
    # Find service
    service = next((s for s in api_services if s["id"] == service_id), None)
    if not service:
        raise HTTPException(status_code=404, detail=f"API service with ID '{service_id}' not found")
    
    # Calculate date range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Get service-specific stats
    request_stats = analytics.get_request_stats(
        start_time=start_time,
        end_time=end_time,
        service_id=service_id
    )
    
    # Get daily stats
    daily_stats = analytics.get_daily_request_stats(
        start_time=start_time,
        end_time=end_time,
        service_id=service_id
    )
    
    # Calculate metrics
    total_requests = request_stats.get("total_requests", 0)
    successful_requests = request_stats.get("successful_requests", 0)
    avg_response_time = request_stats.get("avg_response_time", 0)
    success_rate = (successful_requests / total_requests) if total_requests > 0 else 0
    
    # Calculate uptime (assuming the API was available if there was at least one successful request per day)
    days_with_data = len(daily_stats.get("daily_stats", []))
    days_with_success = sum(1 for day in daily_stats.get("daily_stats", []) if day.get("success_count", 0) > 0)
    uptime_percentage = (days_with_success / days_with_data * 100) if days_with_data > 0 else 0
    
    # Create metrics object
    metrics = ServiceMetricsModel(
        service_id=service_id,
        uptime_percentage=uptime_percentage,
        avg_response_time_ms=avg_response_time,
        success_rate=success_rate,
        request_count=total_requests,
        error_count=total_requests - successful_requests,
        last_checked=datetime.now()
    )
    
    return {
        "service": {
            "id": service_id,
            "name": service["name"],
            "description": service.get("description", "")
        },
        "metrics": metrics,
        "period": {
            "days": days,
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        "daily_stats": daily_stats.get("daily_stats", [])
    }


async def send_webhook(webhook, event_type, payload):
    """Send a webhook notification."""
    if not webhook.get("active", True):
        return
    
    # Check if webhook subscribes to this event
    if event_type not in webhook.get("events", []):
        return
    
    # Prepare webhook data
    webhook_data = {
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
        "webhook_id": webhook["id"],
        "payload": payload
    }
    
    # Add signature if webhook has a secret
    if webhook.get("secret"):
        import hmac
        import hashlib
        import json
        
        payload_str = json.dumps(webhook_data)
        signature = hmac.new(
            webhook["secret"].encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-Hub-Signature": f"sha256={signature}"
        }
    else:
        headers = {
            "Content-Type": "application/json"
        }
    
    # Send webhook
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook["url"],
                json=webhook_data,
                headers=headers
            )
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "webhook_id": webhook["id"],
                "event": event_type
            }
    except Exception as e:
        logger.error(f"Error sending webhook {webhook['id']}: {e}")
        return {
            "success": False,
            "error": str(e),
            "webhook_id": webhook["id"],
            "event": event_type
        }


async def notify_webhooks(event_type, payload):
    """Notify all webhooks about an event."""
    tasks = []
    
    for webhook in webhooks:
        tasks.append(send_webhook(webhook, event_type, payload))
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@router.get("/webhooks", summary="List webhooks")
async def list_webhooks():
    """List all webhooks."""
    return {
        "webhooks": webhooks,
        "count": len(webhooks)
    }


@router.post("/webhooks", summary="Register a webhook")
async def register_webhook(webhook: WebhookModel):
    """Register a new webhook."""
    # Generate ID if not provided
    if not webhook.id:
        webhook.id = str(uuid.uuid4())
    
    # Check if webhook with this ID already exists
    if any(w["id"] == webhook.id for w in webhooks):
        raise HTTPException(status_code=400, detail=f"Webhook with ID '{webhook.id}' already exists")
    
    # Add webhook
    webhook_dict = webhook.dict()
    webhooks.append(webhook_dict)
    
    # Save webhooks
    if not save_webhooks(webhooks):
        raise HTTPException(status_code=500, detail="Failed to save webhook")
    
    # Notify about new webhook
    asyncio.create_task(
        notify_webhooks(
            "webhook.created", 
            {"webhook": webhook_dict}
        )
    )
    
    return {
        "message": "Webhook registered successfully",
        "webhook": webhook_dict
    }


@router.get("/webhooks/{webhook_id}", summary="Get webhook details")
async def get_webhook(webhook_id: str = Path(..., description="Webhook ID")):
    """Get details of a specific webhook."""
    webhook = next((w for w in webhooks if w["id"] == webhook_id), None)
    if not webhook:
        raise HTTPException(status_code=404, detail=f"Webhook with ID '{webhook_id}' not found")
    
    return webhook


@router.put("/webhooks/{webhook_id}", summary="Update webhook")
async def update_webhook(
    webhook: WebhookModel,
    webhook_id: str = Path(..., description="Webhook ID")
):
    """Update an existing webhook."""
    # Check if IDs match
    if webhook.id and webhook.id != webhook_id:
        raise HTTPException(status_code=400, detail="Webhook ID in path and request body must match")
    
    # Set ID from path
    webhook.id = webhook_id
    
    # Find webhook index
    webhook_index = next((i for i, w in enumerate(webhooks) if w["id"] == webhook_id), None)
    if webhook_index is None:
        raise HTTPException(status_code=404, detail=f"Webhook with ID '{webhook_id}' not found")
    
    # Update webhook
    webhook_dict = webhook.dict()
    old_webhook = webhooks[webhook_index]
    webhooks[webhook_index] = webhook_dict
    
    # Save webhooks
    if not save_webhooks(webhooks):
        raise HTTPException(status_code=500, detail="Failed to save webhook")
    
    # Notify about updated webhook
    asyncio.create_task(
        notify_webhooks(
            "webhook.updated", 
            {
                "webhook": webhook_dict,
                "old_webhook": old_webhook
            }
        )
    )
    
    return {
        "message": "Webhook updated successfully",
        "webhook": webhook_dict
    }


@router.delete("/webhooks/{webhook_id}", summary="Delete webhook")
async def delete_webhook(webhook_id: str = Path(..., description="Webhook ID")):
    """Delete a webhook."""
    # Find webhook index
    webhook_index = next((i for i, w in enumerate(webhooks) if w["id"] == webhook_id), None)
    if webhook_index is None:
        raise HTTPException(status_code=404, detail=f"Webhook with ID '{webhook_id}' not found")
    
    # Remove webhook
    webhook = webhooks.pop(webhook_index)
    
    # Save webhooks
    if not save_webhooks(webhooks):
        raise HTTPException(status_code=500, detail="Failed to save webhooks")
    
    # Notify about deleted webhook
    asyncio.create_task(
        notify_webhooks(
            "webhook.deleted", 
            {"webhook": webhook}
        )
    )
    
    return {
        "message": "Webhook deleted successfully",
        "webhook": webhook
    }


@router.post("/webhooks/test", summary="Test a webhook")
async def test_webhook(
    url: str = Body(..., embed=True, description="Webhook URL to test"),
    secret: Optional[str] = Body(None, embed=True, description="Webhook secret for signature")
):
    """Send a test event to a webhook URL."""
    # Create test webhook
    test_webhook = {
        "id": "test-webhook",
        "url": url,
        "secret": secret,
        "events": ["test.event"],
        "active": True
    }
    
    # Send test event
    test_payload = {
        "message": "This is a test webhook event",
        "timestamp": datetime.now().isoformat()
    }
    
    result = await send_webhook(test_webhook, "test.event", test_payload)
    
    if result.get("success", False):
        return {
            "message": "Test webhook sent successfully",
            "status_code": result.get("status_code"),
            "webhook_url": url
        }
    else:        raise HTTPException(
            status_code=500,
            detail=f"Failed to send test webhook: {result.get('error', 'Unknown error')}"
        )
