"""
Advanced rate limiting for the API Hub.

This module provides rate limiting functionality for the API Hub,
with configurable limits, storage backends, and monitoring.
"""
import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Response, Depends, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
import redis
from redis.exceptions import RedisError
import lmdb
import pickle
import hashlib
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration"""
    name: str
    limit: int
    window: int  # in seconds
    description: str = None
    
    def __post_init__(self):
        """Validate rate limit configuration"""
        if self.limit <= 0:
            raise ValueError(f"Rate limit must be positive, got {self.limit}")
        if self.window <= 0:
            raise ValueError(f"Rate limit window must be positive, got {self.window}")


class RateLimitStorage:
    """Base class for rate limit storage backends"""
    
    async def get_counter(self, key: str) -> Tuple[int, float]:
        """Get counter value and expiration
        
        Args:
            key: Counter key
            
        Returns:
            Tuple of (value, expiration_timestamp)
        """
        raise NotImplementedError
    
    async def increment_counter(self, key: str, window: int) -> Tuple[int, float]:
        """Increment counter
        
        Args:
            key: Counter key
            window: Window size in seconds
            
        Returns:
            Tuple of (new_value, expiration_timestamp)
        """
        raise NotImplementedError
    
    async def reset_counter(self, key: str):
        """Reset counter
        
        Args:
            key: Counter key
        """
        raise NotImplementedError


class InMemoryRateLimitStorage(RateLimitStorage):
    """In-memory storage for rate limits"""
    
    def __init__(self):
        """Initialize in-memory storage"""
        self.counters: Dict[str, Tuple[int, float]] = {}
    
    async def get_counter(self, key: str) -> Tuple[int, float]:
        """Get counter value and expiration
        
        Args:
            key: Counter key
            
        Returns:
            Tuple of (value, expiration_timestamp)
        """
        if key not in self.counters:
            return 0, 0
            
        count, expiry = self.counters[key]
        
        # Check if expired
        if time.time() > expiry:
            return 0, 0
            
        return count, expiry
    
    async def increment_counter(self, key: str, window: int) -> Tuple[int, float]:
        """Increment counter
        
        Args:
            key: Counter key
            window: Window size in seconds
            
        Returns:
            Tuple of (new_value, expiration_timestamp)
        """
        count, expiry = await self.get_counter(key)
        
        # Check if expired
        if time.time() > expiry:
            count = 0
            expiry = time.time() + window
        
        # Increment counter
        count += 1
        
        # Store updated counter
        self.counters[key] = (count, expiry)
        
        return count, expiry
    
    async def reset_counter(self, key: str):
        """Reset counter
        
        Args:
            key: Counter key
        """
        if key in self.counters:
            del self.counters[key]


class RedisRateLimitStorage(RateLimitStorage):
    """Redis storage for rate limits"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str = None):
        """Initialize Redis storage
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
        """
        self.redis = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=False)
    
    async def get_counter(self, key: str) -> Tuple[int, float]:
        """Get counter value and expiration
        
        Args:
            key: Counter key
            
        Returns:
            Tuple of (value, expiration_timestamp)
        """
        try:
            # Get counter value
            value = self.redis.get(key)
            if value is None:
                return 0, 0
                
            # Get TTL in seconds
            ttl = self.redis.ttl(key)
            if ttl < 0:
                return 0, 0
                
            # Calculate expiration timestamp
            expiry = time.time() + ttl
            
            return int(value), expiry
        except RedisError as e:
            logger.error(f"Redis error: {e}")
            return 0, 0
    
    async def increment_counter(self, key: str, window: int) -> Tuple[int, float]:
        """Increment counter
        
        Args:
            key: Counter key
            window: Window size in seconds
            
        Returns:
            Tuple of (new_value, expiration_timestamp)
        """
        try:
            # Increment counter
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            result = pipe.execute()
            
            new_value = int(result[0])
            ttl = result[1]
            
            # Set expiration if not already set
            if ttl < 0:
                self.redis.expire(key, window)
                ttl = window
                
            # Calculate expiration timestamp
            expiry = time.time() + ttl
            
            return new_value, expiry
        except RedisError as e:
            logger.error(f"Redis error: {e}")
            return 1, time.time() + window
    
    async def reset_counter(self, key: str):
        """Reset counter
        
        Args:
            key: Counter key
        """
        try:
            self.redis.delete(key)
        except RedisError as e:
            logger.error(f"Redis error when resetting counter: {e}")


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, limit: RateLimit, key: str, reset_at: float):
        """Initialize rate limit exception
        
        Args:
            limit: Rate limit configuration
            key: Rate limit key
            reset_at: Reset timestamp
        """
        self.limit = limit
        self.key = key
        self.reset_at = reset_at
        self.reset_window = int(reset_at - time.time())
        message = f"Rate limit exceeded: {limit.name} ({limit.limit} requests per {limit.window}s). Try again in {self.reset_window}s."
        super().__init__(message)


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, storage: RateLimitStorage = None):
        """Initialize rate limiter
        
        Args:
            storage: Storage backend
        """
        self.storage = storage or InMemoryRateLimitStorage()
        self.limits: Dict[str, RateLimit] = {}
        self.key_funcs: Dict[str, Callable] = {}
        
    def add_limit(self, limit: RateLimit, key_func: Callable = None):
        """Add rate limit
        
        Args:
            limit: Rate limit configuration
            key_func: Function to generate rate limit key
        """
        self.limits[limit.name] = limit
        if key_func:
            self.key_funcs[limit.name] = key_func
            
        logger.info(f"Added rate limit: {limit.name} ({limit.limit} per {limit.window}s)")
    
    def get_limit(self, name: str) -> Optional[RateLimit]:
        """Get rate limit by name
        
        Args:
            name: Rate limit name
            
        Returns:
            Rate limit configuration or None
        """
        return self.limits.get(name)
    
    def list_limits(self) -> List[RateLimit]:
        """List all rate limits
        
        Returns:
            List of rate limits
        """
        return list(self.limits.values())
    
    def get_key_func(self, name: str) -> Callable:
        """Get key function for rate limit
        
        Args:
            name: Rate limit name
            
        Returns:
            Key function
        """
        return self.key_funcs.get(name, lambda req: f"{name}:default")
    
    async def is_rate_limited(self, request: Request, limit_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is rate limited
        
        Args:
            request: FastAPI request
            limit_name: Rate limit name
            
        Returns:
            Tuple of (is_limited, headers)
        """
        limit = self.get_limit(limit_name)
        if not limit:
            return False, {}
            
        # Generate key
        key_func = self.get_key_func(limit_name)
        key = key_func(request)
        
        # Check current count
        count, reset_at = await self.storage.get_counter(key)
        
        # Calculate remaining
        remaining = max(0, limit.limit - count)
        
        headers = {
            "X-RateLimit-Limit": str(limit.limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_at)),
        }
        
        if count >= limit.limit:
            headers["Retry-After"] = str(int(reset_at - time.time()))
            return True, headers
            
        # Increment counter
        new_count, reset_at = await self.storage.increment_counter(key, limit.window)
        
        # Update remaining
        headers["X-RateLimit-Remaining"] = str(max(0, limit.limit - new_count))
        
        return new_count > limit.limit, headers
        
    async def reset_limit(self, request: Request, limit_name: str):
        """Reset rate limit for request
        
        Args:
            request: FastAPI request
            limit_name: Rate limit name
        """
        limit = self.get_limit(limit_name)
        if not limit:
            return
            
        # Generate key
        key_func = self.get_key_func(limit_name)
        key = key_func(request)
        
        # Reset counter
        await self.storage.reset_counter(key)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(
        self,
        app: ASGIApp,
        limiter: RateLimiter,
        limit_name: str = "default",
        status_code: int = status.HTTP_429_TOO_MANY_REQUESTS,
        exclude_paths: List[str] = None
    ):
        """Initialize rate limit middleware
        
        Args:
            app: ASGI application
            limiter: Rate limiter
            limit_name: Rate limit name
            status_code: HTTP status code for rate limit errors
            exclude_paths: List of paths to exclude from rate limiting
        """
        super().__init__(app)
        self.limiter = limiter
        self.limit_name = limit_name
        self.status_code = status_code
        self.exclude_paths = exclude_paths or []
        
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request
        
        Args:
            request: FastAPI request
            call_next: Next middleware
        
        Returns:
            Response
        """
        # Check if path is excluded
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)
            
        # Check rate limit
        is_limited, headers = await self.limiter.is_rate_limited(request, self.limit_name)
        
        if is_limited:
            # Rate limit exceeded
            content = {
                "detail": "Rate limit exceeded",
                "limit": self.limiter.get_limit(self.limit_name).limit,
                "window": self.limiter.get_limit(self.limit_name).window,
                "reset": headers.get("X-RateLimit-Reset")
            }
            
            response = Response(
                content=json.dumps(content),
                status_code=self.status_code,
                media_type="application/json"
            )
            
            # Add rate limit headers
            for name, value in headers.items():
                response.headers[name] = value
                
            return response
            
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for name, value in headers.items():
            response.headers[name] = value
            
        return response


# Helper functions for key generation
def get_ip_key(request: Request) -> str:
    """Get rate limit key based on client IP
    
    Args:
        request: FastAPI request
        
    Returns:
        Rate limit key
    """
    ip = request.client.host
    return f"ip:{ip}"

def get_user_key(request: Request) -> str:
    """Get rate limit key based on user ID
    
    Args:
        request: FastAPI request
        
    Returns:
        Rate limit key
    """
    user_id = request.state.user.id if hasattr(request.state, "user") else "anonymous"
    return f"user:{user_id}"

def get_api_key(request: Request) -> str:
    """Get rate limit key based on API key
    
    Args:
        request: FastAPI request
        
    Returns:
        Rate limit key
    """
    # Try to get API key from header or query param
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        api_key = request.query_params.get("api_key", "anonymous")
        
    return f"api_key:{api_key}"


# Create default rate limiter
def create_default_rate_limiter() -> RateLimiter:
    """Create default rate limiter
    
    Returns:
        Rate limiter instance
    """
    # Try to use Redis if available
    try:
        storage = RedisRateLimitStorage()
        # Test connection
        redis_info = storage.redis.info()
        logger.info(f"Using Redis for rate limiting: {redis_info.get('redis_version')}")
    except Exception as e:
        logger.warning(f"Could not connect to Redis, using in-memory storage for rate limiting: {e}")
        storage = InMemoryRateLimitStorage()
        
    limiter = RateLimiter(storage=storage)
    
    # Add default limits
    limiter.add_limit(
        RateLimit(
            name="default",
            limit=100,
            window=60,
            description="Default API limit"
        ),
        key_func=get_ip_key
    )
    
    limiter.add_limit(
        RateLimit(
            name="strict",
            limit=10,
            window=60,
            description="Strict API limit for sensitive endpoints"
        ),
        key_func=get_ip_key
    )
    
    limiter.add_limit(
        RateLimit(
            name="auth",
            limit=5,
            window=60,
            description="Authentication limit"
        ),
        key_func=get_ip_key
    )
    
    limiter.add_limit(
        RateLimit(
            name="user",
            limit=1000,
            window=3600,
            description="Per-user API limit"
        ),
        key_func=get_user_key
    )
    
    limiter.add_limit(
        RateLimit(
            name="api_key",
            limit=10000,
            window=86400,  # 24 hours
            description="Per-API-key limit"
        ),
        key_func=get_api_key
    )
    
    return limiter


# Global limiter instance
_rate_limiter = None

def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance
    
    Returns:
        Rate limiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = create_default_rate_limiter()
    return _rate_limiter
    

# Dependency for rate limiting endpoints
async def check_rate_limit(
    request: Request,
    limit_name: str = "default",
    limiter: RateLimiter = Depends(get_rate_limiter)
):
    """Check rate limit for endpoint
    
    Args:
        request: FastAPI request
        limit_name: Rate limit name
        limiter: Rate limiter instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    is_limited, headers = await limiter.is_rate_limited(request, limit_name)
    
    if is_limited:
        limit = limiter.get_limit(limit_name)
        reset_at = float(headers.get("X-RateLimit-Reset", 0))
        
        # Update response headers
        for name, value in headers.items():
            request.state.rate_limit_headers = headers
            
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit.name} ({limit.limit} requests per {limit.window}s). Try again in {int(reset_at - time.time())}s."
        )
    
    # Store headers in request state for response
    request.state.rate_limit_headers = headers
