from typing import Dict, Optional, List, Union, Callable
from dataclasses import dataclass, field
import time
import asyncio
import redis.asyncio as redis
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
from fastapi import Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimitStrategy(str, Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitRule:
    limit: int
    window: int  # in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None
    cost: int = 1

@dataclass
class RateLimitConfig:
    redis_url: str = "redis://localhost:6379/0"
    default_window: int = 60  # 1 minute
    default_limit: int = 100
    enable_dynamic_rules: bool = True
    sync_interval: int = 10
    cleanup_interval: int = 300
    default_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    rules: Dict[str, RateLimitRule] = field(default_factory=dict)

class RateLimiter:
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.redis = redis.Redis.from_url(self.config.redis_url)
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background tasks for maintenance"""
        asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically clean up expired rate limit data"""
        while True:
            try:
                await self._cleanup_expired_data()
                await asyncio.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)

    async def _cleanup_expired_data(self):
        """Clean up expired rate limit data"""
        now = time.time()
        
        # Get all keys
        async for key in self.redis.scan_iter("ratelimit:*"):
            try:
                # Check if key has expired data
                if await self.redis.zremrangebyscore(key, 0, now - 3600) > 0:
                    # If no remaining data, remove the key
                    if await self.redis.zcard(key) == 0:
                        await self.redis.delete(key)
            except Exception as e:
                logger.error(f"Error cleaning up key {key}: {e}")

    def _get_rule(self, endpoint: str) -> RateLimitRule:
        """Get rate limit rule for an endpoint"""
        return self.config.rules.get(
            endpoint,
            RateLimitRule(
                limit=self.config.default_limit,
                window=self.config.default_window,
                strategy=self.config.default_strategy
            )
        )

    async def is_rate_limited(
        self,
        key: str,
        rule: RateLimitRule
    ) -> bool:
        """Check if a key is rate limited"""
        try:
            if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, rule)
            elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, rule)
            elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, rule)
            elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._check_leaky_bucket(key, rule)
            else:
                raise ValueError(f"Unknown rate limit strategy: {rule.strategy}")
        
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False

    async def _check_fixed_window(
        self,
        key: str,
        rule: RateLimitRule
    ) -> bool:
        """Check rate limit using fixed window strategy"""
        window_key = f"{key}:{int(time.time() / rule.window)}"
        
        # Increment counter for current window
        count = await self.redis.incr(window_key)
        
        # Set expiration if this is a new window
        if count == 1:
            await self.redis.expire(window_key, rule.window)
        
        return count > rule.limit

    async def _check_sliding_window(
        self,
        key: str,
        rule: RateLimitRule
    ) -> bool:
        """Check rate limit using sliding window strategy"""
        now = time.time()
        
        # Add current request
        await self.redis.zadd(key, {str(now): now})
        
        # Remove old requests
        await self.redis.zremrangebyscore(key, 0, now - rule.window)
        
        # Count requests in window
        count = await self.redis.zcard(key)
        
        # Set key expiration
        await self.redis.expire(key, rule.window)
        
        return count > rule.limit

    async def _check_token_bucket(
        self,
        key: str,
        rule: RateLimitRule
    ) -> bool:
        """Check rate limit using token bucket strategy"""
        now = time.time()
        bucket_key = f"bucket:{key}"
        
        # Get current bucket state
        bucket = await self.redis.hgetall(bucket_key)
        
        if not bucket:
            # Initialize bucket
            tokens = rule.limit
            last_update = now
        else:
            tokens = float(bucket[b'tokens'])
            last_update = float(bucket[b'last_update'])
            
            # Add new tokens based on time passed
            time_passed = now - last_update
            new_tokens = time_passed * (rule.limit / rule.window)
            tokens = min(rule.limit, tokens + new_tokens)

        # Check if enough tokens
        if tokens < rule.cost:
            return True

        # Consume tokens
        tokens -= rule.cost
        
        # Update bucket
        await self.redis.hmset(bucket_key, {
            'tokens': tokens,
            'last_update': now
        })
        await self.redis.expire(bucket_key, rule.window)
        
        return False

    async def _check_leaky_bucket(
        self,
        key: str,
        rule: RateLimitRule
    ) -> bool:
        """Check rate limit using leaky bucket strategy"""
        now = time.time()
        bucket_key = f"leaky:{key}"
        
        # Get current bucket state
        bucket = await self.redis.hgetall(bucket_key)
        
        if not bucket:
            # Initialize bucket
            water_level = rule.cost
            last_update = now
        else:
            water_level = float(bucket[b'water_level'])
            last_update = float(bucket[b'last_update'])
            
            # Calculate leaked water
            time_passed = now - last_update
            leak_rate = rule.limit / rule.window
            leaked = time_passed * leak_rate
            water_level = max(0, water_level - leaked)

        # Check if adding request would overflow
        if water_level + rule.cost > rule.limit:
            return True

        # Add request to bucket
        water_level += rule.cost
        
        # Update bucket
        await self.redis.hmset(bucket_key, {
            'water_level': water_level,
            'last_update': now
        })
        await self.redis.expire(bucket_key, rule.window)
        
        return False

    def rate_limit(
        self,
        limit: Optional[int] = None,
        window: Optional[int] = None,
        strategy: Optional[RateLimitStrategy] = None,
        key_func: Optional[Callable] = None
    ):
        """Decorator for rate limiting endpoints"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get request object
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    for arg in kwargs.values():
                        if isinstance(arg, Request):
                            request = arg
                            break
                
                if not request:
                    raise ValueError("Could not find request object")

                # Get rate limit key
                if key_func:
                    key = key_func(request)
                else:
                    key = self._default_key_func(request)

                # Get or create rule
                rule = RateLimitRule(
                    limit=limit or self.config.default_limit,
                    window=window or self.config.default_window,
                    strategy=strategy or self.config.default_strategy
                )

                # Check rate limit
                if await self.is_rate_limited(key, rule):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def _default_key_func(self, request: Request) -> str:
        """Default function to generate rate limit key"""
        return f"ratelimit:{request.client.host}:{request.url.path}"

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        limiter: RateLimiter,
        exclude_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.limiter = limiter
        self.exclude_paths = set(exclude_paths or [])

    async def dispatch(self, request: Request, call_next):
        # Check if path should be excluded
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Get rate limit rule
        rule = self.limiter._get_rule(request.url.path)
        
        # Generate key
        key = self.limiter._default_key_func(request)
        
        # Check rate limit
        if await self.limiter.is_rate_limited(key, rule):
            return HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        # Continue processing request
        response = await call_next(request)
        
        # Add rate limit headers
        window_stats = await self._get_window_stats(key, rule)
        response.headers['X-RateLimit-Limit'] = str(rule.limit)
        response.headers['X-RateLimit-Remaining'] = str(
            max(0, rule.limit - window_stats['current'])
        )
        response.headers['X-RateLimit-Reset'] = str(
            int(window_stats['reset_time'])
        )
        
        return response

    async def _get_window_stats(
        self,
        key: str,
        rule: RateLimitRule
    ) -> Dict[str, Union[int, float]]:
        """Get current window statistics"""
        now = time.time()
        
        if rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            window_key = f"{key}:{int(now / rule.window)}"
            current = int(await self.limiter.redis.get(window_key) or 0)
            reset_time = (int(now / rule.window) + 1) * rule.window
        else:
            current = await self.limiter.redis.zcard(key)
            reset_time = now + rule.window

        return {
            'current': current,
            'reset_time': reset_time
        }
