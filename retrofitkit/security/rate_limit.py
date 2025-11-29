"""
Rate Limiting Middleware for POLYMORPH-LITE

Redis-based rate limiting to prevent API abuse and ensure fair usage.
Supports per-user, per-IP, and per-endpoint rate limits.
"""
import time
import logging
from typing import Optional, Callable
from functools import wraps
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    
    Supports:
    - Per-user rate limits
    - Per-IP rate limits
    - Per-endpoint rate limits
    - Configurable windows and limits
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_limit: int = 100,
        default_window: int = 60,
        enabled: bool = True
    ):
        self.redis_url = redis_url
        self.default_limit = default_limit
        self.default_window = default_window
        self.enabled = enabled
        self._redis: Optional[redis.Redis] = None
        
    async def connect(self) -> None:
        """Connect to Redis."""
        if not self.enabled:
            logger.info("Rate limiting disabled")
            return
            
        try:
            self._redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            logger.warning("Rate limiting will be disabled")
            self.enabled = False
            
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis")
            
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique identifier for rate limit (e.g., "user:123", "ip:1.2.3.4")
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        if not self.enabled or not self._redis:
            # If Redis unavailable, allow all requests
            return True, limit, 0
            
        now = int(time.time())
        window_start = now - window
        
        try:
            # Use Redis sorted set for sliding window
            pipe = self._redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = await pipe.execute()
            current_count = results[1]
            
            allowed = current_count < limit
            remaining = max(0, limit - current_count - 1)
            reset_time = now + window
            
            return allowed, remaining, reset_time
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # On error, allow request (fail open)
            return True, limit, 0
            
    async def reset_limit(self, key: str) -> None:
        """Reset rate limit for a key (admin function)."""
        if self._redis:
            try:
                await self._redis.delete(key)
                logger.info(f"Reset rate limit for {key}")
            except Exception as e:
                logger.error(f"Failed to reset rate limit: {e}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")
    return _rate_limiter


async def init_rate_limiter(
    redis_url: str = "redis://localhost:6379",
    default_limit: int = 100,
    default_window: int = 60,
    enabled: bool = True
) -> RateLimiter:
    """Initialize global rate limiter."""
    global _rate_limiter
    _rate_limiter = RateLimiter(
        redis_url=redis_url,
        default_limit=default_limit,
        default_window=default_window,
        enabled=enabled
    )
    await _rate_limiter.connect()
    return _rate_limiter


async def shutdown_rate_limiter() -> None:
    """Shutdown global rate limiter."""
    global _rate_limiter
    if _rate_limiter:
        await _rate_limiter.disconnect()


# Rate limit decorator
def rate_limit(
    limit: Optional[int] = None,
    window: Optional[int] = None,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        limit: Maximum requests (default: from config)
        window: Time window in seconds (default: from config)
        key_func: Function to generate rate limit key from request
        
    Example:
        @rate_limit(limit=10, window=60)
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get('request')
                
            if not request:
                # No request object, skip rate limiting
                return await func(*args, **kwargs)
                
            limiter = get_rate_limiter()
            
            # Determine rate limit key
            if key_func:
                key = key_func(request)
            else:
                # Default: use user email if authenticated, else IP
                user = getattr(request.state, 'user', None)
                if user and 'email' in user:
                    key = f"user:{user['email']}"
                else:
                    client_ip = request.client.host if request.client else "unknown"
                    key = f"ip:{client_ip}"
                    
            # Add endpoint to key
            key = f"{key}:{request.url.path}"
            
            # Use provided limits or defaults
            req_limit = limit or limiter.default_limit
            req_window = window or limiter.default_window
            
            # Check rate limit
            allowed, remaining, reset_time = await limiter.check_rate_limit(
                key, req_limit, req_window
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": req_limit,
                        "window": req_window,
                        "reset_at": reset_time
                    }
                )
                
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(req_limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(reset_time)
                
            return response
            
        return wrapper
    return decorator


# Middleware for global rate limiting
async def rate_limit_middleware(request: Request, call_next):
    """
    Global rate limiting middleware.
    
    Applies basic rate limiting to all requests.
    Individual endpoints can override with @rate_limit decorator.
    """
    try:
        limiter = get_rate_limiter()
    except RuntimeError:
        # Rate limiter not initialized, skip
        return await call_next(request)
        
    if not limiter.enabled:
        return await call_next(request)
        
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
        
    # Determine key
    user = getattr(request.state, 'user', None)
    if user and 'email' in user:
        key = f"global:user:{user['email']}"
    else:
        client_ip = request.client.host if request.client else "unknown"
        key = f"global:ip:{client_ip}"
        
    # Check global rate limit (more permissive)
    allowed, remaining, reset_time = await limiter.check_rate_limit(
        key,
        limit=1000,  # 1000 requests
        window=60    # per minute
    )
    
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Global rate limit exceeded",
                "limit": 1000,
                "window": 60,
                "reset_at": reset_time
            },
            headers={
                "X-RateLimit-Limit": "1000",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time),
                "Retry-After": str(reset_time - int(time.time()))
            }
        )
        
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = "1000"
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_time)
    
    return response
