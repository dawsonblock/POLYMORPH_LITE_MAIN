"""
Tests for rate limiting middleware.

Tests:
- Rate limit enforcement
- Per-user limits
- Per-IP limits
- Per-endpoint limits
- Rate limit headers
- Redis fallback
- Limit reset
"""
import pytest
import asyncio
from fastapi import FastAPI, Request, Depends
from fastapi.testclient import TestClient
from retrofitkit.security.rate_limit import (
    RateLimiter,
    init_rate_limiter,
    shutdown_rate_limiter,
    rate_limit,
    rate_limit_middleware
)


@pytest.fixture
async def rate_limiter():
    """Create rate limiter for testing."""
    limiter = await init_rate_limiter(
        redis_url="redis://localhost:6379",
        default_limit=10,
        default_window=60,
        enabled=True
    )
    yield limiter
    await shutdown_rate_limiter()


@pytest.fixture
def app_with_rate_limit():
    """Create FastAPI app with rate limiting."""
    app = FastAPI()
    
    @app.get("/test")
    @rate_limit(limit=5, window=60)
    async def test_endpoint(request: Request):
        return {"message": "success"}
        
    @app.get("/unlimited")
    async def unlimited_endpoint():
        return {"message": "unlimited"}
        
    return app


@pytest.mark.asyncio
async def test_rate_limiter_basic(rate_limiter):
    """Test basic rate limiting."""
    key = "test:user:1"
    limit = 5
    window = 60
    
    # First 5 requests should succeed
    for i in range(5):
        allowed, remaining, reset = await rate_limiter.check_rate_limit(key, limit, window)
        assert allowed is True
        assert remaining == 4 - i
        
    # 6th request should fail
    allowed, remaining, reset = await rate_limiter.check_rate_limit(key, limit, window)
    assert allowed is False
    assert remaining == 0


@pytest.mark.asyncio
async def test_rate_limiter_window_expiry(rate_limiter):
    """Test that rate limits reset after window expires."""
    key = "test:user:2"
    limit = 3
    window = 1  # 1 second window
    
    # Use up limit
    for i in range(3):
        allowed, _, _ = await rate_limiter.check_rate_limit(key, limit, window)
        assert allowed is True
        
    # Should be blocked
    allowed, _, _ = await rate_limiter.check_rate_limit(key, limit, window)
    assert allowed is False
    
    # Wait for window to expire
    await asyncio.sleep(1.1)
    
    # Should be allowed again
    allowed, _, _ = await rate_limiter.check_rate_limit(key, limit, window)
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_different_keys(rate_limiter):
    """Test that different keys have independent limits."""
    limit = 5
    window = 60
    
    # User 1 uses up limit
    for i in range(5):
        allowed, _, _ = await rate_limiter.check_rate_limit("user:1", limit, window)
        assert allowed is True
        
    # User 1 should be blocked
    allowed, _, _ = await rate_limiter.check_rate_limit("user:1", limit, window)
    assert allowed is False
    
    # User 2 should still be allowed
    allowed, _, _ = await rate_limiter.check_rate_limit("user:2", limit, window)
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_reset(rate_limiter):
    """Test rate limit reset."""
    key = "test:user:3"
    limit = 5
    window = 60
    
    # Use up limit
    for i in range(5):
        await rate_limiter.check_rate_limit(key, limit, window)
        
    # Should be blocked
    allowed, _, _ = await rate_limiter.check_rate_limit(key, limit, window)
    assert allowed is False
    
    # Reset limit
    await rate_limiter.reset_limit(key)
    
    # Should be allowed again
    allowed, _, _ = await rate_limiter.check_rate_limit(key, limit, window)
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limiter_disabled():
    """Test that rate limiter can be disabled."""
    limiter = await init_rate_limiter(enabled=False)
    
    # All requests should be allowed
    for i in range(100):
        allowed, _, _ = await limiter.check_rate_limit("test", 5, 60)
        assert allowed is True
        
    await shutdown_rate_limiter()


def test_rate_limit_decorator(app_with_rate_limit):
    """Test rate limit decorator on endpoint."""
    client = TestClient(app_with_rate_limit)
    
    # First 5 requests should succeed
    for i in range(5):
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "5"
        
    # 6th request should fail with 429
    response = client.get("/test")
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]["error"]


def test_rate_limit_headers(app_with_rate_limit):
    """Test that rate limit headers are present."""
    client = TestClient(app_with_rate_limit)
    
    response = client.get("/test")
    assert response.status_code == 200
    
    # Check headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers
    
    assert response.headers["X-RateLimit-Limit"] == "5"
    assert int(response.headers["X-RateLimit-Remaining"]) >= 0


def test_unlimited_endpoint(app_with_rate_limit):
    """Test that endpoints without decorator are not rate limited."""
    client = TestClient(app_with_rate_limit)
    
    # Should be able to make many requests
    for i in range(20):
        response = client.get("/unlimited")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_rate_limiter_concurrent_requests(rate_limiter):
    """Test rate limiter with concurrent requests."""
    key = "test:concurrent"
    limit = 10
    window = 60
    
    async def make_request():
        return await rate_limiter.check_rate_limit(key, limit, window)
        
    # Make 15 concurrent requests
    results = await asyncio.gather(*[make_request() for _ in range(15)])
    
    # Exactly 10 should be allowed
    allowed_count = sum(1 for allowed, _, _ in results if allowed)
    assert allowed_count == 10


@pytest.mark.asyncio
async def test_rate_limiter_redis_unavailable():
    """Test graceful fallback when Redis unavailable."""
    limiter = await init_rate_limiter(
        redis_url="redis://invalid:9999",  # Invalid Redis URL
        enabled=True
    )
    
    # Should fall back to allowing all requests
    for i in range(100):
        allowed, _, _ = await limiter.check_rate_limit("test", 5, 60)
        assert allowed is True
        
    await shutdown_rate_limiter()
