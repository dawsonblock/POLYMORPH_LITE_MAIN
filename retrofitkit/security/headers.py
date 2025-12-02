"""
Security headers middleware for FastAPI.
Implements OWASP recommended security headers.
"""
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Awaitable


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all HTTP responses.
    
    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security (HSTS)
    - Content-Security-Policy (CSP)
    - Permissions-Policy
    - Referrer-Policy
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # XSS Protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # HSTS - Force HTTPS for 1 year
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Needed for React
            "style-src 'self' 'unsafe-inline'",  # Needed for styled components
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss:",  # WebSocket support
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # Permissions Policy (formerly Feature Policy)
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)

        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Remove server header for security obscurity
        if "Server" in response.headers:
            del response.headers["Server"]

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token-bucket rate limiter per client IP with automatic cleanup.

    This is in-process only. For multi-instance deployments,
    adapt this to use Redis.

    Config:
      - requests: max tokens in bucket
      - window_sec: refill window in seconds
      - max_buckets: maximum number of tracked clients (LRU eviction)
      - cleanup_interval: how often to run cleanup (in requests)
    """

    __slots__ = ('requests', 'window_sec', 'max_buckets', 'cleanup_interval', 
                 '_buckets', '_request_count')

    def __init__(self, app, requests: int = 60, window_sec: int = 60, 
                 max_buckets: int = 10000, cleanup_interval: int = 1000):
        super().__init__(app)
        self.requests = requests
        self.window_sec = window_sec
        self.max_buckets = max_buckets
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, dict[str, float]] = {}
        self._request_count = 0

    def _cleanup_expired_buckets(self, now: float) -> None:
        """Remove expired buckets to prevent memory leaks."""
        # Remove buckets that haven't been accessed in 2x the window
        expiry_threshold = now - (self.window_sec * 2)
        expired_keys = [
            key for key, bucket in self._buckets.items()
            if bucket["last"] < expiry_threshold
        ]
        for key in expired_keys:
            del self._buckets[key]
        
        # If still too many, remove oldest entries (LRU eviction)
        if len(self._buckets) > self.max_buckets:
            sorted_keys = sorted(
                self._buckets.keys(),
                key=lambda k: self._buckets[k]["last"]
            )
            for key in sorted_keys[:len(self._buckets) - self.max_buckets]:
                del self._buckets[key]

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Use only client IP for rate limiting (not path) to reduce memory
        client_ip = request.client.host if request.client else "unknown"
        
        now = time.time()
        
        # Periodic cleanup to prevent memory leaks
        self._request_count += 1
        if self._request_count >= self.cleanup_interval:
            self._cleanup_expired_buckets(now)
            self._request_count = 0

        bucket = self._buckets.get(client_ip)

        if bucket is None:
            bucket = {"tokens": float(self.requests), "last": now}
        else:
            # Refill tokens
            elapsed = now - bucket["last"]
            refill = (elapsed / self.window_sec) * self.requests
            bucket["tokens"] = min(float(self.requests), bucket["tokens"] + refill)
            bucket["last"] = now

        # Consume token
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            self._buckets[client_ip] = bucket
            response = await call_next(request)
            remaining = int(bucket["tokens"])
            reset_in = max(0, int(self.window_sec - (time.time() - bucket["last"])))
            response.headers["X-RateLimit-Limit"] = str(self.requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_in)
            return response
        else:
            # Rate limited
            self._buckets[client_ip] = bucket
            reset_in = max(0, int(self.window_sec - (time.time() - bucket["last"])))
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after_sec": reset_in,
                },
                headers={
                    "Retry-After": str(reset_in),
                    "X-RateLimit-Limit": str(self.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_in),
                },
            )
