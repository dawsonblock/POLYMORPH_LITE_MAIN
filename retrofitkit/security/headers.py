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
    Simple token-bucket rate limiter per (ip, path).

    This is in-process only. For multi-instance deployments,
    adapt this to use Redis.

    Config:
      - requests: max tokens in bucket
      - window_sec: refill window in seconds
    """

    def __init__(self, app, requests: int = 60, window_sec: int = 60):
        super().__init__(app)
        self.requests = requests
        self.window_sec = window_sec
        self._buckets: dict[tuple[str, str], dict[str, float]] = {}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        key = (client_ip, request.url.path)

        now = time.time()
        bucket = self._buckets.get(key)

        if bucket is None:
            # Initialize full bucket
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
            self._buckets[key] = bucket
            response = await call_next(request)
            remaining = int(bucket["tokens"])
            reset_in = max(0, int(self.window_sec - (time.time() - bucket["last"])))
            response.headers["X-RateLimit-Limit"] = str(self.requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_in)
            return response
        else:
            # Rate limited
            self._buckets[key] = bucket
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
