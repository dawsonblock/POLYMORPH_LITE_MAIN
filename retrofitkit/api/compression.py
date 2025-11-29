"""
Response compression middleware for API performance.

Adds gzip compression to API responses to reduce bandwidth and improve load times.
"""
from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.gzip import GZipMiddleware as StarletteGZipMiddleware
import gzip
import io


class GZipMiddleware:
    """
    GZip compression middleware.
    
    Compresses responses larger than minimum_size bytes.
    Adds Content-Encoding: gzip header.
    """
    
    def __init__(self, app, minimum_size: int = 500, compression_level: int = 6):
        """
        Initialize GZip middleware.
        
        Args:
            app: FastAPI application
            minimum_size: Minimum response size to compress (bytes)
            compression_level: Compression level (1-9, higher = more compression)
        """
        self.app = app
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        
    async def __call__(self, scope, receive, send):
        """Process request with compression."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        # Check if client accepts gzip
        headers = dict(scope.get("headers", []))
        accept_encoding = headers.get(b"accept-encoding", b"").decode()
        
        if "gzip" not in accept_encoding:
            await self.app(scope, receive, send)
            return
            
        # Use Starlette's GZip middleware
        middleware = StarletteGZipMiddleware(
            self.app,
            minimum_size=self.minimum_size,
            compresslevel=self.compression_level
        )
        await middleware(scope, receive, send)


def add_compression_middleware(app, minimum_size: int = 500):
    """
    Add compression middleware to FastAPI app.
    
    Usage:
        from retrofitkit.api.compression import add_compression_middleware
        
        app = FastAPI()
        add_compression_middleware(app)
    """
    app.add_middleware(
        StarletteGZipMiddleware,
        minimum_size=minimum_size,
        compresslevel=6
    )
