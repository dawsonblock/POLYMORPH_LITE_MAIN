"""
Organization Context Middleware for Multi-Tenant Enforcement.

Extracts org_id from JWT claims and injects into request state.
All downstream dependencies can access org context via request.state.org_id.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import jwt
import logging

logger = logging.getLogger(__name__)


class OrgContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts and validates organization context from JWT.
    
    Sets request.state.org_id and request.state.user_email for downstream use.
    """
    
    # Paths that don't require org context
    EXEMPT_PATHS = {
        "/health",
        "/healthz",
        "/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/auth/login",
        "/auth/token",
        "/auth/refresh",
    }
    
    def __init__(self, app, secret_key: str, algorithm: str = "HS256"):
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    async def dispatch(self, request: Request, call_next):
        # Check if path is exempt
        if self._is_exempt(request.url.path):
            return await call_next(request)
        
        # Extract and validate JWT
        org_id, user_email = self._extract_org_context(request)
        
        if org_id is None:
            # Allow request to proceed, but org_id will be None
            # Individual endpoints can choose to require it
            request.state.org_id = None
            request.state.user_email = user_email
        else:
            request.state.org_id = org_id
            request.state.user_email = user_email
            logger.debug(f"Org context: org_id={org_id}, user={user_email}")
        
        response = await call_next(request)
        return response
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from org context requirement."""
        # Exact match
        if path in self.EXEMPT_PATHS:
            return True
        # Prefix match for /auth/* paths
        if path.startswith("/auth/"):
            return True
        # Static files
        if path.startswith("/static/") or path.startswith("/_next/"):
            return True
        return False
    
    def _extract_org_context(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        """
        Extract org_id and user_email from JWT.
        
        Returns:
            (org_id, user_email) - both may be None
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            return None, None
        
        token = auth_header.split(" ", 1)[1].strip()
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            org_id = payload.get("org_id")
            user_email = payload.get("sub") or payload.get("email")
            return org_id, user_email
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token")
            return None, None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None, None


def require_org_context(request: Request) -> str:
    """
    FastAPI dependency that requires org_id to be present.
    
    Usage:
        @router.get("/data")
        async def get_data(org_id: str = Depends(require_org_context)):
            ...
    """
    org_id = getattr(request.state, "org_id", None)
    if org_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization context required"
        )
    return org_id


def get_org_context(request: Request) -> Optional[str]:
    """
    FastAPI dependency that returns org_id if present, None otherwise.
    
    Usage:
        @router.get("/data")
        async def get_data(org_id: Optional[str] = Depends(get_org_context)):
            ...
    """
    return getattr(request.state, "org_id", None)


def get_user_email(request: Request) -> Optional[str]:
    """Get user email from request context."""
    return getattr(request.state, "user_email", None)
