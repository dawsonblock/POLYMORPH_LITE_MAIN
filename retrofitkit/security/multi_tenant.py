"""
Multi-Tenant Enforcement Middleware for POLYMORPH-LITE

Ensures strict organization isolation for multi-lab SaaS deployments.
All database queries automatically scoped to user's organization.
"""
import logging
from typing import Optional, List
from fastapi import Request, HTTPException, status, Depends
from sqlalchemy.orm import Session
from sqlalchemy import and_

logger = logging.getLogger(__name__)


class OrgContext:
    """
    Organization context for current request.
    
    Injected into request.state by middleware.
    Used to scope all database queries.
    """
    
    def __init__(self, org_id: str, org_name: str, user_email: str):
        self.org_id = org_id
        self.org_name = org_name
        self.user_email = user_email
        
    def __repr__(self) -> str:
        return f"OrgContext(org_id={self.org_id}, org_name={self.org_name}, user={self.user_email})"


async def get_org_context(request: Request) -> OrgContext:
    """
    Dependency to get organization context from request.
    
    Usage:
        @router.get("/samples")
        async def list_samples(org: OrgContext = Depends(get_org_context)):
            # org.org_id is automatically available
            ...
    """
    org_context: Optional[OrgContext] = getattr(request.state, 'org_context', None)
    if not org_context:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Organization context not available"
        )
    return org_context


async def multi_tenant_middleware(request: Request, call_next):
    """
    Multi-tenant enforcement middleware.
    
    Extracts organization from authenticated user and injects into request state.
    All subsequent database queries should use this org_id for filtering.
    """
    # Skip for public endpoints
    public_paths = ["/health", "/metrics", "/docs", "/openapi.json", "/auth/login"]
    if request.url.path in public_paths or request.url.path.startswith("/auth/"):
        return await call_next(request)
        
    # Get authenticated user from request state
    user = getattr(request.state, 'user', None)
    
    if not user:
        # No user authenticated, skip org context
        # (auth middleware will handle this)
        return await call_next(request)
        
    # Extract org_id from user
    org_id = user.get('org_id')
    org_name = user.get('org_name', 'Unknown')
    user_email = user.get('email', 'unknown')
    
    if not org_id:
        logger.error(f"User {user_email} has no org_id")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User not associated with any organization"
        )
        
    # Create and inject org context
    org_context = OrgContext(
        org_id=org_id,
        org_name=org_name,
        user_email=user_email
    )
    request.state.org_context = org_context
    
    logger.debug(f"Request scoped to {org_context}")
    
    response = await call_next(request)
    return response


def scope_to_org(query, org_context: OrgContext, model_class):
    """
    Helper to scope SQLAlchemy query to organization.
    
    Usage:
        query = session.query(Sample)
        query = scope_to_org(query, org_context, Sample)
    """
    if hasattr(model_class, 'org_id'):
        return query.filter(model_class.org_id == org_context.org_id)
    else:
        logger.warning(f"Model {model_class.__name__} has no org_id field")
        return query


class OrgScopedSession:
    """
    Database session wrapper that automatically scopes queries to organization.
    
    Usage:
        scoped_session = OrgScopedSession(session, org_context)
        samples = scoped_session.query(Sample).all()
        # Automatically filtered to org
    """
    
    def __init__(self, session: Session, org_context: OrgContext):
        self.session = session
        self.org_context = org_context
        
    def query(self, *entities):
        """Create query automatically scoped to org."""
        query = self.session.query(*entities)
        
        # Auto-scope if model has org_id
        for entity in entities:
            if hasattr(entity, 'org_id'):
                query = query.filter(entity.org_id == self.org_context.org_id)
                
        return query
        
    def add(self, instance):
        """Add instance, automatically setting org_id."""
        if hasattr(instance, 'org_id') and not instance.org_id:
            instance.org_id = self.org_context.org_id
        return self.session.add(instance)
        
    def __getattr__(self, name):
        """Delegate other methods to underlying session."""
        return getattr(self.session, name)


def get_org_scoped_session(
    session: Session,
    org_context: OrgContext = Depends(get_org_context)
) -> OrgScopedSession:
    """
    Dependency to get org-scoped database session.
    
    Usage:
        @router.get("/samples")
        async def list_samples(
            db: OrgScopedSession = Depends(get_org_scoped_session)
        ):
            samples = db.query(Sample).all()
            # Automatically filtered to user's org
    """
    return OrgScopedSession(session, org_context)


def enforce_org_access(
    resource_org_id: str,
    org_context: OrgContext,
    resource_type: str = "resource"
) -> None:
    """
    Enforce that user can only access resources in their org.
    
    Raises HTTPException if org mismatch.
    
    Usage:
        sample = db.query(Sample).filter(Sample.id == sample_id).first()
        enforce_org_access(sample.org_id, org_context, "sample")
    """
    if resource_org_id != org_context.org_id:
        logger.warning(
            f"Org access violation: user {org_context.user_email} "
            f"(org {org_context.org_id}) attempted to access {resource_type} "
            f"in org {resource_org_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_type.capitalize()} not found"
            # Don't reveal that resource exists in different org
        )


def get_allowed_orgs(user: dict) -> List[str]:
    """
    Get list of org IDs user has access to.
    
    For now, users only have access to their primary org.
    Future: support cross-org access for admins.
    """
    org_id = user.get('org_id')
    if not org_id:
        return []
    return [org_id]


def check_org_permission(
    user: dict,
    org_id: str,
    permission: str = "read"
) -> bool:
    """
    Check if user has permission in specified org.
    
    Args:
        user: Authenticated user dict
        org_id: Organization ID to check
        permission: Permission type (read, write, admin)
        
    Returns:
        True if user has permission
    """
    # Check if user belongs to org
    if user.get('org_id') != org_id:
        return False
        
    # Check permission based on role
    user_roles = user.get('roles', [])
    
    if permission == "admin":
        return "admin" in user_roles
    elif permission == "write":
        return any(role in ["admin", "scientist", "technician"] for role in user_roles)
    else:  # read
        return True  # All authenticated users can read in their org
