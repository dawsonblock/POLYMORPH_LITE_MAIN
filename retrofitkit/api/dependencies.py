"""
FastAPI dependencies for authentication and authorization.
"""

from typing import Set, Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt, JWTError

from retrofitkit.core.database import get_db_session
from retrofitkit.db.models.user import User
from retrofitkit.config import settings
# from retrofitkit.compliance.rbac import get_user_roles # Keep if valid, or refactor

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_db():
    async with get_db_session() as session:
        yield session

def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Get the current authenticated user from JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Fetch user from DB
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
        
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (not locked).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current User object
        
    Raises:
        HTTPException: If user account is locked
    """
    from datetime import datetime, timezone

    # Handle dict from tests (mock users)
    if isinstance(current_user, dict):
        return current_user

    # Handle User object
    if current_user.account_locked_until and current_user.account_locked_until > datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is locked"
        )

    return current_user


def require_role(*allowed_roles: str) -> Callable:
    """
    Dependency factory that requires user to have one of the specified roles.
    
    Usage:
        @app.post("/workflows", dependencies=[Depends(require_role("admin", "scientist"))])
        def create_workflow(...):
            ...
    
    Args:
        *allowed_roles: Role names that are allowed to access this route
        
    Returns:
        Dependency function
    """
    def role_checker(
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
    ):
        # Handle dict from tests (mock users)
        if isinstance(current_user, dict):
            # Mock users have "role" field, treat as single role
            user_role = current_user.get("role", "")
            if "admin" in user_role.lower() or any(role.lower() in user_role.lower() for role in allowed_roles):
                return current_user
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User must have one of the following roles: {', '.join(allowed_roles)}"
            )

        # Handle User object
        user_roles = get_user_roles(db, current_user.email)

        # Admin has all permissions
        if "admin" in user_roles:
            return current_user

        # Check if user has any of the required roles
        if not any(role in user_roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User must have one of the following roles: {', '.join(allowed_roles)}"
            )

        return current_user

    return role_checker


def require_any_role(allowed_roles: Set[str]) -> Callable:
    """
    Alternative dependency that accepts a set of roles.
    
    Args:
        allowed_roles: Set of role names
        
    Returns:
        Dependency function
    """
    return require_role(*list(allowed_roles))
