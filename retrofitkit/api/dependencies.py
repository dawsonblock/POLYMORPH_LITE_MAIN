"""
FastAPI dependencies for authentication and authorization.
"""


from typing import Generator, Optional, Dict, List, Set, Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt import PyJWTError as JWTError
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from retrofitkit.db.session import get_db
from retrofitkit.core.config import get_config
from retrofitkit.db.models.user import User
from retrofitkit.compliance.rbac import get_user_roles


# Simple TokenData model
class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access"
    }
)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> Dict:
    """
    Get current user from JWT token.
    """
    config = get_config()
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            config.security.jwt_secret, 
            algorithms=[config.security.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, scopes=payload.get("scopes", []))
    except (JWTError, ValidationError):
        raise credentials_exception
        
    # Check DB
    # Note: simple User model lookup
    # result = await db.execute(select(User).where(User.username == username))
    # user = result.scalars().first()
    
    # For now, we return the payload as the user context to avoid DB hit on every request
    # unless we need strict user validation.
    # TODO: Implement strict DB check if needed
    
    user = {
        "username": username,
        "email": username, # Assuming username is email
        "scopes": token_data.scopes,
        "is_active": True
    }
    
    return user

async def get_current_active_user(
    current_user: Dict = Depends(get_current_user)
) -> Dict:
    if not current_user.get("is_active"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(*allowed_roles: str) -> Callable:
    """
    Dependency for role-based access control.
    
    Usage::
    
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
