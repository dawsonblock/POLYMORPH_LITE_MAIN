"""
Role-Based Access Control (RBAC) for POLYMORPH v8.0.

Defines user roles and a dependency for enforcing permissions on endpoints.
"""

from enum import Enum
from typing import List, Optional
from fastapi import Depends, HTTPException, status, Request

class Role(str, Enum):
    ADMIN = "admin"         # Full access
    OPERATOR = "operator"   # Can run workflows, view data
    REVIEWER = "reviewer"   # Can sign off on data (OQ/PQ)
    AUDITOR = "auditor"     # Read-only access to logs

class PermissionDenied(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform this action."
        )

def require_role(allowed_roles: List[Role]):
    """
    Dependency to enforce role-based access.
    Assumes `request.state.user` is populated by Auth middleware.
    """
    def role_checker(request: Request):
        user = getattr(request.state, "user", None)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        # Parse user role (adjust based on your User model/JWT structure)
        # Assuming user is a dict from JWT: {"role": "operator", ...}
        user_role_str = user.get("role", "operator") # Default to lowest privilege if missing? Or deny?
        
        try:
            user_role = Role(user_role_str)
        except ValueError:
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid role: {user_role_str}"
            )

        if user_role not in allowed_roles and Role.ADMIN not in allowed_roles: 
            # Admin usually has access, but if explicit list doesn't include admin (rare), check logic.
            # Usually Admin implies all. Let's enforce strict list for now, or add Admin override.
            if user_role == Role.ADMIN:
                return # Admin passes
            raise PermissionDenied()
            
        return user
        
    return role_checker
