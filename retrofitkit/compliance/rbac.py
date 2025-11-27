"""
Role-Based Access Control (RBAC) helper functions.
"""

from typing import Set, Optional
from sqlalchemy.orm import Session
from retrofitkit.db.models.rbac import Role, UserRole
from retrofitkit.db.models.user import User
import uuid


def seed_default_roles(db: Session):
    """
    Create default roles if they don't exist.
    
    Default roles:
    - admin: Full system access
    - scientist: Can create/edit workflows, samples, runs
    - technician: Can execute workflows, manage inventory
    - compliance: Read-only access to audit logs and compliance features
    """
    default_roles = [
        {
            "role_name": "admin",
            "description": "System administrator with full access",
            "permissions": {"all": ["create", "read", "update", "delete"]}
        },
        {
            "role_name": "scientist",
            "description": "Scientist with workflow and sample management rights",
            "permissions": {
                "workflows": ["create", "read", "update"],
                "samples": ["create", "read", "update"],
                "runs": ["create", "read"],
                "devices": ["read"]
            }
        },
        {
            "role_name": "technician",
            "description": "Lab technician with execution and inventory rights",
            "permissions": {
                "runs": ["create", "read"],
                "inventory": ["create", "read", "update"],
                "calibration": ["create", "read"],
                "devices": ["read"]
            }
        },
        {
            "role_name": "compliance",
            "description": "Compliance officer with read-only audit access",
            "permissions": {
                "audit": ["read"],
                "runs": ["read"],
                "samples": ["read"],
                "workflows": ["read"]
            }
        }
    ]

    for role_data in default_roles:
        existing = db.query(Role).filter(Role.role_name == role_data["role_name"]).first()
        if not existing:
            role = Role(
                id=uuid.uuid4(),
                role_name=role_data["role_name"],
                description=role_data["description"],
                permissions=role_data["permissions"]
            )
            db.add(role)

    db.commit()


def assign_role(db: Session, user_email: str, role_name: str, assigned_by: Optional[str] = None) -> bool:
    """
    Assign a role to a user.
    
    Args:
        db: Database session
        user_email: User's email address
        role_name: Name of the role to assign
        assigned_by: Email of the user making the assignment
        
    Returns:
        True if role was assigned, False if user/role not found or already assigned
    """
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return False

    role = db.query(Role).filter(Role.role_name == role_name).first()
    if not role:
        return False

    # Check if already assigned
    existing = db.query(UserRole).filter(
        UserRole.user_email == user_email,
        UserRole.role_id == role.id
    ).first()

    if existing:
        return False  # Already assigned

    user_role = UserRole(
        user_email=user_email,
        role_id=role.id,
        assigned_by=assigned_by
    )
    db.add(user_role)
    db.commit()
    return True


def get_user_roles(db: Session, user_email: str) -> Set[str]:
    """
    Get all role names assigned to a user.
    
    Args:
        db: Database session
        user_email: User's email address
        
    Returns:
        Set of role names
    """
    user_roles = db.query(UserRole).filter(UserRole.user_email == user_email).all()
    role_ids = [ur.role_id for ur in user_roles]

    if not role_ids:
        return set()

    roles =db.query(Role).filter(Role.id.in_(role_ids)).all()
    return {role.role_name for role in roles}


def user_has_role(db: Session, user_email: str, required_role: str) -> bool:
    """
    Check if a user has a specific role.
    
    Args:
        db: Database session
        user_email: User's email address
        required_role: Role name to check for
        
    Returns:
        True if user has the role, False otherwise
    """
    user_roles = get_user_roles(db, user_email)
    return required_role in user_roles or "admin" in user_roles  # Admins have all roles


def user_has_any_role(db: Session, user_email: str, required_roles: Set[str]) -> bool:
    """
    Check if a user has any of the specified roles.
    
    Args:
        db: Database session
        user_email: User's email address
        required_roles: Set of role names
        
    Returns:
        True if user has at least one of the roles
    """
    user_roles = get_user_roles(db, user_email)
    if "admin" in user_roles:
        return True
    return bool(user_roles & required_roles)
