"""
User management with SQLAlchemy instead of raw SQLite.

Migrated from direct SQLite operations to use the unified database layer.
"""

import bcrypt
import pyotp
import time
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from retrofitkit.db.models.user import User
from retrofitkit.compliance.audit import write_audit_event


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get user by email address.
    
    Args:
        db: Database session
        email: User's email address
        
    Returns:
        User object or None if not found
    """
    return db.query(User).filter(User.email == email).first()


def create_user(
    db: Session,
    email: str,
    password: str,
    full_name: str,
    role: str = "scientist",
    is_superuser: bool = False
) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        email: User's email address
        password: Plain text password (will be hashed)
        full_name: User's full name
        role: User's role (default: scientist)
        is_superuser: Whether user is a superuser
        
    Returns:
        Created User object
    """
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    
    user = User(
        email=email,
        name=full_name,
        role=role if not is_superuser else "admin",
        password_hash=hashed_password,
        password_history=[hashed_password.hex()],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Write audit event
    write_audit_event(
        db=db,
        actor_id="system",
        event_type="USER_CREATED",
        entity_type="user",
        entity_id=email,
        payload={"role": user.role, "name": full_name}
    )
    
    return user


def authenticate_user(
    db: Session,
    email: str,
    password: str,
    mfa_token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Authenticate user with email and password.
    
    Args:
        db: Database session
        email: User's email address
        password: Plain text password
        mfa_token: Optional MFA token if MFA is enabled
        
    Returns:
        User info dict if authenticated, None if failed, or dict with mfa_required=True
    """
    user = get_user_by_email(db, email)
    
    if not user:
        write_audit_event(
            db=db,
            actor_id="system",
            event_type="LOGIN_FAILED",
            entity_type="user",
            entity_id=email,
            payload={"reason": "user_not_found"}
        )
        return None
    
    # Check account lock
    if user.account_locked_until:
        if user.account_locked_until > datetime.utcnow():
            write_audit_event(
                db=db,
                actor_id=email,
                event_type="LOGIN_FAILED",
                entity_type="user",
                entity_id=email,
                payload={"reason": "account_locked"}
            )
            return None
        else:
            # Lock expired, reset
            user.account_locked_until = None
            user.failed_login_attempts = 0
            db.commit()

    # Verify password
    try:
        if not bcrypt.checkpw(password.encode(), user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            
            # Check if should lock
            if user.failed_login_attempts >= 5:
                user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
                write_audit_event(
                    db=db,
                    actor_id="system",
                    event_type="ACCOUNT_LOCKED",
                    entity_type="user",
                    entity_id=email,
                    payload={"reason": f"Account locked after {user.failed_login_attempts} failed attempts"}
                )
            else:
                write_audit_event(
                    db=db,
                    actor_id=email,
                    event_type="LOGIN_FAILED",
                    entity_type="user",
                    entity_id=email,
                    payload={"reason": "invalid_password", "attempts": user.failed_login_attempts}
                )
            
            db.commit()
            return None
    except Exception:
        # Handle hashing errors or malformed hashes safely
        write_audit_event(
            db=db,
            actor_id="system",
            event_type="LOGIN_FAILED",
            entity_type="user",
            entity_id=email,
            payload={"reason": "password_verification_error"}
        )
        return None

    # Success - reset counters if there were previous failed attempts or a lock
    if user.failed_login_attempts > 0 or user.account_locked_until:
        user.failed_login_attempts = 0
        user.account_locked_until = None
        db.commit()
        write_audit_event(
            db=db,
            actor_id=email,
            event_type="LOGIN_FAILED",
            entity_type="user",
            entity_id=email,
            payload={"reason": "invalid_password", "attempts": user.failed_login_attempts}
        )
        return None
    
    # Check MFA if enabled
    if user.mfa_secret:
        if not mfa_token:
            write_audit_event(
                db=db,
                actor_id=email,
                event_type="MFA_REQUIRED",
                entity_type="user",
                entity_id=email,
                payload={}
            )
            return {"mfa_required": True}
        
        totp = pyotp.TOTP(user.mfa_secret)
        if not totp.verify(mfa_token):
            write_audit_event(
                db=db,
                actor_id=email,
                event_type="LOGIN_FAILED",
                entity_type="user",
                entity_id=email,
                payload={"reason": "invalid_mfa"}
            )
            return None
    
    # Reset failed login attempts
    user.failed_login_attempts = 0
    user.account_locked_until = None
    db.commit()
    
    write_audit_event(
        db=db,
        actor_id=email,
        event_type="LOGIN_SUCCESS",
        entity_type="user",
        entity_id=email,
        payload={"role": user.role}
    )
    
    return {
        "email": user.email,
        "name": user.name,
        "role": user.role
    }


def enable_mfa(db: Session, email: str) -> Optional[str]:
    """
    Enable MFA for a user and return the secret.
    
    Args:
        db: Database session
        email: User's email address
        
    Returns:
        MFA secret or None if user not found
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    
    secret = pyotp.random_base32()
    user.mfa_secret = secret
    db.commit()
    
    write_audit_event(
        db=db,
        actor_id=email,
        event_type="MFA_ENABLED",
        entity_type="user",
        entity_id=email,
        payload={}
    )
    
    return secret


# Legacy class wrapper for backwards compatibility
class Users:
    """Legacy wrapper for backwards compatibility with existing code."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
    
    def create(self, email: str, name: str, role: str, password: str):
        """Create user - legacy interface."""
        if self.db:
            create_user(self.db, email, password, name, role)
    
    def enable_mfa(self, email: str) -> str:
        """Enable MFA - legacy interface."""
        if self.db:
            return enable_mfa(self.db, email) or ""
        return ""
    
    def authenticate(self, email: str, password: str, mfa_token: Optional[str] = None):
        """Authenticate - legacy interface."""
        if self.db:
            return authenticate_user(self.db, email, password, mfa_token)
        return None
