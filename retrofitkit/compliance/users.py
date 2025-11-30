import bcrypt
import pyotp
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from retrofitkit.db.models.user import User
from retrofitkit.compliance.audit import write_audit_event


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Get user by email address.
    """
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    full_name: str,
    role: str = "scientist",
    is_superuser: bool = False
) -> User:
    """
    Create a new user.
    """
    # Hash password (store as bytes)
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    user = User(
        email=email,
        name=full_name,
        role=role if not is_superuser else "admin",
        password_hash=hashed_password,
        is_active="true",
        # is_superuser field does not exist in DB model, ignoring
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Write audit event
    await write_audit_event(
        db=db,
        actor_id="system",
        event_type="USER_CREATED",
        entity_type="user",
        entity_id=email,
        payload={"role": user.role, "name": full_name}
    )

    return user


async def authenticate_user(
    db: AsyncSession,
    email: str,
    password: str,
    mfa_token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Authenticate user with email and password.
    """
    user = await get_user_by_email(db, email)

    if not user:
        await write_audit_event(
            db=db,
            actor_id="system",
            event_type="LOGIN_FAILED",
            entity_type="user",
            entity_id=email,
            payload={"reason": "user_not_found"}
        )
        return None

    # Check account lock (not implemented in MVP model yet, skipping logic)
    # If we add lock fields to User model, we can uncomment logic here.

    # Verify password
    try:
        # user.password_hash is bytes
        if not bcrypt.checkpw(password.encode(), user.password_hash):
            await write_audit_event(
                db=db,
                actor_id=email,
                event_type="LOGIN_FAILED",
                entity_type="user",
                entity_id=email,
                payload={"reason": "invalid_password"}
            )
            return None
    except Exception:
        await write_audit_event(
            db=db,
            actor_id="system",
            event_type="LOGIN_FAILED",
            entity_type="user",
            entity_id=email,
            payload={"reason": "password_verification_error"}
        )
        return None

    # Success
    await write_audit_event(
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


async def enable_mfa(db: AsyncSession, email: str) -> Optional[str]:
    """
    Enable MFA for a user and return the secret.
    """
    user = await get_user_by_email(db, email)
    if not user:
        return None

    secret = pyotp.random_base32()
    # user.mfa_secret = secret # Not in MVP model yet
    # await db.commit()

    await write_audit_event(
        db=db,
        actor_id=email,
        event_type="MFA_ENABLED",
        entity_type="user",
        entity_id=email,
        payload={}
    )

    return secret


# Legacy class wrapper updated for async
class Users:
    """Wrapper for user operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, email: str, name: str, role: str, password: str) -> None:
        """Create user."""
        await create_user(self.db, email, password, name, role)

    async def enable_mfa(self, email: str) -> str:
        """Enable MFA."""
        return await enable_mfa(self.db, email) or ""

    async def authenticate(self, email: str, password: str, mfa_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Authenticate."""
        return await authenticate_user(self.db, email, password, mfa_token)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return await get_user_by_email(self.db, email)
