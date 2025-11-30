from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import bcrypt

from retrofitkit.core.database import get_db_session
from retrofitkit.db.models.user import User
from retrofitkit.db.models.audit import AuditEvent as AuditLog
from retrofitkit.compliance.tokens import create_access_token

router = APIRouter()

class Login(BaseModel):
    email: str
    password: str

@router.post("/login")
async def login(payload: Login, db: AsyncSession = Depends(get_db_session)) -> dict:
    # Fetch user
    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        # Audit failure (mock for now or use async audit)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Check password
    # Note: bcrypt is CPU bound, should ideally run in executor, but fine for MVP
    try:
        # User.password_hash is LargeBinary (bytes)
        # payload.password is str
        if not bcrypt.checkpw(payload.password.encode(), user.password_hash):
             raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
    except Exception:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Check active
    if user.is_active != "true":
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account inactive"
        )

    # Generate Token
    token = create_access_token({"sub": user.email, "role": user.role})
    
    # Audit Success
    # audit = AuditLog(event="LOGIN_SUCCESS", actor=user.email, subject=user.email, hash="...")
    # db.add(audit)
    # await db.commit()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.email, # Use email as ID since DB model uses email as PK
            "username": user.name or user.email,
            "email": user.email,
            "role": user.role,
            "isActive": True
        }
    }
