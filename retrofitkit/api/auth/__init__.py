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

from retrofitkit.config import settings

# OIDC Settings (Defaults)
if not hasattr(settings, "OIDC_ENABLED"):
    settings.OIDC_ENABLED = False
    settings.OIDC_CLIENT_ID = "placeholder_client_id"
    settings.OIDC_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    settings.API_BASE_URL = "http://localhost:8001/api/v1"

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

# --- OIDC Skeleton ---
from fastapi.responses import RedirectResponse
from retrofitkit.config import settings

@router.get("/login/oidc")
async def login_oidc():
    """
    Initiate OIDC login flow.
    Redirects to the Identity Provider (IdP).
    """
    if not settings.OIDC_ENABLED:
        raise HTTPException(status_code=501, detail="OIDC not enabled")
        
    # Skeleton: Construct redirect URL
    # In production, use authlib.oauth2.client
    # return await oauth.google.authorize_redirect(request, redirect_uri)
    
    client_id = settings.OIDC_CLIENT_ID
    idp_url = settings.OIDC_AUTH_URL
    redirect_uri = f"{settings.API_BASE_URL}/auth/callback/oidc"
    
    # Mock redirect for now
    target = f"{idp_url}?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope=openid email profile"
    return RedirectResponse(url=target)

@router.get("/callback/oidc")
async def callback_oidc(code: str, db: AsyncSession = Depends(get_db_session)):
    """
    Handle OIDC callback.
    Exchange code for token, find/create user, issue JWT.
    """
    if not settings.OIDC_ENABLED:
        raise HTTPException(status_code=501, detail="OIDC not enabled")

    # Skeleton: Exchange code for token
    # token = await oauth.google.authorize_access_token(request)
    # user_info = token.get('userinfo')
    
    # Mock User Info
    user_info = {
        "email": "mock_oidc_user@example.com",
        "name": "Mock OIDC User",
        "role": "operator" # Default role
    }
    
    # Logic:
    # 1. Check if user exists in DB
    # 2. If not, create (JIT provisioning) or reject
    # 3. Issue local JWT
    
    # For skeleton, just return a mock success
    return {
        "message": "OIDC Login Successful (Skeleton)",
        "user": user_info,
        "local_token": "mock_jwt_token"
    }
