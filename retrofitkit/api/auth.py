from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from retrofitkit.compliance.users import Users
from retrofitkit.compliance.tokens import create_access_token

router = APIRouter()

class Login(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(payload: Login):
    user = Users().authenticate(payload.email, payload.password)
    if not user:
        # Check if it was due to lockout (optional: could be generic for security, 
        # but for internal lab OS, specific feedback is helpful)
        # We'll re-query briefly to see if locked, or just return 401.
        # Given the Users().authenticate returns None for both invalid and locked,
        # let's check lock status explicitly if we want to give a 423.
        # However, Users().authenticate handles the logic.
        # To provide 423, we'd need Users().authenticate to raise or return status.
        # For now, we'll stick to 401 to avoid enumeration, unless we want to be friendly.
        # Let's check explicitly for the specific error case.
        
        from retrofitkit.database.models import User, get_session
        from datetime import datetime
        session = get_session()
        db_user = session.query(User).filter(User.email == payload.email).first()
        if db_user and db_user.account_locked_until and db_user.account_locked_until > datetime.utcnow():
             session.close()
             raise HTTPException(
                 status_code=status.HTTP_423_LOCKED,
                 detail=f"Account locked until {db_user.account_locked_until.isoformat()}"
             )
        session.close()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or account locked"
        )
    token = create_access_token({"sub": user["email"], "role": user["role"]})
    return {
        "access_token": token, 
        "token_type": "bearer",
        "user": {
            "id": "1",  # Mock ID since we use email as PK
            "username": user["name"],
            "email": user["email"],
            "role": user["role"],
            "isActive": True
        }
    }
