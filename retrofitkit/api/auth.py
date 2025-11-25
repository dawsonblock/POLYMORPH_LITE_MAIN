from fastapi import APIRouter, Depends, HTTPException
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
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["email"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}
