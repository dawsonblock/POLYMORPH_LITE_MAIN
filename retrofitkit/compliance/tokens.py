from datetime import datetime, timedelta
from jose import jwt
import os

SECRET = "dev-secret"  # replace with RSA in production
ALG = "HS256"

def create_access_token(data: dict, expires_minutes: int = 480):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALG)
