from datetime import datetime, timedelta
from jose import jwt
import os

# FIX: Secure Token Handling
ENV = os.getenv("ENVIRONMENT", "development")
SECRET = os.getenv("SECRET_KEY")

if not SECRET:
    if ENV == "production":
        raise RuntimeError("FATAL: SECRET_KEY environment variable not set in Production mode.")
    else:
        print("WARNING: Using default insecure key for development.")
        SECRET = "dev-secret-do-not-use-in-prod"

ALG = "HS256"

def create_access_token(data: dict, expires_minutes: int = 480):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALG)
