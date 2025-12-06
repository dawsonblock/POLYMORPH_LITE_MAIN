from fastapi import HTTPException, Header
import jwt
from jwt import PyJWTError as JWTError
from retrofitkit.compliance.tokens import SECRET_KEY, ALG

def get_current_user(authorization: str = Header(default=None, alias="Authorization")):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALG])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    email = payload.get("sub") or payload.get("email")
    role = payload.get("role") or "Operator"
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return {"email": email, "role": role}
