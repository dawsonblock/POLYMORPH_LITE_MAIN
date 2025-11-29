import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from retrofitkit.compliance.users import authenticate_user, create_user
from retrofitkit.db.models.user import User
from retrofitkit.compliance.tokens import create_access_token, decode_token

# ... (rest of imports)

# --- JWT Tests ---
def test_jwt_expiry():
    # Create token with short expiry
    data = {"sub": "test@example.com", "role": "scientist"}
    token = create_access_token(data, expires_minutes=-1) # Expired
    
    # Verify should fail (raise exception or return None depending on implementation)
    # decode_token raises HTTPException, so we should expect that or catch it
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        decode_token(token)

def test_jwt_valid():
    data = {"sub": "test@example.com", "role": "scientist"}
    token = create_access_token(data, expires_minutes=5)
    
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "test@example.com"
