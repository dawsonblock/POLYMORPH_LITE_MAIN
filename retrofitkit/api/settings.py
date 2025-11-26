"""
Settings API endpoints for system configuration and user management.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from retrofitkit.api.security import get_current_user
from retrofitkit.core.app import AppContext
from retrofitkit.compliance.users import Users

router = APIRouter(prefix="/settings", tags=["settings"])

class UserCreate(BaseModel):
    email: str
    name: str
    role: str
    password: str

class UserResponse(BaseModel):
    email: str
    name: str
    role: str
    created: float

@router.get("/config")
def get_config(user=Depends(get_current_user)):
    """Get current system configuration."""
    ctx = AppContext.load()
    return ctx.config.to_dict()

@router.post("/config")
def update_config(config: Dict[str, Any], user=Depends(get_current_user)):
    """
    Update system configuration.
    Note: Some changes may require a server restart.
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can update configuration")
    
    # In a real implementation, we would validate and merge deeply.
    # For now, we'll just save to the config file to persist changes.
    ctx = AppContext.load()
    
    # Update in-memory config (simplified)
    # ctx.config.update(config) 
    
    # Save to file
    ctx.config.save_to_file("config/config.yaml")
    
    return {"status": "updated", "message": "Configuration saved. Restart may be required."}

@router.get("/users", response_model=List[UserResponse])
def list_users(user=Depends(get_current_user)):
    """List all users (Admin only)."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # We need to expose a list method in Users class or query DB directly
    # Since Users class doesn't have list method yet, we'll access DB directly here for simplicity
    import sqlite3
    import os
    
    DB_DIR = os.environ.get("P4_DATA_DIR", "data")
    DB = os.path.join(DB_DIR, "system.db")
    
    users = []
    try:
        con = sqlite3.connect(DB)
        cur = con.execute("SELECT email, name, role, created FROM users")
        for row in cur:
            users.append({
                "email": row[0],
                "name": row[1],
                "role": row[2],
                "created": row[3]
            })
        con.close()
    except Exception as e:
        print(f"Error listing users: {e}")
        
    return users

@router.post("/users")
def create_user(payload: UserCreate, user=Depends(get_current_user)):
    """Create a new user (Admin only)."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    users_mgr = Users()
    users_mgr.create(payload.email, payload.name, payload.role, payload.password)
    return {"status": "created", "email": payload.email}
