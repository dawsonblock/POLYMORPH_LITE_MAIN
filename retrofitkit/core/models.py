from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime, timezone

Base = declarative_base()

class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    definition = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(String, default="true") # "true" or "false"

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, unique=True, index=True, nullable=False)
    workflow_version_id = Column(String, ForeignKey("workflow_versions.id"), nullable=False)
    status = Column(String, index=True, default="pending")
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    operator = Column(String, nullable=True) # Email or ID
    
    # Relationships
    workflow_version = relationship("WorkflowVersion")

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(String, default="true")
    is_superuser = Column(String, default="false")
    role = Column(String, default="user")

class AuditLog(Base):
    __tablename__ = "audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(Float, default=lambda: datetime.now(timezone.utc).timestamp(), index=True)
    event = Column(String, nullable=False, index=True)
    actor = Column(String, nullable=False, index=True)
    subject = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    hash = Column(String, nullable=False, index=True)
    prev_hash = Column(String, nullable=True)
