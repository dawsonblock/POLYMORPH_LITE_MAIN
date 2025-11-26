"""
Production-ready audit logging with PostgreSQL persistence.

Migrated from SQLite to PostgreSQL for:
- Production reliability
- Container restart persistence  
- Multi-process concurrency
- Better data integrity
"""
import os
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography import x509

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://polymorph:polymorph_pass@localhost:5432/polymorph_db"
)

# Fallback to SQLite for local development
if "postgresql" not in DATABASE_URL and "sqlite" not in DATABASE_URL:
    DB_DIR = os.environ.get("P4_DATA_DIR", "data")
    os.makedirs(DB_DIR, exist_ok=True)
    DATABASE_URL = f"sqlite:///{os.path.join(DB_DIR, 'audit.db')}"

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class AuditLog(Base):
    """SQLAlchemy ORM model for audit logs."""
    __tablename__ = 'audit'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(Float, nullable=False)
    event = Column(String(255), nullable=False)
    actor = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    details = Column(Text, nullable=True)
    prev_hash = Column(String(64), nullable=True)
    hash = Column(String(64), nullable=False)
    signature = Column(LargeBinary, nullable=True)
    public_key = Column(LargeBinary, nullable=True)
    ca_cert = Column(LargeBinary, nullable=True)
    meaning = Column(Text, nullable=True)


# Create tables
Base.metadata.create_all(engine)


class Audit:
    """
    Production-ready audit logging with PostgreSQL persistence.
    
    Supports both PostgreSQL (production) and SQLite (local dev).
    """
    
    def __init__(self):
        self._init()
    
    def _init(self):
        """Initialize database tables (already handled by Base.metadata.create_all)."""
        # Tables are created automatically by SQLAlchemy
        pass
    
    def log(self, event: str, actor: str, subject: str, details: str = "") -> int:
        """
        Log an audit event with chain-of-custody hash.
        
        Args:
            event: Event type (e.g., "RECIPE_START", "USER_LOGIN")
            actor: Who performed the action
            subject: What was affected
            details: Additional context (JSON string)
            
        Returns:
            Audit log entry ID
        """
        session = SessionLocal()
        try:
            ts = time.time()
            
            # Get previous hash for chain-of-custody
            prev_entry = session.query(AuditLog).order_by(AuditLog.id.desc()).first()
            prev_hash = prev_entry.hash if prev_entry else "GENESIS"
            
            # Compute current hash
            data = f"{ts}{event}{actor}{subject}{details}{prev_hash}"
            current_hash = hashlib.sha256(data.encode()).hexdigest()
            
            # Create new entry
            entry = AuditLog(
                ts=ts,
                event=event,
                actor=actor,
                subject=subject,
                details=details,
                prev_hash=prev_hash,
                hash=current_hash
            )
            
            session.add(entry)
            session.commit()
            return entry.id
            
        finally:
            session.close()
    
    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs.
        
        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of audit log entries as dictionaries
        """
        session = SessionLocal()
        try:
            entries = session.query(AuditLog)\
                .order_by(AuditLog.id.desc())\
                .limit(limit)\
                .offset(offset)\
                .all()
            
            return [
                {
                    "id": e.id,
                    "ts": e.ts,
                    "event": e.event,
                    "actor": e.actor,
                    "subject": e.subject,
                    "details": e.details,
                    "hash": e.hash,
                }
                for e in entries
            ]
        finally:
            session.close()

class CompliantAuditTrail(Audit):
    def __init__(self, ca_key: Optional[rsa.RSAPrivateKey] = None, ca_cert: Optional[x509.Certificate] = None):
        super().__init__()
        self.ca_key = ca_key
        self.ca_cert = ca_cert

    def log_event_signed(self, event_type: str, data: dict, user_id: str, user_key: rsa.RSAPrivateKey, meaning: str = "Approved"):
        prev = self._last_hash()
        # Build record
        record = {**data, "timestamp": datetime.utcnow().isoformat(), "user": user_id, "meaning": meaning, "prev_hash": prev}
        record_json = json.dumps(record, sort_keys=True)

        # Hash + sign
        digest = hashes.Hash(hashes.SHA256())
        digest.update(record_json.encode())
        signature = user_key.sign(digest.finalize(), padding.PKCS1v15(), hashes.SHA256())
        
        # Calculate chain hash
        h = hashlib.sha256((prev + record_json).encode()).hexdigest()

        # Store with certificate
        con = sqlite3.connect(DB)
        con.execute("""INSERT INTO audit (ts, event, actor, subject, details, prev_hash, hash, signature, public_key, ca_cert, meaning) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (time.time(), event_type, user_id, "signed_event", record_json, prev, h, signature, 
                     user_key.public_key().public_bytes(encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo),
                     self.ca_cert.public_bytes(encoding=serialization.Encoding.PEM) if self.ca_cert else None,
                     meaning))
        con.commit()
        con.close()
