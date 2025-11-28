import pytest
import os
import sys

# Force SQLite for testing
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["P4_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["P4_ENVIRONMENT"] = "testing"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retrofitkit.db.session import engine, SessionLocal
from retrofitkit.db.base import Base
# Import all models to ensure they are registered with Base
from retrofitkit.db.models import user, device, sample, workflow, audit, rbac, org, inventory, calibration

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    from retrofitkit.db.base import Base
    from retrofitkit.db.models.user import User
    from retrofitkit.database.models import WorkflowVersion, WorkflowExecution, ConfigSnapshot
    
    """Create tables for all tests."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Debug: Check if tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    print(f"DEBUG: Created tables: {inspector.get_table_names()}")
    
    # Generate dummy keys for signature tests
    import os
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    
    key_dir = "data/config/keys"
    os.makedirs(key_dir, exist_ok=True)
    private_key_path = os.path.join(key_dir, "private.pem")
    public_key_path = os.path.join(key_dir, "public.pem")
    
    if not os.path.exists(private_key_path):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with open(private_key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(public_key_path, "wb") as f:
            f.write(key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    
    yield
    # Drop tables
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    """Provide a transactional scope for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    
    yield session
    
    session.close()
    # Rollback any uncommitted changes
    transaction.rollback()
    connection.close()
    
    # Explicitly clean up all tables since we use StaticPool
    # This ensures a clean state for the next test even if commits happened
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
        conn.commit()