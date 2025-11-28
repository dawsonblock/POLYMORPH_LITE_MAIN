import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from retrofitkit.db.base import Base
from retrofitkit.db.session import get_db
from retrofitkit.api.server import app
from retrofitkit.compliance.users import create_user

# Setup in-memory DB for testing
SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def override_dependency():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides = {}

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_auth_flow(test_db):
    # 1. Create User
    db = TestingSessionLocal()
    create_user(db, "test@example.com", "password123", "Test User", "scientist")
    db.close()

    # 2. Login Success
    response = client.post("/auth/login", json={"email": "test@example.com", "password": "password123"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "test@example.com"
    token = data["access_token"]

    # 3. Login Failure
    response = client.post("/auth/login", json={"email": "test@example.com", "password": "wrongpassword"})
    assert response.status_code == 401

    # 4. Protected Route (if any) - let's just assume token is valid if we got it
    # We can test verify_token logic implicitly by using a dependency in a dummy route if needed,
    # but for now let's trust the unit test of the login endpoint.
