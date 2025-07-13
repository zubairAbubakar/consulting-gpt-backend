import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from httpx import AsyncClient

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test configuration
TEST_CONFIG = {
    "PUBMED_API_KEY": "test_api_key",
    "OPENAI_API_KEY": "test_openai_key",
    "SERPAPI_API_KEY": "test_serpapi_key",
    "DATABASE_URL": "sqlite:///:memory:",
    "TEST_DATABASE_URL": "sqlite:///./test.db"
}

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(
        TEST_CONFIG["DATABASE_URL"],
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine

@pytest.fixture(scope="session")
def TestingSessionLocal(test_engine):
    """Create test database session factory"""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture
def db_session(TestingSessionLocal, test_engine):
    """Create a fresh database session for each test"""
    # Create all tables for the test
    from app.models.base import Base
    Base.metadata.create_all(bind=test_engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Clean up: drop all tables after test
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def client(db_session):
    """Create FastAPI test client for integration testing"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from app.api.v1 import technologies
    from app.db.database import get_db
    
    # Create a test-specific FastAPI app without lifespan events
    test_app = FastAPI(
        title="Consulting GPT API - Test",
        description="Test version of backend API for Consulting GPT application",
        version="1.0.0"
    )
    
    # Configure CORS
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include only the technologies router for testing
    test_app.include_router(technologies.router, prefix="/api/v1/technologies", tags=["technologies"])
    
    @test_app.get("/")
    async def root():
        return {"message": "Welcome to Consulting GPT API - Test"}
    
    # Override the database dependency to use our test database
    def override_get_db():
        yield db_session
    
    test_app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(test_app) as test_client:
        yield test_client
    
    # Clean up override after test
    test_app.dependency_overrides.clear()

@pytest.fixture
async def async_client():
    """Create async FastAPI test client"""
    from main import app
    async with AsyncClient(app=app, base_url="http://test") as async_test_client:
        yield async_test_client

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_paper_service_responses():
    """Fixture to provide common mock responses"""
    return {
        "search_response": {
            "esearchresult": {
                "count": "2",
                "retmax": "2",
                "idlist": ["123456", "789012"]
            }
        },
        "fetch_response": {
            "articles": [
                {
                    "title": "Test Article 1",
                    "abstract": "Test abstract 1",
                    "authors": ["Author 1"],
                    "journal": "Test Journal",
                    "pub_date": "2023-01-01",
                    "pubmed_id": "123456"
                },
                {
                    "title": "Test Article 2", 
                    "abstract": "Test abstract 2",
                    "authors": ["Author 2"],
                    "journal": "Test Journal 2",
                    "pub_date": "2023-01-02",
                    "pubmed_id": "789012"
                }
            ]
        }
    }

@pytest.fixture
def mock_patent_service_responses():
    """Fixture to provide common mock patent responses"""
    return {
        "search_response": {
            "patents": [
                {
                    "title": "Test Patent 1",
                    "abstract": "Test patent abstract 1",
                    "inventors": ["Inventor 1"],
                    "patent_number": "US123456",
                    "publication_date": "2023-01-01"
                },
                {
                    "title": "Test Patent 2",
                    "abstract": "Test patent abstract 2", 
                    "inventors": ["Inventor 2"],
                    "patent_number": "US789012",
                    "publication_date": "2023-01-02"
                }
            ]
        }
    }

@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests"""
    with patch("app.core.config.settings") as mock_settings:
        # Set required attributes
        mock_settings.DATABASE_URL = TEST_CONFIG["DATABASE_URL"]
        mock_settings.OPENAI_API_KEY = TEST_CONFIG["OPENAI_API_KEY"]
        mock_settings.SERPAPI_API_KEY = TEST_CONFIG["SERPAPI_API_KEY"]
        # For integration tests, we need real database config
        if os.getenv("TESTING") == "true":
            mock_settings.POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "db-test")
            mock_settings.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
            mock_settings.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
            mock_settings.POSTGRES_DB = os.getenv("POSTGRES_DB", "consulting_gpt_test")
            mock_settings.SQLALCHEMY_DATABASE_URI = f"postgresql://{mock_settings.POSTGRES_USER}:{mock_settings.POSTGRES_PASSWORD}@{mock_settings.POSTGRES_SERVER}/{mock_settings.POSTGRES_DB}"
        else:
            mock_settings.SQLALCHEMY_DATABASE_URI = TEST_CONFIG["DATABASE_URL"]
        yield mock_settings

@pytest.fixture
def sample_technology_data():
    """Sample technology data for testing"""
    return {
        "name": "AI Medical Diagnostic System",
        "abstract": "An AI system for medical diagnosis",
        "num_of_axes": 5
    }

# Legacy fixtures for backward compatibility
@pytest.fixture(scope="session")
def engine(test_engine):
    return test_engine

@pytest.fixture
def db(db_session):
    return db_session

@pytest.fixture
def paper_service(db):
    from app.services.paper_service import PaperService
    service = PaperService(db)
    service.api_key = TEST_CONFIG["PUBMED_API_KEY"]
    return service