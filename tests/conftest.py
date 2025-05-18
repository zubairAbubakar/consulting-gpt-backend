import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test configuration
TEST_CONFIG = {
    "PUBMED_API_KEY": "test_api_key",  # Mock API key for testing
    "DATABASE_URL": "sqlite:///:memory:"
}

@pytest.fixture(scope="session")
def engine():
    return create_engine(TEST_CONFIG["DATABASE_URL"])

@pytest.fixture(scope="session")
def TestingSessionLocal(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db(TestingSessionLocal):
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def paper_service(db):
    from app.services.paper_service import PaperService
    service = PaperService(db)
    service.api_key = TEST_CONFIG["PUBMED_API_KEY"]
    return service

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
        }
    }

@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests"""
    with patch("app.core.config.settings") as mock_settings:
        # Set required attributes
        mock_settings.DATABASE_URL = "sqlite:///:memory:"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.SERPAPI_API_KEY = "test-key"
        yield mock_settings