import pytest
from unittest.mock import Mock, AsyncMock
from app.services.patent_service import PatentService
from app.models.technology import PatentSearch, PatentResult

@pytest.fixture
def db():
    """Mock database session"""
    return Mock()

@pytest.fixture
def mock_google_search():
    """Mock GoogleSearch response"""
    mock = Mock()
    mock.get_dict.return_value = {
        "organic_results": [
            {
                "patent_id": "US123456",
                "title": "Test Patent",
                "link": "https://patents.google.com/patent/US123456",
                "publication_date": "2025-01-01"
            }
        ]
    }
    return mock

@pytest.fixture
def patent_service(db, monkeypatch):
    """Patent service with mocked dependencies"""
    service = PatentService(db)
    service.api_key = "test_key"
    return service

@pytest.mark.asyncio
async def test_search_patents_success(patent_service, mock_google_search, monkeypatch):
    # Mock GoogleSearch
    monkeypatch.setattr("app.services.patent_service.GoogleSearch", Mock(return_value=mock_google_search))
    
    # Mock get_patent_details
    patent_service._get_patent_details = Mock(return_value={
        "abstract": "Test abstract",
        "title": "Test Patent",
        "patent_number": "US123456",
        "publication_date": "2025-01-01"
    })
    
    result = await patent_service.search_patents(1, "test query")
    
    assert result is not None
    assert isinstance(result, PatentSearch)
    assert result.technology_id == 1
    assert result.search_query == "test query"
    
    # Verify database calls
    patent_service.db.add.assert_called()
    patent_service.db.commit.assert_called()

@pytest.mark.asyncio
async def test_search_patents_error(patent_service, monkeypatch):
    # Mock GoogleSearch to raise exception
    monkeypatch.setattr("app.services.patent_service.GoogleSearch", Mock(side_effect=Exception("API Error")))
    
    result = await patent_service.search_patents(1, "test query")
    
    assert result is None
    patent_service.db.rollback.assert_called_once()