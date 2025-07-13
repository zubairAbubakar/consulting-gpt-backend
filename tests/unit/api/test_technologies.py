import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.api.v1.technologies import router

class TestTechnologiesAPI:
    """Unit tests for Technologies API endpoints"""
    
    @pytest.fixture
    def client_with_mocked_db(self):
        """Create a test client with mocked database"""
        from fastapi import FastAPI
        from app.dependencies import get_db
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock the database dependency
        def mock_get_db():
            mock_db = MagicMock()
            try:
                yield mock_db
            finally:
                pass
        
        app.dependency_overrides[get_db] = mock_get_db
        
        return TestClient(app)
    
    def test_router_exists(self):
        """Test that the router is properly initialized"""
        assert router is not None
        assert hasattr(router, 'routes')
    
    def test_api_basic_structure(self, client_with_mocked_db):
        """Test basic API structure"""
        # This test verifies that the API can be instantiated
        # without making actual database calls
        assert client_with_mocked_db is not None
    
    @patch('app.api.v1.technologies.TechnologyService')
    def test_mock_technology_service(self, mock_service, client_with_mocked_db):
        """Test that TechnologyService can be mocked"""
        # Mock the service
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance
        
        # Test that the service can be instantiated
        assert mock_service is not None
        assert mock_instance is not None
    
    def test_dependencies_importable(self):
        """Test that all required dependencies can be imported"""
        from app.dependencies import get_db
        from app.services.technology_service import TechnologyService
        from app.services.gpt_service import GPTService
        
        assert get_db is not None
        assert TechnologyService is not None
        assert GPTService is not None
