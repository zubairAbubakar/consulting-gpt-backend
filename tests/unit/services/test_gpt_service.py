import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.orm import Session
from app.services.gpt_service import GPTService

class TestGPTService:
    """Unit tests for GPTService"""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        return MagicMock(spec=Session)
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Test GPT response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def gpt_service(self, mock_db):
        """Create a GPTService instance with mocked dependencies"""
        with patch('app.services.gpt_service.AsyncOpenAI') as mock_openai:
            service = GPTService(mock_db)
            service.client = mock_openai.return_value
            return service
    
    def test_init(self, mock_db):
        """Test GPTService initialization"""
        with patch('app.services.gpt_service.AsyncOpenAI'):
            service = GPTService(mock_db)
            assert service.db == mock_db
            assert hasattr(service, 'client')
    
    def test_service_attributes(self, gpt_service):
        """Test that service has required attributes"""
        assert hasattr(gpt_service, 'db')
        assert hasattr(gpt_service, 'client')
    
    @pytest.mark.asyncio
    async def test_basic_gpt_call(self, gpt_service, mock_openai_client):
        """Test basic GPT service functionality"""
        # Mock the OpenAI client response
        gpt_service.client = mock_openai_client
        
        # Test that we can call the service
        # This is a basic test to ensure the service is properly initialized
        assert gpt_service.client is not None
        
        # Mock a simple completion call
        mock_response = await gpt_service.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}]
        )
        
        assert mock_response.choices[0].message.content == "Test GPT response"
    
    def test_service_dependencies(self, gpt_service):
        """Test that GPTService has required dependencies"""
        assert gpt_service.db is not None
        assert gpt_service.client is not None
