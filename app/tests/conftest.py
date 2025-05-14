import pytest
from unittest.mock import patch
import os

@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests"""
    with patch("app.core.config.settings") as mock_settings:
        # Set required attributes
        mock_settings.DATABASE_URL = "sqlite:///:memory:"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.SERPAPI_API_KEY = "test-key"
        yield mock_settings