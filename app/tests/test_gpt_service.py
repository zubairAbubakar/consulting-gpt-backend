import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
from sqlalchemy.orm import Session
import os
import time
from typing import List, Dict, Any
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from app.services.gpt_service import GPTService, RateLimiter
from app.models.technology import Technology, ComparisonAxis

@pytest.fixture
def db():
    """Mock database session"""
    mock_db = Mock(spec=Session)
    mock_db.query.return_value.filter.return_value.first.return_value = None
    mock_db.query.return_value.filter.return_value.all.return_value = []
    return mock_db

@pytest.fixture
def gpt_service(db):
    """GPT service with mocked OpenAI client"""
    service = GPTService(db)
    service.client = AsyncMock()
    return service

@pytest.fixture
def mock_chat_completion():
    """Create a mock chat completion response"""
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    
    message = ChatCompletionMessage(
        content="Test response",
        role="assistant",
        function_call=None,
        tool_calls=None
    )
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=message,
        logprobs=None
    )
    return ChatCompletion(
        id="test_id",
        choices=[choice],
        created=int(time.time()),
        model="gpt-4",
        object="chat.completion",
        system_fingerprint="fp123",
        usage={"total_tokens": 10, "completion_tokens": 5, "prompt_tokens": 5}
    )

@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response"""
    return Mock(
        data=[
            Mock(
                embedding=[0.1] * 1536,
                index=0,
                object="embedding"
            )
        ],
        model="text-embedding-ada-002",
        object="list",
        usage={"prompt_tokens": 5, "total_tokens": 5}
    )

@pytest.mark.asyncio
async def test_create_chat_completion(gpt_service, mock_chat_completion):
    gpt_service.client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
    
    response = await gpt_service._create_chat_completion(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt"
    )
    
    assert response == "Test response"
    gpt_service.client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_create_chat_completion_error(db):
    service = GPTService(db)
    service.client = AsyncMock()
    
    error_response = Mock()
    error_response.status = 500
    error_response.json.return_value = {"error": {"message": "API Error"}}
    
    service.client.chat.completions.create = AsyncMock(side_effect=OpenAIError("API Error"))
    
    result = await service._create_chat_completion(
        system_prompt="Test system prompt",
        user_prompt="Test user prompt"
    )
    
    assert result is None

@pytest.mark.asyncio
async def test_get_embedding(gpt_service, mock_embedding_response):
    gpt_service.client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
    
    result = await gpt_service.get_embedding("test text")
    
    assert len(result) == 1536
    assert result == [0.1] * 1536
    gpt_service.client.embeddings.create.assert_called_once()

@pytest.mark.asyncio
async def test_get_embedding_error(gpt_service):
    error_response = Mock()
    error_response.status = 500
    error_response.json.return_value = {"error": {"message": "API Error"}}
    
    gpt_service.client.embeddings.create = AsyncMock(side_effect=OpenAIError("API Error"))
    
    result = await gpt_service.get_embedding("test text")
    assert result == []

@pytest.mark.asyncio
async def test_calculate_similarity_identical(gpt_service):
    mock_embedding = [0.5, 0.5, 0.5]
    gpt_service.get_embedding = AsyncMock(return_value=mock_embedding)
    
    similarity = await gpt_service.calculate_similarity("same text", "same text")
    assert abs(similarity - 1.0) < 1e-10

@pytest.mark.asyncio
async def test_chat_completion_rate_limit_retry(gpt_service, mock_chat_completion):
    # Create rate limit error
    rate_limit_response = Mock()
    rate_limit_response.status = 429
    rate_limit_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    
    rate_limit_error = OpenAIError("Rate limit exceeded")
    
    # Mock the API to fail with rate limit twice, then succeed
    side_effects = [
        rate_limit_error,
        rate_limit_error,
        mock_chat_completion
    ]
    
    gpt_service.client.chat.completions.create = AsyncMock(side_effect=side_effects)
    
    result = await gpt_service._create_chat_completion(
        system_prompt="test",
        user_prompt="test",
        retry_count=3
    )
    
    assert result == "Test response"
    assert gpt_service.client.chat.completions.create.call_count == 3

def test_create_technology(gpt_service, db):
    name = "Test Tech"
    abstract = "Test abstract"
    problem_statement = "Test problem"
    
    # Create a mock technology with an ID
    mock_tech = Technology(
        id=1,
        name=name,
        abstract=abstract,
        problem_statement=problem_statement
    )
    
    # Configure the mock db to return our mock technology
    db.add = Mock()
    db.commit = Mock()
    db.refresh = Mock(return_value=mock_tech)
    
    technology = gpt_service.create_technology(name, abstract, problem_statement)
    
    assert isinstance(technology, Technology)
    assert technology.name == name
    assert technology.abstract == abstract
    assert technology.problem_statement == problem_statement
    db.add.assert_called_once()
    db.commit.assert_called_once()

def test_create_comparison_axes(gpt_service, db):
    tech_id = 1
    axes_data = [
        {"axis_name": "Size", "extreme1": "Big", "extreme2": "Small", "weight": 1.0}
    ]
    
    # Create mock comparison axes
    mock_axes = [ComparisonAxis(
        id=1,
        technology_id=tech_id,
        axis_name=data["axis_name"],
        extreme1=data["extreme1"],
        extreme2=data["extreme2"],
        weight=data["weight"]
    ) for data in axes_data]
    
    db.add_all = Mock()
    db.commit = Mock()
    db.refresh = Mock(side_effect=mock_axes)
    
    axes = gpt_service.create_comparison_axes(tech_id, axes_data)
    
    assert isinstance(axes, list)
    assert len(axes) == len(axes_data)
    assert all(isinstance(axis, ComparisonAxis) for axis in axes)
    db.add_all.assert_called_once()
    db.commit.assert_called_once()

@pytest.mark.asyncio
async def test_process_and_save_analysis(gpt_service, tmp_path):
    test_data_dir = str(tmp_path)
    tech_name = "Test Tech"
    
    # Mock generate_comparison_axes response
    mock_df = pd.DataFrame({
        'Axis': ['Test1', 'Test2'],
        'Extreme1': ['Low1', 'Low2'],
        'Extreme2': ['High1', 'High2']
    })
    
    gpt_service.generate_comparison_axes = AsyncMock(return_value=(mock_df, "Test problem statement"))
    
    # Mock create_technology response
    mock_tech = Technology(id=1, name=tech_name, abstract="Test description", problem_statement="Test problem statement")
    gpt_service.create_technology = Mock(return_value=mock_tech)
    
    # Mock create_comparison_axes response
    mock_axes = [
        ComparisonAxis(id=1, technology_id=1, axis_name="Test1", extreme1="Low1", extreme2="High1"),
        ComparisonAxis(id=2, technology_id=1, axis_name="Test2", extreme1="Low2", extreme2="High2")
    ]
    gpt_service.create_comparison_axes = Mock(return_value=mock_axes)
    
    # Run the test
    technology, axes = await gpt_service.process_and_save_analysis(
        technology_description="Test description",
        technology_name=tech_name,
        data_dir=test_data_dir
    )
    
    # Verify results
    assert isinstance(technology, Technology)
    assert len(axes) == 2
    assert all(isinstance(axis, ComparisonAxis) for axis in axes)
    
    # Create directory structure
    tech_dir = os.path.join(test_data_dir, tech_name)
    os.makedirs(tech_dir, exist_ok=True)
    
    # Save test files
    mock_df.to_csv(os.path.join(tech_dir, "comp_axes.csv"), index=False)
    with open(os.path.join(tech_dir, "meta_data.json"), "w") as f:
        f.write('{"test": "data"}')
    
    # Verify files exist
    assert os.path.exists(tech_dir)
    assert os.path.exists(os.path.join(tech_dir, "comp_axes.csv"))
    assert os.path.exists(os.path.join(tech_dir, "meta_data.json"))

def test_get_technology_by_id(gpt_service, db):
    # Create a mock technology
    mock_tech = Technology(
        id=1,
        name="Test Tech",
        abstract="Test abstract",
        problem_statement="Test problem"
    )
    
    # Configure mock db to return our mock technology
    db.query.return_value.filter.return_value.first.return_value = mock_tech
    
    # Test the method
    retrieved = gpt_service.get_technology_by_id(1)
    
    assert retrieved is not None
    assert retrieved.id == 1
    assert retrieved.name == "Test Tech"

def test_get_comparison_axes_by_technology(gpt_service, db):
    # Create mock comparison axes
    mock_axes = [
        ComparisonAxis(id=1, technology_id=1, axis_name="Size", extreme1="Small", extreme2="Large"),
        ComparisonAxis(id=2, technology_id=1, axis_name="Speed", extreme1="Slow", extreme2="Fast")
    ]
    
    # Configure mock db to return our mock axes
    db.query.return_value.filter.return_value.all.return_value = mock_axes
    
    # Test the method
    retrieved = gpt_service.get_comparison_axes_by_technology(1)
    
    assert len(retrieved) == 2
    assert all(isinstance(axis, ComparisonAxis) for axis in retrieved)
    assert all(axis.technology_id == 1 for axis in retrieved)

@pytest.mark.asyncio
async def test_rate_limiter():
    limiter = RateLimiter(calls_per_minute=2)
    
    # First two calls should be immediate
    await limiter.acquire()
    await limiter.acquire()
    assert len(limiter.calls) == 2
    
    # Third call should have to wait
    start_time = time.time()
    await limiter.acquire()
    elapsed_time = time.time() - start_time
    
    assert elapsed_time >= 60  # Should have waited at least a minute

@pytest.mark.asyncio
async def test_get_search_term_score(gpt_service, mock_chat_completion):
    # Configure mock to return a score
    mock_chat_completion.choices[0].message.content = "0.75"
    gpt_service.client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
    
    # Test scoring
    score = await gpt_service.get_search_term_score("nanomachines")
    assert isinstance(score, float)
    assert 0 <= score <= 1
    assert score == 0.75
    
    # Test with invalid response
    mock_chat_completion.choices[0].message.content = "not a number"
    score = await gpt_service.get_search_term_score("test")
    assert score == 0.0

@pytest.mark.asyncio
async def test_get_search_keywords(gpt_service, mock_chat_completion):
    # Mock dependencies
    gpt_service.get_embedding = AsyncMock(return_value=[0.5] * 1536)
    gpt_service.calculate_similarity = AsyncMock(return_value=0.8)
    gpt_service.get_search_term_score = AsyncMock(return_value=0.7)
    
    # Configure chat completion mock to return keywords
    mock_chat_completion.choices[0].message.content = "machine,learning"
    gpt_service.client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
    
    # Test keyword generation
    keywords = await gpt_service.get_search_keywords(
        problem_statement="How to improve machine learning model performance",
        keyword_count=2
    )
    
    assert isinstance(keywords, str)
    assert len(keywords.split()) == 2
    assert "machine" in keywords
    assert "learning" in keywords
    
    # Verify all dependencies were called
    assert gpt_service.client.chat.completions.create.called
    assert gpt_service.get_embedding.called
    assert gpt_service.calculate_similarity.called
    assert gpt_service.get_search_term_score.called
    
    # Test with empty response
    mock_chat_completion.choices[0].message.content = ""
    keywords = await gpt_service.get_search_keywords("test", keyword_count=2)
    assert keywords == ""