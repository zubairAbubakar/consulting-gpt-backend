import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.services.medical_assessment_service import MedicalAssessmentService
from app.services.gpt_service import GPTService
from app.models.technology import Guidelines


@pytest.mark.asyncio
async def test_pubmed_guidelines_integration_simple(db_session):
    """Simple test for PubMed guidelines method existence and basic functionality"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.8")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test that the method exists and can be called
    assert hasattr(service, '_fetch_pubmed_guidelines')
    assert hasattr(service, 'get_medical_guidelines_from_pubmed')
    
    # Test with mocked network call to avoid actual API calls
    with patch.object(service, '_fetch_pubmed_guidelines', return_value=[]):
        guidelines = await service.get_medical_guidelines_from_pubmed(
            "ADA", "diabetes technology", "diabetes treatment"
        )
        assert isinstance(guidelines, list)


@pytest.mark.asyncio
async def test_xml_parsing_helper_method(db_session):
    """Test the XML parsing helper method"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test XML parsing
    test_xml = """
    <PubmedArticle>
        <MedlineCitation>
            <Article>
                <Abstract>
                    <AbstractText>Test abstract content for clinical guidelines.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """
    
    abstract = service._extract_abstract_from_xml(test_xml)
    assert "Test abstract content" in abstract


@pytest.mark.asyncio
async def test_guideline_scoring_method(db_session):
    """Test the guideline relevance scoring method"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.75")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test scoring
    score = await service._score_guideline_relevance(
        "Diabetes Treatment Guidelines",
        "Comprehensive diabetes management approach",
        "diabetes treatment options"
    )
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score == 0.75


@pytest.mark.asyncio
async def test_pubmed_integration_no_results(db_session):
    """Test PubMed API when no results are found"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Mock aiohttp response for PubMed with no results
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_search_response = Mock()
        mock_search_response.status = 200
        mock_search_response.json = AsyncMock(return_value={
            'esearchresult': {
                'idlist': []  # No results
            }
        })
        
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_search_response)
        
        # Test the PubMed integration
        guidelines = await service._fetch_pubmed_guidelines("very rare condition")
        
        # Verify no results
        assert len(guidelines) == 0


@pytest.mark.asyncio
async def test_pubmed_integration_api_error(db_session):
    """Test PubMed API error handling"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Mock aiohttp response for PubMed API error
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = Mock()
        mock_response.status = 500
        
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        
        # Test the PubMed integration should handle errors gracefully
        guidelines = await service._fetch_pubmed_guidelines("test condition")
        
        # Verify error handling
        assert len(guidelines) == 0


@pytest.mark.asyncio
async def test_cms_coverage_policies_integration(db_session):
    """Test CMS coverage policies integration"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.8")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test CMS coverage policies fetch
    guidelines = await service._fetch_cms_coverage_policies("diabetes treatment")
    
    # Verify results
    assert isinstance(guidelines, list)
    if guidelines:  # Should have some mock guidelines for diabetes
        assert len(guidelines) > 0
        assert all(g.source == "CMS" for g in guidelines)
        assert all(hasattr(g, 'relevance_score') for g in guidelines)


@pytest.mark.asyncio
async def test_multi_source_guidelines_integration(db_session):
    """Test multi-source guidelines integration (PubMed + CMS)"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.75")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test that multi-source method exists
    assert hasattr(service, 'get_medical_guidelines_from_official_sources')
    
    # Mock both PubMed and CMS calls to avoid actual API calls
    with patch.object(service, '_fetch_pubmed_guidelines', return_value=[]) as mock_pubmed, \
         patch.object(service, '_fetch_cms_coverage_policies', return_value=[]) as mock_cms:
        
        guidelines = await service.get_medical_guidelines_from_official_sources(
            "ADA", "diabetes technology", "diabetes treatment"
        )
        
        # Verify both sources were called
        mock_pubmed.assert_called_once_with("diabetes treatment")
        mock_cms.assert_called_once_with("diabetes treatment")
        assert isinstance(guidelines, list)
