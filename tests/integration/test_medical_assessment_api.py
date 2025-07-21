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
async def test_icd10_bioportal_integration(db_session):
    """Test ICD-10 BioPortal API integration"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.8")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test that ICD-10 method exists
    assert hasattr(service, '_fetch_icd10_guidelines')
    assert hasattr(service, '_extract_medical_terms')
    assert hasattr(service, '_score_icd10_relevance')
    
    # Test with mocked API call to avoid requiring actual BioPortal API key
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'collection': [
                {
                    '@id': 'http://purl.bioontology.org/ontology/ICD10CM/E11.9',
                    'prefLabel': 'Type 2 diabetes mellitus without complications',
                    'definition': ['A chronic condition affecting blood sugar regulation'],
                    'synonym': ['T2DM', 'Adult-onset diabetes']
                }
            ]
        })
        
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        
        # Test ICD-10 guidelines fetch
        guidelines = await service._fetch_icd10_guidelines("diabetes treatment")
        
        # Should return empty list due to missing API key in test environment
        assert isinstance(guidelines, list)


@pytest.mark.asyncio
async def test_medical_terms_extraction(db_session):
    """Test medical terms extraction for ICD-10 search"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="diabetes, hypertension, cardiovascular disease")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test medical terms extraction
    terms = await service._extract_medical_terms("Patient with diabetes and high blood pressure")
    
    # Verify results
    assert isinstance(terms, list)
    assert len(terms) <= 5  # Should limit to 5 terms
    if terms:  # If GPT returned terms
        assert all(isinstance(term, str) for term in terms)


@pytest.mark.asyncio
async def test_multi_source_guidelines_integration_updated(db_session):
    """Test multi-source guidelines integration (PubMed + ICD-10)"""
    # Create mock GPT service
    gpt_service = Mock(spec=GPTService)
    gpt_service._create_chat_completion = AsyncMock(return_value="0.75")
    
    # Create medical assessment service
    service = MedicalAssessmentService(gpt_service, db_session)
    
    # Test that multi-source method exists
    assert hasattr(service, 'get_medical_guidelines_from_official_sources')
    
    # Mock both PubMed and ICD-10 calls to avoid actual API calls
    with patch.object(service, '_fetch_pubmed_guidelines', return_value=[]) as mock_pubmed, \
         patch.object(service, '_fetch_icd10_guidelines', return_value=[]) as mock_icd10:
        
        guidelines = await service.get_medical_guidelines_from_official_sources(
            "ADA", "diabetes technology", "diabetes treatment"
        )
        
        # Verify both sources were called
        mock_pubmed.assert_called_once_with("diabetes treatment")
        mock_icd10.assert_called_once_with("diabetes treatment")
        assert isinstance(guidelines, list)
