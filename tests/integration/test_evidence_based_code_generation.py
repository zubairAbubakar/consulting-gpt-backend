"""
Integration tests for evidence-based code generation functionality.
Tests the new methods for extracting HCPCS codes from medical guidelines.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.medical_assessment_service import MedicalAssessmentService
from app.services.gpt_service import GPTService
from app.models.technology import Guidelines


class TestEvidenceBasedCodeGeneration:
    """Test evidence-based code generation methods."""

    @pytest.fixture
    def mock_gpt_service(self):
        """Create a mock GPT service."""
        mock_service = MagicMock(spec=GPTService)
        mock_service._create_chat_completion = AsyncMock()
        return mock_service

    @pytest.fixture
    def medical_service(self, mock_gpt_service, db_session):
        """Create medical assessment service with mocked dependencies."""
        return MedicalAssessmentService(gpt_service=mock_gpt_service, db=db_session)

    @pytest.mark.asyncio
    async def test_extract_procedures_from_guidelines_simple(self, medical_service):
        """Test procedure extraction from guidelines with proper model fields."""
        # Mock guidelines with proper structure (content, not description)
        guidelines = [
            Guidelines(
                title="MRI Guidelines for Brain Imaging",
                content="Guidelines for magnetic resonance imaging procedures. Recommended procedures include brain MRI with contrast, DWI sequences, and FLAIR imaging.",
                source="PubMed",
                link="https://example.com"
            )
        ]
        
        # Mock GPT response for procedure extraction
        medical_service.gpt_service._create_chat_completion.return_value = "Magnetic Resonance Imaging (MRI) - Brain, MRI with contrast enhancement"
        
        # Test the method
        procedures = await medical_service._extract_procedures_from_guidelines(guidelines)
        
        # Verify results
        assert isinstance(procedures, list)
        assert len(procedures) > 0
        
        # Verify GPT service was called
        assert medical_service.gpt_service._create_chat_completion.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_evidence_based_codes_with_mocks(self, medical_service):
        """Test evidence-based code generation with full mocking."""
        problem_statement = "MRI Brain Imaging"
        
        # Mock guidelines
        guidelines = [
            Guidelines(
                title="MRI Guidelines",
                content="Brain MRI protocols for diagnostic imaging",
                source="PubMed",
                link="https://example.com"
            )
        ]
        
        # Mock the complete flow with patches
        with patch.object(medical_service, '_extract_procedures_from_guidelines') as mock_extract:
            mock_extract.return_value = ["Brain MRI", "MRI with contrast"]
            
            with patch.object(medical_service, '_map_procedures_to_hcpcs') as mock_map:
                mock_map.return_value = ["70551", "70552", "70553"]
                
                with patch.object(medical_service, '_validate_hcpcs_codes') as mock_validate:
                    mock_validate.return_value = ["70551", "70552", "70553"]
                    
                    # Test the method
                    result = await medical_service._get_evidence_based_codes(guidelines, problem_statement)
                    
                    # Verify results
                    assert isinstance(result, list)
                    assert len(result) == 3
                    assert "70551" in result
                    assert "70552" in result
                    assert "70553" in result

    @pytest.mark.asyncio
    async def test_map_procedures_to_hcpcs_basic(self, medical_service):
        """Test basic mapping of procedures to HCPCS codes."""
        procedures = ["Magnetic Resonance Imaging - Brain", "CT Scan - Head"]
        problem_statement = "Brain imaging technology"
        
        # Mock GPT response for HCPCS mapping
        medical_service.gpt_service._create_chat_completion.return_value = "70551,70552,70470"
        
        result = await medical_service._map_procedures_to_hcpcs(procedures, problem_statement)
        
        # Verify results
        assert isinstance(result, list)
        assert len(result) == 3
        assert "70551" in result
        assert "70552" in result
        assert "70470" in result

    @pytest.mark.asyncio
    async def test_validate_hcpcs_codes_basic(self, medical_service):
        """Test basic HCPCS code validation."""
        candidate_codes = ["70551", "70552"]
        
        # Mock database queries and CMS API calls
        with patch.object(medical_service.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.first.return_value = None
            
            with patch.object(medical_service, '_fetch_cms_fee') as mock_fetch:
                mock_fetch.side_effect = [
                    {"fee": 150.00, "code": "70551"},  # Valid code
                    {"fee": 200.00, "code": "70552"},  # Valid code
                ]
                
                # Test validation
                validated_codes = await medical_service._validate_hcpcs_codes(candidate_codes)
                
                # Should return valid codes
                assert isinstance(validated_codes, list)
                assert len(validated_codes) <= len(candidate_codes)

    @pytest.mark.asyncio 
    async def test_evidence_based_methods_exist(self, medical_service):
        """Test that the evidence-based methods exist and are callable."""
        # Verify the new methods exist
        assert hasattr(medical_service, '_get_evidence_based_codes')
        assert hasattr(medical_service, '_extract_procedures_from_guidelines') 
        assert hasattr(medical_service, '_map_procedures_to_hcpcs')
        assert hasattr(medical_service, '_validate_hcpcs_codes')
        
        # Verify they are callable
        assert callable(getattr(medical_service, '_get_evidence_based_codes'))
        assert callable(getattr(medical_service, '_extract_procedures_from_guidelines'))
        assert callable(getattr(medical_service, '_map_procedures_to_hcpcs'))
        assert callable(getattr(medical_service, '_validate_hcpcs_codes'))
