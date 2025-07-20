#!/usr/bin/env python3
"""
Simple test script to verify PubMed API integration
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_pubmed_fetch():
    """Test the PubMed API fetch functionality"""
    from app.services.medical_assessment_service import MedicalAssessmentService
    from app.services.gpt_service import GPTService
    from unittest.mock import Mock
    
    # Create mock database and GPT service
    mock_db = Mock()
    mock_gpt_service = Mock()
    
    # Mock the GPT service _create_chat_completion method
    async def mock_chat_completion(system_prompt, user_prompt, temperature=0.3):
        return "0.8"  # Mock relevance score
    
    mock_gpt_service._create_chat_completion = mock_chat_completion
    
    # Create service instance
    service = MedicalAssessmentService(mock_gpt_service, mock_db)
    
    # Test PubMed fetch
    print("Testing PubMed API integration...")
    problem_statement = "diabetes treatment"
    
    try:
        guidelines = await service._fetch_pubmed_guidelines(problem_statement)
        
        print(f"✅ Successfully fetched {len(guidelines)} guidelines from PubMed")
        
        if guidelines:
            first_guideline = guidelines[0]
            print(f"📄 First guideline: {first_guideline.title[:100]}...")
            print(f"🔗 Link: {first_guideline.link}")
            print(f"⭐ Relevance score: {first_guideline.relevance_score}")
            print(f"📝 Content preview: {first_guideline.content[:200]}...")
        else:
            print("⚠️  No guidelines found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_pubmed_fetch())
    if success:
        print("\n🎉 PubMed integration test passed!")
    else:
        print("\n💥 PubMed integration test failed!")
        sys.exit(1)
