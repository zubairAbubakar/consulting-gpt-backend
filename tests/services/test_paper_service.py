import pytest
import asyncio
from aioresponses import aioresponses
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.services.paper_service import PaperService
from app.models.technology import RelatedPaper

# Sample XML response from PubMed API
SAMPLE_PUBMED_XML = """<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation Status="MEDLINE">
            <Article>
                <Journal>
                    <Title>Nature Nanotechnology</Title>
                    <JournalIssue>
                        <PubDate>
                            <Year>2023</Year>
                            <Month>05</Month>
                            <Day>15</Day>
                        </PubDate>
                    </JournalIssue>
                </Journal>
                <ArticleTitle>Test Article Title</ArticleTitle>
                <Abstract>
                    <AbstractText>Main abstract text</AbstractText>
                    <AbstractText Label="BACKGROUND">Background information</AbstractText>
                    <AbstractText Label="METHODS">Research methods</AbstractText>
                </Abstract>
                <AuthorList CompleteYN="Y">
                    <Author ValidYN="Y">
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                    <Author ValidYN="Y">
                        <LastName>Doe</LastName>
                        <ForeName>Jane</ForeName>
                    </Author>
                </AuthorList>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>
"""

# Sample JSON response from PubMed search API
SAMPLE_SEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "retmax": "2",
        "idlist": ["123456", "789012"]
    }
}

@pytest.fixture
def engine():
    return create_engine('sqlite:///:memory:')

@pytest.fixture
def TestingSessionLocal(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

@pytest.fixture
def db(TestingSessionLocal):
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        
@pytest.fixture
def db():
    # Mock database session
    return Session()

@pytest.fixture
def paper_service(db):
    return PaperService(db)




@pytest.mark.asyncio
async def test_fetch_paper_details_error(paper_service):
    with aioresponses() as m:
        # Mock an error response
        paper_id = "123456"
        fetch_url = f"{paper_service.base_url}/efetch.fcgi"
        m.get(fetch_url, status=500)

        # Test error handling
        result = await paper_service.fetch_paper_details(paper_id)
        assert result is None

@pytest.mark.asyncio
async def test_search_papers_empty_response(paper_service):
    with aioresponses() as m:
        # Mock empty search results
        search_url = f"{paper_service.base_url}/efetch.fcgi"
        m.get(
            search_url,
            payload={"esearchresult": {"count": "0", "idlist": []}},
            status=200
        )

        result = await paper_service.search_papers("nonexistent query")
        assert result == []    