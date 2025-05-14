from typing import List, Optional, Dict
from serpapi import GoogleSearch
from sqlalchemy.orm import Session
from datetime import datetime
from app.core.config import settings
from app.models.technology import PatentSearch, PatentResult, Technology
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatentService:
    """Service for searching and managing patent information"""
    
    def __init__(self, db: Session):
        self.db = db
        self.api_key = settings.SERPAPI_API_KEY
        self.max_pages = 3  # Maximum number of pages to fetch per search
        
    async def search_patents(self, technology_id: int, search_query: str) -> Optional[PatentSearch]:
        """
        Search for patents using SerpAPI and store results
        
        Args:
            technology_id: ID of the technology being researched
            search_query: Query string to search for patents
            
        Returns:
            PatentSearch object containing results or None if error
        """
        try:
            # Create search record
            patent_search = PatentSearch(
                technology_id=technology_id,
                search_query=search_query,
                search_date=datetime.utcnow()
            )
            self.db.add(patent_search)
            self.db.commit()
            self.db.refresh(patent_search)
            
            # Configure search parameters
            params = {
                "engine": "google_patents",
                "q": f"AB=({search_query})",
                "api_key": self.api_key,
                "page": 1
            }
            
            page_count = 0
            while page_count < self.max_pages:
                # Update page parameter
                params["page"] = page_count + 1
                
                # Execute search
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Process results
                organic_results = results.get("organic_results", [])
                
                # Store results
                for result in organic_results:
                    patent_result = PatentResult(
                        search_id=patent_search.id,
                        patent_id=result.get("patent_id", ""),
                        title=result.get("title", ""),
                        abstract=self._get_patent_details(result["patent_id"]).get("abstract", ""),
                        publication_date=result.get("publication_date", ""),
                        url=result.get("link", "")
                    )
                    self.db.add(patent_result)
                
                # Check for next page
                pagination = results.get("pagination", {})
                serpapi_pagination = results.get("serpapi_pagination", {})
                if not (pagination.get("next") or serpapi_pagination.get("next")):
                    break
                    
                page_count += 1
                
            self.db.commit()
            return patent_search
            
        except Exception as e:
            logger.error(f"Error during patent search: {e}")
            self.db.rollback()
            return None
            
    def _get_patent_details(self, patent_id: str) -> Dict:
        """
        Fetch detailed patent information using SerpAPI
        
        Args:
            patent_id: The patent ID to fetch details for
            
        Returns:
            Dictionary containing patent details
        """
        try:
            params = {
                "engine": "google_patents_details",
                "patent_id": patent_id,
                "api_key": self.api_key
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            return {
                "abstract": results.get("abstract", ""),
                "title": results.get("title", ""),
                "patent_number": results.get("patent_number", ""),
                "publication_date": results.get("publication_date", "")
            }
            
        except Exception as e:
            logger.error(f"Error fetching patent details for {patent_id}: {e}")
            return {}