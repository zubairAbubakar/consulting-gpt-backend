from typing import List, Optional, Dict, Any
from serpapi import GoogleSearch
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from app.core.config import settings
from app.models.technology import PatentSearch, PatentResult, Technology
import logging
import asyncio
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatentService:
    """Service for searching and managing patent information"""
    
    def __init__(self, db: Session):
        self.db = db
        self.api_key = settings.SERPAPI_API_KEY
        self.max_pages = 1  # Maximum number of pages to fetch per search
        self.sem = Semaphore(5)  # Limit concurrent API calls
        
    async def search_patents(self, technology_id: int, search_query: str) -> Optional[PatentSearch]:
        """
        Search for patents using SerpAPI and store basic results
        
        Args:
            technology_id: ID of the technology being researched
            search_query: Query string to search for patents
            
        Returns:
            PatentSearch object containing results or None if error
        """
        try:
            
            now = datetime.now(timezone.utc)
            patent_search = PatentSearch(
                technology_id=technology_id,
                search_query=search_query,
                search_date=now,
                created_at=now,
                updated_at=now
            )
            self.db.add(patent_search)
            self.db.commit()
            self.db.refresh(patent_search)
            
            # Configure search parameters
            params = {
                "engine": "google_patents",
                "q": f"({search_query})",
                "api_key": self.api_key,
                "page": 1
            }
            
            page_count = 0
            total_results = 0
            max_results = 10  # Maximum results to collect
            
            while page_count < self.max_pages and total_results < max_results:
                # Execute search with rate limiting
                async with self.sem:
                    search = await self._execute_search_with_retry(params)
                    if not search:
                        break
                
                # Process results
                organic_results = search.get("organic_results", [])
                
                # Store basic results
                for result in organic_results:
                    patent_result = PatentResult(
                        search_id=patent_search.id,
                        patent_id=result.get("patent_id", ""),
                        title=result.get("title", ""),
                        abstract="",  # Will be fetched separately
                        publication_date=result.get("publication_date", ""),
                        url=result.get("link", ""),
                        created_at=now,
                        updated_at=now
                    )
                    self.db.add(patent_result)
                    total_results += 1
                    
                    if total_results >= max_results:
                        break
                
                # Check for next page
                pagination = search.get("pagination", {})
                serpapi_pagination = search.get("serpapi_pagination", {})
                
                if not (pagination.get("next") or serpapi_pagination.get("next")):
                    break
                    
                page_count += 1
                params["page"] = page_count + 1
                
                # Add a delay between pages to avoid rate limits
                await asyncio.sleep(2)
                
            self.db.commit()
            
            # Log search statistics
            logger.info(f"Patent search completed: {total_results} results from {page_count + 1} pages")
            return patent_search
            
        except Exception as e:
            logger.error(f"Error during patent search: {e}")
            self.db.rollback()
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
    )
    async def _execute_search_with_retry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a SerpAPI search with retry logic
        
        Args:
            params: Search parameters for SerpAPI
            
        Returns:
            Dictionary of search results or empty dict on failure
        """
        try:
            search = GoogleSearch(params)
            return search.get_dict()
        except Exception as e:
            logger.warning(f"SerpAPI search failed, retrying: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ValueError))
    )
    async def fetch_patent_details(self, patent_id: str) -> Dict:
        """
        Fetch detailed information about a specific patent
        
        Args:
            patent_id: The Google Patents ID
            
        Returns:
            Dictionary containing patent details
        """
        try:
            async with self.sem:  # Limit concurrent API calls
                params = {
                    "engine": "google_patents_details",
                    "patent_id": patent_id,
                    "api_key": self.api_key
                }
                
                search = GoogleSearch(params)
                results = search.get_dict()
                
                # Handle potentially missing or malformed data
                inventors = results.get("inventors", [])
                if not isinstance(inventors, list):
                    inventors = []
                    
                assignees = results.get("assignees", [])
                if not isinstance(assignees, list):
                    assignees = []
                    
                classifications = results.get("classifications", [])
                if not isinstance(classifications, list):
                    classifications = []
                
                # Extract all relevant patent information with safe type checking
                return {
                    "abstract": str(results.get("abstract", "")),
                    "title": str(results.get("title", "")),
                    "patent_number": str(results.get("patent_number", "")),
                    "publication_date": str(results.get("publication_date", "")),
                    "filing_date": str(results.get("filing_date", "")),
                    "inventors": ", ".join(str(inv.get("name", "")) if isinstance(inv, dict) else "" for inv in inventors),
                    "assignees": ", ".join(str(asn.get("name", "")) if isinstance(asn, dict) else "" for asn in assignees),
                    "classifications": ", ".join(str(cls.get("code", "")) if isinstance(cls, dict) else "" for cls in classifications),
                    "citations_count": len(results.get("citations", [])) if isinstance(results.get("citations"), list) else 0,
                    "cited_by_count": len(results.get("cited_by", [])) if isinstance(results.get("cited_by"), list) else 0,
                    "status": str(results.get("status", ""))
                }
                
        except Exception as e:
            logger.error(f"Error fetching patent details from API for {patent_id}: {e}")
            # Return empty dict with default values instead of raising
            return {
                "abstract": "",
                "title": "",
                "patent_number": patent_id,
                "publication_date": "",
                "filing_date": "",
                "inventors": "",
                "assignees": "",
                "classifications": "",
                "citations_count": 0,
                "cited_by_count": 0,
                "status": ""
            }
            
    async def update_patent_details(self, patent_result_id: int) -> bool:
        """
        Update a patent result with detailed information
        
        Args:
            patent_result_id: ID of the PatentResult to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get the patent result
            patent_result = self.db.query(PatentResult).get(patent_result_id)
            if not patent_result:
                logger.error(f"Patent result {patent_result_id} not found")
                return False
                
            # Fetch details
            details = await self.fetch_patent_details(patent_result.patent_id)
            
            # Update the result
            patent_result.abstract = details.get("abstract", "")
            patent_result.title = details.get("title") or patent_result.title
            patent_result.updated_at = datetime.now(timezone.utc)
            
            # Add additional fields if they exist in your database model
            # If these fields don't exist, you'll need to modify your database schema
            if hasattr(patent_result, "inventors"):
                patent_result.inventors = details.get("inventors", "")
            if hasattr(patent_result, "assignees"):
                patent_result.assignees = details.get("assignees", "")
            if hasattr(patent_result, "filing_date"):
                patent_result.filing_date = details.get("filing_date", "")
            if hasattr(patent_result, "status"):
                patent_result.status = details.get("status", "")
            
            self.db.commit()
            logger.info(f"Updated patent details for {patent_result.patent_id}")
            return True
                
        except Exception as e:
            logger.error(f"Error updating patent details: {e}")
            self.db.rollback()
            return False
            
    async def process_patent_queue(self, search_id: int) -> Dict[str, Any]:
        """
        Process all patents from a search and update their details
        
        Args:
            search_id: ID of the PatentSearch to process
            
        Returns:
            Stats about the processing
        """
        try:
            # Get all patents from the search
            patent_results = self.db.query(PatentResult).filter(
                PatentResult.search_id == search_id,
                PatentResult.abstract == ""  # Only process those without details
            ).all()
            
            logger.info(f"Processing {len(patent_results)} patents for search {search_id}")
            
            # Process in parallel with rate limiting
            tasks = []
            for patent in patent_results:
                tasks.append(self.update_patent_details(patent.id))
                
            # Execute in batches of 5 to control concurrency
            batch_size = 5
            completed = 0
            failed = 0
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Patent detail update failed: {result}")
                        failed += 1
                    elif result is True:
                        completed += 1
                    else:
                        failed += 1
                
                # Add delay between batches
                if i + batch_size < len(tasks):
                    await asyncio.sleep(2)
            
            return {
                "total": len(patent_results),
                "completed": completed,
                "failed": failed
            }
                
        except Exception as e:
            logger.error(f"Error processing patent queue: {e}")
            return {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "error": str(e)
            }