import aiohttp
import asyncio
from typing import List, Dict, Optional
import logging
from sqlalchemy.orm import Session
from app.models.technology import RelatedPaper
from tenacity import retry, stop_after_attempt, wait_exponential
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class PaperService:
    def __init__(self, db: Session):
        self.db = db
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.api_key = "YOUR_NCBI_API_KEY"  # Optional but recommended
        self.sem = asyncio.Semaphore(5)  # Limit concurrent requests

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search_papers(self, search_query: str) -> List[Dict]:
        """
        Search for papers using PubMed API
        """
        try:
            async with self.sem:
                search_url = f"{self.base_url}/esearch.fcgi"
                params = {
                    "db": "pubmed",
                    "term": search_query,
                    "retmode": "json",
                    "retmax": 10
                }
                if self.api_key:
                    params["api_key"] = self.api_key

                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            paper_ids = data.get("esearchresult", {}).get("idlist", [])
                            return paper_ids
                        else:
                            logger.error(f"Error searching papers: {response.status}")
                            return []

        except Exception as e:
            logger.error(f"Error in paper search: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_paper_details(self, paper_id: str) -> Optional[Dict]:
        """
        Fetch paper details using PubMed API
        Returns paper details in a structured format
        """
        try:
            async with self.sem:
                fetch_url = f"{self.base_url}/efetch.fcgi"
                params = {
                    "db": "pubmed",
                    "id": paper_id,
                }
                if self.api_key:
                    params["api_key"] = self.api_key

                async with aiohttp.ClientSession() as session:
                    async with session.get(fetch_url, params=params) as response:
                        if response.status == 200:
                            xml_content = await response.text()
                            return self._parse_paper_xml(xml_content)
                        else:
                            logger.error(f"Error fetching paper details: {response.status}")
                            return None

        except Exception as e:
            logger.error(f"Error fetching paper details: {e}")
            return None

    def _parse_paper_xml(self, xml_content: str) -> Optional[Dict]:
        """
        Parse PubMed XML response into structured data
        """
        try:
            root = ET.fromstring(xml_content)
            article = root.find(".//Article")
            if article is None:
                return None

            # Extract abstract
            abstract_text = []
            abstract = article.find(".//Abstract")
            if abstract is not None:
                for abstract_part in abstract.findall(".//AbstractText"):
                    label = abstract_part.get('Label')
                    text = abstract_part.text or ''
                    if label:
                        abstract_text.append(f"{label}: {text}")
                    else:
                        abstract_text.append(text)

            # Extract authors
            author_list = []
            authors = article.findall(".//Author")
            for author in authors:
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None and fore_name is not None:
                    author_list.append(f"{fore_name.text} {last_name.text}")

            # Extract journal info
            journal = article.find(".//Journal")
            journal_title = journal.find(".//Title").text if journal is not None and journal.find(".//Title") is not None else ""

            # Extract publication date
            pub_date = journal.find(".//PubDate") if journal is not None else None
            year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else ""
            month = pub_date.find("Month").text if pub_date is not None and pub_date.find("Month") is not None else ""
            day = pub_date.find("Day").text if pub_date is not None and pub_date.find("Day") is not None else ""
            
            publication_date = "-".join(filter(None, [year, month, day]))

            return {
                "title": article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "",
                "abstract": "\n".join(abstract_text),
                "authors": ", ".join(author_list),
                "journal": journal_title,
                "publication_date": publication_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/",
                "citation_count": 0  # Could be updated with separate API call if needed
            }

        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing paper details: {e}")
            return None