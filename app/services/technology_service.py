from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import json
import asyncio
import logging

from app.models.technology import Technology, RelatedTechnology, RelatedPaper, PatentSearch, PatentResult, ComparisonAxis
from app.schemas.technology import TechnologyCreate, TechnologyRead, RelatedTechnologyRead
from app.services.gpt_service import GPTService
from app.services.patent_service import PatentService
from app.services.paper_service import PaperService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnologyService:
    def __init__(self, db: Session):
        self.db = db
        self.gpt_service = GPTService(db)
        self.patent_service = PatentService(db)
        self.paper_service = PaperService(db)
    
    async def create_technology(self, name: str, abstract: str = None, num_of_axes: int = 3) -> Technology:
        """
        Create a new technology record
        """
        try:
            # Create technology object first with known values
            db_technology = Technology(
                name=name,
                abstract=abstract,
                num_of_axes=num_of_axes
            )
            
            # Generate problem statement using the abstract
            if abstract:
                problem_statement = await self.gpt_service.generate_problem_statement(abstract)
                db_technology.problem_statement = problem_statement
            
            # Save to database
            self.db.add(db_technology)
            self.db.commit()
            self.db.refresh(db_technology)

            # Generate and save comparison axes
            axes_df, _ = await self.gpt_service.generate_comparison_axes(
                technology_name=name,
                problem_statement=problem_statement,
                num_axes=num_of_axes,
                technology_description=abstract
            )

            # Debug print to check DataFrame structure
            # print(f"DataFrame columns: {axes_df.columns.tolist()}")
            # print(f"DataFrame content:\n{axes_df}")

            # Save comparison axes to database
            if not axes_df.empty and 'Axis' in axes_df.columns:
                for _, row in axes_df.iterrows():
                    comparison_axis = ComparisonAxis(
                        technology_id=db_technology.id,
                        axis_name=row['Axis'],          # Use original column name
                        extreme1=row['Extreme1'],       # Use original column name
                        extreme2=row['Extreme2'],       # Use original column name
                        weight=1.0  # Default weight
                    )
                    self.db.add(comparison_axis)
                
                self.db.commit()
                self.db.refresh(db_technology)
            else:
                print("Error: DataFrame is empty or missing required columns")
            
            return db_technology
            
        except Exception as e:
            print(f"Error creating technology: {e}")
            self.db.rollback()
            return None
    
    def get_all_technologies(self) -> List[Technology]:
        """
        Get all technologies from the database
        """
        return self.db.query(Technology).all()
    
    def get_technology_by_id(self, technology_id: int) -> Optional[Technology]:
        """
        Get a technology by ID
        """
        return self.db.query(Technology).filter(Technology.id == technology_id).first()
    
    def get_related_technologies(self, technology_id: int) -> List[RelatedTechnology]:
        """
        Get related technologies for a specific technology
        """
        return self.db.query(RelatedTechnology).filter(
            RelatedTechnology.technology_id == technology_id
        ).all()
    
    async def generate_search_keywords(self, technology_id: int) -> bool:
        """
        Generate search keywords for a technology
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Generate keywords based on the technology name and problem statement
            keywords = await self.gpt_service.generate_search_keywords(                 
                technology.problem_statement,
                keyword_count=4
            )
            
            # Update the technology with the new keywords
            if keywords:
                technology.search_keywords = keywords
                self.db.commit()
                return True
            
            return False
        except Exception as e:
            print(f"Error generating search keywords: {e}")
            return False
    
    async def generate_comparison_axes(self, technology_id: int) -> bool:
        """
        Generate comparison axes for a technology
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Generate axes based on the technology name and problem statement
            axes_df = await self.gpt_service.generate_comparison_axes(
                technology.name,
                technology.problem_statement,
                technology.num_of_axes,
                technology.abstract
            )
            
            # Clear existing axes
            for axis in technology.comparison_axes:
                self.db.delete(axis)

            print(f"generated axes: {axes_df} ")
            # Save comparison axes to database
            for _, row in axes_df.iterrows():
                comparison_axis = ComparisonAxis(
                    technology_id=technology.id,
                    axis_name=row['axis_name'],
                    extreme1=row['extreme1'],
                    extreme2=row['extreme2'],
                    weight=row.get("weight", 1.0)
                )
                self.db.add(comparison_axis)
            
            self.db.commit()   

            return True
        except Exception as e:
            print(f"Error generating comparison axes: {e}")
            return False
    
    async def search_patents(self, technology_id: int) -> bool:
        """
        Search for patents related to a technology
        """
        try:
            # Get the technology
            technology = self.get_technology_by_id(technology_id)
            if not technology or not technology.search_keywords:
                print("No technology or search keywords found")
                return False

            print(f"Searching patents with keywords: {technology.search_keywords}")

            # Search patents using patent service first to ensure we have results
            search_results = await self.patent_service.search_patents(technology.id, technology.search_keywords)
            if not search_results:
                print("No patent results found")
                return False
            
            return True

        except Exception as e:
            print(f"Error searching patents: {e}")
            self.db.rollback()
            return False
    
    async def process_patent_results(self, technology_id: int) -> bool:
        """
        Process patent search results to create related technology entries
        """
        # Get the technology
        technology = self.get_technology_by_id(technology_id)
        if not technology:
            return False
            
        try:
            # Get the latest patent search for this technology
            patent_search = self.db.query(PatentSearch)\
                .filter(PatentSearch.technology_id == technology_id)\
                .order_by(PatentSearch.search_date.desc())\
                .first()
            
            if not patent_search:
                return False
                
            # Get the patent results for this search
            patent_results = self.db.query(PatentResult)\
                .filter(PatentResult.search_id == patent_search.id)\
                .all()
            
            print(f"Processing {len(patent_results)} patent results")
            
            # Process the results
            for result in patent_results:
                # Check if related technology already exists
                existing = self.db.query(RelatedTechnology)\
                    .filter(
                        RelatedTechnology.technology_id == technology_id,
                        RelatedTechnology.document_id == result.patent_id
                    ).first()
                
                if not existing:
                    # Create new related technology
                    patent_details = await self.patent_service.fetch_patent_details(result.patent_id)
                    if patent_details:
                        related_tech = RelatedTechnology(
                            technology_id=technology_id,
                            name=patent_details.get('title', ''),
                            abstract=patent_details.get('abstract', ''),
                            document_id=result.patent_id,
                            type="patent",
                            url=result.url,
                            publication_date=patent_details.get('publication_date', ''),
                            inventors=patent_details.get('inventors', ''),
                            assignees=patent_details.get('assignees', '')
                        )
                        self.db.add(related_tech)
                        print(f"Added new related technology: {related_tech.name}")
            
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error processing patent results: {e}")
            self.db.rollback()
            return False
        

    async def search_related_papers(self, technology_id: int) -> bool:
        """
        Search and store related papers for a technology with complete details
        """
        try:
            # Get the technology
            technology = self.get_technology_by_id(technology_id)
            if not technology or not technology.search_keywords:
                logger.error("No technology or search keywords found")
                return False
    
            # Search for papers
            paper_ids = await self.paper_service.search_papers(technology.search_keywords)
            if not paper_ids:
                logger.error("No papers found")
                return False
    
            papers_added = 0
            # Process each paper
            for paper_id in paper_ids:
                # Check if paper already exists
                existing = self.db.query(RelatedPaper).filter(
                    RelatedPaper.technology_id == technology_id,
                    RelatedPaper.paper_id == paper_id
                ).first()
    
                if not existing:
                    # Fetch paper details immediately
                    details = await self.paper_service.fetch_paper_details(paper_id)
                    if details:
                        # Create new paper entry with complete details
                        paper = RelatedPaper(
                            technology_id=technology_id,
                            paper_id=paper_id,
                            title=details["title"],
                            abstract=details["abstract"],
                            authors=details["authors"],
                            publication_date=details["publication_date"],
                            journal=details["journal"],
                            url=details["url"],
                            citation_count=details["citation_count"]
                        )
                        self.db.add(paper)
                        papers_added += 1
                        logger.info(f"Added paper: {details['title']}")
    
            if papers_added > 0:
                self.db.commit()
                logger.info(f"Added {papers_added} new papers")
            return True
    
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            self.db.rollback()
            return False        