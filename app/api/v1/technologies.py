from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.schemas.related_paper import RelatedPaperRead
from app.services.technology_service import TechnologyService
from app.models.technology import PatentSearch, PatentResult, RelatedPaper  
from app.schemas.technology import (
    TechnologyCreate,
    TechnologyRead,
    RelatedTechnologyRead
)

router = APIRouter()

@router.post("/", response_model=TechnologyRead)
async def create_technology(
    technology: TechnologyCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new technology and initiate the setup process
    This endpoint will:
    1. Create the technology record
    2. Generate search keywords
    3. Generate comparison axes
    4. Search for related patents
    5. Process patent results to create related technology entries
    """
    service = TechnologyService(db)
    
    # Create technology with initial data - now properly awaited
    tech = await service.create_technology(
        name=technology.name,
        abstract=technology.abstract,
        num_of_axes=technology.num_of_axes or 5  # Default to 5 if not specified
    )
    
    if not tech:
        raise HTTPException(status_code=500, detail="Failed to create technology")
    
    # Run remaining tasks in background
    background_tasks.add_task(
        complete_technology_setup_background,
        tech.id,
        db
    )
    
    return tech

@router.get("/", response_model=List[TechnologyRead])
def get_all_technologies(
    db: Session = Depends(get_db)
):
    """Get all technologies"""
    service = TechnologyService(db)
    return service.get_all_technologies()

@router.get("/{technology_id}", response_model=TechnologyRead)
def get_technology(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific technology by ID"""
    service = TechnologyService(db)
    technology = service.get_technology_by_id(technology_id)
    
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    return technology

@router.get("/{technology_id}/related", response_model=List[RelatedTechnologyRead])
def get_related_technologies(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Get related technologies for a specific technology"""
    service = TechnologyService(db)
    
    # Verify technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    # Get related technologies
    return service.get_related_technologies(technology_id)

@router.post("/{technology_id}/regenerate-keywords")
async def regenerate_search_keywords(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Regenerate search keywords for a technology"""
    service = TechnologyService(db)
    
    # Verify technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    # Run in background
    background_tasks.add_task(
        regenerate_keywords_background,
        technology_id,
        db
    )
    
    return {"message": "Regenerating search keywords in background"}

@router.post("/{technology_id}/regenerate-axes")
async def regenerate_comparison_axes(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Regenerate comparison axes for a technology"""
    service = TechnologyService(db)
    
    # Verify technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    # Run in background
    background_tasks.add_task(
        regenerate_axes_background,
        technology_id,
        db
    )
    
    return {"message": "Regenerating comparison axes in background"}

@router.post("/{technology_id}/search-patents")
async def search_patents(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Search for patents related to a technology"""
    service = TechnologyService(db)
    
    # Verify technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    if not technology.search_keywords:
        raise HTTPException(status_code=400, detail="Technology has no search keywords")
        
    # Run in background
    background_tasks.add_task(
        search_patents_background,
        technology_id,
        db
    )
    
    return {"message": "Searching for patents in background"}

@router.post("/{technology_id}/search-papers")
async def search_papers(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Search for related papers in background
    """
    service = TechnologyService(db)
    background_tasks.add_task(search_papers_background, technology_id, db)
    return {"message": "Searching for papers in background"}

@router.get("/{technology_id}/papers")
async def get_papers(
    technology_id: int,
    db: Session = Depends(get_db)
) -> List[RelatedPaperRead]:
    """
    Get all related papers for a technology
    """
    service = TechnologyService(db)
    papers = service.db.query(RelatedPaper).filter(
        RelatedPaper.technology_id == technology_id
    ).all()
    return papers

# Background task functions

async def complete_technology_setup_background(technology_id: int, db: Session):
    """Complete technology setup in background"""
    service = TechnologyService(db)
    try:
        # Generate search keywords
        await service.generate_search_keywords(technology_id)
        
        # # Generate comparison axes
        # await service.generate_comparison_axes(technology_id)
        
        # Search patents
        print(f"Starting patent search for technology {technology_id}")
        search_success = await service.search_patents(technology_id)
        print(f"Patent search completed with success: {search_success}")
        
        if search_success:
            try:
                print(f"Starting patent results processing for technology {technology_id}")
                processing_success = await service.process_patent_results(technology_id)
                print(f"Patent results processing completed: {processing_success}")
            except Exception as process_error:
                print(f"Error processing patent results: {process_error}")
                # Continue execution even if processing fails
        else:
            print("Patent search failed, skipping results processing")

        # Search papers
        print(f"Starting paper search for technology {technology_id}")
        await service.search_related_papers(technology_id)
        print(f"Paper search completed for technology {technology_id}")
            
    except Exception as e:
        print(f"Error in background task for technology {technology_id}: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

async def regenerate_keywords_background(technology_id: int, db: Session):
    """Regenerate search keywords in background"""
    service = TechnologyService(db)
    await service.generate_search_keywords(technology_id)

async def regenerate_axes_background(technology_id: int, db: Session):
    """Regenerate comparison axes in background"""
    service = TechnologyService(db)
    await service.generate_comparison_axes(technology_id)

async def search_patents_background(technology_id: int, db: Session):
    """Search patents in background"""
    service = TechnologyService(db)
    try:
        print(f"Starting patent search for technology {technology_id}")
        success = await service.search_patents(technology_id)
        print(f"Patent search completed with success: {success}")
        
        if success:
            try:
                # Get the latest patent search results
                patent_search = db.query(PatentSearch)\
                    .filter(PatentSearch.technology_id == technology_id)\
                    .order_by(PatentSearch.search_date.desc())\
                    .first()
                
                if patent_search and patent_search.search_results:
                    print(f"Found {len(patent_search.search_results)} patent results to process")
                    await service.process_patent_results(technology_id)
                    print("Patent results processing completed")
                else:
                    print("No patent results found to process")
            except Exception as process_error:
                print(f"Error processing patent results: {str(process_error)}")
                import traceback
                print(f"Process error traceback: {traceback.format_exc()}")
        else:
            print(f"Patent search failed for technology {technology_id}")
            
    except Exception as e:
        print(f"Error in patent search background task: {str(e)}")
        import traceback
        print(f"Search error traceback: {traceback.format_exc()}")

# Background task functions
async def search_papers_background(technology_id: int, db: Session):
    """Search papers in background"""
    service = TechnologyService(db)
    await service.search_related_papers(technology_id)
