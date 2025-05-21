import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.dependencies import get_db
from app.schemas.market_analysis import MarketAnalysisRead
from app.schemas.pca_component import PCAResultRead
from app.schemas.related_paper import RelatedPaperRead
from app.services.technology_service import TechnologyService
from app.models.technology import MarketAnalysis, PCAResult, PatentSearch, PatentResult, RelatedPaper, ClusterResult, ClusterMember   
from app.schemas.technology import (
    ComparisonAxisRead,
    TechnologyCreate,
    TechnologyRead,
    RelatedTechnologyRead
)
from app.services.visualization_service import VisualizationService

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@router.post("/{technology_id}/market-analysis")
async def perform_market_analysis(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger market analysis in background
    """
    service = TechnologyService(db)
    background_tasks.add_task(market_analysis_background, technology_id, db)
    return {"message": "Market analysis started in background"}

@router.get("/{technology_id}/market-analysis")
async def get_market_analysis(
    technology_id: int,
    db: Session = Depends(get_db)
) -> List[MarketAnalysisRead]:
    """
    Get market analysis results
    """
    service = TechnologyService(db)
    analyses = service.db.query(MarketAnalysis).filter(
        MarketAnalysis.technology_id == technology_id
    ).all()
    return analyses

@router.get("/{technology_id}/comparison-axes", response_model=List[ComparisonAxisRead])
def get_comparison_axes(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all comparison axes for a technology
    """
    service = TechnologyService(db)
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    axes = service.get_comparison_axes(technology_id)
    return axes

@router.post("/{technology_id}/pca", response_model=PCAResultRead)
async def create_pca_analysis(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger PCA analysis for a technology
    """
    service = TechnologyService(db)
    
    # Check if technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
    
    # Perform PCA analysis
    pca_result = await service.perform_pca_analysis(technology_id)
    if not pca_result:
        raise HTTPException(
            status_code=400,
            detail="Could not perform PCA analysis. Ensure market analysis has been completed."
        )
    
    # Generate component descriptions in background
    background_tasks.add_task(
        service.describe_pca_components,
        pca_result,
        technology
    )
    
    return pca_result

@router.get("/{technology_id}/pca", response_model=List[PCAResultRead])
async def get_pca_results(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """
    Get PCA analysis results for a technology
    """
    service = TechnologyService(db)
    
    # Check if technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
    
    try:
        # Get PCA results
        results = db.query(PCAResult).filter(
            PCAResult.technology_id == technology_id
        ).all()
        
        # Return empty list if no results found
        return results if results else []
        
    except Exception as e:
        logger.warning(f"Error fetching PCA results: {e}")
        # Return empty list instead of throwing an error
        return []

@router.get("/{technology_id}/visualization", response_model=Dict)
async def get_visualization_data(
    technology_id: int,
    viz_type: str = "pca",  # or "raw"
    db: Session = Depends(get_db)
):
    """
    Get visualization data for a technology
    """
    service = TechnologyService(db)
    viz_service = VisualizationService()
    
    if viz_type == "pca":
        pca_result = db.query(PCAResult)\
            .filter(PCAResult.technology_id == technology_id)\
            .order_by(PCAResult.id.desc())\
            .first()
            
        if not pca_result:
            raise HTTPException(
                status_code=404,
                detail="No PCA results found"
            )
            
        return viz_service.prepare_pca_plot_data(pca_result)
        
    else:  # raw axes plot
        analyses = db.query(MarketAnalysis)\
            .filter(MarketAnalysis.technology_id == technology_id)\
            .all()
            
        return viz_service.prepare_axes_plot_data(analyses)

@router.post("/{technology_id}/clustering")
async def create_clustering(
    technology_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger clustering analysis
    """
    service = TechnologyService(db)
    
    # Ensure technology exists
    technology = service.get_technology_by_id(technology_id)
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
    
    # Run clustering in background
    background_tasks.add_task(service.perform_clustering, technology_id)
    
    return {"message": "Clustering analysis started"}

@router.get("/{technology_id}/clusters")
async def get_clusters(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """
    Get clustering results
    """
    service = TechnologyService(db)
    
    clusters = db.query(ClusterResult)\
        .filter(ClusterResult.technology_id == technology_id)\
        .all()
        
    return clusters

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

        # Perform market analysis
        print(f"Starting market analysis for technology {technology_id}")
        await service.perform_market_analysis(technology_id)
            
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

async def market_analysis_background(technology_id: int, db: Session):
    """Run market analysis in background"""
    service = TechnologyService(db)
    await service.perform_market_analysis(technology_id)