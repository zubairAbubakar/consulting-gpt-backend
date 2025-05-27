import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session, joinedload
from app.dependencies import get_db
from app.models.dental_fee_schedule import DentalFeeSchedule
from app.schemas.cluster import ClusterDetailRead
from app.schemas.market_analysis import MarketAnalysisRead
from app.schemas.pca_component import PCAResultRead
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.schemas.related_paper import RelatedPaperRead
from app.schemas.visualization import VisualizationParams
from app.services.gpt_service import GPTService
from app.services.medical_assessment_service import MedicalAssessmentService
from app.services.recommendation_service import RecommendationService
from app.services.technology_service import TechnologyService
from app.models.technology import ( 
    MarketAnalysis, 
    PCAResult, 
    PatentSearch, 
    RelatedPaper, 
    ClusterResult, 
    ClusterMember, 
    Technology,
    MedicalAssessment,
    BillableItem 
)
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
        service.describe_pca_components_background,
        technology_id,
        pca_result.id
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

@router.get("/{technology_id}/clusters/{cluster_id}", response_model=ClusterDetailRead)
async def get_cluster_details(
    technology_id: int,
    cluster_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific cluster including all member technologies
    """
    cluster = db.query(ClusterResult)\
        .options(
            joinedload(ClusterResult.cluster_members)
            .joinedload(ClusterMember.related_technology)
        )\
        .filter(
            ClusterResult.technology_id == technology_id,
            ClusterResult.id == cluster_id
        ).first()
        
    if not cluster:
        raise HTTPException(
            status_code=404,
            detail="Cluster not found"
        )

    # Transform cluster members to include technology details
    members = []
    for member in cluster.cluster_members:
        members.append({
            "id": member.id,
            "cluster_id": member.cluster_id,
            "technology_id": member.technology_id,
            "distance_to_center": member.distance_to_center,
            "technology_name": member.related_technology.name,
            "technology_abstract": member.related_technology.abstract
        })

    return {
        "id": cluster.id,
        "technology_id": cluster.technology_id,
        "name": cluster.name,
        "description": cluster.description,
        "contains_target": cluster.contains_target,
        "center_x": cluster.center_x,
        "center_y": cluster.center_y,
        "cluster_spread": cluster.cluster_spread,
        "technology_count": cluster.technology_count,
        "created_at": cluster.created_at,
        "members": members
    }


@router.get("/{technology_id}/visualization")
async def get_visualization_data(
    technology_id: int,
    show_clusters: bool = Query(False, description="Toggle cluster visualization"),
    viz_type: str = Query("pca", description="Visualization type: pca, raw, or combined"),
    selected_axes: Optional[List[str]] = Query(None, description="List of axes to show"),
    show_labels: bool = Query(True, description="Show technology labels"),
    show_annotations: bool = Query(True, description="Show additional annotations"),
    db: Session = Depends(get_db)
):
    """
    Get enhanced visualization data with interactive features
    """
    viz_service = VisualizationService()
    
    # Get all required data
    pca_result = db.query(PCAResult)\
        .filter(PCAResult.technology_id == technology_id)\
        .order_by(PCAResult.id.desc())\
        .first()
    
    cluster_results = None
    if show_clusters:
        cluster_results = db.query(ClusterResult)\
            .options(joinedload(ClusterResult.cluster_members)
                    .joinedload(ClusterMember.related_technology))\
            .filter(ClusterResult.technology_id == technology_id)\
            .all()
    
    market_analyses = None
    if viz_type in ["raw", "combined"]:
        market_analyses = db.query(MarketAnalysis)\
            .options(joinedload(MarketAnalysis.comparison_axis))\
            .filter(MarketAnalysis.technology_id == technology_id)\
            .all()
    
    return viz_service.prepare_visualization_data(
        pca_result=pca_result,
        market_analyses=market_analyses,
        cluster_results=cluster_results,
        show_clusters=show_clusters,
        selected_axes=selected_axes,
        show_labels=show_labels,
        show_annotations=show_annotations
    )

@router.get("/{technology_id}/visualization/silhouette")
async def get_silhouette_analysis(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Get silhouette analysis for clustering"""
    viz_service = VisualizationService()
    
    pca_result = db.query(PCAResult)\
        .filter(PCAResult.technology_id == technology_id)\
        .order_by(PCAResult.id.desc())\
        .first()
    
    if not pca_result:
        raise HTTPException(
            status_code=404,
            detail="No PCA results found"
        )
        
    return viz_service.prepare_silhouette_analysis(pca_result)

@router.post(
    "/{technology_id}/recommendations",
    response_model=RecommendationResponse
)
async def generate_recommendations(
    technology_id: int,
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Generate commercialization recommendations"""
    
    # Get technology details
    technology = db.query(Technology)\
        .filter(Technology.id == technology_id)\
        .first()
    
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")
        
    # Get market analysis data
    market_analyses = db.query(MarketAnalysis)\
        .options(joinedload(MarketAnalysis.comparison_axis))\
        .filter(MarketAnalysis.technology_id == technology_id)\
        .all()
        
    # Format market data
    market_data = {
        "axes": [
            {
                "name": analysis.comparison_axis.axis_name,
                "score": analysis.score,
                "technology": analysis.related_technology.name
            }
            for analysis in market_analyses
        ]
    }
    
    # Initialize services with db session
    gpt_service = GPTService(db)
    recommendation_service = RecommendationService(gpt_service, db)
    
    recommendations = await recommendation_service.generate_recommendations(
        technology_id=technology_id,
        name=technology.name,
        problem_statement=technology.problem_statement,
        abstract=technology.abstract,
        current_stage=request.current_stage,
        market_data=market_data
    )
    
    return recommendations


@router.post("/{technology_id}/medical-assessment")
async def create_medical_assessment(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Generate medical assessment including fee calculations"""
    
    # Get technology details
    technology = db.query(Technology)\
        .filter(Technology.id == technology_id)\
        .first()
    
    if not technology:
        raise HTTPException(status_code=404, detail="Technology not found")

    medical_service = MedicalAssessmentService(GPTService(db), db)
    
    assessment = await medical_service.create_medical_assessment(
        technology_id=technology_id,
        problem_statement=technology.problem_statement,
        technology_name=technology.name
    )
    
    if not assessment:
        raise HTTPException(
            status_code=400,
            detail="No relevant medical associations found"
        )
    
    return assessment


@router.get("/{technology_id}/billable-items")
async def get_billable_items(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Get billable items for a technology's medical assessment"""
    
    # First get the medical assessment for this technology
    assessment = db.query(MedicalAssessment)\
        .filter(MedicalAssessment.technology_id == technology_id)\
        .order_by(MedicalAssessment.created_at.desc())\
        .first()
    
    if not assessment:
        raise HTTPException(
            status_code=404,
            detail="No medical assessment found for this technology"
        )
    
    # Get the billable items
    billable_items = db.query(BillableItem)\
        .filter(BillableItem.assessment_id == assessment.id)\
        .all()
    
    return {
        "assessment_id": assessment.id,
        "medical_association": assessment.medical_association,
        "billable_items": billable_items,
        "total_fee": sum(item.fee for item in billable_items if item.fee)
    }

@router.get("/{technology_id}/classify-association")
async def classify_medical_association(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """Classify problem statement to relevant medical association"""
      # Get technology details
    technology = db.query(Technology)\
        .filter(Technology.id == technology_id)\
        .first()
    
    if not technology.problem_statement:
        raise HTTPException(status_code=404, detail="Problem statement not found")
    
    medical_assessment_service = MedicalAssessmentService(GPTService(db), db)
    
    acronym = await medical_assessment_service.classify_medical_association(technology.problem_statement)
    
    if acronym == "No medical associations found":
        raise HTTPException(
            status_code=404,
            detail="No relevant medical association found"
        )
        
    return {"acronym": acronym}

@router.get("/dental/{code}")
async def get_dental_fee(
    code: str,
    db: Session = Depends(get_db)
):
    """Get dental fee for a specific code"""
    fee = db.query(DentalFeeSchedule)\
        .filter(DentalFeeSchedule.code == code)\
        .first()
        
    if not fee:
        raise HTTPException(
            status_code=404,
            detail=f"Fee not found for code {code}"
        )
        
    return fee

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