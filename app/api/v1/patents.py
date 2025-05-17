from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.services.patent_service import PatentService
from app.models.technology import PatentSearch, PatentResult
from app.schemas.patent import PatentSearchResponse, PatentResultBase, PatentSearchBase
from app.dependencies import get_db

router = APIRouter()

@router.post("/search/{technology_id}", response_model=PatentSearchResponse)
async def search_patents(
    technology_id: int,
    search_query: str,
    db: Session = Depends(get_db)
):
    """
    Search for patents related to a technology and store results
    """
    patent_service = PatentService(db)
    result = await patent_service.search_patents(technology_id, search_query)
    
    if not result:
        raise HTTPException(status_code=500, detail="Patent search failed")
    
    return result

@router.get("/search/{search_id}/results", response_model=List[PatentResultBase])
async def get_search_results(
    search_id: int,
    db: Session = Depends(get_db)
):
    """
    Get results for a specific patent search
    """
    results = db.query(PatentResult).filter(PatentResult.search_id == search_id).all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this search")
        
    return results

@router.get("/technology/{technology_id}/searches", response_model=List[PatentSearchBase])
async def get_technology_searches(
    technology_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all patent searches for a specific technology
    """
    searches = db.query(PatentSearch).filter(PatentSearch.technology_id == technology_id).all()
    return searches

@router.patch("/result/{patent_result_id}/details", response_model=PatentResultBase)
async def fetch_patent_details(
    patent_result_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch and update detailed information for a specific patent result
    """
    patent_service = PatentService(db)
    
    # Get the patent result
    patent_result = db.query(PatentResult).get(patent_result_id)
    if not patent_result:
        raise HTTPException(status_code=404, detail="Patent result not found")
    
    # Update the details
    success = await patent_service.update_patent_details(patent_result_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update patent details")
    
    # Return the updated patent result
    db.refresh(patent_result)
    return patent_result

@router.patch("/search/{search_id}/details", response_model=List[PatentResultBase])
async def fetch_search_details(
    search_id: int,
    db: Session = Depends(get_db)
):
    """
    Fetch and update details for all patents in a search
    """
    # Get all patent results for this search
    patent_results = db.query(PatentResult).filter(PatentResult.search_id == search_id).all()
    if not patent_results:
        raise HTTPException(status_code=404, detail="No results found for this search")
    
    patent_service = PatentService(db)
    
    # Update details for each patent result
    update_tasks = []
    for patent_result in patent_results:
        update_tasks.append(patent_service.update_patent_details(patent_result.id))
    
    # Wait for all updates to complete
    import asyncio
    await asyncio.gather(*update_tasks)
    
    # Refresh and return the updated patent results
    updated_results = db.query(PatentResult).filter(PatentResult.search_id == search_id).all()
    return updated_results