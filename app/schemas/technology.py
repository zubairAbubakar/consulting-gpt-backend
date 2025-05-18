from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime

class ComparisonAxisBase(BaseModel):
    axis_name: str
    extreme1: str
    extreme2: str
    weight: float = 1.0

class ComparisonAxisCreate(ComparisonAxisBase):
    pass

class ComparisonAxisRead(ComparisonAxisBase):
    id: int
    technology_id: int

    class Config:
        from_attributes = True

class RelatedTechnologyBase(BaseModel):
    name: str
    abstract: str
    document_id: str
    type: str
    cluster: Optional[int] = None
    url: Optional[str] = None
    publication_date: Optional[str] = None
    inventors: Optional[str] = None
    assignees: Optional[str] = None
    col: Optional[float] = 0.0

class RelatedTechnologyRead(RelatedTechnologyBase):
    id: int
    technology_id: int

    class Config:
        from_attributes = True

class RelatedPaperBase(BaseModel):
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = 0
    col: Optional[float] = 0.0

class RelatedPaperRead(RelatedPaperBase):
    id: int
    technology_id: int

    class Config:
        from_attributes = True

class AnalysisResultBase(BaseModel):
    score: float
    explanation: str
    confidence: float

class AnalysisResultRead(AnalysisResultBase):
    id: int
    technology_id: int
    related_technology_id: int
    axis_id: int

    class Config:
        from_attributes = True

class TechnologyBase(BaseModel):
    name: str
    abstract: str
    num_of_axes: int = 5

class TechnologyCreate(TechnologyBase):
    pass

class TechnologyRead(TechnologyBase):
    id: int
    search_keywords: Optional[str] = None
    problem_statement: Optional[str] = None

    class Config:
        from_attributes = True

class TechnologyDetailRead(TechnologyRead):
    comparison_axes: List[ComparisonAxisRead] = []
    related_technologies: List[RelatedTechnologyRead] = []
    related_papers: List[RelatedPaperRead] = []

    class Config:
        from_attributes = True

class TechnologySearchQuery(BaseModel):
    query: str

class PatentSearchCreate(BaseModel):
    technology_id: int
    search_query: str