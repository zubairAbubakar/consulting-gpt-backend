from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

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