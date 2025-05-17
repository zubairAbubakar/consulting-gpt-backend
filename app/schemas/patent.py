from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class PatentResultBase(BaseModel):
    id: int
    search_id: int
    patent_id: str
    title: str
    abstract: str
    publication_date: str
    url: str

    class Config:
        from_attributes = True

class PatentSearchBase(BaseModel):
    id: int
    technology_id: int
    search_query: str
    search_date: datetime

    class Config:
        from_attributes = True

class PatentSearchResponse(PatentSearchBase):
    search_results: List[PatentResultBase]

    class Config:
        from_attributes = True