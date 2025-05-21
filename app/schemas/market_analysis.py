from typing import Optional

from pydantic import BaseModel


class MarketAnalysisBase(BaseModel):
    score: float
    explanation: Optional[str] = None
    confidence: float

class MarketAnalysisRead(MarketAnalysisBase):
    id: int
    technology_id: int
    related_technology_id: int
    axis_id: int

    class Config:
        from_attributes = True