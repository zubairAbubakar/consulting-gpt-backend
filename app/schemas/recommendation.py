from pydantic import BaseModel
from typing import Optional

class RecommendationRequest(BaseModel):
    current_stage: str
    stage_description: Optional[str] = None

class RecommendationResponse(BaseModel):
    general_assessment: str
    logistical_showstoppers: str
    market_showstoppers: str
    current_stage: str
    created_at: str