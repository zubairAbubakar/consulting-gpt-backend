from typing import Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

class AnalysisStatusBase(BaseModel):
    status: str = Field(..., description="Current status: pending, processing, complete, or error")
    component_name: str = Field(..., description="Name of the analysis component")

class AnalysisStatusCreate(AnalysisStatusBase):
    technology_id: int = Field(..., description="ID of the technology being analyzed")
    error_message: Optional[str] = Field(None, description="Error message if status is error")

class AnalysisStatusRead(AnalysisStatusBase):
    id: int = Field(..., description="Unique ID of the status record")
    technology_id: int = Field(..., description="ID of the technology being analyzed")
    started_at: Optional[Union[datetime, str]] = Field(None, description="When processing started (ISO format)")
    completed_at: Optional[Union[datetime, str]] = Field(None, description="When processing completed (ISO format)")
    error_message: Optional[str] = Field(None, description="Error message if status is error")

    class Config:
        from_attributes = True

class AnalysisStatusSummary(BaseModel):
    status: str = Field(..., description="Current status: pending, processing, complete, or error")
    startedAt: Optional[str] = Field(None, description="When processing started (ISO format)")
    completedAt: Optional[str] = Field(None, description="When processing completed (ISO format)")
    errorMessage: Optional[str] = Field(None, description="Error message if status is error")

class AnalysisStatusResponse(BaseModel):
    technologyId: int = Field(..., description="ID of the technology being analyzed")
    components: Dict[str, AnalysisStatusSummary] = Field(
        ...,
        description="Map of component names to their status details"
    )
    overall: str = Field(..., description="Overall status of the analysis process")
    pollingRecommendation: Dict[str, int] = Field(
        ..., 
        description="Recommendations for client polling behavior"
    )

    class Config:
        schema_extra = {
            "example": {
                "technologyId": 123,
                "components": {
                    "comparisonAxes": {
                        "status": "complete",
                        "startedAt": "2025-06-29T10:15:30Z",
                        "completedAt": "2025-06-29T10:16:45Z",
                        "errorMessage": None
                    },
                    "relatedPatents": {
                        "status": "processing",
                        "startedAt": "2025-06-29T10:16:50Z",
                        "completedAt": None,
                        "errorMessage": None
                    },
                    "pcaVisualization": {
                        "status": "pending",
                        "startedAt": None,
                        "completedAt": None,
                        "errorMessage": None
                    }
                },
                "overall": "processing",
                "pollingRecommendation": {
                    "intervalMs": 7000
                }
            }
        }

