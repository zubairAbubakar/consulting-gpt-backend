from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

class VisualizationParams(BaseModel):
    show_clusters: bool = False
    viz_type: Literal["pca", "raw", "combined"] = "pca"
    selected_axes: Optional[List[str]] = None
    show_labels: bool = True
    show_annotations: bool = True

class TechnologyDetail(BaseModel):
    name: str
    abstract: str
    cluster_name: Optional[str] = None
    market_scores: Dict[str, float] = {}

class InteractiveFeatures(BaseModel):
    click_details: Dict[str, TechnologyDetail]
    annotations: Dict[str, bool]
    selected_axes: List[str]