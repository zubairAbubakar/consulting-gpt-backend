from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

class ClusterMemberBase(BaseModel):
    technology_id: int
    distance_to_center: float

class ClusterMemberRead(ClusterMemberBase):
    id: int
    cluster_id: int
    technology_name: str  # Added for convenience
    technology_abstract: str

    class Config:
        from_attributes = True

class ClusterResultBase(BaseModel):
    technology_id: int
    name: str
    description: str
    contains_target: bool
    center_x: float
    center_y: float
    cluster_spread: float
    technology_count: int

class ClusterResultCreate(ClusterResultBase):
    pass

class ClusterResultRead(ClusterResultBase):
    id: int
    created_at: datetime
    members: List[ClusterMemberRead]

    class Config:
        from_attributes = True

class ClusterDetailRead(ClusterResultRead):
    """Extended cluster information including member details"""
    members: List[ClusterMemberRead]