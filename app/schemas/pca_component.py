# ...existing code...

from typing import Dict, List, Optional
from pydantic import BaseModel


class PCAComponentBase(BaseModel):
    component_number: int
    explained_variance_ratio: float
    description: Optional[str] = None

class PCAResultBase(BaseModel):
    technology_id: int
    components: List[PCAComponentBase]
    transformed_data: Dict[str, List[float]]  # technology_name -> [pc1, pc2, ...]
    total_variance_explained: float

class PCAResultRead(PCAResultBase):
    id: int

    class Config:
        from_attributes = True