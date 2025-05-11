from datetime import datetime
from typing import Any
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import Column, DateTime

class BaseModel:
    id: Any
    __name__: str

    # Generate tablename automatically
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()
    
    # Common columns for all models
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base = declarative_base(cls=BaseModel)