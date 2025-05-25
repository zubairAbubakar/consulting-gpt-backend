from sqlalchemy import Column, Integer, String

from app.models.base import Base


class MedicalAssociation(Base):
    __tablename__ = "medical_associations"

    id = Column(Integer, primary_key=True, index=True)
    acronym = Column(String(50), unique=True, index=True)  
    organization = Column(String(500)) 
