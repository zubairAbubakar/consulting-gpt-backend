from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.db.database import Base

class Technology(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True)
    abstract = Column(Text)
    problem_statement = Column(Text)
    
    # Relationships
    comparison_axes = relationship("ComparisonAxis", back_populates="technology")
    related_technologies = relationship("RelatedTechnology", back_populates="technology")
    analysis_results = relationship("AnalysisResult", back_populates="technology")

class ComparisonAxis(Base):
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    axis_name = Column(String(255))
    extreme1 = Column(String(255))
    extreme2 = Column(String(255))
    
    # Relationship
    technology = relationship("Technology", back_populates="comparison_axes")
    analysis_results = relationship("AnalysisResult", back_populates="axis")

class RelatedTechnology(Base):
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    name = Column(String(255))
    abstract = Column(Text)
    document_id = Column(String(255))
    type = Column(String(50))  # 'patent' or 'paper'
    cluster = Column(Integer, nullable=True)
    
    # Relationship
    technology = relationship("Technology", back_populates="related_technologies")
    analysis_results = relationship("AnalysisResult", back_populates="related_technology")

class AnalysisResult(Base):
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    axis_id = Column(Integer, ForeignKey("comparisonaxis.id"))
    related_technology_id = Column(Integer, ForeignKey("relatedtechnology.id"))
    score = Column(Float)
    
    # Relationships
    technology = relationship("Technology", back_populates="analysis_results")
    axis = relationship("ComparisonAxis", back_populates="analysis_results")
    related_technology = relationship("RelatedTechnology", back_populates="analysis_results")