from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float, DateTime
from sqlalchemy.orm import relationship
from app.models.base import Base
from datetime import datetime

class Technology(Base):
    __tablename__ = "technology"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True)
    abstract = Column(Text)
    problem_statement = Column(Text)
    search_keywords = Column(Text, nullable=True)  # Adding search keywords column

    # Relationships
    comparison_axes = relationship("ComparisonAxis", back_populates="technology")
    related_technologies = relationship("RelatedTechnology", back_populates="technology")
    analysis_results = relationship("AnalysisResult", back_populates="technology")

class ComparisonAxis(Base):
    __tablename__ = "comparison_axis"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    axis_name = Column(String(255))
    extreme1 = Column(String(255))
    extreme2 = Column(String(255))
    weight = Column(Float, default=1.0)  # Added weight for importance
    
    # Relationship
    technology = relationship("Technology", back_populates="comparison_axes")
    analysis_results = relationship("AnalysisResult", back_populates="axis")

class RelatedTechnology(Base):
    __tablename__ = "related_technology"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    name = Column(String(255))
    abstract = Column(Text)
    document_id = Column(String(255))
    type = Column(String(50))  # 'patent' or 'paper'
    cluster = Column(Integer, nullable=True)
    url = Column(String(512), nullable=True)  # Added URL field
    
    # Relationship
    technology = relationship("Technology", back_populates="related_technologies")
    analysis_results = relationship("AnalysisResult", back_populates="related_technology")

class AnalysisResult(Base):
    __tablename__ = "analysis_result"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    related_technology_id = Column(Integer, ForeignKey("related_technology.id"))
    axis_id = Column(Integer, ForeignKey("comparison_axis.id"))
    score = Column(Float)
    explanation = Column(Text)
    confidence = Column(Float)
    
    # Relationships
    technology = relationship("Technology", back_populates="analysis_results")
    related_technology = relationship("RelatedTechnology", back_populates="analysis_results")
    axis = relationship("ComparisonAxis", back_populates="analysis_results")