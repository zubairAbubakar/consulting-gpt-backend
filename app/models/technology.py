from sqlalchemy import Column, Integer, String, Text, ForeignKey, Float, DateTime, JSON
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
    num_of_axes = Column(Integer, default=5)  # Default number of axes for technology comparison

    # Relationships
    comparison_axes = relationship("ComparisonAxis", back_populates="technology")
    related_technologies = relationship("RelatedTechnology", back_populates="technology")
    analysis_results = relationship("AnalysisResult", back_populates="technology")
    patent_searches = relationship("PatentSearch", back_populates="technology")
    related_papers = relationship("RelatedPaper", back_populates="technology")

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
    publication_date = Column(String(255))
    inventors = Column(JSON)
    assignees = Column(JSON)
    col = Column(Float, default=0.0)  # Column for storing additional information
    
    # Relationship
    technology = relationship("Technology", back_populates="related_technologies")
    analysis_results = relationship("AnalysisResult", back_populates="related_technology")

class RelatedPaper(Base):
    __tablename__ = "related_paper"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    paper_id = Column(String(255))  # PubMed ID
    title = Column(String(512))
    abstract = Column(Text)
    authors = Column(String(512))
    publication_date = Column(String(255))
    journal = Column(String(255))
    url = Column(String(512))
    citation_count = Column(Integer, default=0)
    col = Column(Float, default=0.0)  # Column for storing additional information  
    
    # Relationships
    technology = relationship("Technology", back_populates="related_papers")

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

class PatentSearch(Base):
    __tablename__ = "patent_search"
    
    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    search_query = Column(String(512))
    search_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    technology = relationship("Technology", back_populates="patent_searches")
    search_results = relationship("PatentResult", back_populates="search")

    def __init__(self, **kwargs):
        # Ensure search_query is stored as a normal string
        if 'search_query' in kwargs and isinstance(kwargs['search_query'], (list, tuple)):
            kwargs['search_query'] = ' '.join(kwargs['search_query'])
        super().__init__(**kwargs)    

class PatentResult(Base):
    __tablename__ = "patent_result"
    
    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, ForeignKey("patent_search.id"))
    patent_id = Column(String(255))  # The Google Patents ID
    title = Column(String(512))
    abstract = Column(Text)
    publication_date = Column(String(255))
    url = Column(String(512))
    
    # Relationships
    search = relationship("PatentSearch", back_populates="search_results")