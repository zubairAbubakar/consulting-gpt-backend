from sqlalchemy import Boolean, Column, Integer, String, Text, ForeignKey, Float, DateTime, JSON
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
    market_analysis_summary = Column(Text, nullable=True)  # Summary insight of market analysis

    # Relationships
    comparison_axes = relationship("ComparisonAxis", back_populates="technology")
    related_technologies = relationship("RelatedTechnology", back_populates="technology")
    market_analyses = relationship("MarketAnalysis", back_populates="technology")
    patent_searches = relationship("PatentSearch", back_populates="technology")
    related_papers = relationship("RelatedPaper", back_populates="technology")
    pca_results = relationship("PCAResult", back_populates="technology")
    cluster_results = relationship("ClusterResult", back_populates="technology")
    recommendations = relationship("Recommendation", back_populates="technology")
    medical_assessments = relationship("MedicalAssessment", back_populates="technology")
    analysis_status = relationship("AnalysisStatus", back_populates="technology")

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
    market_analyses = relationship("MarketAnalysis", back_populates="comparison_axis")

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
    market_analyses = relationship("MarketAnalysis", back_populates="related_technology")
    cluster_memberships = relationship("ClusterMember", back_populates="related_technology")

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

class MarketAnalysis(Base):
    __tablename__ = "market_analysis"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    related_technology_id = Column(Integer, ForeignKey("related_technology.id"))
    axis_id = Column(Integer, ForeignKey("comparison_axis.id"))
    score = Column(Float)  # -1 to 1 score
    explanation = Column(Text, nullable=True)
    confidence = Column(Float)
    
    # Relationships
    technology = relationship("Technology", back_populates="market_analyses")
    related_technology = relationship("RelatedTechnology", back_populates="market_analyses")
    comparison_axis = relationship("ComparisonAxis", back_populates="market_analyses")

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

class PCAResult(Base):
    __tablename__ = "pca_results"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    components = Column(JSON)  # Stores component info including variance ratios
    transformed_data = Column(JSON)  # Stores transformed coordinates
    total_variance_explained = Column(Float)

    technology = relationship("Technology", back_populates="pca_results")

class AnalysisStatus(Base):
    __tablename__ = "analysis_status"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    component_name = Column(String(255)) 
    status = Column(String(100))  # e.g., processing, 'pending', 'complete', 'error'
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)  # Nullable for ongoing processes    
    error_message = Column(Text, nullable=True)  # Nullable for successful processes

    technology = relationship("Technology", back_populates="analysis_status")    

class ClusterResult(Base):
    __tablename__ = "cluster_results"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    name = Column(String(255))
    description = Column(Text)
    contains_target = Column(Boolean, default=False)
    center_x = Column(Float)  # X coordinate of cluster center
    center_y = Column(Float)  # Y coordinate of cluster center
    cluster_spread = Column(Float)  # Average distance to center
    technology_count = Column(Integer)  # Number of technologies in cluster

    # Relationships
    technology = relationship("Technology", back_populates="cluster_results")
    cluster_members = relationship("ClusterMember", back_populates="cluster")

class ClusterMember(Base):
    __tablename__ = "cluster_members"

    id = Column(Integer, primary_key=True, index=True)
    cluster_id = Column(Integer, ForeignKey("cluster_results.id"))
    technology_id = Column(Integer, ForeignKey("related_technology.id"))
    distance_to_center = Column(Float)  # Distance to cluster center
    
    # Relationships
    cluster = relationship("ClusterResult", back_populates="cluster_members")
    related_technology = relationship(
        "RelatedTechnology",
        back_populates="cluster_memberships",
        foreign_keys=[technology_id]
    )

class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    general_assessment = Column(Text)
    logistical_showstoppers = Column(Text)
    market_showstoppers = Column(Text)
    current_stage = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    technology = relationship("Technology", back_populates="recommendations")

class MedicalAssessment(Base):
    __tablename__ = "medical_assessments"

    id = Column(Integer, primary_key=True, index=True)
    technology_id = Column(Integer, ForeignKey("technology.id"))
    medical_association = Column(String(255))
    guidelines = Column(Text)
    recommendations = Column(Text)
    
    # Relationships
    technology = relationship("Technology", back_populates="medical_assessments")
    billable_items = relationship("BillableItem", back_populates="medical_assessment")
    guidelines = relationship("Guidelines", back_populates="medical_assessment")

class BillableItem(Base):
    __tablename__ = "billable_items"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("medical_assessments.id"))
    hcpcs_code = Column(String(50))
    description = Column(Text)
    fee = Column(Float)
    
    # Relationships
    medical_assessment = relationship("MedicalAssessment", back_populates="billable_items")

class Guidelines(Base):
    __tablename__ = "guidelines"

    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("medical_assessments.id"))
    title = Column(String(500))
    link = Column(String(1000))
    relevance_score = Column(Float)
    content = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)  # Track source: PubMed, CMS, FDA, etc.

    # Relationships
    medical_assessment = relationship("MedicalAssessment", back_populates="guidelines")

