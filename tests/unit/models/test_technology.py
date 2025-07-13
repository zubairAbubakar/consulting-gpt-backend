import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.base import Base
from app.models.technology import Technology, ComparisonAxis

class TestTechnologyModel:
    """Unit tests for Technology model"""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create in-memory SQLite engine for testing"""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def db_session(self, engine):
        """Create database session for each test"""
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            yield session
        finally:
            session.close()
    
    def test_technology_model_creation(self, db_session):
        """Test creating a Technology model instance"""
        tech = Technology(
            name="Test Technology",
            abstract="Test abstract",
            num_of_axes=3
        )
        
        db_session.add(tech)
        db_session.commit()
        
        # Verify the technology was created
        assert tech.id is not None
        assert tech.name == "Test Technology"
        assert tech.abstract == "Test abstract"
        assert tech.num_of_axes == 3
    
    def test_technology_model_attributes(self):
        """Test Technology model has required attributes"""
        tech = Technology()
        
        # Test that all expected attributes exist
        assert hasattr(tech, 'id')
        assert hasattr(tech, 'name')
        assert hasattr(tech, 'abstract')
        assert hasattr(tech, 'problem_statement')
        assert hasattr(tech, 'search_keywords')
        assert hasattr(tech, 'num_of_axes')
        assert hasattr(tech, 'market_analysis_summary')
    
    def test_technology_model_relationships(self):
        """Test Technology model has required relationships"""
        tech = Technology()
        
        # Test that all expected relationships exist
        assert hasattr(tech, 'comparison_axes')
        assert hasattr(tech, 'related_technologies')
        assert hasattr(tech, 'market_analyses')
        assert hasattr(tech, 'patent_searches')
        assert hasattr(tech, 'related_papers')
        assert hasattr(tech, 'pca_results')
        assert hasattr(tech, 'cluster_results')
        assert hasattr(tech, 'recommendations')
        assert hasattr(tech, 'medical_assessments')
        assert hasattr(tech, 'analysis_status')
    
    def test_technology_default_values(self, db_session):
        """Test Technology model default values"""
        tech = Technology(name="Test Default Values", abstract="Test")
        db_session.add(tech)
        db_session.commit()
        
        # Test default values (applies after saving to DB)
        assert tech.num_of_axes == 5  # Default value
    
    def test_comparison_axis_model(self, db_session):
        """Test ComparisonAxis model creation"""
        # First create a technology
        tech = Technology(
            name="Test Technology for Axis",
            abstract="Test abstract"
        )
        db_session.add(tech)
        db_session.commit()
        
        # Create a comparison axis
        axis = ComparisonAxis(
            technology_id=tech.id
        )
        db_session.add(axis)
        db_session.commit()
        
        assert axis.id is not None
        assert axis.technology_id == tech.id
