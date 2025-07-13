"""Integration tests for Technology API endpoints"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.technology import Technology


class TestTechnologyAPIIntegration:
    """Integration tests for the Technology API"""

    @pytest.fixture(autouse=True)
    def setup_database(self, db_session: Session):
        """Setup test database with clean state"""
        # Clear any existing data
        db_session.query(Technology).delete()
        db_session.commit()
        self.db = db_session

    def test_create_technology_integration(self, client: TestClient):
        """Test creating a technology through the API"""
        technology_data = {
            "name": "Integration Test Technology",
            "abstract": "A technology created through integration testing",
            "problem_statement": "Testing the full API workflow"
        }

        response = client.post("/technologies/", json=technology_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == technology_data["name"]
        assert data["abstract"] == technology_data["abstract"]
        assert "id" in data
        assert data["num_of_axes"] == 5  # Default value

    def test_get_technology_integration(self, client: TestClient):
        """Test retrieving a technology through the API"""
        # First create a technology
        tech = Technology(
            name="Get Test Technology",
            abstract="Technology for get testing",
            num_of_axes=3
        )
        self.db.add(tech)
        self.db.commit()
        self.db.refresh(tech)

        # Now retrieve it via API
        response = client.get(f"/technologies/{tech.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == tech.id
        assert data["name"] == "Get Test Technology"
        assert data["num_of_axes"] == 3

    def test_list_technologies_integration(self, client: TestClient):
        """Test listing technologies through the API"""
        # Create multiple technologies
        techs = [
            Technology(name="Tech 1", abstract="Abstract 1"),
            Technology(name="Tech 2", abstract="Abstract 2"),
            Technology(name="Tech 3", abstract="Abstract 3")
        ]
        
        for tech in techs:
            self.db.add(tech)
        self.db.commit()

        # List via API
        response = client.get("/technologies/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        names = [tech["name"] for tech in data]
        assert "Tech 1" in names
        assert "Tech 2" in names
        assert "Tech 3" in names

    def test_technology_not_found_integration(self, client: TestClient):
        """Test 404 handling for non-existent technology"""
        response = client.get("/technologies/99999")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_create_invalid_technology_integration(self, client: TestClient):
        """Test validation errors when creating invalid technology"""
        invalid_data = {
            "name": "",  # Empty name should fail validation
            "abstract": "Valid abstract"
        }

        response = client.post("/technologies/", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data


class TestTechnologyDatabaseIntegration:
    """Test technology operations with real database interactions"""

    @pytest.fixture(autouse=True)
    def setup_database(self, db_session: Session):
        """Setup test database with clean state"""
        db_session.query(Technology).delete()
        db_session.commit()
        self.db = db_session

    def test_technology_crud_operations(self):
        """Test full CRUD operations on Technology model"""
        # Create
        tech = Technology(
            name="CRUD Test Technology",
            abstract="Testing CRUD operations",
            problem_statement="Full database testing",
            num_of_axes=7
        )
        self.db.add(tech)
        self.db.commit()
        self.db.refresh(tech)
        
        assert tech.id is not None
        assert tech.name == "CRUD Test Technology"
        assert tech.num_of_axes == 7

        # Read
        retrieved_tech = self.db.query(Technology).filter(
            Technology.id == tech.id
        ).first()
        assert retrieved_tech is not None
        assert retrieved_tech.name == "CRUD Test Technology"

        # Update
        retrieved_tech.abstract = "Updated abstract"
        self.db.commit()
        
        updated_tech = self.db.query(Technology).filter(
            Technology.id == tech.id
        ).first()
        assert updated_tech.abstract == "Updated abstract"

        # Delete
        self.db.delete(updated_tech)
        self.db.commit()
        
        deleted_tech = self.db.query(Technology).filter(
            Technology.id == tech.id
        ).first()
        assert deleted_tech is None

    def test_technology_unique_constraints(self):
        """Test database unique constraints"""
        # Create first technology
        tech1 = Technology(
            name="Unique Test",
            abstract="First technology"
        )
        self.db.add(tech1)
        self.db.commit()

        # Try to create duplicate
        tech2 = Technology(
            name="Unique Test",  # Same name
            abstract="Second technology"
        )
        self.db.add(tech2)
        
        # Should raise integrity error
        with pytest.raises(Exception):  # IntegrityError or similar
            self.db.commit()

    def test_technology_relationships(self):
        """Test technology model relationships"""
        tech = Technology(
            name="Relationship Test",
            abstract="Testing relationships"
        )
        self.db.add(tech)
        self.db.commit()
        self.db.refresh(tech)

        # Test that relationships exist (even if empty)
        assert hasattr(tech, 'comparison_axes')
        assert hasattr(tech, 'related_technologies')
        assert hasattr(tech, 'market_analyses')
        assert tech.comparison_axes == []
        assert tech.related_technologies == []
