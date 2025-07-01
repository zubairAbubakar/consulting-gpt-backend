from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.technology import AnalysisStatus, Technology
from app.schemas.analysis_status import AnalysisStatusCreate

class AnalysisStatusService:
    def __init__(self, db: Session):
        self.db = db

    def create_status(
        self, 
        technology_id: int, 
        component_name: str, 
        status: str = "pending",
        error_message: Optional[str] = None
    ) -> AnalysisStatus:
        """
        Create a new status record for a specific component of a technology analysis
        """
        # Check if the technology exists
        technology = self.db.query(Technology).filter(Technology.id == technology_id).first()
        if not technology:
            raise ValueError(f"Technology with id {technology_id} not found")
        
        # Create the status object
        db_status = AnalysisStatus(
            technology_id=technology_id,
            component_name=component_name,
            status=status,
            started_at=datetime.utcnow() if status == "processing" else None,
            completed_at=datetime.utcnow() if status in ["complete", "error"] else None,
            error_message=error_message
        )
        
        self.db.add(db_status)
        self.db.commit()
        self.db.refresh(db_status)
        return db_status
    
    def update_status(
        self, 
        technology_id: int, 
        component_name: str, 
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[AnalysisStatus]:
        """
        Update the status of a component for a technology analysis
        """
        # Find the existing status record
        db_status = self.db.query(AnalysisStatus).filter(
            AnalysisStatus.technology_id == technology_id,
            AnalysisStatus.component_name == component_name
        ).first()
        
        if not db_status:
            # If no record exists, create a new one
            return self.create_status(technology_id, component_name, status, error_message)
        
        # Update the status
        db_status.status = status
        
        # Update timestamps based on status
        if status == "processing" and not db_status.started_at:
            db_status.started_at = datetime.utcnow()
            
        if status in ["complete", "error"]:
            db_status.completed_at = datetime.utcnow()
            
        if error_message:
            db_status.error_message = error_message
        
        self.db.commit()
        self.db.refresh(db_status)
        return db_status
    
    def get_status(self, technology_id: int, component_name: str) -> Optional[AnalysisStatus]:
        """
        Get the status of a specific component for a technology
        """
        return self.db.query(AnalysisStatus).filter(
            AnalysisStatus.technology_id == technology_id,
            AnalysisStatus.component_name == component_name
        ).first()
    
    def get_all_statuses(self, technology_id: int) -> List[AnalysisStatus]:
        """
        Get all status records for a technology
        """
        return self.db.query(AnalysisStatus).filter(
            AnalysisStatus.technology_id == technology_id
        ).all()
    
    def initialize_all_statuses(self, technology_id: int) -> Dict[str, str]:
        """
        Initialize status records for all analysis components of a technology
        Returns a dictionary of component names and their statuses
        """
        # Define all possible components
        components = [
            "comparisonAxes",
            "relatedPatents", 
            "relatedPapers",
            "marketAnalysis", 
            "pcaVisualization", 
            "medicalAssessment",
            "clusterAnalysis"
        ]
        
        statuses = {}
        for component in components:
            status = self.create_status(technology_id, component)
            statuses[component] = status.status
            
        return statuses
    
    def get_status_summary(self, technology_id: int) -> Dict[str, Any]:
        """
        Get a summary of all component statuses for a technology
        Returns a structured response with overall status and polling recommendations
        """
        all_statuses = self.get_all_statuses(technology_id)
        
        # If no statuses found, initialize them
        if not all_statuses:
            self.initialize_all_statuses(technology_id)
            all_statuses = self.get_all_statuses(technology_id)
        
        components = {}
        for status in all_statuses:
            components[status.component_name] = {
                "status": status.status,
                "startedAt": status.started_at.isoformat() if status.started_at else None,
                "completedAt": status.completed_at.isoformat() if status.completed_at else None,
                "errorMessage": status.error_message
            }
        
        # Calculate overall status
        statuses = [s.status for s in all_statuses]
        overall = self._calculate_overall_status(statuses)
        
        # Calculate recommended polling interval
        polling_interval = self._calculate_polling_interval(statuses)
        
        return {
            "technologyId": technology_id,
            "components": components,
            "overall": overall,
            "pollingRecommendation": {
                "intervalMs": polling_interval
            }
        }
    
    def _calculate_overall_status(self, statuses: List[str]) -> str:
        """
        Calculate the overall status based on component statuses
        """
        if not statuses:
            return "pending"
            
        if any(s == "error" for s in statuses):
            return "error"
            
        if all(s == "complete" for s in statuses):
            return "complete"
            
        if any(s == "processing" for s in statuses):
            return "processing"
            
        return "pending"
    
    def _calculate_polling_interval(self, statuses: List[str]) -> int:
        """
        Calculate the recommended polling interval based on status
        Returns interval in milliseconds
        """
        # If most components are still pending, poll less frequently
        if statuses.count("pending") > len(statuses) // 2:
            return 10000  # 10 seconds
        
        # If many components are processing, poll more frequently
        if statuses.count("processing") > 1:
            return 5000  # 5 seconds
        
        # Default polling interval
        return 7000  # 7 seconds
