import csv
import logging
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session
from app.db.base import Base
from app.db.database import engine
from app.models.dental_fee_schedule import DentalFeeSchedule
from app.models.medical_association import MedicalAssociation

logger = logging.getLogger(__name__)

def init_db() -> None:
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

async def init_dental_fees(db: Session) -> None:
    """Initialize dental fee schedule if empty"""
    try:
        # First ensure tables exist
        init_db()

        # Check if already populated
        if db.query(DentalFeeSchedule).first():
            logger.info("Dental fee schedule already populated")
            return {"message": "Dental fee schedule already exists"}

        csv_path = Path(__file__).parent.parent.parent / "data" / "dental_fee_schedule.csv"

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dental fee schedule CSV not found at {csv_path}")

        # Read CSV file with specific numeric handling
        df = pd.read_csv(csv_path)
        
        # Clean numeric columns by removing commas and converting to float
        numeric_columns = [
            'Average Fee', 
            'Standard\nDeviation\n$',
            '10th\n$', 
            '25th\n$', 
            'Median\n50th\n$',
            '75th\n$',
            '90th\n$'
        ]
        
        for col in numeric_columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
        # Process each row
        for _, row in df.iterrows():
            fee_item = DentalFeeSchedule(
                code=row['Code'],
                description=row['Description of Service'],
                average_fee=float(row['Average Fee']),
                std_deviation=float(row['Standard\nDeviation\n$']),
                percentile_10th=float(row['10th\n$']),
                percentile_25th=float(row['25th\n$']),
                percentile_50th=float(row['Median\n50th\n$']),
                percentile_75th=float(row['75th\n$']),
                percentile_90th=float(row['90th\n$']),
                num_responses=int(row['Number of\nResponses'])
            )
            db.add(fee_item)

        db.commit()
        logger.info("Dental fee schedule populated successfully")
        return {"message": "Dental fee schedule populated successfully"}

    except Exception as e:
        logger.error(f"Error populating dental fee schedule: {e}")
        db.rollback()
        raise

async def init_medical_associations(db: Session) -> None:
    """Initialize medical associations table if empty"""
    try:
        # First ensure tables exist
        init_db()
        
        # Check if already populated
        if db.query(MedicalAssociation).first():
            logger.info("Medical associations table already populated")
            return
            
        # Get path to CSV file
        csv_path = Path(__file__).parent.parent.parent / "data" / "medical_associations.csv"
        
        if not csv_path.exists():
            logger.warning(f"Medical associations CSV not found at {csv_path}")
            return
            
        # Read from CSV and populate
        with open(csv_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                association = MedicalAssociation(
                    acronym=row['Acronym'],
                    organization=row['Organization']
                )
                db.add(association)
        
        db.commit()
        logger.info("Medical associations table populated successfully")
        
    except Exception as e:
        logger.error(f"Error initializing medical associations: {e}")
        db.rollback()