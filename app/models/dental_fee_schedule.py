from sqlalchemy import Column, Integer, String, Float
from app.models.base import Base

class DentalFeeSchedule(Base):
    __tablename__ = "dental_fee_schedules"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(10), unique=True, index=True)
    description = Column(String(500))
    average_fee = Column(Float)
    std_deviation = Column(Float)
    percentile_10th = Column(Float)
    percentile_25th = Column(Float)
    percentile_50th = Column(Float)
    percentile_75th = Column(Float)
    percentile_90th = Column(Float)
    num_responses = Column(Integer)
    