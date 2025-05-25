from sqlalchemy import Column, Float, Integer, String, Text
from app.models.base import Base


class FeeSchedule(Base):
    __tablename__ = "fee_schedules"

    id = Column(Integer, primary_key=True, index=True)
    hcpcs_code = Column(String(50), unique=True, index=True)
    description = Column(Text)
    fee = Column(Float)
