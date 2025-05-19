from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Create SQLAlchemy engine
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import base and models to ensure they are registered
from app.models.base import Base
from app.models import Technology, ComparisonAxis, RelatedTechnology

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()