from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Consulting GPT"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    POSTGRES_SERVER: str = "db"  # Changed from "db" to "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "consulting_gpt"
    DATABASE_URL: str = None #"postgresql://postgres:postgres@localhost/consulting_gpt"
    
    # External APIs
    OPENAI_API_KEY: str = "test-key"  # Default for testing
    SERPAPI_API_KEY: str = "test-key"  # Default for testing
    BIOPORTAL_API_KEY: str = "test-key"  # Default for testing, get free key from bioportal.bioontology.org
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]  # Frontend URL
    
    model_config = ConfigDict(env_file=".env", case_sensitive=True)

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()