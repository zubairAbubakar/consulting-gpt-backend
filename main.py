from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import patents, technologies
from app.db.database import SessionLocal
from app.db.init_data import init_dental_fees, init_medical_associations

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize data
    db = SessionLocal()
    try:
        await init_dental_fees(db)
        await init_medical_associations(db)
    finally:
        db.close()
    
    yield  # Server is running and handling requests
    
    # Shutdown: Clean up resources if needed
    pass

app = FastAPI(
    title="Consulting GPT API",
    description="Backend API for Consulting GPT application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(patents.router, prefix="/api/v1/patents", tags=["patents"])
app.include_router(technologies.router, prefix="/api/v1/technologies", tags=["technologies"])

@app.get("/")
async def root():
    return {"message": "Welcome to Consulting GPT API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)