from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.cv_router import router as cv_router
from app.core.database import init_database, db_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Initializing database...")
    init_database()
    print("Database initialization completed.")
    print("Note: Run 'alembic upgrade head' to apply migrations")
    
    yield
    
    # Shutdown
    print("Closing database connections...")
    await db_manager.close()
    print("Database connections closed.")


app = FastAPI(
    title="CV Analysis API",
    description="CV parsing and analysis using Ollama",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cv_router, prefix="/api/v1", tags=["cv"])

@app.get("/")
def read_root():
    return {
        "message": "CV Analysis and Job Matching API",
        "version": "0.1.0",
        "endpoints": {
            "analyze_cv": "/api/v1/analyze-cv",
            "upload_cv": "/api/v1/cv/upload",
            "list_candidates": "/api/v1/candidates",
            "match_job": "/api/v1/match-job",
            "explain_match": "/api/v1/explain-match/{resume_id}",
            "health": "/api/v1/health",
            "models": "/api/v1/models"
        },
        "docs": "/docs"
    }