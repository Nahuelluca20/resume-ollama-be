from fastapi import FastAPI
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

app.include_router(cv_router, prefix="/api/v1", tags=["cv"])

@app.get("/")
def read_root():
    return {
        "message": "CV Analysis API",
        "version": "0.1.0",
        "endpoints": {
            "analyze_cv": "/api/v1/analyze-cv",
            "health": "/api/v1/health"
        }
    }