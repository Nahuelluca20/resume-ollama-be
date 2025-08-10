from fastapi import FastAPI
from app.api.cv_router import router as cv_router

app = FastAPI(
    title="CV Analysis API",
    description="CV parsing and analysis using Ollama",
    version="0.1.0"
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