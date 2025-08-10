import time
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.pdf_service import PDFService
from app.services.ollama_service import OllamaService
from app.schemas.cv_schemas import CVAnalysisResponse, HealthResponse, ErrorResponse

router = APIRouter()


@router.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    model: Optional[str] = Query(default=None, description="Ollama model to use for analysis")
):
    """
    Upload a PDF CV and get structured analysis using Ollama.
    
    - **file**: PDF file to analyze (max 10MB)
    - **model**: Optional Ollama model name (defaults to llama3.2)
    """
    start_time = time.time()
    
    try:
        # Validate file
        PDFService.validate_pdf_file(file.content_type, file.size)
        
        # Read PDF content
        pdf_content = await file.read()
        
        # Extract text from PDF
        extracted_text = await PDFService.extract_text_from_pdf(pdf_content)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text content found in PDF"
            )
        
        # Analyze with Ollama
        ollama_model = model or OllamaService.DEFAULT_MODEL
        analysis_result = await OllamaService.analyze_cv_with_ollama(
            extracted_text, 
            ollama_model
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create text preview
        text_preview = (
            extracted_text[:500] + "..." 
            if len(extracted_text) > 500 
            else extracted_text
        )
        
        # Structure response
        response = CVAnalysisResponse(
            success=True,
            filename=file.filename,
            file_size=file.size or len(pdf_content),
            extracted_text_preview=text_preview,
            analysis=analysis_result.get("analysis"),
            metadata={
                "model_used": analysis_result.get("model_used"),
                "raw_text_length": analysis_result.get("raw_text_length"),
                "parsing_error": analysis_result.get("parsing_error"),
                "raw_response": analysis_result.get("raw_response", "")[:1000]  # Limit raw response
            },
            processing_time=processing_time,
            created_at=datetime.utcnow()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint to verify Ollama connectivity and available models.
    """
    try:
        health_data = OllamaService.check_ollama_health()
        
        return HealthResponse(
            status=health_data["status"],
            ollama_available=health_data["ollama_available"],
            available_models=health_data["available_models"],
            error=health_data.get("error")
        )
        
    except Exception as e:
        return HealthResponse(
            status="error",
            ollama_available=False,
            error=str(e),
            available_models=[]
        )


@router.get("/models")
def list_models():
    """
    List available Ollama models for CV analysis.
    """
    try:
        health_data = OllamaService.check_ollama_health()
        
        if health_data["ollama_available"]:
            return {
                "success": True,
                "models": health_data["available_models"],
                "default_model": OllamaService.DEFAULT_MODEL
            }
        else:
            return {
                "success": False,
                "error": health_data.get("error", "Ollama service unavailable"),
                "models": []
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )