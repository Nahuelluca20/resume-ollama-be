import time
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from app.services.pdf_service import PDFService
from app.services.ollama_service import OllamaService
from app.services.database_service import DatabaseService
from app.services.embedding_service import EmbeddingService
from app.schemas.cv_schemas import CVAnalysisResponse, HealthResponse, ErrorResponse
from app.core.database import get_db_session

router = APIRouter()


@router.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    model: Optional[str] = Query(default=None, description="Ollama model to use for analysis"),
    store_in_db: bool = Query(default=True, description="Store analysis results in database"),
    generate_embeddings: bool = Query(default=True, description="Generate embeddings for semantic search"),
    db: AsyncSession = Depends(get_db_session)
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
        
        # Handle case where analysis is None (parsing failed)
        analysis = analysis_result.get("analysis")
        if analysis is None:
            # Create empty analysis structure when parsing fails
            from app.schemas.cv_schemas import CVAnalysisSchema, PersonalInfoSchema
            analysis = CVAnalysisSchema(
                personal_info=PersonalInfoSchema(),
                summary=None,
                skills=[],
                experience=[],
                education=[],
                certifications=[],
                languages=[]
            )
        
        # Store in database if requested
        resume_id = None
        embeddings_generated = False
        
        if store_in_db and analysis:
            try:
                db_service = DatabaseService(db)
                resume = await db_service.store_cv_analysis(
                    filename=file.filename,
                    file_content=pdf_content,
                    raw_text=extracted_text,
                    analysis=analysis
                )
                resume_id = resume.id
                
                # Generate and store embeddings if requested
                if generate_embeddings:
                    try:
                        embeddings = await EmbeddingService.generate_cv_embeddings(
                            analysis, extracted_text
                        )
                        
                        # Store each embedding section
                        for section_type, embedding in embeddings.items():
                            if section_type == "full_text":
                                content = extracted_text
                            elif section_type == "summary":
                                content = analysis.summary
                            elif section_type == "skills":
                                content = " ".join([skill.name for skill in analysis.skills if skill.name])
                            elif section_type == "experience":
                                content = " | ".join([
                                    f"{exp.role} at {exp.company}" for exp in analysis.experience 
                                    if exp.role and exp.company
                                ])
                            elif section_type == "education":
                                content = " | ".join([
                                    f"{edu.degree} in {edu.field} from {edu.institution}" 
                                    for edu in analysis.education 
                                    if edu.degree and edu.institution
                                ])
                            elif section_type == "certifications":
                                content = " | ".join(analysis.certifications)
                            else:
                                content = ""
                            
                            if content and embedding:
                                await db_service.store_embedding(
                                    resume_id=resume.id,
                                    section_type=section_type,
                                    content=content,
                                    embedding=embedding,
                                    model_name=EmbeddingService.DEFAULT_EMBEDDING_MODEL
                                )
                        
                        embeddings_generated = True
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate embeddings: {str(e)}")
                        
            except Exception as e:
                print(f"Warning: Failed to store in database: {str(e)}")
        
        # Structure response
        response = CVAnalysisResponse(
            success=True,
            filename=file.filename,
            file_size=file.size or len(pdf_content),
            extracted_text_preview=text_preview,
            analysis=analysis,
            metadata={
                "model_used": analysis_result.get("model_used"),
                "raw_text_length": analysis_result.get("raw_text_length"),
                "parsing_error": analysis_result.get("parsing_error"),
                "raw_response": analysis_result.get("raw_response", "")[:1000],  # Limit raw response
                "stored_in_db": store_in_db and resume_id is not None,
                "resume_id": resume_id,
                "embeddings_generated": embeddings_generated
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