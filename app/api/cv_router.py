import time
import uuid
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi.responses import JSONResponse

from app.services.pdf_service import PDFService
from app.services.ollama_service import OllamaService
from app.services.database_service import DatabaseService
from app.services.embedding_service import EmbeddingService
from app.services.matching_service import MatchingService
from app.schemas.cv_schemas import CVAnalysisResponse, HealthResponse, ErrorResponse, JobDescriptionRequest, JobMatchResponse, CandidateListResponse, CandidateSummary
from app.core.database import get_db_session
from app.models.cv_models import CVProcessingJob, Resume, Candidate, Experience, Education, Skill, ResumeSkill

router = APIRouter()


@router.post("/cv/upload")
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Query(default=None, description="Ollama model to use for analysis"),
    store_in_db: bool = Query(default=True, description="Store analysis results in database"),
    generate_embeddings: bool = Query(default=True, description="Generate embeddings for semantic search"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a PDF CV and start background processing.
    Returns job_id immediately for status polling.
    """
    try:
        # Validate file immediately
        PDFService.validate_pdf_file(file.content_type, file.size)
        
        # Read file content
        file_content = await file.read()
        
        # Create processing job
        job_id = str(uuid.uuid4())
        job = CVProcessingJob(
            id=job_id,
            original_filename=file.filename or "uploaded_cv.pdf",
            file_size=file.size or len(file_content),
            status="uploaded",
            current_step="File uploaded, queued for processing",
            estimated_completion=datetime.utcnow() + timedelta(seconds=30),
            file_content=file_content
        )
        
        # Save job to database
        db.add(job)
        await db.commit()
        
        # Start background processing
        background_tasks.add_task(
            process_cv_background,
            job_id,
            file_content,
            file.filename or "uploaded_cv.pdf",
            model or OllamaService.DEFAULT_MODEL,
            store_in_db,
            generate_embeddings
        )
        
        return {
            "job_id": job_id,
            "status": "uploaded",
            "estimated_time": "30 seconds",
            "message": "CV uploaded successfully. Processing started.",
            "poll_url": f"/api/cv/status/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/cv/status/{job_id}")
async def get_cv_status(
    job_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get processing status for a CV job."""
    try:
        result = await db.execute(select(CVProcessingJob).where(CVProcessingJob.id == job_id))
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response = {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress_percentage,
            "current_step": job.current_step,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "estimated_completion": job.estimated_completion
        }
        
        if job.status == "completed" and job.resume_id:
            response["cv_id"] = job.resume_id
            response["redirect_url"] = f"/api/cv/{job.resume_id}"
            response["success"] = True
        
        if job.status == "failed":
            response["error"] = job.error_message
            response["success"] = False
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")


@router.get("/cv/{cv_id}")
async def get_cv_analysis(
    cv_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Get completed CV analysis results."""
    try:
        # Get resume with related data
        result = await db.execute(
            select(Resume)
            .where(Resume.id == cv_id)
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="CV not found")
        
        # Get candidate info
        candidate_result = await db.execute(
            select(Candidate).where(Candidate.id == resume.candidate_id)
        )
        candidate = candidate_result.scalar_one_or_none()
        
        # Get related data through database service
        db_service = DatabaseService(db)
        
        return {
            "cv_id": cv_id,
            "filename": resume.original_filename,
            "summary": resume.summary,
            "candidate": {
                "name_hash": candidate.name_hash if candidate else None,
                "email_hash": candidate.email_hash if candidate else None,
                "phone_hash": candidate.phone_hash if candidate else None,
                "location": candidate.location if candidate else None,
            },
            "raw_text_preview": resume.raw_text[:500] + "..." if len(resume.raw_text) > 500 else resume.raw_text,
            "created_at": resume.created_at,
            "updated_at": resume.updated_at,
            "metadata": {
                "file_hash": resume.file_hash,
                "raw_text_length": len(resume.raw_text)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving CV: {str(e)}")


@router.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    model: Optional[str] = Query(
        default=None, description="Ollama model to use for analysis"
    ),
    store_in_db: bool = Query(
        default=True, description="Store analysis results in database"
    ),
    generate_embeddings: bool = Query(
        default=True, description="Generate embeddings for semantic search"
    ),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Upload a PDF CV and get structured analysis using Ollama.

    - **file**: PDF file to analyze (max 10MB)
    - **model**: Optional Ollama model name (defaults to gpt-oss:20b)
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
            raise HTTPException(status_code=400, detail="No text content found in PDF")

        # Analyze with Ollama
        ollama_model = model or OllamaService.DEFAULT_MODEL
        analysis_result = await OllamaService.analyze_cv_with_ollama(
            extracted_text, ollama_model
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
                languages=[],
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
                    analysis=analysis,
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
                                content = " ".join(
                                    [
                                        skill.name
                                        for skill in analysis.skills
                                        if skill.name
                                    ]
                                )
                            elif section_type == "experience":
                                content = " | ".join(
                                    [
                                        f"{exp.role} at {exp.company}"
                                        for exp in analysis.experience
                                        if exp.role and exp.company
                                    ]
                                )
                            elif section_type == "education":
                                content = " | ".join(
                                    [
                                        f"{edu.degree} in {edu.field} from {edu.institution}"
                                        for edu in analysis.education
                                        if edu.degree and edu.institution
                                    ]
                                )
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
                                    model_name=EmbeddingService.DEFAULT_EMBEDDING_MODEL,
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
                "raw_response": analysis_result.get("raw_response", "")[
                    :1000
                ],  # Limit raw response
                "stored_in_db": store_in_db and resume_id is not None,
                "resume_id": resume_id,
                "embeddings_generated": embeddings_generated,
            },
            processing_time=processing_time,
            created_at=datetime.utcnow(),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
            error=health_data.get("error"),
        )

    except Exception as e:
        return HealthResponse(
            status="error", ollama_available=False, error=str(e), available_models=[]
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
                "default_model": OllamaService.DEFAULT_MODEL,
            }
        else:
            return {
                "success": False,
                "error": health_data.get("error", "Ollama service unavailable"),
                "models": [],
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/match-job", response_model=JobMatchResponse)
async def match_job_to_cvs(
    job_request: JobDescriptionRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Match a job description against stored CVs and return ranked candidates.
    
    - **job_description**: The job posting text to match against
    - **required_skills**: List of required skills (optional)
    - **preferred_skills**: List of preferred skills (optional)
    - **minimum_experience_years**: Minimum years of experience required (optional)
    - **location_preference**: Preferred candidate location (optional)
    - **max_results**: Maximum number of matches to return (default: 10)
    """
    start_time = time.time()
    
    try:
        # Find matches using the matching service
        match_results = await MatchingService.find_matches_with_session(job_request, db)
        
        if not match_results["success"]:
            raise HTTPException(status_code=500, detail="Matching service failed")
        
        processing_time = time.time() - start_time
        
        # Create job description preview (first 200 characters)
        job_preview = (
            job_request.job_description[:200] + "..."
            if len(job_request.job_description) > 200
            else job_request.job_description
        )
        
        return JobMatchResponse(
            success=True,
            job_description_preview=job_preview,
            total_cvs_analyzed=match_results["total_cvs_analyzed"],
            matches=match_results["matches"],
            processing_time=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching job to CVs: {str(e)}")


@router.post("/explain-match/{resume_id}")
async def explain_match(
    resume_id: int,
    job_request: JobDescriptionRequest,
    model: Optional[str] = Query(default=None, description="Ollama model to use for explanation"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Generate a detailed explanation of why a specific CV matches a job description.
    
    - **resume_id**: ID of the resume to explain the match for
    - **job_request**: Job description and requirements to match against
    - **model**: Optional Ollama model name (defaults to gpt-oss:20b)
    """
    try:
        # Get the resume and candidate data
        result = await db.execute(
            select(Resume, Candidate).join(Candidate).where(Resume.id == resume_id)
        )
        resume_candidate = result.first()
        
        if not resume_candidate:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        resume, candidate = resume_candidate
        
        # Calculate match scores
        job_embedding = await EmbeddingService.generate_job_embedding(job_request.job_description)
        match = await MatchingService._calculate_match_score(
            resume, candidate, job_request, job_embedding, db
        )
        
        if not match:
            raise HTTPException(status_code=500, detail="Could not calculate match scores")
        
        # Prepare data for explanation
        candidate_summary = resume.summary or "No summary available"
        experience_summary = " | ".join([
            f"{exp.role} at {exp.company}" for exp in match.relevant_experience
            if exp.role and exp.company
        ]) or "No relevant experience found"
        
        # Generate explanation using Ollama
        explanation = await OllamaService.generate_match_explanation(
            job_description=job_request.job_description,
            candidate_summary=candidate_summary,
            matched_skills=match.matched_skills,
            experience_summary=experience_summary,
            match_scores={
                "overall_score": match.match_score.overall_score,
                "skill_match_score": match.match_score.skill_match_score,
                "experience_match_score": match.match_score.experience_match_score,
                "semantic_similarity_score": match.match_score.semantic_similarity_score
            },
            model=model or OllamaService.DEFAULT_MODEL
        )
        
        # Update the match object with the explanation
        match.match_score.explanation = explanation
        
        return {
            "success": True,
            "resume_id": resume_id,
            "candidate_name": match.candidate_name,
            "match": match,
            "detailed_explanation": explanation,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating match explanation: {str(e)}")


@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(
    page: int = Query(1, ge=1, description="Page number (starting from 1)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of candidates per page"),
    skill_filter: Optional[str] = Query(None, description="Filter candidates by skill name"),
    location_filter: Optional[str] = Query(None, description="Filter candidates by location"),
    min_experience: Optional[int] = Query(None, ge=0, description="Minimum years of experience"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    List all candidates with their summary information.
    
    - **page**: Page number for pagination (default: 1)
    - **page_size**: Number of candidates per page (default: 10, max: 100)
    - **skill_filter**: Filter candidates by skill name (partial match)
    - **location_filter**: Filter candidates by location (partial match)
    - **min_experience**: Minimum years of experience
    """
    try:
        # Build base query
        query = select(Resume, Candidate).join(Candidate)
        
        # Apply filters if provided
        if location_filter:
            query = query.where(Candidate.location.ilike(f"%{location_filter}%"))
        
        # Get total count for pagination
        count_query = select(Resume.id).join(Candidate)
        if location_filter:
            count_query = count_query.where(Candidate.location.ilike(f"%{location_filter}%"))
        
        total_result = await db.exec(count_query)
        total_candidates = len(total_result.all())
        
        # Calculate pagination
        total_pages = (total_candidates + page_size - 1) // page_size
        offset = (page - 1) * page_size
        
        # Apply pagination
        query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.exec(query)
        resumes_candidates = result.all()
        
        candidates_summary = []
        
        for resume, candidate in resumes_candidates:
            # Get latest experience
            exp_query = select(Experience).where(Experience.resume_id == resume.id).order_by(Experience.order_index.desc())
            exp_result = await db.exec(exp_query)
            latest_experience = exp_result.first()
            
            # Get top skills (limit to 5)
            skills_query = select(ResumeSkill, Skill).join(Skill).where(ResumeSkill.resume_id == resume.id).limit(5)
            skills_result = await db.exec(skills_query)
            skills_data = skills_result.all()
            top_skills = [skill.name for resume_skill, skill in skills_data if skill.name]
            
            # Apply skill filter if provided
            if skill_filter:
                skill_match = any(skill_filter.lower() in skill.lower() for skill in top_skills)
                if not skill_match:
                    continue
            
            # Get highest education level
            edu_query = select(Education).where(Education.resume_id == resume.id).order_by(Education.order_index.desc())
            edu_result = await db.exec(edu_query)
            latest_education = edu_result.first()
            
            # Calculate years of experience (simplified)
            years_experience = None
            if latest_experience and latest_experience.start_date and latest_experience.end_date:
                try:
                    import re
                    start_year = int(re.search(r'\d{4}', latest_experience.start_date).group())
                    end_year = int(re.search(r'\d{4}', latest_experience.end_date).group())
                    years_experience = end_year - start_year if end_year >= start_year else None
                except (ValueError, AttributeError):
                    years_experience = None
            
            # Apply experience filter
            if min_experience is not None and years_experience is not None:
                if years_experience < min_experience:
                    continue
            
            candidate_summary = CandidateSummary(
                resume_id=resume.id,
                candidate_id=candidate.id,
                name=getattr(candidate, 'name', None),  # Handle hashed names
                email=getattr(candidate, 'email', None),  # Handle hashed emails
                location=candidate.location,
                summary=resume.summary,
                top_skills=top_skills,
                years_of_experience=years_experience,
                latest_role=latest_experience.role if latest_experience else None,
                latest_company=latest_experience.company if latest_experience else None,
                education_level=latest_education.degree if latest_education else None,
                created_at=resume.created_at,
                updated_at=resume.updated_at
            )
            
            candidates_summary.append(candidate_summary)
        
        # Update total count after filtering
        actual_total = len(candidates_summary) if skill_filter or min_experience else total_candidates
        actual_total_pages = (actual_total + page_size - 1) // page_size if actual_total > 0 else 1
        
        return CandidateListResponse(
            success=True,
            total_candidates=actual_total,
            candidates=candidates_summary,
            page=page,
            page_size=page_size,
            total_pages=actual_total_pages,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing candidates: {str(e)}")


async def update_job_status(
    job_id: str,
    status: str,
    progress: int,
    step: str,
    resume_id: Optional[int] = None,
    error_message: Optional[str] = None
):
    """Helper function to update job status in database."""
    from app.core.database import get_db_session
    
    async for db in get_db_session():
        try:
            result = await db.execute(select(CVProcessingJob).where(CVProcessingJob.id == job_id))
            job = result.scalar_one_or_none()
            
            if job:
                job.status = status
                job.progress_percentage = progress
                job.current_step = step
                job.updated_at = datetime.utcnow()
                
                if resume_id:
                    job.resume_id = resume_id
                if error_message:
                    job.error_message = error_message
                
                await db.commit()
        except Exception as e:
            print(f"Error updating job status: {e}")
        finally:
            await db.close()
        break


async def process_cv_background(
    job_id: str,
    file_content: bytes,
    filename: str,
    model: str,
    store_in_db: bool,
    generate_embeddings: bool
):
    """Background task to process CV with progress updates."""
    try:
        # Update status: extracting
        await update_job_status(job_id, "extracting", 20, "Extracting text from PDF")
        
        # Extract text from PDF
        extracted_text = await PDFService.extract_text_from_pdf(file_content)
        
        if not extracted_text.strip():
            await update_job_status(
                job_id, "failed", 0, "Text extraction failed", 
                error_message="No text content found in PDF"
            )
            return
        
        # Update status: analyzing
        await update_job_status(job_id, "analyzing", 50, "Analyzing CV with LLM")
        
        # Analyze with Ollama (the slow part)
        analysis_result = await OllamaService.analyze_cv_with_ollama(extracted_text, model)
        
        if not analysis_result.get("analysis"):
            await update_job_status(
                job_id, "failed", 0, "LLM analysis failed",
                error_message=analysis_result.get("parsing_error", "LLM analysis failed")
            )
            return
        
        # Update status: storing
        await update_job_status(job_id, "storing", 80, "Storing results in database")
        
        # Store in database if requested
        resume_id = None
        if store_in_db:
            async for db in get_db_session():
                try:
                    db_service = DatabaseService(db)
                    resume = await db_service.store_cv_analysis(
                        filename=filename,
                        file_content=file_content,
                        raw_text=extracted_text,
                        analysis=analysis_result["analysis"],
                    )
                    resume_id = resume.id
                    
                    # Generate embeddings if requested
                    if generate_embeddings:
                        await update_job_status(
                            job_id, "generating_embeddings", 90, "Generating embeddings for semantic search"
                        )
                        
                        try:
                            embeddings = await EmbeddingService.generate_cv_embeddings(
                                analysis_result["analysis"], extracted_text
                            )
                            
                            # Store each embedding section
                            for section_type, embedding in embeddings.items():
                                if section_type == "full_text":
                                    content = extracted_text
                                elif section_type == "summary":
                                    content = analysis_result["analysis"].get("summary", "")
                                elif section_type == "skills":
                                    skills = analysis_result["analysis"].get("skills", [])
                                    content = " ".join([skill.get("name", "") for skill in skills if skill.get("name")])
                                elif section_type == "experience":
                                    experiences = analysis_result["analysis"].get("experience", [])
                                    content = " | ".join([
                                        f"{exp.get('role', '')} at {exp.get('company', '')}"
                                        for exp in experiences
                                        if exp.get('role') and exp.get('company')
                                    ])
                                elif section_type == "education":
                                    educations = analysis_result["analysis"].get("education", [])
                                    content = " | ".join([
                                        f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('institution', '')}"
                                        for edu in educations
                                        if edu.get('degree') and edu.get('institution')
                                    ])
                                elif section_type == "certifications":
                                    content = " | ".join(analysis_result["analysis"].get("certifications", []))
                                else:
                                    content = ""
                                
                                if content and embedding:
                                    await db_service.store_embedding(
                                        resume_id=resume.id,
                                        section_type=section_type,
                                        content=content,
                                        embedding=embedding,
                                        model_name=EmbeddingService.DEFAULT_EMBEDDING_MODEL,
                                    )
                        
                        except Exception as e:
                            print(f"Warning: Failed to generate embeddings: {str(e)}")
                            # Continue processing even if embeddings fail
                    
                except Exception as e:
                    await update_job_status(
                        job_id, "failed", 0, "Database storage failed",
                        error_message=f"Failed to store in database: {str(e)}"
                    )
                    return
                finally:
                    await db.close()
                break
        
        # Complete successfully
        await update_job_status(
            job_id, "completed", 100, "Processing complete",
            resume_id=resume_id
        )
        
    except Exception as e:
        await update_job_status(
            job_id, "failed", 0, "Processing failed",
            error_message=str(e)
        )
