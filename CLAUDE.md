# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CV/Resume parsing and matching system built with FastAPI and Ollama. The system processes CV documents, extracts structured information using local LLMs, and matches candidates to job postings with semantic similarity scoring.

## Development Commands

### Package Management
- Install dependencies: `uv install`
- Add new dependency: `uv add <package-name>`

### Running the Application  
- Start development server: `uvicorn app.main:app --reload`
- Start production server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### Testing
Currently no test framework is configured. Consider adding pytest for testing.

## Architecture

The system follows a microservices architecture with these core components:

### API Layer (FastAPI)
- **app/main.py**: Main FastAPI application entry point
- **app/api/cv_router.py**: CV analysis endpoints with PDF processing
- **Implemented endpoints**: 
  - `/analyze-cv`: Upload and analyze PDF CVs with Ollama
  - `/health`: Check Ollama service health and available models
  - `/models`: List available Ollama models for analysis

### Model Runtime (Ollama)
- **app/services/ollama_service.py**: Ollama integration service
- Local LLM integration for:
  - Structured CV data extraction with JSON output format
  - Zero-shot classification of skills and experience
  - Personal info, education, and experience parsing
- Default model: `gpt-oss:20b`
- Health checking and model listing functionality

### Storage Architecture
- **app/models/cv_models.py**: SQLAlchemy database models for CVs and embeddings
- **app/core/database.py**: Database configuration and session management
- **app/services/database_service.py**: Database operations for CV storage
- **PostgreSQL + pgvector**: Structured data and vector search capabilities
- **Implemented features**: CV analysis storage, embeddings storage, database session management

### Data Processing Pipeline (Implemented)
1. **PDF file validation**: File type and size validation (app/services/pdf_service.py)
2. **PDF text extraction**: Using PyMuPDF for text extraction
3. **Structured data extraction**: Ollama LLM parsing with Pydantic schemas
4. **Database storage**: CV analysis results and metadata storage
5. **Embedding generation**: Optional semantic embeddings (app/services/embedding_service.py)
6. **Vector storage**: pgvector integration for similarity search

## Key Technologies (Implemented)

- **PDF Processing**: PyMuPDF for text extraction and validation
- **LLM Integration**: Ollama Python client for local model inference
- **Data Schemas**: Pydantic for structured data validation and parsing
- **Database**: SQLAlchemy with async PostgreSQL + pgvector support
- **Embeddings**: Configurable embedding service for semantic search
- **API Framework**: FastAPI with async/await support

## Data Models (Implemented)

### Pydantic Schemas (app/schemas/cv_schemas.py)
- **PersonalInfoSchema**: Name, email, phone, location extraction
- **ExperienceSchema**: Company, role, dates, description parsing
- **EducationSchema**: Institution, degree, field, years
- **SkillSchema**: Name, category, proficiency level, experience years
- **CVAnalysisSchema**: Complete structured CV analysis
- **CVAnalysisResponse**: API response with metadata and processing info

### Database Models (app/models/cv_models.py)
- **Resume**: CV metadata, file content, and parsed analysis storage
- **ResumeEmbedding**: Vector embeddings for semantic search with pgvector support

### Matching Logic
- Semantic similarity via vector cosine similarity
- Skills overlap scoring (Jaccard/weighted)
- Experience fit analysis (years vs requirements)
- Constraint validation (language, location, authorization)

## Current Implementation Status

### ✅ Completed Features
- **PDF CV Upload & Analysis**: Full pipeline from PDF upload to structured data extraction
- **Ollama LLM Integration**: Local model inference with structured JSON output
- **Database Storage**: PostgreSQL with async SQLAlchemy and pgvector support
- **Embedding Generation**: Optional semantic embeddings for search capabilities
- **Job Matching System**: Dynamic job description matching against stored CVs
- **LLM Match Explanations**: Detailed AI-generated explanations of candidate-job fit
- **Similarity Scoring**: Multi-factor scoring (semantic, skills, experience) with weighted algorithms
- **Background Processing**: Async CV processing with job status polling
- **Health Monitoring**: Ollama service health checks and model availability
- **Structured Data Schemas**: Comprehensive Pydantic models for CV data
- **Error Handling**: Robust error handling with fallback mechanisms

### 🔄 API Endpoints

#### CV Management
- `POST /analyze-cv`: Upload PDF and get structured analysis (sync)
- `POST /cv/upload`: Upload PDF for background processing (async with polling)
- `GET /cv/status/{job_id}`: Poll status of background CV processing job
- `GET /cv/{cv_id}`: Get completed CV analysis results
- `GET /candidates`: List all candidates with summary information, filtering, and pagination

#### Job Matching
- `POST /match-job`: Match job description against stored CVs and get ranked candidates
- `POST /explain-match/{resume_id}`: Generate detailed LLM explanation of why a CV matches a job

#### System
- `GET /health`: Check Ollama service and available models
- `GET /models`: List available Ollama models

### 🎯 Next Steps
- User interface for CV management and job matching
- Batch processing optimization
- Advanced filtering and search capabilities
- Match result caching and performance optimization

## Development Notes

- Project uses modern Python (>=3.11) with full async/await support
- Structured around service-oriented architecture with clean separation
- Comprehensive error handling and graceful degradation
- Configurable embedding models and Ollama integration
- Database-first approach with migration support