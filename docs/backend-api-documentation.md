# CV Analysis API - Backend Developer Documentation

This document provides comprehensive backend/API documentation for developers working on the CV Analysis and Job Matching system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Database Models](#database-models)
3. [Service Layer](#service-layer)
4. [API Endpoints](#api-endpoints)
5. [Configuration](#configuration)
6. [Development Setup](#development-setup)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

## Architecture Overview

The system follows a clean architecture pattern with clear separation of concerns:

```
app/
├── main.py                     # FastAPI application entry point
├── api/
│   └── cv_router.py           # CV analysis and job matching routes
├── core/
│   └── database.py            # Database configuration and session management
├── models/
│   └── cv_models.py           # SQLModel database models
├── schemas/
│   └── cv_schemas.py          # Pydantic request/response schemas
└── services/
    ├── database_service.py    # Database operations and PII handling
    ├── embedding_service.py   # Vector embeddings via Ollama
    ├── matching_service.py    # Job matching algorithms
    ├── ollama_service.py      # LLM integration for CV analysis
    └── pdf_service.py         # PDF text extraction
```

### Key Technologies

- **FastAPI**: Web framework with automatic OpenAPI documentation
- **SQLModel**: Database ORM with Pydantic integration
- **PostgreSQL + pgvector**: Database with vector similarity support
- **Ollama**: Local LLM inference for CV analysis and embeddings
- **PyMuPDF**: PDF text extraction
- **Async/Await**: Full asynchronous processing support

## Database Models

### Core Models

#### Candidate Model
```python
class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name_hash: Optional[str] = Field(index=True)          # SHA256 hash of name
    email_hash: Optional[str] = Field(index=True)         # SHA256 hash of email
    phone_hash: Optional[str] = None                      # SHA256 hash of phone
    location: Optional[str] = None                        # Stored in plain text
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

**Key Features:**
- PII data (name, email, phone) is hashed for privacy protection
- Location stored as plain text for filtering/matching
- Automatic deduplication based on email or name hash
- One-to-many relationship with resumes

#### Resume Model
```python
class Resume(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id")
    original_filename: str
    file_hash: str = Field(unique=True)                   # SHA256 of file content
    raw_text: str                                         # Extracted PDF text
    summary: Optional[str] = None                         # LLM-generated summary
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

**Key Features:**
- File deduplication via SHA256 hash
- Stores complete raw text for reference
- Optional AI-generated professional summary
- Foreign key relationship to candidate

#### Experience Model
```python
class Experience(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None                      # Free-form date string
    end_date: Optional[str] = None                        # Free-form date string
    description: Optional[str] = None
    order_index: int = 0                                  # Preserve order from CV
```

#### Education Model
```python
class Education(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[str] = None                      # Free-form year string
    end_year: Optional[str] = None                        # Free-form year string
    order_index: int = 0                                  # Preserve order from CV
```

#### Skill Models
```python
class Skill(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)            # Normalized skill name
    category: Optional[str] = None                        # Technical, Soft, etc.
    aliases: Optional[str] = None                         # Alternative names

class ResumeSkill(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    skill_id: int = Field(foreign_key="skill.id")
    proficiency_level: Optional[str] = None               # Beginner/Expert etc.
    years_experience: Optional[int] = None               # Years of experience
```

**Key Features:**
- Normalized skill names for consistent matching
- Many-to-many relationship between resumes and skills
- Skill proficiency and experience tracking

### Vector Storage Models

#### ResumeEmbedding Model
```python
class ResumeEmbedding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id", index=True)
    section_type: str = Field(index=True)                 # 'summary', 'skills', 'experience', etc.
    section_content: str = Field(sa_column=Column(Text))  # Original text content
    embedding: List[float] = Field(sa_column=Column(Vector(768)))  # 768-dim vector
    model_name: str                                       # Embedding model used
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

**Section Types:**
- `full_text`: Complete CV text
- `summary`: Professional summary section
- `skills`: Concatenated skills text
- `experience`: Combined experience descriptions
- `education`: Education background text
- `certifications`: Certifications list

### Job Processing Models

#### CVProcessingJob Model
```python
class CVProcessingJob(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    original_filename: str
    file_size: int
    status: str = Field(default="uploaded")               # Processing status
    progress_percentage: int = Field(default=0)          # 0-100 progress
    current_step: Optional[str] = None                    # Current processing step
    error_message: Optional[str] = None                   # Error details
    resume_id: Optional[int] = None                       # Final resume ID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None       # ETA for completion
    file_content: Optional[bytes] = None                  # Temporary file storage
```

**Status Values:**
- `uploaded`: File received and queued
- `extracting`: Extracting text from PDF
- `analyzing`: Running LLM analysis
- `storing`: Saving to database
- `generating_embeddings`: Creating vector embeddings
- `completed`: Processing finished successfully
- `failed`: Processing failed with error

## Service Layer

### DatabaseService (`app.services.database_service`)

Handles all database operations with PII protection and data normalization.

#### Key Methods:

```python
class DatabaseService:
    async def get_or_create_candidate(
        self, name: str, email: str, phone: str, location: str
    ) -> Candidate:
        """Get existing candidate or create new one based on PII hashes."""
    
    async def create_resume(
        self, candidate_id: int, filename: str, file_content: bytes, 
        raw_text: str, summary: str
    ) -> Resume:
        """Create resume with file deduplication."""
    
    async def store_cv_analysis(
        self, filename: str, file_content: bytes, raw_text: str, analysis: Dict
    ) -> Resume:
        """Store complete CV analysis results."""
    
    async def store_embedding(
        self, resume_id: int, section_type: str, content: str, 
        embedding: List[float], model_name: str
    ) -> ResumeEmbedding:
        """Store vector embedding for resume section."""
```

#### PII Hashing (`PIIHasher`)

```python
class PIIHasher:
    @staticmethod
    def hash_pii(value: str, salt: str = "cv_ollama_salt") -> str:
        """Hash PII data with salt for privacy protection."""
        combined = f"{salt}:{value.lower().strip()}"
        return hashlib.sha256(combined.encode()).hexdigest()
```

**Privacy Features:**
- All PII (names, emails, phone numbers) stored as SHA256 hashes
- Consistent salt for deterministic hashing
- Phone numbers normalized (digits only) before hashing
- Location data stored in plain text for filtering

### OllamaService (`app.services.ollama_service`)

Integrates with Ollama for LLM-based CV analysis and explanations.

#### Key Methods:

```python
class OllamaService:
    DEFAULT_MODEL = "gpt-oss:20b"
    
    @staticmethod
    async def analyze_cv_with_ollama(cv_text: str, model: str) -> Dict[str, Any]:
        """Analyze CV text using Ollama LLM with structured JSON output."""
    
    @staticmethod
    async def generate_match_explanation(
        job_description: str, candidate_summary: str, matched_skills: List[str],
        experience_summary: str, match_scores: Dict, model: str
    ) -> str:
        """Generate detailed explanation of candidate-job match."""
    
    @staticmethod
    def check_ollama_health() -> Dict[str, Any]:
        """Check Ollama service health and available models."""
```

#### Structured Output Schema

The LLM is prompted to return JSON following this exact structure:

```json
{
  "personal_info": {
    "name": "string or null",
    "email": "string or null", 
    "phone": "string or null",
    "location": "string or null"
  },
  "summary": "string or null",
  "skills": [
    {
      "name": "string",
      "category": "string or null",
      "proficiency_level": "string or null",
      "years_experience": "number or null"
    }
  ],
  "experience": [
    {
      "company": "string or null",
      "role": "string or null", 
      "start_date": "string or null",
      "end_date": "string or null",
      "description": "string or null"
    }
  ],
  "education": [...],
  "certifications": ["string"],
  "languages": ["string"]
}
```

### EmbeddingService (`app.services.embedding_service`)

Generates vector embeddings using nomic-embed-text via Ollama.

#### Key Methods:

```python
class EmbeddingService:
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_DIMENSION = 768
    
    @staticmethod
    async def generate_embedding(text: str, model: str) -> List[float]:
        """Generate 768-dimensional embedding for text."""
    
    @staticmethod
    async def generate_cv_embeddings(
        analysis: Dict, raw_text: str, model: str
    ) -> Dict[str, List[float]]:
        """Generate embeddings for different CV sections."""
    
    @staticmethod
    async def generate_job_embedding(job_description: str) -> List[float]:
        """Generate embedding for job description."""
    
    @staticmethod
    def calculate_cosine_similarity(
        embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
```

**Embedding Sections Generated:**
- `full_text`: Complete CV text (primary for matching)
- `summary`: Professional summary only
- `skills`: Concatenated skill names with categories
- `experience`: Combined experience descriptions
- `education`: Education background text
- `certifications`: Certification names

### MatchingService (`app.services.matching_service`)

Implements multi-factor scoring algorithm for job-candidate matching.

#### Key Methods:

```python
class MatchingService:
    @staticmethod
    async def find_matches_with_session(
        job_request: JobDescriptionRequest, session: AsyncSession
    ) -> Dict[str, Any]:
        """Find matching CVs for job description."""
    
    @staticmethod
    async def _calculate_match_score(
        resume: Resume, candidate: Candidate, job_request: JobDescriptionRequest,
        job_embedding: List[float], session: AsyncSession
    ) -> Optional[CVMatch]:
        """Calculate comprehensive match score."""
```

#### Scoring Algorithm

**Overall Score Calculation:**
```python
overall_score = (
    semantic_score * 0.4 +      # Vector similarity
    skill_match_score * 0.35 +  # Skills overlap
    experience_match_score * 0.25 # Experience fit
)
```

**Semantic Similarity (40% weight):**
- Cosine similarity between job embedding and resume full_text embedding
- Uses nomic-embed-text 768-dimensional vectors

**Skill Match Score (35% weight):**
- Required skills match ratio * 0.7
- Preferred skills match ratio * 0.3
- Supports partial matching (substring matching)

**Experience Match Score (25% weight):**
- Years of experience score * 0.6
- Relevance score (keyword overlap) * 0.4
- Minimum experience threshold enforcement

### PDFService (`app.services.pdf_service`)

Handles PDF processing and validation using PyMuPDF.

#### Key Methods:

```python
class PDFService:
    @staticmethod
    async def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
    
    @staticmethod
    def validate_pdf_file(file_content_type: str, file_size: int) -> None:
        """Validate PDF file type and size (max 10MB)."""
```

## API Endpoints

### System Health Endpoints

#### `GET /` - API Overview
Returns basic API information and available endpoints.

#### `GET /api/v1/health` - Health Check
```python
@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify Ollama connectivity and available models."""
```

**Response Schema:**
```python
class HealthResponse(BaseModel):
    status: str                    # "healthy" or "unhealthy"
    ollama_available: bool         # Ollama service status
    available_models: List[str]    # List of available models
    error: Optional[str]           # Error message if unhealthy
```

#### `GET /api/v1/models` - List Models
Returns available Ollama models for CV analysis.

### CV Processing Endpoints

#### `POST /api/v1/analyze-cv` - Synchronous CV Analysis
```python
@router.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(
    file: UploadFile = File(...),
    model: Optional[str] = Query(default=None),
    store_in_db: bool = Query(default=True),
    generate_embeddings: bool = Query(default=True),
    db: AsyncSession = Depends(get_db_session)
):
```

**Processing Steps:**
1. Validate PDF file (type, size < 10MB)
2. Extract text using PyMuPDF
3. Analyze with Ollama LLM
4. Store in database (if requested)
5. Generate embeddings (if requested)
6. Return structured analysis

#### `POST /api/v1/cv/upload` - Asynchronous CV Upload
```python
@router.post("/cv/upload")
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Optional[str] = Query(default=None),
    store_in_db: bool = Query(default=True),
    generate_embeddings: bool = Query(default=True),
    db: AsyncSession = Depends(get_db_session)
):
```

**Background Processing Pipeline:**
1. File validation and job creation
2. Background task: `process_cv_background()`
   - Text extraction (20% progress)
   - LLM analysis (50% progress)
   - Database storage (80% progress)
   - Embedding generation (90% progress)
   - Completion (100% progress)

#### `GET /api/v1/cv/status/{job_id}` - Job Status Polling
Returns processing status for background jobs.

**Status Flow:**
```
uploaded → extracting → analyzing → storing → generating_embeddings → completed
                                                                    ↘ failed
```

#### `GET /api/v1/cv/{cv_id}` - Get CV Analysis
Returns completed CV analysis results with candidate information.

### Job Matching Endpoints

#### `POST /api/v1/match-job` - Job Matching
```python
@router.post("/match-job", response_model=JobMatchResponse)
async def match_job_to_cvs(
    job_request: JobDescriptionRequest,
    db: AsyncSession = Depends(get_db_session)
):
```

**Request Schema:**
```python
class JobDescriptionRequest(BaseModel):
    job_description: str                        # Job posting text
    required_skills: Optional[List[str]] = []   # Must-have skills
    preferred_skills: Optional[List[str]] = []  # Nice-to-have skills
    minimum_experience_years: Optional[int]     # Min experience required
    location_preference: Optional[str]          # Preferred location
    max_results: Optional[int] = 10            # Result limit
```

**Matching Process:**
1. Generate job description embedding
2. Retrieve all resumes with candidates
3. Calculate match scores for each resume
4. Filter by minimum threshold (0.1)
5. Sort by overall score (descending)
6. Return top matches (limited by max_results)

#### `POST /api/v1/explain-match/{resume_id}` - Match Explanation
Generates detailed LLM explanation for why a specific candidate matches a job.

**Process:**
1. Retrieve resume and candidate data
2. Calculate match scores and metrics
3. Generate natural language explanation using Ollama
4. Return detailed analysis with scores

### Candidate Management Endpoints

#### `GET /api/v1/candidates` - List Candidates
```python
@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    skill_filter: Optional[str] = Query(None),
    location_filter: Optional[str] = Query(None),
    min_experience: Optional[int] = Query(None, ge=0),
    db: AsyncSession = Depends(get_db_session)
):
```

**Features:**
- Pagination support
- Skill name filtering (partial match)
- Location filtering (partial match)
- Minimum experience filtering
- Returns candidate summaries with top skills and latest position

## Configuration

### Database Configuration (`app.core.database`)

```python
class DatabaseConfig:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5433")
    DB_USER = os.getenv("DB_USER", "cvollama")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "cvollama123")
    DB_NAME = os.getenv("DB_NAME", "cv_ollama")
    
    DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SYNC_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5433
DB_USER=cvollama
DB_PASSWORD=cvollama123
DB_NAME=cv_ollama
DB_ECHO=false                    # Enable SQL query logging

# Ollama
OLLAMA_HOST=http://localhost:11434

# Application
PYTHONPATH=.
```

### Model Configuration

**Default Models:**
- **Analysis Model**: `gpt-oss:20b` (configurable per request)
- **Embedding Model**: `nomic-embed-text` (768 dimensions)

**Model Requirements:**
- Analysis model must support structured JSON output
- Embedding model must produce consistent vector dimensions
- Both models must be available in local Ollama instance

## Development Setup

### Prerequisites

1. **Python 3.11+** with UV package manager
2. **PostgreSQL 15+** with pgvector extension
3. **Ollama** with required models installed
4. **Docker** (optional, for containerized database)

### Installation Steps

```bash
# 1. Clone repository and install dependencies
uv install

# 2. Start PostgreSQL with pgvector (using Docker)
docker-compose up -d postgres

# 3. Run database migrations
alembic upgrade head

# 4. Install Ollama models
ollama pull gpt-oss:20b
ollama pull nomic-embed-text

# 5. Start development server
uvicorn app.main:app --reload
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

### Testing

```bash
# Run tests (when test framework is added)
pytest

# Test Ollama connectivity
curl http://localhost:11434/api/tags

# Test API health
curl http://localhost:8000/api/v1/health
```

## Error Handling

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid file, validation errors)
- **404**: Not Found (CV not found, job not found)
- **413**: Payload Too Large (file > 10MB)
- **422**: Unprocessable Entity (schema validation errors)
- **500**: Internal Server Error (service failures)

### Error Response Format

```python
class ErrorResponse(BaseModel):
    success: bool = False
    error: str                 # Brief error message
    detail: Optional[str]      # Detailed error information
```

### Common Error Scenarios

#### PDF Processing Errors
- **Invalid file type**: Only PDF files accepted
- **File too large**: Maximum 10MB limit
- **Corrupted PDF**: Unable to extract text
- **Empty PDF**: No extractable text content

#### Ollama Service Errors
- **Service unavailable**: Ollama not running
- **Model not found**: Requested model not installed
- **Analysis timeout**: LLM processing exceeded timeout
- **Invalid JSON**: LLM returned malformed JSON

#### Database Errors
- **Connection failure**: Database not accessible
- **Constraint violation**: Duplicate file hash
- **Migration required**: Database schema outdated

### Error Recovery Strategies

```python
# Graceful degradation for embeddings
try:
    embeddings = await EmbeddingService.generate_cv_embeddings(analysis, text)
    embeddings_generated = True
except Exception as e:
    print(f"Warning: Failed to generate embeddings: {str(e)}")
    embeddings_generated = False
    # Continue processing without embeddings
```

## Performance Considerations

### Database Optimization

**Indexes:**
- `candidate.email_hash` - Fast candidate lookup
- `candidate.name_hash` - Alternative candidate lookup
- `skill.name` - Unique constraint and fast skill matching
- `resumeembedding.resume_id` - Fast embedding retrieval
- `resumeembedding.section_type` - Filtered embedding queries

**Query Optimization:**
- Use async database sessions for non-blocking I/O
- Implement connection pooling with appropriate limits
- Use `select()` with explicit column selection for large tables
- Implement pagination for all list endpoints

### Vector Search Performance

```python
# Example optimized vector similarity query (for future implementation)
async def find_similar_resumes(
    self, query_embedding: List[float], limit: int = 10
) -> List[ResumeEmbedding]:
    """Optimized vector similarity search using pgvector operators."""
    query = text("""
        SELECT re.*, (re.embedding <=> :query_embedding) as distance
        FROM resumeembedding re
        WHERE re.section_type = 'full_text'
        ORDER BY re.embedding <=> :query_embedding
        LIMIT :limit
    """)
    
    result = await self.session.execute(
        query, {"query_embedding": query_embedding, "limit": limit}
    )
    return result.fetchall()
```

### Memory Management

**Large File Handling:**
- Stream PDF processing to avoid loading entire file in memory
- Temporary file storage for background jobs
- Cleanup file content after processing completion

**Background Job Management:**
- Limit concurrent background jobs
- Implement job cleanup for completed/failed jobs
- Monitor memory usage during LLM processing

### Caching Strategies

**Model Results Caching:**
- Cache Ollama model list responses
- Cache frequently accessed embeddings
- Implement Redis for distributed caching (future)

**Database Query Caching:**
- Cache skill lookups (skills rarely change)
- Cache candidate profile data
- Implement query result caching for expensive operations

### Monitoring and Observability

**Key Metrics to Monitor:**
- CV processing time (by stage)
- Database query performance
- Ollama response times
- Memory usage during processing
- Background job queue length
- Error rates by endpoint

**Logging Strategy:**
```python
import logging

logger = logging.getLogger(__name__)

# Log processing stages
logger.info(f"Starting CV analysis for file: {filename}")
logger.info(f"Text extraction completed in {extraction_time:.2f}s")
logger.info(f"LLM analysis completed in {analysis_time:.2f}s")
logger.warning(f"Embedding generation failed: {error}")
logger.error(f"Critical error in CV processing: {error}")
```

### Scalability Considerations

**Horizontal Scaling:**
- Stateless API design enables load balancing
- Database connection pooling for multiple instances
- Background job processing can be scaled independently

**Performance Bottlenecks:**
- **LLM Processing**: Slowest component, consider model optimization
- **Vector Operations**: May require specialized vector database at scale
- **PDF Text Extraction**: CPU-intensive, consider worker pools
- **Database Write Operations**: Batch operations where possible

**Future Optimizations:**
- Implement asynchronous job queue (Redis + Celery)
- Use dedicated vector database (Pinecone, Weaviate)
- Optimize LLM prompts for faster inference
- Implement result caching and CDN for static content