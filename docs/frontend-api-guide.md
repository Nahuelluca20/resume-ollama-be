# CV Analysis API - Frontend Integration Guide

This guide provides comprehensive documentation for integrating the CV Analysis and Job Matching API with a Next.js frontend application.

## API Base Configuration

**Local Development URL:** `http://127.0.0.1:8000`

**API Prefix:** `/api/v1`

**Full Base URL:** `http://127.0.0.1:8000/api/v1`

## Suggested Frontend Application Sections

### 1. Dashboard/Overview Page (`/`)
- System health status
- Recent CV uploads
- Quick statistics (total candidates, recent matches)
- Navigation to main features

### 2. CV Upload & Management (`/cv`)
- **Upload CV**: Single file upload with drag-and-drop
- **CV List**: Paginated list of all uploaded CVs
- **CV Details**: Individual CV analysis results
- **Processing Status**: Real-time status updates for background processing

### 3. Job Matching (`/jobs`)
- **Create Job Posting**: Form to enter job description and requirements
- **Match Results**: Ranked candidate list with scores
- **Match Explanation**: Detailed AI explanation for specific matches

### 4. Candidate Directory (`/candidates`)
- **Candidate List**: Searchable and filterable candidate directory
- **Candidate Profile**: Detailed view of individual candidates
- **Advanced Filters**: Skills, location, experience level

### 5. Settings/System (`/settings`)
- **Model Configuration**: Available Ollama models
- **System Health**: Service status monitoring

## API Endpoints Documentation

### System Health & Info

#### Get API Overview
```http
GET /
```

**Response:**
```json
{
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
```

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "available_models": ["gpt-oss:20b", "llama2:7b"],
  "error": null
}
```

#### List Available Models
```http
GET /api/v1/models
```

**Response:**
```json
{
  "success": true,
  "models": ["gpt-oss:20b", "llama2:7b"],
  "default_model": "gpt-oss:20b"
}
```

### CV Upload & Analysis

#### Upload CV (Synchronous)
```http
POST /api/v1/analyze-cv
Content-Type: multipart/form-data
```

**Request:**
- `file`: PDF file (max 10MB)
- `model`: Optional Ollama model name
- `store_in_db`: Boolean (default: true)
- `generate_embeddings`: Boolean (default: true)

**Frontend Implementation:**
```typescript
const uploadCV = async (file: File, options?: {
  model?: string;
  storeInDb?: boolean;
  generateEmbeddings?: boolean;
}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const params = new URLSearchParams();
  if (options?.model) params.append('model', options.model);
  if (options?.storeInDb !== undefined) params.append('store_in_db', options.storeInDb.toString());
  if (options?.generateEmbeddings !== undefined) params.append('generate_embeddings', options.generateEmbeddings.toString());

  const response = await fetch(`http://127.0.0.1:8000/api/v1/analyze-cv?${params}`, {
    method: 'POST',
    body: formData,
  });

  return response.json();
};
```

**Response:**
```json
{
  "success": true,
  "filename": "john_doe_cv.pdf",
  "file_size": 2456789,
  "extracted_text_preview": "John Doe\nSoftware Engineer...",
  "analysis": {
    "personal_info": {
      "name": "John Doe",
      "email": "john@example.com",
      "phone": "+1234567890",
      "location": "San Francisco, CA"
    },
    "summary": "Experienced software engineer...",
    "skills": [
      {
        "name": "Python",
        "category": "Programming",
        "proficiency_level": "Expert",
        "years_experience": 5
      }
    ],
    "experience": [...],
    "education": [...],
    "certifications": [...],
    "languages": [...]
  },
  "metadata": {
    "model_used": "gpt-oss:20b",
    "stored_in_db": true,
    "resume_id": 123,
    "embeddings_generated": true
  },
  "processing_time": 15.7,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Upload CV (Asynchronous with Polling)
```http
POST /api/v1/cv/upload
Content-Type: multipart/form-data
```

**Frontend Implementation:**
```typescript
const uploadCVAsync = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://127.0.0.1:8000/api/v1/cv/upload', {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  return result.job_id;
};

const pollJobStatus = async (jobId: string) => {
  const response = await fetch(`http://127.0.0.1:8000/api/v1/cv/status/${jobId}`);
  return response.json();
};

// Usage example for polling
const handleAsyncUpload = async (file: File) => {
  const jobId = await uploadCVAsync(file);
  
  const pollInterval = setInterval(async () => {
    const status = await pollJobStatus(jobId);
    
    if (status.status === 'completed') {
      clearInterval(pollInterval);
      // Redirect to CV details page
      router.push(`/cv/${status.cv_id}`);
    } else if (status.status === 'failed') {
      clearInterval(pollInterval);
      // Handle error
      console.error(status.error);
    }
    // Update progress UI
    setProgress(status.progress);
  }, 2000);
};
```

**Initial Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "uploaded",
  "estimated_time": "30 seconds",
  "message": "CV uploaded successfully. Processing started.",
  "poll_url": "/api/cv/status/550e8400-e29b-41d4-a716-446655440000"
}
```

#### Get Job Status
```http
GET /api/v1/cv/status/{job_id}
```

**Response (In Progress):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "analyzing",
  "progress": 50,
  "current_step": "Analyzing CV with LLM",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z",
  "estimated_completion": "2024-01-15T10:30:30Z"
}
```

**Response (Completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "current_step": "Processing complete",
  "cv_id": 123,
  "redirect_url": "/api/cv/123",
  "success": true
}
```

#### Get CV Analysis Results
```http
GET /api/v1/cv/{cv_id}
```

**Response:**
```json
{
  "cv_id": 123,
  "filename": "john_doe_cv.pdf",
  "summary": "Experienced software engineer...",
  "candidate": {
    "name_hash": "hashed_name",
    "email_hash": "hashed_email",
    "phone_hash": "hashed_phone",
    "location": "San Francisco, CA"
  },
  "raw_text_preview": "John Doe\nSoftware Engineer...",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "file_hash": "abc123...",
    "raw_text_length": 5000
  }
}
```

### Candidate Management

#### List Candidates
```http
GET /api/v1/candidates?page=1&page_size=10&skill_filter=python&location_filter=san%20francisco&min_experience=3
```

**Frontend Implementation:**
```typescript
const listCandidates = async (filters: {
  page?: number;
  pageSize?: number;
  skillFilter?: string;
  locationFilter?: string;
  minExperience?: number;
}) => {
  const params = new URLSearchParams();
  if (filters.page) params.append('page', filters.page.toString());
  if (filters.pageSize) params.append('page_size', filters.pageSize.toString());
  if (filters.skillFilter) params.append('skill_filter', filters.skillFilter);
  if (filters.locationFilter) params.append('location_filter', filters.locationFilter);
  if (filters.minExperience) params.append('min_experience', filters.minExperience.toString());

  const response = await fetch(`http://127.0.0.1:8000/api/v1/candidates?${params}`);
  return response.json();
};
```

**Response:**
```json
{
  "success": true,
  "total_candidates": 50,
  "candidates": [
    {
      "resume_id": 123,
      "candidate_id": 456,
      "name": "John Doe",
      "email": "john@example.com",
      "location": "San Francisco, CA",
      "summary": "Experienced software engineer...",
      "top_skills": ["Python", "React", "AWS", "Docker", "PostgreSQL"],
      "years_of_experience": 5,
      "latest_role": "Senior Software Engineer",
      "latest_company": "Tech Corp",
      "education_level": "Bachelor's in Computer Science",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "page": 1,
  "page_size": 10,
  "total_pages": 5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Job Matching

#### Match Job to CVs
```http
POST /api/v1/match-job
Content-Type: application/json
```

**Request:**
```json
{
  "job_description": "We are looking for a Senior Python Developer with experience in FastAPI, PostgreSQL, and AWS...",
  "required_skills": ["Python", "FastAPI", "PostgreSQL"],
  "preferred_skills": ["AWS", "Docker", "React"],
  "minimum_experience_years": 3,
  "location_preference": "San Francisco",
  "max_results": 10
}
```

**Frontend Implementation:**
```typescript
const matchJob = async (jobData: {
  jobDescription: string;
  requiredSkills?: string[];
  preferredSkills?: string[];
  minimumExperienceYears?: number;
  locationPreference?: string;
  maxResults?: number;
}) => {
  const response = await fetch('http://127.0.0.1:8000/api/v1/match-job', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_description: jobData.jobDescription,
      required_skills: jobData.requiredSkills || [],
      preferred_skills: jobData.preferredSkills || [],
      minimum_experience_years: jobData.minimumExperienceYears,
      location_preference: jobData.locationPreference,
      max_results: jobData.maxResults || 10,
    }),
  });

  return response.json();
};
```

**Response:**
```json
{
  "success": true,
  "job_description_preview": "We are looking for a Senior Python Developer...",
  "total_cvs_analyzed": 45,
  "matches": [
    {
      "resume_id": 123,
      "candidate_name": "John Doe",
      "candidate_email": "john@example.com",
      "candidate_location": "San Francisco, CA",
      "match_score": {
        "overall_score": 0.85,
        "skill_match_score": 0.9,
        "experience_match_score": 0.8,
        "semantic_similarity_score": 0.85,
        "explanation": null
      },
      "matched_skills": ["Python", "FastAPI", "PostgreSQL", "AWS"],
      "relevant_experience": [
        {
          "company": "Tech Corp",
          "role": "Senior Software Engineer",
          "start_date": "2020-01",
          "end_date": "2024-01",
          "description": "Led development of Python/FastAPI applications..."
        }
      ],
      "years_of_experience": 5
    }
  ],
  "processing_time": 2.3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Explain Match
```http
POST /api/v1/explain-match/{resume_id}
Content-Type: application/json
```

**Request:**
```json
{
  "job_description": "We are looking for a Senior Python Developer...",
  "required_skills": ["Python", "FastAPI"],
  "preferred_skills": ["AWS", "Docker"]
}
```

**Frontend Implementation:**
```typescript
const explainMatch = async (resumeId: number, jobData: {
  jobDescription: string;
  requiredSkills?: string[];
  preferredSkills?: string[];
}) => {
  const response = await fetch(`http://127.0.0.1:8000/api/v1/explain-match/${resumeId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      job_description: jobData.jobDescription,
      required_skills: jobData.requiredSkills || [],
      preferred_skills: jobData.preferredSkills || [],
    }),
  });

  return response.json();
};
```

**Response:**
```json
{
  "success": true,
  "resume_id": 123,
  "candidate_name": "John Doe",
  "match": {
    "resume_id": 123,
    "candidate_name": "John Doe",
    "match_score": {
      "overall_score": 0.85,
      "skill_match_score": 0.9,
      "experience_match_score": 0.8,
      "semantic_similarity_score": 0.85,
      "explanation": "This candidate is an excellent match for the Senior Python Developer position. They have extensive experience with Python (5+ years) and have worked extensively with FastAPI in their current role at Tech Corp. Their background includes building scalable web applications using PostgreSQL databases, which aligns perfectly with the job requirements..."
    },
    "matched_skills": ["Python", "FastAPI", "PostgreSQL", "AWS"],
    "relevant_experience": [...],
    "years_of_experience": 5
  },
  "detailed_explanation": "This candidate is an excellent match...",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Data Types (TypeScript)

```typescript
interface PersonalInfo {
  name?: string;
  email?: string;
  phone?: string;
  location?: string;
}

interface Experience {
  company?: string;
  role?: string;
  start_date?: string;
  end_date?: string;
  description?: string;
}

interface Education {
  institution?: string;
  degree?: string;
  field?: string;
  start_year?: string;
  end_year?: string;
}

interface Skill {
  name?: string;
  category?: string;
  proficiency_level?: string;
  years_experience?: number;
}

interface CVAnalysis {
  personal_info: PersonalInfo;
  summary?: string;
  skills: Skill[];
  experience: Experience[];
  education: Education[];
  certifications: string[];
  languages: string[];
}

interface MatchScore {
  overall_score: number;
  skill_match_score: number;
  experience_match_score: number;
  semantic_similarity_score: number;
  explanation?: string;
}

interface CVMatch {
  resume_id: number;
  candidate_name?: string;
  candidate_email?: string;
  candidate_location?: string;
  match_score: MatchScore;
  matched_skills: string[];
  relevant_experience: Experience[];
  years_of_experience?: number;
}

interface JobDescription {
  job_description: string;
  required_skills?: string[];
  preferred_skills?: string[];
  minimum_experience_years?: number;
  location_preference?: string;
  max_results?: number;
}
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "detail": "Error message description",
  "status_code": 400
}
```

**Common Error Codes:**
- `400`: Bad Request (invalid file, missing parameters)
- `404`: Not Found (CV not found, job not found)
- `413`: Payload Too Large (file size exceeds 10MB)
- `422`: Unprocessable Entity (validation errors)
- `500`: Internal Server Error (server-side issues)

**Frontend Error Handling Example:**
```typescript
const handleApiError = (error: any) => {
  if (error.status === 413) {
    return "File too large. Please upload a file smaller than 10MB.";
  } else if (error.status === 400) {
    return error.detail || "Invalid request. Please check your input.";
  } else if (error.status === 404) {
    return "Resource not found.";
  } else if (error.status >= 500) {
    return "Server error. Please try again later.";
  }
  return "An unexpected error occurred.";
};
```

## Next.js App Router File Structure Suggestion

```
app/
├── page.tsx                    # Dashboard
├── layout.tsx                  # Root layout
├── cv/
│   ├── page.tsx               # CV list
│   ├── upload/
│   │   └── page.tsx           # Upload CV form
│   ├── [id]/
│   │   └── page.tsx           # CV details
│   └── status/
│       └── [jobId]/
│           └── page.tsx       # Processing status
├── candidates/
│   ├── page.tsx               # Candidate directory
│   └── [id]/
│       └── page.tsx           # Candidate profile
├── jobs/
│   ├── page.tsx               # Job list
│   ├── new/
│   │   └── page.tsx           # Create job posting
│   └── match/
│       └── [id]/
│           └── page.tsx       # Match results
├── settings/
│   └── page.tsx               # System settings
└── components/
    ├── cv/
    ├── candidates/
    ├── jobs/
    └── ui/
```

## Best Practices

1. **File Upload**: Use drag-and-drop interface with progress indicators
2. **Async Processing**: Implement real-time polling with WebSocket alternative
3. **Pagination**: Always implement pagination for large datasets
4. **Caching**: Cache API responses where appropriate (candidate lists, model info)
5. **Loading States**: Show loading spinners and progress bars
6. **Error Boundaries**: Implement React error boundaries for better UX
7. **Validation**: Client-side validation before API calls
8. **Responsive Design**: Ensure mobile-friendly interface
9. **Accessibility**: Follow WCAG guidelines for form inputs and navigation

## Rate Limiting & Performance

- **CV Analysis**: CPU-intensive operation, consider queueing for multiple uploads
- **Job Matching**: Can analyze multiple CVs, implement pagination for large datasets
- **Real-time Updates**: Use polling with exponential backoff for status updates
- **File Uploads**: Implement chunked uploads for large files if needed