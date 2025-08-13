from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class PersonalInfoSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None


class ExperienceSchema(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class EducationSchema(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None


class SkillSchema(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    proficiency_level: Optional[str] = None
    years_experience: Optional[int] = None


class CVAnalysisSchema(BaseModel):
    personal_info: PersonalInfoSchema
    summary: Optional[str] = None
    skills: List[SkillSchema] = []
    experience: List[ExperienceSchema] = []
    education: List[EducationSchema] = []
    certifications: List[str] = []
    languages: List[str] = []


class CVAnalysisResponse(BaseModel):
    success: bool
    filename: str
    file_size: int
    extracted_text_preview: str
    analysis: CVAnalysisSchema
    metadata: Dict[str, Any] = {}
    processing_time: Optional[float] = None
    created_at: datetime


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    available_models: List[str] = []
    error: Optional[str] = None


class JobDescriptionRequest(BaseModel):
    job_description: str
    required_skills: Optional[List[str]] = []
    preferred_skills: Optional[List[str]] = []
    minimum_experience_years: Optional[int] = None
    location_preference: Optional[str] = None
    max_results: Optional[int] = 10


class MatchScore(BaseModel):
    overall_score: float
    skill_match_score: float
    experience_match_score: float
    semantic_similarity_score: float
    explanation: Optional[str] = None


class CVMatch(BaseModel):
    resume_id: int
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    candidate_location: Optional[str] = None
    match_score: MatchScore
    matched_skills: List[str] = []
    relevant_experience: List[ExperienceSchema] = []
    years_of_experience: Optional[int] = None


class JobMatchResponse(BaseModel):
    success: bool
    job_description_preview: str
    total_cvs_analyzed: int
    matches: List[CVMatch] = []
    processing_time: Optional[float] = None
    timestamp: datetime


class CandidateSummary(BaseModel):
    resume_id: int
    candidate_id: int
    name: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    top_skills: List[str] = []
    years_of_experience: Optional[int] = None
    latest_role: Optional[str] = None
    latest_company: Optional[str] = None
    education_level: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CandidateListResponse(BaseModel):
    success: bool
    total_candidates: int
    candidates: List[CandidateSummary] = []
    page: int = 1
    page_size: int = 10
    total_pages: int
    timestamp: datetime