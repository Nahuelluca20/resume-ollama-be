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
    name: str
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