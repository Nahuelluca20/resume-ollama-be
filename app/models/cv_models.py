from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import Text
from pgvector.sqlalchemy import Vector


class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name_hash: Optional[str] = Field(index=True)
    email_hash: Optional[str] = Field(index=True)
    phone_hash: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    resumes: List["Resume"] = Relationship(back_populates="candidate")


class Resume(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(foreign_key="candidate.id")
    original_filename: str
    file_hash: str = Field(unique=True)
    raw_text: str
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    candidate: Candidate = Relationship(back_populates="resumes")
    experiences: List["Experience"] = Relationship(back_populates="resume")
    educations: List["Education"] = Relationship(back_populates="resume")
    skills: List["ResumeSkill"] = Relationship(back_populates="resume")


class Experience(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    company: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    order_index: int = 0
    
    resume: Resume = Relationship(back_populates="experiences")


class Education(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    institution: Optional[str] = None
    degree: Optional[str] = None
    field: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None
    order_index: int = 0
    
    resume: Resume = Relationship(back_populates="educations")


class Skill(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    category: Optional[str] = None
    aliases: Optional[str] = None
    
    resume_skills: List["ResumeSkill"] = Relationship(back_populates="skill")


class ResumeSkill(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id")
    skill_id: int = Field(foreign_key="skill.id")
    proficiency_level: Optional[str] = None
    years_experience: Optional[int] = None
    
    resume: Resume = Relationship(back_populates="skills")
    skill: Skill = Relationship(back_populates="resume_skills")


class ResumeEmbedding(SQLModel, table=True):
    """Store vector embeddings for semantic search of resume sections."""
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id", index=True)
    section_type: str = Field(index=True)  # 'summary', 'skills', 'experience', 'education', 'full_text'
    section_content: str = Field(sa_column=Column(Text))
    embedding: List[float] = Field(sa_column=Column(Vector(768)))  # Using 768-dimensional embeddings (nomic-embed-text)
    model_name: str  # Track which embedding model was used
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    resume: Resume = Relationship()


class CertificationEmbedding(SQLModel, table=True):
    """Store embeddings for certifications for matching purposes."""
    id: Optional[int] = Field(default=None, primary_key=True)
    resume_id: int = Field(foreign_key="resume.id", index=True)
    certification_text: str
    embedding: List[float] = Field(sa_column=Column(Vector(768)))
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    resume: Resume = Relationship()