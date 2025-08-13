import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlmodel import Session

from app.models.cv_models import (
    Candidate, Resume, Experience, Education, Skill, ResumeSkill,
    ResumeEmbedding, CertificationEmbedding
)


class PlainTextValidator:
    """Utility class for validating personal information."""
    
    # Common placeholder values that should be treated as None
    PLACEHOLDER_VALUES = {
        "name", "full name", "candidate name", "your name", "insert name",
        "email", "email address", "your email", "candidate email", "insert email",
        "phone", "phone number", "contact number", "your phone", "telephone",
        "location", "city", "address", "your location", "insert location"
    }
    
    @staticmethod
    def is_placeholder(value: str) -> bool:
        """Check if a value is a placeholder."""
        if not value:
            return True
        normalized = value.lower().strip()
        return normalized in PlainTextValidator.PLACEHOLDER_VALUES or len(normalized) < 2
    
    @staticmethod
    def clean_name(name: str) -> Optional[str]:
        """Clean and validate candidate name."""
        if not name or PlainTextValidator.is_placeholder(name):
            return None
        return name.strip()
    
    @staticmethod
    def clean_email(email: str) -> Optional[str]:
        """Clean and validate candidate email."""
        if not email or PlainTextValidator.is_placeholder(email):
            return None
        # Basic email validation
        email = email.strip().lower()
        if "@" not in email or "." not in email:
            return None
        return email
    
    @staticmethod
    def clean_phone(phone: str) -> Optional[str]:
        """Clean and validate candidate phone number."""
        if not phone or PlainTextValidator.is_placeholder(phone):
            return None
        # Clean phone number (remove spaces, dashes, parentheses) but keep original format
        cleaned_digits = ''.join(filter(str.isdigit, phone))
        if len(cleaned_digits) < 7:  # Minimum reasonable phone number length
            return None
        return phone.strip()


class DatabaseService:
    """Service for database operations related to CV storage and retrieval."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_or_create_candidate(
        self, 
        name: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        location: Optional[str] = None
    ) -> Candidate:
        """Get existing candidate or create new one based on PII hashes."""
        
        # Clean and validate PII data
        clean_name = PlainTextValidator.clean_name(name)
        clean_email = PlainTextValidator.clean_email(email)
        clean_phone = PlainTextValidator.clean_phone(phone)
        
        # Filter out placeholder location values
        valid_location = location if location and not PlainTextValidator.is_placeholder(location) else None
        
        # Try to find existing candidate by email or name
        query = select(Candidate)
        conditions = []
        
        if clean_email:
            conditions.append(Candidate.email == clean_email)
        if clean_name and not clean_email:  # Only use name if no email
            conditions.append(Candidate.name == clean_name)
        
        if conditions:
            query = query.where(conditions[0])
            result = await self.session.execute(query)
            candidate = result.scalar_one_or_none()
            
            if candidate:
                # Update location if provided and different
                if valid_location and candidate.location != valid_location:
                    candidate.location = valid_location
                    candidate.updated_at = datetime.utcnow()
                    await self.session.flush()
                return candidate
        
        # Create new candidate
        candidate = Candidate(
            name=clean_name,
            email=clean_email,
            phone=clean_phone,
            location=valid_location,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.session.add(candidate)
        await self.session.flush()  # Flush to get ID without committing
        return candidate
    
    async def create_resume(
        self,
        candidate_id: int,
        filename: str,
        file_content: bytes,
        raw_text: str,
        summary: Optional[str] = None
    ) -> Resume:
        """Create a new resume record."""
        
        # Generate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if resume with same hash already exists
        query = select(Resume).where(Resume.file_hash == file_hash)
        result = await self.session.execute(query)
        existing_resume = result.scalar_one_or_none()
        
        if existing_resume:
            return existing_resume
        
        resume = Resume(
            candidate_id=candidate_id,
            original_filename=filename,
            file_hash=file_hash,
            raw_text=raw_text,
            summary=summary,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.session.add(resume)
        await self.session.flush()  # Flush to get ID without committing
        return resume
    
    async def store_experience(self, resume_id: int, experiences: List[Dict]) -> List[Experience]:
        """Store experience records for a resume."""
        experience_records = []
        
        for idx, exp_data in enumerate(experiences):
            experience = Experience(
                resume_id=resume_id,
                company=exp_data.get("company"),
                role=exp_data.get("role"),
                start_date=exp_data.get("start_date"),
                end_date=exp_data.get("end_date"),
                description=exp_data.get("description"),
                order_index=idx
            )
            experience_records.append(experience)
            self.session.add(experience)
        
        await self.session.flush()  # Flush to get IDs without committing
        
        return experience_records
    
    async def store_education(self, resume_id: int, educations: List[Dict]) -> List[Education]:
        """Store education records for a resume."""
        education_records = []
        
        for idx, edu_data in enumerate(educations):
            education = Education(
                resume_id=resume_id,
                institution=edu_data.get("institution"),
                degree=edu_data.get("degree"),
                field=edu_data.get("field"),
                start_year=str(edu_data.get("start_year")) if edu_data.get("start_year") else None,
                end_year=str(edu_data.get("end_year")) if edu_data.get("end_year") else None,
                order_index=idx
            )
            education_records.append(education)
            self.session.add(education)
        
        await self.session.flush()  # Flush to get IDs without committing
        return education_records
    
    async def get_or_create_skill(self, skill_name: str, category: Optional[str] = None) -> Skill:
        """Get existing skill or create new one."""
        # Normalize skill name for consistent storage
        normalized_name = skill_name.strip()
        
        query = select(Skill).where(Skill.name == normalized_name)
        result = await self.session.execute(query)
        skill = result.scalar_one_or_none()
        
        if not skill:
            skill = Skill(
                name=normalized_name,
                category=category
            )
            self.session.add(skill)
            await self.session.flush()  # Flush to get ID without committing transaction
        
        return skill
    
    async def store_skills(self, resume_id: int, skills: List[Dict]) -> List[ResumeSkill]:
        """Store skills for a resume."""
        resume_skills = []
        
        for skill_data in skills:
            skill_name = skill_data.get("name")
            if not skill_name:
                continue
            
            # Get or create skill
            skill = await self.get_or_create_skill(
                skill_name, 
                skill_data.get("category")
            )
            
            # Create resume-skill relationship
            resume_skill = ResumeSkill(
                resume_id=resume_id,
                skill_id=skill.id,
                proficiency_level=skill_data.get("proficiency_level"),
                years_experience=skill_data.get("years_experience")
            )
            resume_skills.append(resume_skill)
            self.session.add(resume_skill)
        
        await self.session.flush()  # Flush to get IDs without committing
        return resume_skills
    
    async def store_cv_analysis(
        self,
        filename: str,
        file_content: bytes,
        raw_text: str,
        analysis: Dict[str, Any]
    ) -> Resume:
        """Store complete CV analysis results in database."""
        
        # Get or create candidate
        personal_info = analysis.get("personal_info", {})
        candidate = await self.get_or_create_candidate(
            name=personal_info.get("name"),
            email=personal_info.get("email"),
            phone=personal_info.get("phone"),
            location=personal_info.get("location")
        )
        
        # Create resume
        resume = await self.create_resume(
            candidate_id=candidate.id,
            filename=filename,
            file_content=file_content,
            raw_text=raw_text,
            summary=analysis.get("summary")
        )
        
        # Store experiences
        experiences = analysis.get("experience", [])
        if experiences:
            experience_dicts = [
                {
                    "company": exp.get("company"),
                    "role": exp.get("role"),
                    "start_date": exp.get("start_date"),
                    "end_date": exp.get("end_date"),
                    "description": exp.get("description")
                }
                for exp in experiences
            ]
            await self.store_experience(resume.id, experience_dicts)
        
        # Store education
        educations = analysis.get("education", [])
        if educations:
            education_dicts = [
                {
                    "institution": edu.get("institution"),
                    "degree": edu.get("degree"),
                    "field": edu.get("field"),
                    "start_year": edu.get("start_year"),
                    "end_year": edu.get("end_year")
                }
                for edu in educations
            ]
            await self.store_education(resume.id, education_dicts)
        
        # Store skills
        skills = analysis.get("skills", [])
        if skills:
            skill_dicts = [
                {
                    "name": skill.get("name"),
                    "category": skill.get("category"),
                    "proficiency_level": skill.get("proficiency_level"),
                    "years_experience": skill.get("years_experience")
                }
                for skill in skills
            ]
            await self.store_skills(resume.id, skill_dicts)
        
        # Commit all changes at the end
        await self.session.commit()
        return resume
    
    async def store_embedding(
        self,
        resume_id: int,
        section_type: str,
        content: str,
        embedding: List[float],
        model_name: str
    ) -> ResumeEmbedding:
        """Store vector embedding for resume section."""
        resume_embedding = ResumeEmbedding(
            resume_id=resume_id,
            section_type=section_type,
            section_content=content,
            embedding=embedding,
            model_name=model_name,
            created_at=datetime.utcnow()
        )
        
        self.session.add(resume_embedding)
        await self.session.commit()
        await self.session.refresh(resume_embedding)
        return resume_embedding
    
    async def find_similar_resumes(
        self,
        query_embedding: List[float],
        section_type: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[ResumeEmbedding]:
        """Find similar resumes using cosine similarity."""
        # Note: This is a simplified version. In production, you'd use proper vector similarity queries
        # with pgvector's cosine similarity operators (<=> or <#>)
        
        query = select(ResumeEmbedding).where(
            ResumeEmbedding.section_type == section_type
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()