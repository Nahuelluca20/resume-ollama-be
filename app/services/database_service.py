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
from app.schemas.cv_schemas import CVAnalysisSchema


class PIIHasher:
    """Utility class for hashing PII data for privacy."""
    
    @staticmethod
    def hash_pii(value: str, salt: str = "cv_ollama_salt") -> str:
        """Hash PII data with salt for privacy protection."""
        if not value:
            return None
        combined = f"{salt}:{value.lower().strip()}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def hash_name(name: str) -> Optional[str]:
        """Hash candidate name."""
        return PIIHasher.hash_pii(name) if name else None
    
    @staticmethod
    def hash_email(email: str) -> Optional[str]:
        """Hash candidate email."""
        return PIIHasher.hash_pii(email) if email else None
    
    @staticmethod
    def hash_phone(phone: str) -> Optional[str]:
        """Hash candidate phone number."""
        if not phone:
            return None
        # Clean phone number (remove spaces, dashes, parentheses)
        cleaned_phone = ''.join(filter(str.isdigit, phone))
        return PIIHasher.hash_pii(cleaned_phone)


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
        
        # Hash PII data
        name_hash = PIIHasher.hash_name(name)
        email_hash = PIIHasher.hash_email(email)
        phone_hash = PIIHasher.hash_phone(phone)
        
        # Try to find existing candidate by email or name hash
        query = select(Candidate)
        conditions = []
        
        if email_hash:
            conditions.append(Candidate.email_hash == email_hash)
        if name_hash and not email_hash:  # Only use name if no email
            conditions.append(Candidate.name_hash == name_hash)
        
        if conditions:
            query = query.where(conditions[0])
            result = await self.session.execute(query)
            candidate = result.scalar_one_or_none()
            
            if candidate:
                # Update location if provided and different
                if location and candidate.location != location:
                    candidate.location = location
                    candidate.updated_at = datetime.utcnow()
                    await self.session.commit()
                return candidate
        
        # Create new candidate
        candidate = Candidate(
            name_hash=name_hash,
            email_hash=email_hash,
            phone_hash=phone_hash,
            location=location,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.session.add(candidate)
        await self.session.commit()
        await self.session.refresh(candidate)
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
        await self.session.commit()
        await self.session.refresh(resume)
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
        
        await self.session.commit()
        for exp in experience_records:
            await self.session.refresh(exp)
        
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
                start_year=edu_data.get("start_year"),
                end_year=edu_data.get("end_year"),
                order_index=idx
            )
            education_records.append(education)
            self.session.add(education)
        
        await self.session.commit()
        for edu in education_records:
            await self.session.refresh(edu)
        
        return education_records
    
    async def get_or_create_skill(self, skill_name: str, category: Optional[str] = None) -> Skill:
        """Get existing skill or create new one."""
        # Normalize skill name
        normalized_name = skill_name.strip().lower()
        
        query = select(Skill).where(Skill.name == normalized_name)
        result = await self.session.execute(query)
        skill = result.scalar_one_or_none()
        
        if not skill:
            skill = Skill(
                name=normalized_name,
                category=category
            )
            self.session.add(skill)
            await self.session.commit()
            await self.session.refresh(skill)
        
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
        
        await self.session.commit()
        for rs in resume_skills:
            await self.session.refresh(rs)
        
        return resume_skills
    
    async def store_cv_analysis(
        self,
        filename: str,
        file_content: bytes,
        raw_text: str,
        analysis: CVAnalysisSchema
    ) -> Resume:
        """Store complete CV analysis results in database."""
        
        # Get or create candidate
        personal_info = analysis.personal_info
        candidate = await self.get_or_create_candidate(
            name=personal_info.name,
            email=personal_info.email,
            phone=personal_info.phone,
            location=personal_info.location
        )
        
        # Create resume
        resume = await self.create_resume(
            candidate_id=candidate.id,
            filename=filename,
            file_content=file_content,
            raw_text=raw_text,
            summary=analysis.summary
        )
        
        # Store experiences
        if analysis.experience:
            experience_dicts = [
                {
                    "company": exp.company,
                    "role": exp.role,
                    "start_date": exp.start_date,
                    "end_date": exp.end_date,
                    "description": exp.description
                }
                for exp in analysis.experience
            ]
            await self.store_experience(resume.id, experience_dicts)
        
        # Store education
        if analysis.education:
            education_dicts = [
                {
                    "institution": edu.institution,
                    "degree": edu.degree,
                    "field": edu.field,
                    "start_year": edu.start_year,
                    "end_year": edu.end_year
                }
                for edu in analysis.education
            ]
            await self.store_education(resume.id, education_dicts)
        
        # Store skills
        if analysis.skills:
            skill_dicts = [
                {
                    "name": skill.name,
                    "category": skill.category,
                    "proficiency_level": skill.proficiency_level,
                    "years_experience": skill.years_experience
                }
                for skill in analysis.skills
            ]
            await self.store_skills(resume.id, skill_dicts)
        
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