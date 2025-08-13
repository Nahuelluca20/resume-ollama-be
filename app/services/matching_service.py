from typing import List, Dict, Any, Optional
import re
import time
from sqlmodel import Session, select
from fastapi import HTTPException

from app.models.cv_models import Resume, ResumeEmbedding, Skill, ResumeSkill, Experience, Candidate
from app.services.embedding_service import EmbeddingService
from app.schemas.cv_schemas import JobDescriptionRequest, CVMatch, MatchScore, ExperienceSchema
from app.core.database import get_db_session


class MatchingService:
    """Service for matching job descriptions against stored CVs."""
    
    @staticmethod
    async def find_matches_with_session(job_request: JobDescriptionRequest, session: Session) -> Dict[str, Any]:
        """Find matching CVs for a given job description using provided database session."""
        start_time = time.time()
        
        try:
            # Generate job description embedding
            job_embedding = await EmbeddingService.generate_job_embedding(job_request.job_description)
            
            # Get all resumes with their candidates
            query = select(Resume, Candidate).join(Candidate).where(Resume.id.is_not(None))
            result = await session.exec(query)
            resumes_with_candidates = result.all()
            
            if not resumes_with_candidates:
                return {
                    "success": True,
                    "matches": [],
                    "total_cvs_analyzed": 0,
                    "processing_time": time.time() - start_time
                }
            
            matches = []
            
            for resume, candidate in resumes_with_candidates:
                try:
                    match = await MatchingService._calculate_match_score(
                        resume, candidate, job_request, job_embedding, session
                    )
                    if match and match.match_score.overall_score > 0.1:  # Minimum threshold
                        matches.append(match)
                except Exception as e:
                    print(f"Error processing resume {resume.id}: {str(e)}")
                    continue
            
            # Sort matches by overall score (descending)
            matches.sort(key=lambda x: x.match_score.overall_score, reverse=True)
            
            # Limit results
            max_results = job_request.max_results or 10
            matches = matches[:max_results]
            
            return {
                "success": True,
                "matches": matches,
                "total_cvs_analyzed": len(resumes_with_candidates),
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error finding matches: {str(e)}"
            )
    
    @staticmethod
    async def _calculate_match_score(
        resume: Resume, 
        candidate: Candidate, 
        job_request: JobDescriptionRequest,
        job_embedding: List[float],
        session: Session
    ) -> Optional[CVMatch]:
        """Calculate match score between a resume and job description."""
        try:
            # Get resume embeddings
            embedding_query = select(ResumeEmbedding).where(
                ResumeEmbedding.resume_id == resume.id,
                ResumeEmbedding.section_type == "full_text"
            )
            embedding_result = await session.exec(embedding_query)
            resume_embedding_obj = embedding_result.first()
            
            if not resume_embedding_obj:
                return None
            
            # Calculate semantic similarity
            semantic_score = EmbeddingService.calculate_cosine_similarity(
                job_embedding, resume_embedding_obj.embedding
            )
            
            # Get resume skills
            skills_query = select(ResumeSkill, Skill).join(Skill).where(
                ResumeSkill.resume_id == resume.id
            )
            skills_result = await session.exec(skills_query)
            resume_skills = skills_result.all()
            
            # Calculate skill match score
            skill_match_score, matched_skills = MatchingService._calculate_skill_match(
                resume_skills, job_request.required_skills or [], job_request.preferred_skills or []
            )
            
            # Get resume experiences
            experience_query = select(Experience).where(Experience.resume_id == resume.id)
            experiences_result = await session.exec(experience_query)
            experiences = experiences_result.all()
            
            # Calculate experience match score
            experience_match_score, relevant_experiences, total_years = MatchingService._calculate_experience_match(
                experiences, job_request.minimum_experience_years, job_request.job_description
            )
            
            # Calculate overall score (weighted average)
            overall_score = (
                semantic_score * 0.4 +
                skill_match_score * 0.35 +
                experience_match_score * 0.25
            )
            
            # Create match score object
            match_score = MatchScore(
                overall_score=round(overall_score, 3),
                skill_match_score=round(skill_match_score, 3),
                experience_match_score=round(experience_match_score, 3),
                semantic_similarity_score=round(semantic_score, 3)
            )
            
            # Convert experiences to schema format
            relevant_exp_schemas = [
                ExperienceSchema(
                    company=exp.company,
                    role=exp.role,
                    start_date=exp.start_date,
                    end_date=exp.end_date,
                    description=exp.description
                )
                for exp in relevant_experiences
            ]
            
            return CVMatch(
                resume_id=resume.id,
                candidate_name=getattr(candidate, 'name', None),
                candidate_email=getattr(candidate, 'email', None),
                candidate_location=candidate.location,
                match_score=match_score,
                matched_skills=matched_skills,
                relevant_experience=relevant_exp_schemas,
                years_of_experience=total_years
            )
        
        except Exception as e:
            print(f"Error calculating match score for resume {resume.id}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_skill_match(
        resume_skills: List[tuple], 
        required_skills: List[str], 
        preferred_skills: List[str]
    ) -> tuple[float, List[str]]:
        """Calculate skill match score and return matched skills."""
        if not resume_skills:
            return 0.0, []
        
        # Extract skill names from resume
        resume_skill_names = [
            skill.name.lower() for resume_skill, skill in resume_skills
        ]
        
        # Normalize job skills for comparison
        required_skills_lower = [skill.lower() for skill in required_skills]
        preferred_skills_lower = [skill.lower() for skill in preferred_skills]
        
        # Find matches
        matched_required = []
        matched_preferred = []
        
        for resume_skill in resume_skill_names:
            # Check for exact matches or partial matches
            for req_skill in required_skills_lower:
                if req_skill in resume_skill or resume_skill in req_skill:
                    matched_required.append(req_skill)
                    break
            
            for pref_skill in preferred_skills_lower:
                if pref_skill in resume_skill or resume_skill in pref_skill:
                    matched_preferred.append(pref_skill)
                    break
        
        # Calculate score
        required_match_ratio = len(matched_required) / max(len(required_skills_lower), 1)
        preferred_match_ratio = len(matched_preferred) / max(len(preferred_skills_lower), 1)
        
        # Weight required skills more heavily
        skill_score = (required_match_ratio * 0.7) + (preferred_match_ratio * 0.3)
        
        # Return all matched skills for display
        all_matched = list(set(matched_required + matched_preferred))
        
        return skill_score, all_matched
    
    @staticmethod
    def _calculate_experience_match(
        experiences: List[Experience], 
        minimum_experience_years: Optional[int],
        job_description: str
    ) -> tuple[float, List[Experience], Optional[int]]:
        """Calculate experience match score and return relevant experiences."""
        if not experiences:
            return 0.0, [], None
        
        # Calculate total years of experience
        total_years = MatchingService._calculate_total_experience_years(experiences)
        
        # Experience years score
        years_score = 1.0
        if minimum_experience_years:
            if total_years is None:
                years_score = 0.5  # Some uncertainty
            elif total_years < minimum_experience_years:
                years_score = total_years / minimum_experience_years
        
        # Find relevant experiences based on job description keywords
        relevant_experiences = MatchingService._find_relevant_experiences(
            experiences, job_description
        )
        
        # Relevance score based on how many experiences seem relevant
        relevance_score = len(relevant_experiences) / len(experiences) if experiences else 0
        
        # Combined experience score
        experience_score = (years_score * 0.6) + (relevance_score * 0.4)
        
        return experience_score, relevant_experiences, total_years
    
    @staticmethod
    def _calculate_total_experience_years(experiences: List[Experience]) -> Optional[int]:
        """Calculate total years of experience from experience entries."""
        total_months = 0
        
        for exp in experiences:
            try:
                if exp.start_date and exp.end_date:
                    # Simple calculation - assumes YYYY-MM format or just YYYY
                    start_year = int(re.search(r'\d{4}', exp.start_date).group())
                    end_year = int(re.search(r'\d{4}', exp.end_date).group())
                    
                    if end_year >= start_year:
                        total_months += (end_year - start_year) * 12
            except (ValueError, AttributeError):
                # Skip if dates can't be parsed
                continue
        
        return total_months // 12 if total_months > 0 else None
    
    @staticmethod
    def _find_relevant_experiences(
        experiences: List[Experience], 
        job_description: str
    ) -> List[Experience]:
        """Find experiences that are relevant to the job description."""
        if not job_description:
            return experiences
        
        # Extract keywords from job description
        job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
        
        relevant_experiences = []
        
        for exp in experiences:
            # Check if experience contains relevant keywords
            exp_text = " ".join(filter(None, [
                exp.role or "",
                exp.company or "",
                exp.description or ""
            ])).lower()
            
            exp_keywords = set(re.findall(r'\b\w+\b', exp_text))
            
            # Calculate keyword overlap
            common_keywords = job_keywords.intersection(exp_keywords)
            overlap_ratio = len(common_keywords) / len(job_keywords) if job_keywords else 0
            
            # Consider relevant if there's significant keyword overlap
            if overlap_ratio > 0.1:  # At least 10% keyword overlap
                relevant_experiences.append(exp)
        
        return relevant_experiences if relevant_experiences else experiences[:3]  # Return top 3 if none are particularly relevant