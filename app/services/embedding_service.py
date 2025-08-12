import ollama
from typing import List, Dict, Any, Optional
from fastapi import HTTPException


class EmbeddingService:
    """Service for generating embeddings using nomic-embed-text via Ollama."""
    
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    EMBEDDING_DIMENSION = 768  # nomic-embed-text uses 768 dimensions
    
    @staticmethod
    async def generate_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
        """Generate embedding for a text using Ollama."""
        try:
            if not text or not text.strip():
                return [0.0] * EmbeddingService.EMBEDDING_DIMENSION
            
            response = ollama.embeddings(
                model=model,
                prompt=text.strip()
            )
            
            return response["embedding"]
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embedding with Ollama: {str(e)}"
            )
    
    @staticmethod
    async def generate_cv_embeddings(
        analysis: Dict[str, Any],
        raw_text: str,
        model: str = DEFAULT_EMBEDDING_MODEL
    ) -> Dict[str, List[float]]:
        """Generate embeddings for different sections of a CV analysis."""
        embeddings = {}
        
        try:
            # Full text embedding
            if raw_text:
                embeddings["full_text"] = await EmbeddingService.generate_embedding(
                    raw_text, model
                )
            
            # Summary embedding
            summary = analysis.get("summary")
            if summary:
                embeddings["summary"] = await EmbeddingService.generate_embedding(
                    summary, model
                )
            
            # Skills embedding (concatenate all skills)
            skills = analysis.get("skills", [])
            if skills:
                skills_text = " ".join([
                    f"{skill.get('name')} ({skill.get('category')})" if skill.get('category') else skill.get('name')
                    for skill in skills if skill.get('name')
                ])
                if skills_text:
                    embeddings["skills"] = await EmbeddingService.generate_embedding(
                        skills_text, model
                    )
            
            # Experience embedding (concatenate all experience descriptions)
            experiences = analysis.get("experience", [])
            if experiences:
                experience_texts = []
                for exp in experiences:
                    exp_parts = []
                    if exp.get("role"):
                        exp_parts.append(exp.get("role"))
                    if exp.get("company"):
                        exp_parts.append(f"at {exp.get('company')}")
                    if exp.get("description"):
                        exp_parts.append(exp.get("description"))
                    
                    if exp_parts:
                        experience_texts.append(" ".join(exp_parts))
                
                if experience_texts:
                    experience_text = " | ".join(experience_texts)
                    embeddings["experience"] = await EmbeddingService.generate_embedding(
                        experience_text, model
                    )
            
            # Education embedding
            educations = analysis.get("education", [])
            if educations:
                education_texts = []
                for edu in educations:
                    edu_parts = []
                    if edu.get("degree"):
                        edu_parts.append(edu.get("degree"))
                    if edu.get("field"):
                        edu_parts.append(f"in {edu.get('field')}")
                    if edu.get("institution"):
                        edu_parts.append(f"from {edu.get('institution')}")
                    
                    if edu_parts:
                        education_texts.append(" ".join(edu_parts))
                
                if education_texts:
                    education_text = " | ".join(education_texts)
                    embeddings["education"] = await EmbeddingService.generate_embedding(
                        education_text, model
                    )
            
            # Certifications embedding
            certifications = analysis.get("certifications", [])
            if certifications:
                certifications_text = " | ".join(certifications)
                embeddings["certifications"] = await EmbeddingService.generate_embedding(
                    certifications_text, model
                )
            
            return embeddings
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating CV embeddings: {str(e)}"
            )
    
    @staticmethod
    async def generate_job_embedding(job_description: str, model: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
        """Generate embedding for a job description."""
        return await EmbeddingService.generate_embedding(job_description, model)
    
    @staticmethod
    def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def check_embedding_model_availability() -> Dict[str, Any]:
        """Check if the embedding model is available in Ollama."""
        try:
            models = ollama.list()
            available_models = [model["model"] for model in models["models"]]
            
            is_available = EmbeddingService.DEFAULT_EMBEDDING_MODEL in available_models
            
            return {
                "model": EmbeddingService.DEFAULT_EMBEDDING_MODEL,
                "available": is_available,
                "all_models": available_models,
                "dimension": EmbeddingService.EMBEDDING_DIMENSION
            }
            
        except Exception as e:
            return {
                "model": EmbeddingService.DEFAULT_EMBEDDING_MODEL,
                "available": False,
                "error": str(e),
                "all_models": [],
                "dimension": EmbeddingService.EMBEDDING_DIMENSION
            }