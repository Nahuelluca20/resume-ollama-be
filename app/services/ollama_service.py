import json
import ollama
from typing import Dict, Any
from fastapi import HTTPException


class OllamaService:
    DEFAULT_MODEL = "gpt-oss:20b"

    @staticmethod
    async def analyze_cv_with_ollama(
        cv_text: str, model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """Analyze CV text using Ollama LLM model with structured output."""
        try:
            prompt = f"""
            You are an expert CV/resume analyzer. Analyze the following CV text and extract structured information.
            Extract all relevant information including personal details, skills, experience, education, certifications, and languages.
            Be thorough and accurate in your extraction.

            Please respond with a valid JSON object following this exact structure:
            {{
                "personal_info": {{
                    "name": "Full Name or null",
                    "email": "email@example.com or null",
                    "phone": "Phone number or null",
                    "location": "Location or null"
                }},
                "summary": "Professional summary or null",
                "skills": [
                    {{
                        "name": "Skill name",
                        "category": "Technical/Soft/Language/etc or null",
                        "proficiency_level": "Beginner/Intermediate/Advanced/Expert or null",
                        "years_experience": number or null
                    }}
                ],
                "experience": [
                    {{
                        "company": "Company name or null",
                        "role": "Job title or null",
                        "start_date": "Start date or null",
                        "end_date": "End date or null",
                        "description": "Job description or null"
                    }}
                ],
                "education": [
                    {{
                        "institution": "School/University name or null",
                        "degree": "Degree type or null",
                        "field": "Field of study or null",
                        "start_year": "Start year or null",
                        "end_year": "End year or null"
                    }}
                ],
                "certifications": ["Certification 1", "Certification 2"],
                "languages": ["Language 1", "Language 2"]
            }}

            CV Content:
            {cv_text}
            """

            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            print(response)
            analysis_text = response["message"]["content"].strip()

            # Parse JSON response directly
            try:
                analysis = json.loads(analysis_text)

                return {
                    "analysis": analysis,
                    "raw_response": analysis_text,
                    "model_used": model,
                    "raw_text_length": len(cv_text),
                }

            except json.JSONDecodeError as e:
                # Fallback: return raw analysis if JSON parsing fails
                return {
                    "analysis": None,
                    "raw_response": analysis_text,
                    "model_used": model,
                    "raw_text_length": len(cv_text),
                    "parsing_error": str(e),
                }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error analyzing CV with Ollama: {str(e)}"
            )

    @staticmethod
    async def generate_match_explanation(
        job_description: str,
        candidate_summary: str,
        matched_skills: list,
        experience_summary: str,
        match_scores: dict,
        model: str = DEFAULT_MODEL
    ) -> str:
        """Generate a detailed explanation of why a candidate matches a job."""
        try:
            prompt = f"""
            You are an expert recruiter analyzing candidate-job matches. Provide a detailed explanation of why this candidate matches the job posting.

            Job Description:
            {job_description}

            Candidate Profile:
            - Summary: {candidate_summary}
            - Matched Skills: {', '.join(matched_skills) if matched_skills else 'None specifically identified'}
            - Experience Summary: {experience_summary}

            Match Scores:
            - Overall Match: {match_scores.get('overall_score', 0):.1%}
            - Skills Match: {match_scores.get('skill_match_score', 0):.1%}
            - Experience Match: {match_scores.get('experience_match_score', 0):.1%}
            - Semantic Similarity: {match_scores.get('semantic_similarity_score', 0):.1%}

            Please provide:
            1. Key strengths and matches between the candidate and job requirements
            2. Areas where the candidate excels for this role
            3. Any potential gaps or areas for development
            4. Overall assessment and recommendation

            Keep the explanation concise but comprehensive (2-3 paragraphs).
            """

            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            return response["message"]["content"].strip()

        except Exception as e:
            return f"Error generating match explanation: {str(e)}"

    @staticmethod
    def check_ollama_health() -> Dict[str, Any]:
        """Check Ollama service health and available models."""
        try:
            models = ollama.list()
            return {
                "status": "healthy",
                "ollama_available": True,
                "available_models": [model["model"] for model in models["models"]],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "ollama_available": False,
                "error": str(e),
                "available_models": [],
            }
