import json
import ollama
from typing import Dict, Any
from fastapi import HTTPException
from app.schemas.cv_schemas import (
    CVAnalysisSchema,
    PersonalInfoSchema,
    ExperienceSchema,
    EducationSchema,
    SkillSchema,
)


class OllamaService:
    DEFAULT_MODEL = "gpt-oss:20b"

    @staticmethod
    async def analyze_cv_with_ollama(
        cv_text: str, model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """Analyze CV text using Ollama LLM model."""
        try:
            prompt = f"""
            You are an expert CV/resume analyzer. Analyze the following CV text and extract structured information in valid JSON format.

            Return ONLY a JSON object with these exact fields:
            {{
                "personal_info": {{
                    "name": "extracted name or null",
                    "email": "extracted email or null", 
                    "phone": "extracted phone or null",
                    "location": "extracted location or null"
                }},
                "summary": "professional summary or null",
                "skills": [
                    {{
                        "name": "skill name",
                        "category": "technical/soft/language or null",
                        "proficiency_level": "beginner/intermediate/advanced/expert or null",
                        "years_experience": null or number
                    }}
                ],
                "experience": [
                    {{
                        "company": "company name or null",
                        "role": "job title or null", 
                        "start_date": "start date or null",
                        "end_date": "end date or null",
                        "description": "job description or null"
                    }}
                ],
                "education": [
                    {{
                        "institution": "school/university name or null",
                        "degree": "degree type or null",
                        "field": "field of study or null", 
                        "start_year": "start year or null",
                        "end_year": "end year or null"
                    }}
                ],
                "certifications": ["certification1", "certification2"],
                "languages": ["language1", "language2"]
            }}

            CV Content:
            {cv_text}
            
            JSON Response:
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

            analysis_text = response["message"]["content"].strip()

            # Try to parse JSON from the response
            try:
                # Find JSON content between curly braces
                start_idx = analysis_text.find("{")
                end_idx = analysis_text.rfind("}") + 1

                if start_idx != -1 and end_idx != 0:
                    json_str = analysis_text[start_idx:end_idx]
                    parsed_data = json.loads(json_str)

                    # Validate and structure the response
                    analysis = OllamaService._structure_analysis(parsed_data)

                    return {
                        "analysis": analysis,
                        "raw_response": analysis_text,
                        "model_used": model,
                        "raw_text_length": len(cv_text),
                    }
                else:
                    raise ValueError("No valid JSON found in response")

            except (json.JSONDecodeError, ValueError) as e:
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
    def _structure_analysis(data: Dict[str, Any]) -> CVAnalysisSchema:
        """Structure the parsed data into CVAnalysisSchema."""
        try:
            # Parse personal info
            personal_info_data = data.get("personal_info", {})
            personal_info = PersonalInfoSchema(
                name=personal_info_data.get("name"),
                email=personal_info_data.get("email"),
                phone=personal_info_data.get("phone"),
                location=personal_info_data.get("location"),
            )

            # Parse skills
            skills = []
            for skill_data in data.get("skills", []):
                if isinstance(skill_data, dict):
                    skill = SkillSchema(
                        name=skill_data.get("name", ""),
                        category=skill_data.get("category"),
                        proficiency_level=skill_data.get("proficiency_level"),
                        years_experience=skill_data.get("years_experience"),
                    )
                    skills.append(skill)
                elif isinstance(skill_data, str):
                    skill = SkillSchema(name=skill_data)
                    skills.append(skill)

            # Parse experience
            experiences = []
            for exp_data in data.get("experience", []):
                experience = ExperienceSchema(
                    company=exp_data.get("company"),
                    role=exp_data.get("role"),
                    start_date=exp_data.get("start_date"),
                    end_date=exp_data.get("end_date"),
                    description=exp_data.get("description"),
                )
                experiences.append(experience)

            # Parse education
            educations = []
            for edu_data in data.get("education", []):
                education = EducationSchema(
                    institution=edu_data.get("institution"),
                    degree=edu_data.get("degree"),
                    field=edu_data.get("field"),
                    start_year=edu_data.get("start_year"),
                    end_year=edu_data.get("end_year"),
                )
                educations.append(education)

            return CVAnalysisSchema(
                personal_info=personal_info,
                summary=data.get("summary"),
                skills=skills,
                experience=experiences,
                education=educations,
                certifications=data.get("certifications", []),
                languages=data.get("languages", []),
            )

        except Exception as e:
            # Return empty analysis if structuring fails
            return CVAnalysisSchema(
                personal_info=PersonalInfoSchema(),
                skills=[],
                experience=[],
                education=[],
                certifications=[],
                languages=[],
            )

    @staticmethod
    def check_ollama_health() -> Dict[str, Any]:
        """Check Ollama service health and available models."""
        try:
            models = ollama.list()
            return {
                "status": "healthy",
                "ollama_available": True,
                "available_models": [model["name"] for model in models["models"]],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "ollama_available": False,
                "error": str(e),
                "available_models": [],
            }
