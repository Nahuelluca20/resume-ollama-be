import json
import ollama
from typing import Dict, Any
from fastapi import HTTPException
from app.schemas.cv_schemas import (
    CVAnalysisSchema,
)


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
                format=CVAnalysisSchema.model_json_schema(),
            )

            analysis_text = response["message"]["content"].strip()

            # Parse response directly with Pydantic
            try:
                analysis = CVAnalysisSchema.model_validate_json(analysis_text)

                return {
                    "analysis": analysis,
                    "raw_response": analysis_text,
                    "model_used": model,
                    "raw_text_length": len(cv_text),
                }

            except Exception as e:
                # Fallback: return raw analysis if validation fails
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
