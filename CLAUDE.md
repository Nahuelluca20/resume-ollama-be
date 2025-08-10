# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CV/Resume parsing and matching system built with FastAPI and Ollama. The system processes CV documents, extracts structured information using local LLMs, and matches candidates to job postings with semantic similarity scoring.

## Development Commands

### Package Management
- Install dependencies: `uv install`
- Add new dependency: `uv add <package-name>`

### Running the Application  
- Start development server: `uvicorn main:app --reload`
- Start production server: `uvicorn main:app --host 0.0.0.0 --port 8000`

### Testing
Currently no test framework is configured. Consider adding pytest for testing.

## Architecture

The system follows a microservices architecture with these core components:

### API Layer (FastAPI)
- **main.py**: Basic FastAPI application with placeholder endpoints
- Planned endpoints: upload, parsing, embedding, matching, and feedback
- Background worker integration for heavy processing tasks

### Model Runtime (Ollama)
- Local LLM integration for:
  - CV text extraction and normalization
  - Zero-shot classification of skills and experience
  - Semantic matching between CVs and job descriptions
- Lightweight embedding models for vector search

### Planned Storage Architecture
- **Object Storage**: Raw CV files (MinIO locally, S3/R2 for production)
- **PostgreSQL + pgvector**: Structured data and vector search capabilities
- **Redis**: Task queues, caching, and idempotency

### Data Processing Pipeline
1. File ingestion and validation
2. PDF parsing and OCR (for scanned documents)
3. Text cleaning and normalization
4. LLM-based information extraction
5. Embedding generation
6. Vector storage and indexing
7. Match scoring and ranking

## Key Technologies

- **PDF Processing**: PyMuPDF, pdfplumber, unstructured library
- **OCR**: Tesseract + ocrmypdf for scanned documents  
- **Embeddings**: Multilingual models (bge-m3, nomic-embed-text)
- **Task Queues**: Celery with Redis/RabbitMQ
- **Security**: PII detection with spaCy + Presidio

## Data Models (Planned)

### Core Entities
- **candidates**: Personal information (hashed for privacy)
- **resumes**: CV metadata and parsed content
- **resume_chunks**: Text chunks with embeddings and positional data
- **jobs**: Job postings with requirements
- **job_chunks**: Job description chunks with embeddings
- **matches**: Scored matches with explanation factors
- **skills**: Canonical skills taxonomy with aliases

### Matching Logic
- Semantic similarity via vector cosine similarity
- Skills overlap scoring (Jaccard/weighted)
- Experience fit analysis (years vs requirements)
- Constraint validation (language, location, authorization)

## Development Notes

- The current main.py contains only placeholder FastAPI endpoints
- See `docs/MVP.md` for comprehensive system architecture and requirements
- Project uses modern Python (>=3.11) with type hints
- Designed for multilingual CV processing from the start
- Privacy-first approach with PII hashing and configurable retention