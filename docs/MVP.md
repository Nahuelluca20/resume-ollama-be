# Architecture at a Glance

## Frontend (React)

- Upload CVs, show parsed fields, let users correct/approve.
- Display match scores + “why” highlights.

## API (FastAPI)

- Endpoints for upload, parsing, embedding, matching, and feedback.
- Background worker for heavier tasks.

## Model Runtime (Ollama)

- Local LLM(s) for extraction, normalization, and zero-shot classification.
- Lightweight embedding model for search/matching.

## Storage

- **Object storage**: raw files (S3-compatible: MinIO locally, R2/S3 in cloud).
- **Postgres + pgvector**: metadata + vector search (hybrid search = text + vectors).
- **Redis**: queues, caching, idempotency.

## Workers

- Celery/Dramatiq/RQ for async parsing & OCR (keep API fast).

## Observability

- OpenTelemetry + Prometheus/Grafana (or just structured logs initially).

---

# 1. Other Technologies to Use

## Parsing & NLP

- **PDF (digital)**: PyMuPDF (fast + layout info) or pdfplumber.
- **OCR**: Tesseract + ocrmypdf for scanned PDFs/photos.
- **General parser**: `unstructured` (handles PDFs, DOCX, HTML; outputs sections).
- **Language detection**: fastText, langid, or langdetect.
- **NER/PII detection**: spaCy + Presidio.

## Embeddings & Search

- Multilingual embedding model (e.g., bge-m3, nomic-embed-text).
- Vector DB: pgvector (Postgres) → Qdrant/Weaviate if scaling.

## Job/Skills Normalization

- Maintain a skills taxonomy (later integrate ESCO / O\*NET).
- Use LLM for synonyms → canonical skills, extracting years, education, etc.

## Scheduling & Pipelines

- Celery (with Redis/RabbitMQ):
  1. File ingestion
  2. Parse
  3. OCR if needed
  4. Clean
  5. Embed
  6. Persist
  7. Score

## Security & Compliance

- File scanning (ClamAV), encryption-at-rest, signed URLs, RBAC, retention settings.

---

# 2. Reading PDF CVs / Transforming to Text

## Detection

- Check MIME + extension.
- If scanned → run ocrmypdf → extract again.

## Extraction

- PyMuPDF for coordinates.
- `unstructured` for higher-level elements.

## Normalization

- Clean whitespace, fix hyphens, join lines, remove headers/footers.
- Keep page + bbox for UI highlights.

## Fallbacks

- Messy PDFs: render to images → OCR per page.
- DOCX: python-docx or docx2txt.

## Enrichment

- LLM to extract: name, email, phone, location, languages, education, roles, dates, skills, years/skill.
- Detect language and store original + normalized.

---

# 3. Storing CVs (Data Model & Strategy)

## Object Storage

- `/cvs/{org}/{uuid}/original.pdf`
- Derived: OCR PDF, extracted JSON, plain text, thumbnails.
- Signed URLs (short TTL).

## Postgres Schema

- **candidates**: id, name, email_hash, phone_hash, location, created_at, …
- **resumes**: id, candidate_id, file_uri, lang, parsed_json, text_checksum, created_at, is_active
- **resume_chunks**: id, resume_id, chunk_index, text, embedding, page_ref, bbox, section_type
- **jobs**: id, title, company, description, location, required_skills JSONB, nice_to_have JSONB, lang, created_at
- **job_chunks**: id, job_id, text, embedding, section_type
- **matches**: id, resume_id, job_id, score, factors JSONB, created_at
- **skills**: id, canonical_name, aliases JSONB

## Why This Design

- Raw + structured stored together.
- Chunk embeddings → better highlights & retrieval.
- PII hashed for deduplication without storing raw.

## Retention & Privacy

- Configurable TTLs per tenant.
- Soft vs hard delete.
- Audit logs for views/downloads.

---

# Matching Logic

**Score Composition**

- **Semantic similarity**: resume-chunk ↔ job-chunk cosine sims.
- **Skills overlap**: Jaccard/weighted overlap; penalties for missing must-haves.
- **Experience fit**: years extracted vs min required.
- **Constraints**: language, location, authorization (pass/fail).

**Calibration**

- Learn weights (α, β, γ) from feedback (logistic regression works well).

**UI**

- Show top matching chunks with highlights.
- Indicate missing must-haves.
- Short LLM rationale.

---

# MVP Scope (2–3 Weeks)

1. Upload CV → parse (PDF+OCR) → extract basics.
2. Create job postings → embed.
3. Compute match score + show top-K with explanations.
4. Feedback buttons for labeling.
5. Admin tools: re-run parsing, edit skills, export CSV/JSON.

---

# Picking Models in Ollama

- **LLM**: Llama 3.1 8B instruct or Mistral 7B instruct.
- **Embeddings**: multilingual model (e.g., bge-\* or nomic-embed-text).

---

# Key Early Decisions

- Multilingual support? If yes, pick multilingual embeddings now.
- Keep raw files or discard after processing?
- Hybrid search vs vectors-only? (Hybrid preferred).
- Use background jobs for OCR/embedding.
- Maintain chunk references for explainability.
