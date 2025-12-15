"""
MEGA-RAG: Medical Evidence-Guided Augmented Retrieval-Augmented Generation

A hallucination-mitigation system for medical QA using:
- Tri-Brid Retrieval (Vector + BM25 + Graph)
- SEAE Hallucination Auditor
- DISC Self-Correction Module
- LangGraph Orchestration
"""

__version__ = "1.0.0"
__author__ = "MEGA-RAG Team"

from mega_rag.config import (
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    COLLECTION_NAME
)

__all__ = [
    "__version__",
    "GEMINI_MODEL",
    "EMBEDDING_MODEL",
    "COLLECTION_NAME"
]
