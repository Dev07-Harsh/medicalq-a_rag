"""
Retrieval modules for MEGA-RAG.

MEGA-RAG v2.0 Retrieval Pipeline:
- Vector Retriever: Dense embeddings with PubMedBERT
- BM25 Retriever: Sparse keyword matching with medical stopwords
- Graph Retriever: Knowledge graph based on medical entities
- Hybrid Retriever: Tri-Brid fusion with cross-encoder reranking
- Query Reformulator: Improves vague/unclear queries (NEW)
"""
# Avoid circular imports - import modules directly when needed

# Export main classes for convenience
from mega_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from mega_rag.retrieval.query_reformulator import QueryReformulator, QueryAnalysis, ReformulatedQuery

__all__ = [
    "HybridRetriever",
    "RetrievalResult",
    "QueryReformulator",
    "QueryAnalysis",
    "ReformulatedQuery"
]
