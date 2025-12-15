"""
Tri-Brid Hybrid Retriever
Combines Vector, BM25, and Graph retrieval with cross-encoder reranking.
Includes query expansion for improved recall.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import numpy as np

from mega_rag.config import (
    VECTOR_TOP_K,
    BM25_TOP_K,
    GRAPH_TOP_K,
    RERANK_TOP_K,
    CROSS_ENCODER_MODEL,
    ENABLE_QUERY_EXPANSION
)

# Import retrievers directly (no circular import here)
from mega_rag.retrieval.vector_retriever import VectorRetriever
from mega_rag.retrieval.bm25_retriever import BM25Retriever
from mega_rag.retrieval.graph_retriever import GraphRetriever

if TYPE_CHECKING:
    from mega_rag.core.document_processor import Document


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with scores from all methods."""
    content: str
    metadata: dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """
    Tri-Brid Retrieval System combining:
    1. Vector (Dense) - Semantic similarity
    2. BM25 (Sparse) - Keyword matching
    3. Graph - Entity relationships

    With Cross-Encoder reranking for final selection.
    """

    def __init__(
        self,
        vector_weight: float = 0.4,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.3,
        use_reranker: bool = True
    ):
        # =================================================================
        # Dynamic Weight Adjustment when Graph is Disabled
        # =================================================================
        # When GRAPH_TOP_K=0, redistribute graph weight to other retrievers
        # to maintain proper fusion scoring
        if GRAPH_TOP_K == 0:
            total_active_weight = vector_weight + bm25_weight
            self.vector_weight = vector_weight / total_active_weight
            self.bm25_weight = bm25_weight / total_active_weight
            self.graph_weight = 0.0
            print(f"  Graph disabled: weights adjusted to vector={self.vector_weight:.2f}, bm25={self.bm25_weight:.2f}")
        else:
            self.vector_weight = vector_weight
            self.bm25_weight = bm25_weight
            self.graph_weight = graph_weight
            
        self.use_reranker = use_reranker

        # Initialize retrievers
        print("Initializing Tri-Brid Retrieval System...")
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.graph_retriever = GraphRetriever()

        # Initialize cross-encoder for reranking
        if use_reranker:
            print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        else:
            self.cross_encoder = None

        self._is_indexed = False
        self._query_expansion_enabled = ENABLE_QUERY_EXPANSION

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with medical synonyms and related terms.
        Returns a list of query variations for better recall.
        """
        if not self._query_expansion_enabled:
            return [query]

        # Medical term expansions (common synonyms)
        medical_expansions = {
            "hypertension": ["high blood pressure", "elevated blood pressure", "HTN"],
            "diabetes": ["diabetes mellitus", "DM", "blood sugar disorder"],
            "treatment": ["therapy", "management", "intervention", "medication"],
            "medication": ["drug", "medicine", "pharmaceutical", "treatment"],
            "first-line": ["initial", "primary", "first choice", "recommended"],
            "side effects": ["adverse effects", "adverse reactions", "complications"],
            "diagnosis": ["detection", "screening", "assessment", "evaluation"],
            "symptoms": ["signs", "manifestations", "clinical features"],
            "risk factors": ["risk", "predisposing factors", "causes"],
            "prevention": ["prophylaxis", "preventive measures"],
            "cardiovascular": ["heart", "cardiac", "CV"],
            "renal": ["kidney", "nephro"],
            "hepatic": ["liver", "hepato"],
        }

        queries = [query]  # Original query is always first

        # Check if any expansion terms appear in query
        query_lower = query.lower()
        expanded_terms = []

        for term, synonyms in medical_expansions.items():
            if term in query_lower:
                # Add first synonym as alternate query
                expanded_terms.extend(synonyms[:1])

        # Create expanded query variant if synonyms found
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms[:2])
            queries.append(expanded_query)

        return queries[:2]  # Return max 2 queries

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents in all three retrieval systems."""
        print(f"\nIndexing {len(documents)} documents...")

        # Vector index
        print("\n1. Building vector index...")
        self.vector_retriever.add_documents(documents)

        # BM25 index
        print("\n2. Building BM25 index...")
        self.bm25_retriever.add_documents(documents)

        # Graph index
        print("\n3. Building knowledge graph...")
        self.graph_retriever.add_documents(documents)

        self._is_indexed = True
        print("\nIndexing complete!")

    def index_from_pdfs(self, pdf_paths: List[str]) -> None:
        """Process PDFs and index them."""
        from pathlib import Path
        from mega_rag.core.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        documents = processor.process_multiple_pdfs([Path(p) for p in pdf_paths])

        self.index_documents(documents)

    def _normalize_scores(
        self,
        results: List[Tuple[str, float, dict]]
    ) -> Dict[str, Tuple[float, dict]]:
        """Normalize scores to [0, 1] range using min-max scaling."""
        if not results:
            return {}

        scores = [r[1] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = {}
        if max_score == min_score:
            # All scores are identical - assign equal normalized score
            # Use max_score if > 0, otherwise 0.5 as neutral score
            uniform_score = max_score if max_score > 0 else 0.5
            for content, score, metadata in results:
                normalized[content] = (uniform_score, metadata)
        else:
            score_range = max_score - min_score
            for content, score, metadata in results:
                norm_score = (score - min_score) / score_range
                normalized[content] = (norm_score, metadata)

        return normalized

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Tuple[str, float, dict]]],
        k: int = 60
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.
        RRF score = sum(1 / (k + rank)) across all lists
        """
        rrf_scores: Dict[str, float] = {}

        for results in result_lists:
            for rank, (content, score, metadata) in enumerate(results):
                if content not in rrf_scores:
                    rrf_scores[content] = 0
                rrf_scores[content] += 1 / (k + rank + 1)

        return rrf_scores

    def _deduplicate_results(
        self,
        results: List[Tuple[str, float, dict]]
    ) -> List[Tuple[str, float, dict]]:
        """Deduplicate results, keeping the highest score for each content."""
        seen = {}
        for content, score, metadata in results:
            if content not in seen or score > seen[content][1]:
                seen[content] = (content, score, metadata)
        return list(seen.values())

    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        fusion_method: str = "weighted"  # "weighted" or "rrf"
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using Tri-Brid approach with query expansion.

        Args:
            query: The search query
            top_k: Number of final results to return
            fusion_method: "weighted" for weighted combination, "rrf" for reciprocal rank fusion

        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        # Expand query for better recall
        queries = self._expand_query(query)

        # Collect results from all queries
        all_vector_results = []
        all_bm25_results = []
        all_graph_results = []

        for q in queries:
            vector_results = self.vector_retriever.retrieve(q, VECTOR_TOP_K)
            bm25_results = self.bm25_retriever.retrieve(q, BM25_TOP_K)
            graph_results = self.graph_retriever.retrieve(q, GRAPH_TOP_K)

            all_vector_results.extend(vector_results)
            all_bm25_results.extend(bm25_results)
            all_graph_results.extend(graph_results)

        # Deduplicate results (keep highest scores)
        vector_results = self._deduplicate_results(all_vector_results)
        bm25_results = self._deduplicate_results(all_bm25_results)
        graph_results = self._deduplicate_results(all_graph_results)

        # Combine results
        all_contents: Dict[str, RetrievalResult] = {}

        # Process vector results
        vector_normalized = self._normalize_scores(vector_results)
        for content, (score, metadata) in vector_normalized.items():
            if content not in all_contents:
                all_contents[content] = RetrievalResult(
                    content=content, metadata=metadata
                )
            all_contents[content].vector_score = score

        # Process BM25 results
        bm25_normalized = self._normalize_scores(bm25_results)
        for content, (score, metadata) in bm25_normalized.items():
            if content not in all_contents:
                all_contents[content] = RetrievalResult(
                    content=content, metadata=metadata
                )
            all_contents[content].bm25_score = score

        # Process graph results
        graph_normalized = self._normalize_scores(graph_results)
        for content, (score, metadata) in graph_normalized.items():
            if content not in all_contents:
                all_contents[content] = RetrievalResult(
                    content=content, metadata=metadata
                )
            all_contents[content].graph_score = score

        if not all_contents:
            return []

        # Calculate fusion scores
        if fusion_method == "rrf":
            rrf_scores = self._reciprocal_rank_fusion([
                vector_results, bm25_results, graph_results
            ])
            for content, result in all_contents.items():
                result.fusion_score = rrf_scores.get(content, 0)
        else:  # weighted
            for content, result in all_contents.items():
                result.fusion_score = (
                    self.vector_weight * result.vector_score +
                    self.bm25_weight * result.bm25_score +
                    self.graph_weight * result.graph_score
                )

        # Sort by fusion score
        sorted_results = sorted(
            all_contents.values(),
            key=lambda x: x.fusion_score,
            reverse=True
        )

        # Apply cross-encoder reranking
        if self.use_reranker and self.cross_encoder:
            # Get top candidates for reranking
            candidates = sorted_results[:top_k * 2]

            if candidates:
                # Create query-document pairs
                pairs = [(query, r.content) for r in candidates]

                # Get reranking scores
                rerank_scores = self.cross_encoder.predict(pairs)

                # Normalize rerank scores
                min_rs = min(rerank_scores)
                max_rs = max(rerank_scores)
                range_rs = max_rs - min_rs if max_rs != min_rs else 1

                for result, score in zip(candidates, rerank_scores):
                    result.rerank_score = (score - min_rs) / range_rs

                # Sort by rerank score
                sorted_results = sorted(
                    candidates,
                    key=lambda x: x.rerank_score,
                    reverse=True
                )

        return sorted_results[:top_k]

    def save_indices(self) -> None:
        """Save all indices to disk."""
        self.bm25_retriever.save_index()
        self.graph_retriever.save_graph()
        print("All indices saved.")

    def load_indices(self) -> bool:
        """Load all indices from disk."""
        bm25_loaded = self.bm25_retriever.load_index()
        graph_loaded = self.graph_retriever.load_graph()
        vector_count = self.vector_retriever.get_collection_count()

        if bm25_loaded and graph_loaded and vector_count > 0:
            self._is_indexed = True
            return True
        return False

    def get_stats(self) -> dict:
        """Get statistics about all indices."""
        return {
            'vector_docs': self.vector_retriever.get_collection_count(),
            'bm25_docs': self.bm25_retriever.get_document_count(),
            'graph': self.graph_retriever.get_stats()
        }

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed


if __name__ == "__main__":
    # Test hybrid retriever
    retriever = HybridRetriever()

    if retriever.load_indices():
        print(f"Stats: {retriever.get_stats()}")

        results = retriever.retrieve("What is first line treatment for hypertension?")
        print("\n--- Hybrid Retrieval Results ---")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Fusion: {result.fusion_score:.4f}, Rerank: {result.rerank_score:.4f}")
            print(f"   Vector: {result.vector_score:.4f}, BM25: {result.bm25_score:.4f}, Graph: {result.graph_score:.4f}")
            print(f"   Content: {result.content[:200]}...")
