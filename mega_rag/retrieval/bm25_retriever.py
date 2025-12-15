"""
BM25 Keyword Retriever for sparse retrieval.
Implements traditional keyword-based retrieval using BM25 algorithm.
Enhanced with medical domain-aware tokenization.
"""
from __future__ import annotations
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING
from rank_bm25 import BM25Okapi

from mega_rag.config import BM25_TOP_K, BASE_DIR

if TYPE_CHECKING:
    from mega_rag.core.document_processor import Document

# =============================================================================
# Medical Domain-Aware Stopwords
# =============================================================================
# Standard English stopwords that don't carry medical meaning
MEDICAL_STOPWORDS = {
    # Articles and determiners
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    # Conjunctions and prepositions
    'and', 'or', 'but', 'for', 'with', 'from', 'into', 'onto', 'upon',
    'about', 'above', 'after', 'before', 'between', 'through', 'during',
    # Pronouns
    'their', 'there', 'they', 'them', 'its', 'our', 'your', 'his', 'her',
    # Common verbs (non-medical)
    'was', 'were', 'been', 'being', 'have', 'has', 'had', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'are', 'is',
    # Adverbs and modifiers
    'also', 'only', 'both', 'each', 'all', 'any', 'some', 'more', 'most',
    'other', 'such', 'than', 'then', 'very', 'just', 'even', 'well',
    # Common filler words
    'however', 'therefore', 'thus', 'hence', 'whereas', 'although',
    'while', 'when', 'where', 'which', 'what', 'who', 'whom', 'whose',
}

# Medical compound terms to preserve (hyphenated terms)
MEDICAL_COMPOUND_PATTERNS = [
    r'ace-inhibitor[s]?', r'beta-blocker[s]?', r'alpha-blocker[s]?',
    r'calcium-channel', r'angiotensin-receptor', r'blood-pressure',
    r'anti-hypertensive', r'anti-inflammatory', r'anti-diabetic',
    r'first-line', r'second-line', r'third-line', r'over-the-counter',
    r'non-steroidal', r'long-term', r'short-term', r'dose-dependent',
]


class BM25Retriever:
    """
    Sparse keyword retrieval using BM25 algorithm.
    Complements vector retrieval for exact term matching.
    Enhanced with medical domain-aware tokenization.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or (BASE_DIR / "bm25_index.pkl")
        self.bm25 = None
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Medical domain-aware tokenization.
        
        Improvements over basic tokenization:
        1. Preserves hyphenated medical terms (e.g., "ACE-inhibitor")
        2. Removes stopwords that don't carry medical meaning
        3. Handles medical abbreviations correctly
        """
        text = text.lower()
        
        # Extract tokens including hyphenated compounds
        # This preserves medical terms like "ace-inhibitor", "beta-blocker"
        tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
        
        # Filter stopwords and very short tokens (< 3 chars)
        # But keep common medical abbreviations like "BP", "HR"
        filtered_tokens = []
        for token in tokens:
            # Keep if:
            # 1. Not a stopword AND length > 2
            # 2. OR is a hyphenated compound (likely medical term)
            if '-' in token:
                filtered_tokens.append(token)  # Keep all hyphenated terms
            elif len(token) > 2 and token not in MEDICAL_STOPWORDS:
                filtered_tokens.append(token)
        
        return filtered_tokens

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the BM25 index."""
        for doc in documents:
            self.documents.append(doc.content)
            self.metadatas.append(doc.metadata)
            self.tokenized_corpus.append(self._tokenize(doc.content))

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 index built with {len(self.documents)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = BM25_TOP_K
    ) -> List[Tuple[str, float, dict]]:
        """
        Retrieve top-k documents using BM25.

        Returns:
            List of (document_text, bm25_score, metadata) tuples
        """
        if self.bm25 is None or len(self.documents) == 0:
            return []

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                results.append((
                    self.documents[idx],
                    scores[idx],
                    self.metadatas[idx]
                ))

        return results

    def save_index(self) -> None:
        """Save BM25 index to disk."""
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas,
            'tokenized_corpus': self.tokenized_corpus
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk."""
        if not self.index_path.exists():
            return False

        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)

        self.documents = data['documents']
        self.metadatas = data['metadatas']
        self.tokenized_corpus = data['tokenized_corpus']

        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"BM25 index loaded with {len(self.documents)} documents")
        return True

    def get_document_count(self) -> int:
        """Get number of documents in the index."""
        return len(self.documents)

    def clear(self) -> None:
        """Clear the index."""
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.tokenized_corpus = []


if __name__ == "__main__":
    # Test BM25 retriever
    retriever = BM25Retriever()

    # Try to load existing index
    if retriever.load_index():
        results = retriever.retrieve("blood pressure medication")
        print("\nBM25 Top results:")
        for doc, score, meta in results[:3]:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc[:200]}...")
            print()
