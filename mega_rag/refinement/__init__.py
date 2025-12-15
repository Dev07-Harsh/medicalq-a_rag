"""
Refinement modules for hallucination mitigation.

MEGA-RAG v2.0 Refinement Pipeline:
- SEAE: Semantic-Evidential Alignment Evaluation
- DISC: Discrepancy-Identified Self-Clarification
- Chain-of-Thought: Med-PaLM 2 inspired reasoning for complex queries
- Citation Verifier: Post-generation citation validation
"""
# Avoid circular imports - import modules directly when needed

# Export main classes for convenience
from mega_rag.refinement.seae import SEAE, SEAEResult
from mega_rag.refinement.disc import DISC, DISCResult
from mega_rag.refinement.chain_of_thought import MedicalCoTReasoner
from mega_rag.refinement.citation_verifier import CitationVerifier, VerificationResult

__all__ = [
    "SEAE",
    "SEAEResult", 
    "DISC",
    "DISCResult",
    "MedicalCoTReasoner",
    "CitationVerifier",
    "VerificationResult"
]
