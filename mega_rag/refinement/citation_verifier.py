"""
Citation Verification Module

Post-processing step to verify that citations in generated answers
are actually supported by the cited evidence.

This helps catch:
1. Hallucinated citations (citing sources that don't exist)
2. Misattributed claims (claim doesn't match cited source)
3. Unsupported generalizations
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from sentence_transformers import CrossEncoder
import numpy as np

from mega_rag.config import CROSS_ENCODER_MODEL, ENABLE_CITATION_VERIFICATION


@dataclass
class Citation:
    """Represents a citation in the answer."""
    claim: str  # The claim being made
    source_reference: str  # The cited source (e.g., "[Source: WHO Guidelines]")
    start_pos: int  # Position in answer
    end_pos: int


@dataclass
class CitationVerification:
    """Result of verifying a single citation."""
    citation: Citation
    is_verified: bool
    confidence: float
    matching_evidence: Optional[str]
    explanation: str


@dataclass 
class VerificationResult:
    """Overall verification result for an answer."""
    original_answer: str
    verified_answer: str
    total_citations: int
    verified_citations: int
    failed_citations: int
    verification_details: List[CitationVerification]
    overall_citation_accuracy: float
    
    # Aliases for workflow compatibility
    @property
    def unverified_citations(self) -> int:
        return self.failed_citations
    
    @property
    def verification_rate(self) -> float:
        return self.overall_citation_accuracy
    
    @property
    def citation_matches(self) -> List['CitationMatch']:
        """Convert to workflow-compatible format."""
        matches = []
        for detail in self.verification_details:
            matches.append(CitationMatch(
                citation_text=detail.citation.claim,
                claimed_source=detail.citation.source_reference,
                is_verified=detail.is_verified,
                confidence=detail.confidence
            ))
        return matches


@dataclass
class CitationMatch:
    """Workflow-compatible citation match structure."""
    citation_text: str
    claimed_source: str
    is_verified: bool
    confidence: float


class CitationVerifier:
    """
    Verifies that citations in generated answers are supported by evidence.
    
    Process:
    1. Extract all citations from the answer
    2. For each citation, find the matching evidence chunk
    3. Verify the claim is actually supported by that evidence
    4. Flag or remove unsupported citations
    """
    
    def __init__(self, cross_encoder_model: str = CROSS_ENCODER_MODEL):
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.verification_threshold = 0.5  # Minimum score to consider verified
        
    def _extract_citations(self, answer: str) -> List[Citation]:
        """
        Extract all citations from the answer.
        
        Looks for patterns like:
        - [Source: Document Name]
        - [Source: Document Name, Page X]
        - (Source: ...)
        - According to [Document Name]
        """
        citations = []
        
        # Pattern 1: [Source: ...] format
        pattern1 = r'([^.!?]+?)\s*\[Source:\s*([^\]]+)\]'
        for match in re.finditer(pattern1, answer, re.IGNORECASE):
            claim = match.group(1).strip()
            source = match.group(2).strip()
            citations.append(Citation(
                claim=claim,
                source_reference=f"[Source: {source}]",
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Pattern 2: (Source: ...) format
        pattern2 = r'([^.!?]+?)\s*\(Source:\s*([^)]+)\)'
        for match in re.finditer(pattern2, answer, re.IGNORECASE):
            claim = match.group(1).strip()
            source = match.group(2).strip()
            citations.append(Citation(
                claim=claim,
                source_reference=f"(Source: {source})",
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Pattern 3: "According to [Document]" format
        pattern3 = r'[Aa]ccording to\s+\[?([^\],]+)\]?,?\s*([^.!?]+[.!?])'
        for match in re.finditer(pattern3, answer):
            source = match.group(1).strip()
            claim = match.group(2).strip()
            citations.append(Citation(
                claim=claim,
                source_reference=f"According to {source}",
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Deduplicate by position
        seen_positions = set()
        unique_citations = []
        for cit in citations:
            pos_key = (cit.start_pos, cit.end_pos)
            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_citations.append(cit)
        
        return unique_citations
    
    def _find_matching_evidence(
        self,
        citation: Citation,
        evidence_chunks: List[str],
        source_metadata: List[dict]
    ) -> Tuple[Optional[str], float]:
        """
        Find the evidence chunk that matches the cited source.
        
        Returns:
            (matching_evidence, confidence_score)
        """
        if not evidence_chunks:
            return None, 0.0
        
        # Extract source name from citation
        source_name = re.sub(r'[\[\]()]', '', citation.source_reference)
        source_name = source_name.replace('Source:', '').strip().lower()
        
        # First, try exact source matching via metadata
        for i, meta in enumerate(source_metadata):
            filename = meta.get('filename', '').lower()
            source = meta.get('source', '').lower()
            
            # Check if cited source matches metadata
            if source_name in filename or source_name in source:
                return evidence_chunks[i], 0.9
        
        # Fallback: Use semantic similarity to find best match
        pairs = [(citation.claim, evidence) for evidence in evidence_chunks]
        scores = self.cross_encoder.predict(pairs)
        
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        
        # Normalize score
        normalized = (best_score + 5) / 13  # Map from [-5, 8] to [0, 1]
        normalized = max(0, min(1, normalized))
        
        return evidence_chunks[best_idx], normalized
    
    def _verify_claim_against_evidence(
        self,
        claim: str,
        evidence: str
    ) -> Tuple[bool, float, str]:
        """
        Verify that a claim is actually supported by the evidence.
        
        Returns:
            (is_verified, confidence, explanation)
        """
        if not evidence:
            return False, 0.0, "No matching evidence found"
        
        # Use cross-encoder to check entailment
        score = self.cross_encoder.predict([(claim, evidence)])[0]
        
        # Normalize
        normalized = (float(score) + 5) / 13
        normalized = max(0, min(1, normalized))
        
        if normalized >= self.verification_threshold:
            if normalized >= 0.8:
                explanation = "Claim is strongly supported by evidence"
            else:
                explanation = "Claim is moderately supported by evidence"
            return True, normalized, explanation
        else:
            if normalized >= 0.3:
                explanation = "Claim is weakly supported - may need qualification"
            else:
                explanation = "Claim is NOT supported by the cited evidence"
            return False, normalized, explanation
    
    def verify_answer(
        self,
        answer: str,
        evidence_chunks: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> VerificationResult:
        """
        Verify all citations in an answer.
        
        Args:
            answer: The generated answer with citations
            evidence_chunks: The evidence used to generate the answer
            source_metadata: Metadata for each evidence chunk
            
        Returns:
            VerificationResult with details and potentially corrected answer
        """
        if not ENABLE_CITATION_VERIFICATION:
            return VerificationResult(
                original_answer=answer,
                verified_answer=answer,
                total_citations=0,
                verified_citations=0,
                failed_citations=0,
                verification_details=[],
                overall_citation_accuracy=1.0
            )
        
        source_metadata = source_metadata or [{} for _ in evidence_chunks]
        
        # Extract citations
        citations = self._extract_citations(answer)
        
        if not citations:
            return VerificationResult(
                original_answer=answer,
                verified_answer=answer,
                total_citations=0,
                verified_citations=0,
                failed_citations=0,
                verification_details=[],
                overall_citation_accuracy=1.0
            )
        
        # Verify each citation
        verifications = []
        verified_count = 0
        failed_count = 0
        
        for citation in citations:
            # Find matching evidence
            matching_evidence, match_confidence = self._find_matching_evidence(
                citation, evidence_chunks, source_metadata
            )
            
            # Verify claim against evidence
            if matching_evidence:
                is_verified, confidence, explanation = self._verify_claim_against_evidence(
                    citation.claim, matching_evidence
                )
            else:
                is_verified = False
                confidence = 0.0
                explanation = "Could not find matching source in evidence"
            
            verification = CitationVerification(
                citation=citation,
                is_verified=is_verified,
                confidence=confidence,
                matching_evidence=matching_evidence[:200] + "..." if matching_evidence and len(matching_evidence) > 200 else matching_evidence,
                explanation=explanation
            )
            verifications.append(verification)
            
            if is_verified:
                verified_count += 1
            else:
                failed_count += 1
        
        # Generate corrected answer if there are failed citations
        verified_answer = self._generate_corrected_answer(answer, verifications)
        
        # Calculate overall accuracy
        accuracy = verified_count / len(citations) if citations else 1.0
        
        return VerificationResult(
            original_answer=answer,
            verified_answer=verified_answer,
            total_citations=len(citations),
            verified_citations=verified_count,
            failed_citations=failed_count,
            verification_details=verifications,
            overall_citation_accuracy=accuracy
        )
    
    def _generate_corrected_answer(
        self,
        answer: str,
        verifications: List[CitationVerification]
    ) -> str:
        """
        Generate a corrected answer by marking or removing unverified citations.
        """
        corrected = answer
        
        # Sort by position (reverse) to avoid offset issues
        failed_verifications = [v for v in verifications if not v.is_verified]
        failed_verifications.sort(key=lambda v: v.citation.start_pos, reverse=True)
        
        for verification in failed_verifications:
            citation = verification.citation
            
            # Option 1: Mark as unverified (less aggressive)
            # We add a note instead of removing
            original_text = answer[citation.start_pos:citation.end_pos]
            
            if verification.confidence < 0.3:
                # Very low confidence - add strong warning
                replacement = f"{original_text} [⚠️ Citation needs verification]"
            else:
                # Moderate confidence - add mild warning
                replacement = f"{original_text} [Citation partially supported]"
            
            corrected = corrected[:citation.start_pos] + replacement + corrected[citation.end_pos:]
        
        return corrected
    
    def verify_citations(
        self,
        answer: str,
        evidence_chunks: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> VerificationResult:
        """
        Alias for verify_answer() for workflow compatibility.
        """
        return self.verify_answer(answer, evidence_chunks, source_metadata)
    
    def correct_citations(
        self,
        answer: str,
        report: VerificationResult,
        evidence_chunks: List[str]
    ) -> str:
        """
        Correct citations based on verification report.
        Returns the corrected answer from the report or regenerates if needed.
        """
        # If we already have a corrected answer in the report, use it
        if report.verified_answer != report.original_answer:
            return report.verified_answer
        
        # Otherwise, regenerate corrections
        return self._generate_corrected_answer(answer, report.verification_details)


def verify_citations(
    answer: str,
    evidence_chunks: List[str],
    source_metadata: Optional[List[dict]] = None
) -> VerificationResult:
    """Convenience function to verify citations in an answer."""
    verifier = CitationVerifier()
    return verifier.verify_answer(answer, evidence_chunks, source_metadata)


if __name__ == "__main__":
    # Test citation verification
    test_answer = """
    The first-line treatment for hypertension includes thiazide diuretics, 
    ACE inhibitors, and calcium channel blockers [Source: WHO Guidelines].
    
    Patients should maintain a blood pressure below 140/90 mmHg for most adults
    [Source: Hypertension Guidelines, Page 15].
    
    According to recent studies, beta-blockers are no longer recommended as 
    first-line therapy due to inferior outcomes.
    """
    
    test_evidence = [
        "First-line antihypertensive treatment should include thiazide diuretics, ACE inhibitors, ARBs, or calcium channel blockers.",
        "Target blood pressure should be less than 140/90 mmHg for most adult patients with hypertension.",
        "Beta-blockers may be considered for specific indications such as heart failure or post-MI."
    ]
    
    test_metadata = [
        {"filename": "WHO Guidelines.pdf", "page": 10},
        {"filename": "Hypertension Guidelines.pdf", "page": 15},
        {"filename": "Cardiac Treatment.pdf", "page": 5}
    ]
    
    result = verify_citations(test_answer, test_evidence, test_metadata)
    
    print(f"Total citations: {result.total_citations}")
    print(f"Verified: {result.verified_citations}")
    print(f"Failed: {result.failed_citations}")
    print(f"Accuracy: {result.overall_citation_accuracy:.2%}")
    
    print("\nDetails:")
    for v in result.verification_details:
        status = "✓" if v.is_verified else "✗"
        print(f"  {status} [{v.confidence:.2f}] {v.citation.claim[:50]}...")
        print(f"     {v.explanation}")
