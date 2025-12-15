"""mega_rag.refinement.seae

SEAE: Semantic-Evidential Alignment Evaluation
Hallucination Auditor that checks if the answer aligns with retrieved evidence.

NOTE ON CACHING
--------------
This module optionally caches embeddings computed by SentenceTransformer.
When implemented with exact-text keys, caching is deterministic and should not
change alignment scores or system behavior; it only improves performance.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib
import threading
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from mega_rag.config import SEAE_THRESHOLD, EMBEDDING_MODEL, CROSS_ENCODER_MODEL


@dataclass
class SEAEResult:
    """Result from SEAE evaluation."""
    is_aligned: bool
    alignment_score: float
    claim_scores: List[Tuple[str, float]]  # (claim, score) pairs
    misaligned_claims: List[str]
    evidence_coverage: float
    explanation: str


class SEAE:
    """
    Semantic-Evidential Alignment Evaluation (SEAE)

    Evaluates whether generated answers are grounded in retrieved evidence.
    Uses a combination of:
    1. Claim extraction - Break answer into atomic claims
    2. Semantic similarity - Compare claims to evidence
    3. Entailment checking - Verify claims are supported

    This acts as the hallucination auditor in the MEGA-RAG refinement loop.
    """

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        threshold: float = SEAE_THRESHOLD
    ):
        self.threshold = threshold

        # In-memory embedding cache (safe + deterministic).
        # Keyed by: SHA256(embedding_model_name + "\n" + exact_text)
        # Values stored as immutable numpy arrays on CPU.
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._emb_cache_lock = threading.Lock()
        self._emb_cache_max_items = 10_000  # simple bound to avoid unbounded growth

        # Models for evaluation
        print("Loading SEAE models...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        # Used for cache keys.
        self._embedding_model_id = str(embedding_model)

    def _emb_cache_key(self, text: str) -> str:
        """Create a deterministic cache key for an exact input text."""
        payload = (self._embedding_model_id + "\n" + (text or "")).encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()

    def _encode_cached(self, texts: List[str]) -> np.ndarray:
        """Encode texts with SentenceTransformer, reusing cached embeddings when possible.

        Returns a 2D numpy array with shape (len(texts), dim).
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [self._emb_cache_key(t) for t in texts]

        # Fast path: collect cached embeddings
        cached: List[Optional[np.ndarray]] = [None] * len(texts)
        missing_idx: List[int] = []
        with self._emb_cache_lock:
            for i, k in enumerate(keys):
                v = self._emb_cache.get(k)
                if v is None:
                    missing_idx.append(i)
                else:
                    cached[i] = v

        # Encode missing texts in one batch (preserves determinism)
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            embs = self.embedding_model.encode(missing_texts, normalize_embeddings=True)
            embs_np = np.asarray(embs, dtype=np.float32)

            with self._emb_cache_lock:
                # Simple bounded cache: if full, clear (keeps implementation safe/simple)
                if len(self._emb_cache) > self._emb_cache_max_items:
                    self._emb_cache.clear()

                for j, i in enumerate(missing_idx):
                    cached[i] = embs_np[j]
                    self._emb_cache[keys[i]] = embs_np[j]

            # Preserve original ordering
            return np.vstack(cached)

        # All items were already cached
        return np.vstack(cached)

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract atomic claims from an answer.
        Simple approach: split by sentences and common claim delimiters.
        Filters out non-factual statements like "Final Answer: yes/no/maybe".
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        claims = []
        for sent in sentences:
            sent = sent.strip()
            
            # Skip empty or very short sentences
            if not sent or len(sent) < 10:
                continue
            
            sent_lower = sent.lower().strip()
            
            # ============================================================
            # SKIP NON-FACTUAL STATEMENTS (critical for yes/no/maybe tasks)
            # ============================================================
            
            # Skip "Final Answer: yes/no/maybe" in any format
            if re.search(r'final\s*answer\s*[:\s]*(yes|no|maybe)', sent_lower, re.IGNORECASE):
                continue
            
            # Skip standalone yes/no/maybe (possibly with punctuation)
            if re.match(r'^\s*(yes|no|maybe)\s*[.!?]*\s*$', sent_lower):
                continue
                
            # Skip sentences that are ONLY "Final Answer" type content
            if re.match(r'^\s*final\s+answer\s*$', sent_lower):
                continue
            
            # Skip instruction-like sentences (from prompt leakage)
            skip_patterns = [
                r'^\s*this\s+is\s+a\s+yes/no/maybe\s+question',
                r'^\s*your\s+(corrected\s+)?answer\s+must',
                r'^\s*briefly\s+explain',
                r'^\s*end\s+with\s+exactly',
                r'^\s*question\s*:',
                r'^\s*answer\s*:',
                r'^\s*explanation\s*:',
            ]
            if any(re.match(pat, sent_lower) for pat in skip_patterns):
                continue
            
            # Further split compound sentences
            sub_claims = re.split(r'[;,]\s*(?:and|but|or|however|also)\s*', sent)
            for claim in sub_claims:
                claim = claim.strip()
                if not claim or len(claim) < 15:  # Skip very short claims
                    continue
                    
                claim_lower = claim.lower().strip()
                
                # Skip final answer patterns in sub-claims too
                if re.search(r'final\s*answer', claim_lower, re.IGNORECASE):
                    continue
                
                # Skip citation-only content
                if re.match(r'^\s*\[evidence\s*\d+\]', claim_lower):
                    continue
                if re.match(r'^\s*\[source:', claim_lower):
                    continue
                if re.match(r'^\s*\[\d+\]\s*$', claim):
                    continue
                if re.match(r'^\s*(see\s+)?evidence\s*\d+', claim_lower):
                    continue
                
                # Clean leading Yes/No/Maybe markers
                claim_clean = re.sub(r'^(yes|no|maybe|possibly|perhaps)[,\s]+', '', claim, flags=re.IGNORECASE).strip()
                
                if claim_clean and len(claim_clean) >= 15:
                    claims.append(claim_clean)
        
        # If no substantive claims found, check if this is a decision-only answer
        # (e.g., "Final Answer: yes" with no explanation)
        # In this case, we return empty claims - the evaluate() method will handle this
        return claims

    def _is_decision_only_answer(self, answer: str) -> bool:
        """Check if answer contains only a yes/no/maybe decision without explanation."""
        import re
        answer_clean = answer.strip()
        
        # Very short answer (less than 50 chars) that contains a decision
        if len(answer_clean) < 50:
            if re.search(r'\b(yes|no|maybe)\b', answer_clean, re.IGNORECASE):
                return True
        
        # Answer is just "Final Answer: X" possibly with whitespace
        if re.match(r'^\s*final\s*answer\s*[:\s]*(yes|no|maybe)\s*[.!?]?\s*$', answer_clean, re.IGNORECASE):
            return True
            
        return False

    def _compute_claim_evidence_alignment(
        self,
        claim: str,
        evidence_chunks: List[str]
    ) -> Tuple[float, str]:
        """
        Compute alignment score between a claim and evidence.
        Returns (score, best_matching_evidence).
        
        Uses calibrated min-max normalization for cross-encoder scores.
        Cross-encoder (ms-marco-MiniLM) outputs logits typically in range [-10, +10].
        """
        if not evidence_chunks:
            return 0.0, ""

        # Use cross-encoder for more accurate alignment
        pairs = [(claim, evidence) for evidence in evidence_chunks]
        scores = self.cross_encoder.predict(pairs)

        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])

        # =================================================================
        # Calibrated Normalization for Cross-Encoder Scores
        # =================================================================
        # Cross-encoder scores are logits, not probabilities.
        # MS-MARCO MiniLM typically outputs:
        #   - Highly relevant: +5 to +10
        #   - Somewhat relevant: 0 to +5  
        #   - Irrelevant: -5 to 0
        #   - Very irrelevant: -10 to -5
        #
        # We use calibrated bounds based on empirical observation:
        MIN_CE_SCORE = -5.0  # Floor for irrelevant content
        MAX_CE_SCORE = 8.0   # Ceiling for highly relevant content
        
        # Clip to bounds and normalize to [0, 1]
        clipped_score = np.clip(best_score, MIN_CE_SCORE, MAX_CE_SCORE)
        normalized_score = (clipped_score - MIN_CE_SCORE) / (MAX_CE_SCORE - MIN_CE_SCORE)

        return float(normalized_score), evidence_chunks[best_idx]

    def _compute_evidence_coverage(
        self,
        answer: str,
        evidence_chunks: List[str]
    ) -> float:
        """
        Compute how much of the evidence is covered by the answer.
        Prevents hallucination by addition (answer containing info not in evidence).
        """
        if not evidence_chunks:
            return 0.0

        # Embed answer and evidence (cached)
        answer_emb = self._encode_cached([answer])
        evidence_embs = self._encode_cached(evidence_chunks)

        # Compute max similarity to any evidence chunk
        similarities = cosine_similarity(answer_emb, evidence_embs)[0]

        # Average of max similarities gives coverage
        return float(np.mean(similarities))

    def evaluate(
        self,
        question: str,
        answer: str,
        evidence_chunks: List[str],
        debug: bool = False
    ) -> SEAEResult:
        """
        Evaluate answer alignment with evidence.

        Args:
            question: The original question
            answer: The generated answer
            evidence_chunks: Retrieved evidence chunks
            debug: Enable detailed debug output

        Returns:
            SEAEResult with alignment analysis
        """
        if debug:
            print("\n" + "="*60)
            print("[DEBUG SEAE] Starting alignment evaluation")
            print(f"[DEBUG SEAE] Answer length: {len(answer)}")
            print(f"[DEBUG SEAE] Evidence chunks: {len(evidence_chunks)}")
            print("="*60)
        
        if not answer or not evidence_chunks:
            if debug:
                print("[DEBUG SEAE] No answer or evidence provided!")
            return SEAEResult(
                is_aligned=False,
                alignment_score=0.0,
                claim_scores=[],
                misaligned_claims=[],
                evidence_coverage=0.0,
                explanation="No answer or evidence provided"
            )

        # SPECIAL CASE: Decision-only answer (e.g., just "Final Answer: yes")
        # For yes/no/maybe questions, if the model gives only a decision without 
        # explanation, we trust the decision based on evidence coverage alone
        if self._is_decision_only_answer(answer):
            evidence_coverage = self._compute_evidence_coverage(answer, evidence_chunks)
            if debug:
                print(f"[DEBUG SEAE] Decision-only answer detected!")
                print(f"[DEBUG SEAE] Trusting based on evidence coverage: {evidence_coverage:.3f}")
            
            # For decision-only answers, pass if evidence coverage is reasonable
            # This allows the model to answer yes/no/maybe without lengthy explanation
            is_aligned = evidence_coverage >= 0.3  # Lower threshold for decision-only
            
            return SEAEResult(
                is_aligned=is_aligned,
                alignment_score=evidence_coverage,  # Use coverage as proxy for alignment
                claim_scores=[],
                misaligned_claims=[],
                evidence_coverage=evidence_coverage,
                explanation=f"Decision-only answer. Evidence coverage: {evidence_coverage:.2f}"
            )

        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if debug:
            print(f"[DEBUG SEAE] Extracted {len(claims)} claims from answer:")
            for i, claim in enumerate(claims[:5]):
                print(f"  [{i+1}] {claim[:80]}...")

        # If no substantive claims extracted, fall back to evidence coverage
        if not claims:
            evidence_coverage = self._compute_evidence_coverage(answer, evidence_chunks)
            if debug:
                print(f"[DEBUG SEAE] No claims extracted, using evidence coverage: {evidence_coverage:.3f}")
            
            # Pass if evidence coverage is reasonable
            is_aligned = evidence_coverage >= 0.4
            
            return SEAEResult(
                is_aligned=is_aligned,
                alignment_score=evidence_coverage,
                claim_scores=[],
                misaligned_claims=[],
                evidence_coverage=evidence_coverage,
                explanation=f"No extractable claims. Evidence coverage: {evidence_coverage:.2f}"
            )

        # Evaluate each claim
        claim_scores = []
        misaligned_claims = []

        for claim in claims:
            score, best_evidence = self._compute_claim_evidence_alignment(claim, evidence_chunks)
            claim_scores.append((claim, score))

            if score < self.threshold:
                misaligned_claims.append(claim)
            
            if debug:
                status = "✓" if score >= self.threshold else "✗"
                print(f"[DEBUG SEAE] Claim score {score:.3f} {status}: {claim[:60]}...")

        # Compute overall alignment score
        if claim_scores:
            alignment_score = np.mean([s for _, s in claim_scores])
        else:
            alignment_score = 0.0

        # Compute evidence coverage
        evidence_coverage = self._compute_evidence_coverage(answer, evidence_chunks)

        # Determine if answer passes
        # A single low-scoring claim shouldn't fail an otherwise good answer
        is_aligned = alignment_score >= self.threshold
        
        if debug:
            print("-"*60)
            print(f"[DEBUG SEAE] RESULTS:")
            print(f"[DEBUG SEAE]   Alignment Score: {alignment_score:.3f} (threshold: {self.threshold})")
            print(f"[DEBUG SEAE]   Evidence Coverage: {evidence_coverage:.3f}")
            print(f"[DEBUG SEAE]   Is Aligned: {is_aligned}")
            print(f"[DEBUG SEAE]   Misaligned Claims: {len(misaligned_claims)}")
            print("="*60 + "\n")

        # Generate explanation
        if is_aligned:
            explanation = (
                f"Answer is well-aligned with evidence. "
                f"Alignment score: {alignment_score:.2f}, "
                f"Evidence coverage: {evidence_coverage:.2f}"
            )
        else:
            explanation = (
                f"Answer may contain hallucinations. "
                f"Alignment score: {alignment_score:.2f} (threshold: {self.threshold}). "
                f"Found {len(misaligned_claims)} potentially unsupported claims."
            )

        return SEAEResult(
            is_aligned=is_aligned,
            alignment_score=alignment_score,
            claim_scores=claim_scores,
            misaligned_claims=misaligned_claims,
            evidence_coverage=evidence_coverage,
            explanation=explanation
        )

    def get_feedback_for_correction(self, result: SEAEResult) -> str:
        """
        Generate detailed feedback for the DISC module to correct hallucinations.
        Provides claim-by-claim analysis with specific improvement suggestions.
        """
        if result.is_aligned:
            return "The answer is well-supported by evidence. No corrections needed."

        feedback_parts = [
            "=" * 50,
            "HALLUCINATION AUDIT REPORT",
            "=" * 50,
            f"Overall Alignment Score: {result.alignment_score:.2f} (Required: {self.threshold})",
            f"Evidence Coverage: {result.evidence_coverage:.2f}",
            f"Total Claims Analyzed: {len(result.claim_scores)}",
            f"Unsupported Claims Found: {len(result.misaligned_claims)}",
            "",
            "-" * 50,
            "CLAIM-BY-CLAIM ANALYSIS:",
            "-" * 50,
        ]

        # Show all claims with their status
        for i, (claim, score) in enumerate(result.claim_scores, 1):
            status = "✓ SUPPORTED" if score >= self.threshold else "✗ UNSUPPORTED"
            truncated_claim = claim[:100] + "..." if len(claim) > 100 else claim
            feedback_parts.append(f"\n{i}. [{status}] (score: {score:.2f})")
            feedback_parts.append(f"   Claim: \"{truncated_claim}\"")

            if score < self.threshold:
                # Provide specific guidance for unsupported claims
                if score < 0.3:
                    feedback_parts.append("   Action: REMOVE this claim - no evidence support")
                elif score < self.threshold:
                    feedback_parts.append("   Action: MODIFY to match evidence more closely")

        feedback_parts.extend([
            "",
            "-" * 50,
            "CORRECTION INSTRUCTIONS:",
            "-" * 50,
            "1. REMOVE claims marked with '✗ UNSUPPORTED' if evidence score < 0.3",
            "2. MODIFY claims with scores 0.3-0.6 to better match the evidence",
            "3. KEEP claims marked '✓ SUPPORTED' but verify citations",
            "4. If information is not in evidence, state: 'Based on available evidence, this cannot be determined'",
            "5. Do NOT introduce any new information not present in the original evidence",
        ])

        return "\n".join(feedback_parts)


if __name__ == "__main__":
    # Test SEAE
    seae = SEAE()

    # Test case
    question = "What is the first-line treatment for hypertension?"
    answer = (
        "The first-line treatment for hypertension typically includes thiazide diuretics, "
        "ACE inhibitors, ARBs, or calcium channel blockers. "
        "Patients should also be advised to exercise daily and eat less salt."
    )
    evidence = [
        "First-line antihypertensive treatments include thiazide diuretics, "
        "angiotensin-converting enzyme (ACE) inhibitors, angiotensin receptor blockers (ARBs), "
        "and calcium channel blockers (CCBs).",
        "Lifestyle modifications such as reducing sodium intake and increasing physical activity "
        "are recommended as adjunct therapy for hypertension management."
    ]

    result = seae.evaluate(question, answer, evidence)
    print(f"Is aligned: {result.is_aligned}")
    print(f"Alignment score: {result.alignment_score:.4f}")
    print(f"Evidence coverage: {result.evidence_coverage:.4f}")
    print(f"Explanation: {result.explanation}")

    if not result.is_aligned:
        print("\n--- Feedback for correction ---")
        print(seae.get_feedback_for_correction(result))
