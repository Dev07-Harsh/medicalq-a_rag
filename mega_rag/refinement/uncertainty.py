"""
Uncertainty Quantification for MEGA-RAG

Combines retrieval, generation, and verification signals into calibrated confidence.
Supports abstention when confidence is too low for medical safety.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class UncertaintyResult:
    """Combined uncertainty assessment."""
    # Individual confidence signals (0-1, higher = more confident)
    retrieval_confidence: float = 0.0    # Based on retrieval scores
    generation_confidence: float = 0.0   # Based on self-consistency agreement
    alignment_confidence: float = 0.0    # Based on SEAE alignment score

    # Combined score
    overall_confidence: float = 0.0
    confidence_level: str = "LOW"        # LOW, MEDIUM, HIGH

    # Abstention decision
    should_abstain: bool = False
    abstention_reason: str = ""

    # Details
    signal_details: Dict[str, float] = field(default_factory=dict)


class UncertaintyQuantifier:
    """
    Quantifies answer uncertainty from multiple signals.

    Signals:
    1. Retrieval confidence: How relevant were the retrieved chunks?
       - Based on average similarity scores from retrieval
    2. Generation confidence: How consistent is the model?
       - Based on self-consistency voting agreement ratio
    3. Alignment confidence: How well-grounded is the answer?
       - Based on SEAE alignment score

    Abstention thresholds (configurable):
    - If overall confidence < 0.3 -> abstain
    - If retrieval confidence < 0.2 -> abstain (no good evidence)
    """

    def __init__(
        self,
        abstention_threshold: float = 0.3,
        min_retrieval_confidence: float = 0.2,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.abstention_threshold = abstention_threshold
        self.min_retrieval_confidence = min_retrieval_confidence
        # Weights for combining signals
        self.weights = weights or {
            "retrieval": 0.3,
            "generation": 0.3,
            "alignment": 0.4,
        }

    def compute_retrieval_confidence(
        self,
        retrieval_scores: List[float],
        top_k: int = 5,
    ) -> float:
        """Compute confidence from retrieval similarity scores."""
        if not retrieval_scores:
            return 0.0
        # Use top-k scores
        top_scores = sorted(retrieval_scores, reverse=True)[:top_k]
        avg_score = sum(top_scores) / len(top_scores)
        # Normalize: scores typically range 0.3-0.9
        confidence = max(0.0, min(1.0, (avg_score - 0.2) / 0.6))
        return round(confidence, 4)

    def compute_generation_confidence(
        self,
        vote_distribution: Optional[Dict[str, int]] = None,
        num_paths: int = 3,
    ) -> float:
        """Compute confidence from self-consistency voting."""
        if not vote_distribution:
            return 0.5  # No voting data, neutral confidence
        total_votes = sum(vote_distribution.values())
        if total_votes == 0:
            return 0.5
        max_votes = max(vote_distribution.values())
        agreement = max_votes / total_votes
        return round(agreement, 4)

    def compute_alignment_confidence(
        self,
        alignment_score: Optional[float] = None,
    ) -> float:
        """Compute confidence from SEAE alignment score."""
        if alignment_score is None:
            return 0.5  # No alignment data
        return round(max(0.0, min(1.0, alignment_score)), 4)

    def quantify(
        self,
        retrieval_scores: Optional[List[float]] = None,
        vote_distribution: Optional[Dict[str, int]] = None,
        alignment_score: Optional[float] = None,
        num_paths: int = 3,
    ) -> UncertaintyResult:
        """
        Compute overall uncertainty from all available signals.
        Returns UncertaintyResult with confidence scores and abstention decision.
        """
        # Compute individual signals
        retrieval_conf = self.compute_retrieval_confidence(retrieval_scores or [])
        generation_conf = self.compute_generation_confidence(vote_distribution, num_paths)
        alignment_conf = self.compute_alignment_confidence(alignment_score)

        # Weighted combination
        w = self.weights
        overall = (
            w["retrieval"] * retrieval_conf
            + w["generation"] * generation_conf
            + w["alignment"] * alignment_conf
        )
        overall = round(overall, 4)

        # Confidence level
        if overall >= 0.7:
            level = "HIGH"
        elif overall >= 0.4:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Abstention decision
        should_abstain = False
        abstention_reason = ""

        if overall < self.abstention_threshold:
            should_abstain = True
            abstention_reason = (
                f"Overall confidence too low ({overall:.2f} < {self.abstention_threshold})"
            )
        elif retrieval_conf < self.min_retrieval_confidence:
            should_abstain = True
            abstention_reason = (
                f"Insufficient retrieval evidence ({retrieval_conf:.2f} < {self.min_retrieval_confidence})"
            )

        return UncertaintyResult(
            retrieval_confidence=retrieval_conf,
            generation_confidence=generation_conf,
            alignment_confidence=alignment_conf,
            overall_confidence=overall,
            confidence_level=level,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            signal_details={
                "retrieval_weight": w["retrieval"],
                "generation_weight": w["generation"],
                "alignment_weight": w["alignment"],
                "retrieval_scores_count": len(retrieval_scores or []),
                "has_voting": vote_distribution is not None,
                "has_alignment": alignment_score is not None,
            },
        )
