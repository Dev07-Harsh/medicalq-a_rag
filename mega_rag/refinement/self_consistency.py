"""
Self-Consistency Voting Module for Hallucination Reduction

Implements the key insight from Self-RAG and Medprompt papers:
Generate multiple reasoning paths and use majority voting to reduce hallucination.

Reference: 
- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., 2023)
- "Can Generalist Foundation Models Outcompete Special-Purpose Tuning?" (Medprompt, 2023)
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import re


@dataclass
class VotingResult:
    """Result from self-consistency voting."""
    final_decision: str  # yes/no/maybe
    confidence: float  # 0-1, based on agreement
    num_paths: int
    vote_distribution: Dict[str, int]  # {"yes": 2, "no": 1, "maybe": 0}
    reasoning_paths: List[str]
    is_unanimous: bool
    

class SelfConsistencyVoter:
    """
    Self-Consistency Voting for robust decision-making.
    
    Key principle: If a model consistently arrives at the same answer
    through different reasoning paths, that answer is more likely correct.
    
    For yes/no/maybe questions:
    1. Generate N diverse reasoning paths (using temperature variation)
    2. Extract decision from each path
    3. Use majority voting for final decision
    4. Confidence = agreement ratio
    """
    
    def __init__(
        self,
        llm=None,
        num_paths: int = 3,
        temperature_variance: float = 0.2,
        min_agreement_threshold: float = 0.5
    ):
        self.llm = llm
        self.num_paths = num_paths
        self.temperature_variance = temperature_variance
        self.min_agreement_threshold = min_agreement_threshold
        
    def set_llm(self, llm):
        """Set the LLM instance."""
        self.llm = llm
        
    def _create_diverse_prompt(
        self,
        question: str,
        evidence_chunks: List[str],
        path_index: int
    ) -> str:
        """Create diverse prompts for different reasoning paths."""
        
        # Format evidence
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}]: {chunk}"
            for i, chunk in enumerate(evidence_chunks[:8])  # Limit evidence
        ])
        
        # Different reasoning approaches for diversity - all emphasize DECISIVENESS
        approaches = [
            {
                "focus": "evidence-weight",
                "instruction": """Weigh the evidence carefully:
- Count pieces of evidence that SUPPORT the claim
- Count pieces of evidence that CONTRADICT the claim
- If more support than contradict → YES
- If more contradict than support → NO
- Use MAYBE only if exactly equal or completely unrelated"""
            },
            {
                "focus": "direct-answer", 
                "instruction": """Answer directly based on what the evidence shows:
- Does ANY evidence directly answer this question? If yes, use that answer.
- If evidence shows positive results/effects → YES
- If evidence shows negative/null results → NO
- Avoid MAYBE - most research questions have a lean one way or another."""
            },
            {
                "focus": "conclusion-based",
                "instruction": """Look at the conclusions in the evidence:
- What do the study conclusions say?
- Do they support or refute the question's premise?
- A weak positive is still YES, a weak negative is still NO
- MAYBE is only for truly contradictory or missing evidence."""
            }
        ]
        
        approach = approaches[path_index % len(approaches)]
        
        prompt = f"""You are a medical research expert making a DEFINITIVE decision on this question.

{approach['instruction']}

QUESTION: {question}

EVIDENCE:
{evidence_text}

CRITICAL RULES FOR DECISION:
1. You MUST choose YES or NO if there is ANY lean in the evidence
2. YES = Evidence shows positive effect, support, or confirmation (even if weak)
3. NO = Evidence shows negative effect, no effect, or refutation (even if weak)  
4. MAYBE = ONLY if evidence is completely absent or perfectly contradictory
5. When in doubt between yes/no and maybe, choose yes or no based on the slight lean

Most medical research questions have a direction. Be DECISIVE.

Your response MUST end with exactly one of:
Final Answer: yes
Final Answer: no
Final Answer: maybe

Brief reasoning then final answer:"""

        return prompt
    
    def _extract_decision(self, response: str) -> Tuple[str, float]:
        """
        Extract yes/no/maybe decision from a response.
        Returns (decision, confidence).
        """
        response_lower = response.lower().strip()
        
        # Pattern 1: Look for "Final Answer: X" format (most reliable)
        final_answer_match = re.search(
            r'final\s*answer\s*[:\s]*\b(yes|no|maybe)\b',
            response_lower,
            re.IGNORECASE
        )
        
        if final_answer_match:
            decision = final_answer_match.group(1).lower()
            
            # Estimate confidence from response language
            if any(w in response_lower for w in ['clearly', 'strongly', 'definitely', 'certainly']):
                confidence = 0.95
            elif any(w in response_lower for w in ['likely', 'probably', 'suggests']):
                confidence = 0.75
            elif any(w in response_lower for w in ['may', 'might', 'possibly', 'could']):
                confidence = 0.55
            else:
                confidence = 0.7
                
            return decision, confidence
        
        # Pattern 2: Look at the last line for standalone yes/no/maybe
        lines = response_lower.strip().split('\n')
        last_lines = lines[-3:] if len(lines) >= 3 else lines
        
        for line in reversed(last_lines):
            line = line.strip()
            if re.match(r'^(yes|no|maybe)[.!?\s]*$', line):
                return line.rstrip('.!? '), 0.6
        
        # Pattern 3: Count evidence words and infer
        yes_indicators = len(re.findall(
            r'\b(support|confirm|show|demonstrate|indicate|effective|significant|positive)\b',
            response_lower
        ))
        no_indicators = len(re.findall(
            r'\b(not|no|fail|refute|contradict|ineffective|insignificant|negative)\b',
            response_lower
        ))
        
        # Make decision based on indicator counts
        if yes_indicators > no_indicators + 1:
            return 'yes', 0.5
        elif no_indicators > yes_indicators + 1:
            return 'no', 0.5
        else:
            return 'maybe', 0.4
    
    def vote(
        self,
        question: str,
        evidence_chunks: List[str],
        debug: bool = False
    ) -> VotingResult:
        """
        Generate multiple reasoning paths and vote on the answer.
        
        Args:
            question: The yes/no/maybe question
            evidence_chunks: Retrieved evidence
            debug: Enable debug output
            
        Returns:
            VotingResult with final decision and confidence
        """
        if not self.llm:
            raise ValueError("LLM not set. Call set_llm() first.")
        
        if debug:
            print(f"\n{'='*60}")
            print(f"[SELF-CONSISTENCY] Starting voting with {self.num_paths} paths")
            print(f"{'='*60}")
        
        decisions = []
        confidences = []
        reasoning_paths = []
        
        # Generate diverse reasoning paths
        for i in range(self.num_paths):
            prompt = self._create_diverse_prompt(question, evidence_chunks, i)
            
            try:
                # Generate response (some LLMs support temperature, some don't)
                response = self.llm.generate(prompt)
                reasoning_paths.append(response)
                
                # Extract decision
                decision, confidence = self._extract_decision(response)
                decisions.append(decision)
                confidences.append(confidence)
                
                if debug:
                    print(f"\n[PATH {i+1}] Decision: {decision} (conf: {confidence:.2f})")
                    # Show last 200 chars with the answer
                    print(f"[PATH {i+1}] Response snippet: ...{response[-200:]}")
                    
            except Exception as e:
                if debug:
                    print(f"[PATH {i+1}] ERROR: {e}")
                # Fallback to 'maybe' on error
                decisions.append('maybe')
                confidences.append(0.3)
                reasoning_paths.append(f"Error: {e}")
        
        # Count votes
        vote_counts = Counter(decisions)
        
        # Majority voting
        final_decision, majority_count = vote_counts.most_common(1)[0]
        
        # Calculate confidence based on agreement
        agreement_ratio = majority_count / len(decisions)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Combined confidence: agreement * average individual confidence
        combined_confidence = agreement_ratio * avg_confidence
        
        # Unanimous agreement bonus
        is_unanimous = majority_count == len(decisions)
        if is_unanimous:
            combined_confidence = min(1.0, combined_confidence * 1.2)
        
        if debug:
            print(f"\n{'='*60}")
            print(f"[SELF-CONSISTENCY] VOTING RESULTS:")
            print(f"  Vote distribution: {dict(vote_counts)}")
            print(f"  Final decision: {final_decision}")
            print(f"  Agreement: {agreement_ratio:.0%} ({majority_count}/{len(decisions)})")
            print(f"  Confidence: {combined_confidence:.2f}")
            print(f"  Unanimous: {is_unanimous}")
            print(f"{'='*60}\n")
        
        return VotingResult(
            final_decision=final_decision,
            confidence=combined_confidence,
            num_paths=len(decisions),
            vote_distribution=dict(vote_counts),
            reasoning_paths=reasoning_paths,
            is_unanimous=is_unanimous
        )
    
    def vote_with_explanations(
        self,
        question: str,
        evidence_chunks: List[str],
        debug: bool = False
    ) -> Tuple[str, str]:
        """
        Vote and return both the decision and a synthesized explanation.
        
        Returns:
            (final_answer_with_explanation, decision_only)
        """
        result = self.vote(question, evidence_chunks, debug=debug)
        
        # Create synthesized explanation based on majority reasoning
        explanation_parts = []
        for i, (path, decision) in enumerate(zip(result.reasoning_paths, 
                                                   [self._extract_decision(p)[0] for p in result.reasoning_paths])):
            if decision == result.final_decision:
                # Extract key reasoning from paths that match majority
                # Take first substantive sentence (skip empty lines)
                sentences = [s.strip() for s in path.split('.') if len(s.strip()) > 20]
                if sentences:
                    explanation_parts.append(sentences[0] + '.')
        
        # Combine unique explanations
        seen = set()
        unique_explanations = []
        for exp in explanation_parts[:2]:  # Max 2 explanation sentences
            exp_lower = exp.lower()
            if exp_lower not in seen:
                seen.add(exp_lower)
                unique_explanations.append(exp)
        
        explanation = " ".join(unique_explanations) if unique_explanations else "Based on the evidence provided."
        
        # Format final answer
        confidence_text = ""
        if result.is_unanimous:
            confidence_text = " (unanimous agreement)"
        elif result.confidence > 0.7:
            confidence_text = f" (high confidence: {result.confidence:.0%})"
        elif result.confidence < 0.5:
            confidence_text = f" (low confidence: {result.confidence:.0%})"
        
        final_answer = f"{explanation}{confidence_text}\n\nFinal Answer: {result.final_decision}"
        
        return final_answer, result.final_decision


# =============================================================================
# Utility function for workflow integration
# =============================================================================

def apply_self_consistency(
    llm,
    question: str,
    evidence_chunks: List[str],
    num_paths: int = 3,
    debug: bool = False
) -> Tuple[str, str, float]:
    """
    Convenience function to apply self-consistency voting.
    
    Args:
        llm: The LLM instance
        question: The question to answer
        evidence_chunks: Retrieved evidence
        num_paths: Number of reasoning paths (default 3)
        debug: Enable debug output
        
    Returns:
        (full_answer, decision, confidence)
    """
    voter = SelfConsistencyVoter(llm=llm, num_paths=num_paths)
    result = voter.vote(question, evidence_chunks, debug=debug)
    
    # Build answer with explanation
    full_answer, decision = voter.vote_with_explanations(
        question, evidence_chunks, debug=False
    )
    
    return full_answer, decision, result.confidence


if __name__ == "__main__":
    # Test example
    print("Self-Consistency Voter Module")
    print("Use apply_self_consistency() for easy integration")
