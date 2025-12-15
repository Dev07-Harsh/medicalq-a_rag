"""
DISC: Discrepancy-Identified Self-Clarification
Self-correction module that refines answers based on SEAE feedback.
"""
from typing import List, Optional, Callable
from dataclasses import dataclass

from mega_rag.config import DISC_MAX_CORRECTIONS
from mega_rag.refinement.seae import SEAEResult


@dataclass
class DISCResult:
    """Result from DISC correction."""
    original_answer: str
    corrected_answer: str
    iterations: int
    was_corrected: bool
    correction_history: List[str]
    final_alignment_score: float


class DISC:
    """
    Discrepancy-Identified Self-Clarification (DISC)

    Uses feedback from SEAE to iteratively refine answers until they
    are properly grounded in evidence.

    Works by:
    1. Receiving feedback about misaligned claims
    2. Prompting the LLM to correct specific issues
    3. Re-evaluating until alignment threshold is met or max iterations reached
    """

    def __init__(
        self,
        max_corrections: int = DISC_MAX_CORRECTIONS
    ):
        self.max_corrections = max_corrections

    def _is_yes_no_question(self, question: str) -> bool:
        """
        Detect if question expects a yes/no/maybe answer.
        
        Handles multiple PubMedQA formats:
        1. Direct questions: "Does X cause Y?"
        2. Colon format: "Topic description: is it effective?"
        3. Embedded questions: "Study of X: can it improve Y?"
        """
        import re
        question_lower = question.lower().strip()
        
        # Pattern 1: Questions starting with yes/no words
        yes_no_starters = [
            'does ', 'do ', 'is ', 'are ', 'can ', 'will ', 'would ', 'should ',
            'could ', 'has ', 'have ', 'was ', 'were ', 'did '
        ]
        if any(question_lower.startswith(starter) for starter in yes_no_starters):
            return True
        
        # Pattern 2: "Topic: is/does/can X?" format (common in PubMedQA)
        colon_pattern = r':\s*(is|does|do|are|can|will|would|should|could|has|have|was|were|did)\s+'
        if re.search(colon_pattern, question_lower):
            return True
        
        # Pattern 3: Questions ending with "?" that contain yes/no indicators
        if question_lower.endswith('?'):
            yes_no_words = ['is it', 'does it', 'can it', 'are they', 'do they', 
                           'should it', 'will it', 'could it', 'has it', 'have they']
            if any(word in question_lower for word in yes_no_words):
                return True
        
        return False

    def create_correction_prompt(
        self,
        question: str,
        original_answer: str,
        evidence_chunks: List[str],
        seae_feedback: str,
        misaligned_claims: List[str]
    ) -> str:
        """
        Create a simplified prompt for the LLM to correct the answer.
        Designed for better compatibility with local models.
        """
        evidence_text = "\n\n".join([
            f"[EVIDENCE {i+1}]: {chunk}"
            for i, chunk in enumerate(evidence_chunks)
        ])

        # Format misaligned claims simply
        claims_to_fix = ""
        if misaligned_claims:
            claims_to_fix = "\n".join([
                f"- {claim[:100]}..." if len(claim) > 100 else f"- {claim}"
                for claim in misaligned_claims[:3]  # Limit to top 3
            ])
        else:
            claims_to_fix = "None"

        # Add yes/no/maybe format requirement for yes/no questions
        is_yes_no = self._is_yes_no_question(question)
        
        if is_yes_no:
            format_instruction = """
FORMAT: This is a yes/no/maybe question. Your corrected answer MUST:
1. Briefly explain reasoning using ONLY the evidence
2. End with EXACTLY: "Final Answer: yes" OR "Final Answer: no" OR "Final Answer: maybe"
"""
        else:
            format_instruction = ""

        prompt = f"""Fix this answer to match the evidence.

QUESTION: {question}

ORIGINAL ANSWER:
{original_answer}

EVIDENCE:
{evidence_text}

CLAIMS TO FIX OR REMOVE:
{claims_to_fix}

RULES:
1. Use ONLY information from the evidence
2. Remove claims not supported by evidence
3. Cite sources: [Source: Evidence N]
4. Make a best-effort judgment (yes/no) if evidence leans one way
{format_instruction}
Write the corrected answer:"""

        return prompt

    def correct(
        self,
        question: str,
        answer: str,
        evidence_chunks: List[str],
        seae_result: SEAEResult,
        llm_call: Callable[[str], str],
        seae_evaluate: Callable[[str, str, List[str]], SEAEResult]
    ) -> DISCResult:
        """
        Iteratively correct the answer until it passes SEAE evaluation.

        Args:
            question: Original question
            answer: Initial answer to correct
            evidence_chunks: Retrieved evidence
            seae_result: Initial SEAE evaluation result
            llm_call: Function to call LLM with a prompt
            seae_evaluate: Function to evaluate answer with SEAE

        Returns:
            DISCResult with correction details
        """
        if seae_result.is_aligned:
            return DISCResult(
                original_answer=answer,
                corrected_answer=answer,
                iterations=0,
                was_corrected=False,
                correction_history=[],
                final_alignment_score=seae_result.alignment_score
            )

        current_answer = answer
        current_result = seae_result
        correction_history = [answer]

        for iteration in range(self.max_corrections):
            # Generate feedback for correction
            feedback = self._generate_detailed_feedback(current_result)

            # Create correction prompt
            prompt = self.create_correction_prompt(
                question=question,
                original_answer=current_answer,
                evidence_chunks=evidence_chunks,
                seae_feedback=feedback,
                misaligned_claims=current_result.misaligned_claims
            )

            # Get corrected answer from LLM
            corrected_answer = llm_call(prompt)

            # Clean up the response
            corrected_answer = self._clean_response(corrected_answer)
            correction_history.append(corrected_answer)

            # Re-evaluate with SEAE
            new_result = seae_evaluate(question, corrected_answer, evidence_chunks)

            # Check if improvement achieved
            if new_result.is_aligned:
                return DISCResult(
                    original_answer=answer,
                    corrected_answer=corrected_answer,
                    iterations=iteration + 1,
                    was_corrected=True,
                    correction_history=correction_history,
                    final_alignment_score=new_result.alignment_score
                )

            # Check if score improved
            if new_result.alignment_score <= current_result.alignment_score:
                # No improvement, stop iterating
                break

            current_answer = corrected_answer
            current_result = new_result

        # Return best result achieved
        return DISCResult(
            original_answer=answer,
            corrected_answer=current_answer,
            iterations=len(correction_history) - 1,
            was_corrected=current_answer != answer,
            correction_history=correction_history,
            final_alignment_score=current_result.alignment_score
        )

    def _generate_detailed_feedback(self, seae_result: SEAEResult) -> str:
        """Generate concise feedback for LLM correction."""
        feedback = f"Score: {seae_result.alignment_score:.2f}/0.60 required"

        if seae_result.misaligned_claims:
            feedback += f"\nUnsupported claims: {len(seae_result.misaligned_claims)}"

        return feedback

    def _clean_response(self, response: str) -> str:
        """Clean up LLM response to extract just the answer."""
        response = response.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "CORRECTED ANSWER:",
            "Corrected Answer:",
            "Here is the corrected answer:",
            "Here's the corrected answer:",
        ]

        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return response


class RefinementLoop:
    """
    Complete refinement loop combining SEAE and DISC.
    """

    def __init__(
        self,
        seae_module,
        disc_module: DISC,
        max_iterations: int = 3
    ):
        self.seae = seae_module
        self.disc = disc_module
        self.max_iterations = max_iterations

    def refine(
        self,
        question: str,
        initial_answer: str,
        evidence_chunks: List[str],
        llm_call: Callable[[str], str]
    ) -> DISCResult:
        """
        Run the complete refinement loop.
        """
        # Initial SEAE evaluation
        seae_result = self.seae.evaluate(question, initial_answer, evidence_chunks)

        # If already aligned, return as-is
        if seae_result.is_aligned:
            return DISCResult(
                original_answer=initial_answer,
                corrected_answer=initial_answer,
                iterations=0,
                was_corrected=False,
                correction_history=[initial_answer],
                final_alignment_score=seae_result.alignment_score
            )

        # Run DISC correction
        return self.disc.correct(
            question=question,
            answer=initial_answer,
            evidence_chunks=evidence_chunks,
            seae_result=seae_result,
            llm_call=llm_call,
            seae_evaluate=self.seae.evaluate
        )


if __name__ == "__main__":
    # Test DISC (requires LLM integration)
    print("DISC module loaded. Requires LLM integration to test.")
    print("Use with RefinementLoop in the main workflow.")
