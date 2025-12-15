"""
Chain-of-Thought (CoT) Medical Reasoning Module

Inspired by Med-PaLM 2's Ensemble Refinement approach:
1. Generate multiple diverse reasoning paths
2. Use self-consistency voting across paths  
3. Refine final answer using ensemble of reasoning

Reference: "Towards Expert-Level Medical Question Answering with Large Language Models"
(Singhal et al., 2023) - arXiv:2305.09617
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import re

from mega_rag.config import (
    COT_NUM_REASONING_PATHS,
    COT_SELF_CONSISTENCY,
    COT_ENSEMBLE_REFINEMENT,
    ENABLE_CHAIN_OF_THOUGHT
)


@dataclass
class ReasoningPath:
    """Represents a single reasoning path."""
    reasoning_steps: List[str]
    intermediate_conclusions: List[str]
    final_answer: str
    confidence: float
    decision: str  # yes/no/maybe for PubMedQA


@dataclass
class CoTResult:
    """Result from Chain-of-Thought reasoning."""
    question: str
    reasoning_paths: List[ReasoningPath]
    consensus_decision: str
    consensus_confidence: float
    refined_answer: str
    used_ensemble: bool


class MedicalChainOfThought:
    """
    Medical Chain-of-Thought Reasoning with Ensemble Refinement.
    
    Implements Med-PaLM 2 inspired techniques:
    1. Step-by-step medical reasoning prompts
    2. Multiple diverse reasoning paths (self-consistency)
    3. Ensemble refinement for final answer
    
    Key insight from Med-PaLM 2:
    "Ensemble refinement combines multiple chain-of-thought reasoning paths
    and uses a separate refinement step to synthesize the final answer."
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        self.num_paths = COT_NUM_REASONING_PATHS
        self.use_self_consistency = COT_SELF_CONSISTENCY
        self.use_ensemble_refinement = COT_ENSEMBLE_REFINEMENT
        
    def set_llm(self, llm):
        """Set the LLM for generation."""
        self.llm = llm
    
    def detect_complexity(self, question: str) -> Dict:
        """
        Public method to detect question complexity for workflow integration.
        
        Returns:
            Dict with 'is_complex', 'category', and 'indicators'
        """
        is_complex = self._is_complex_question(question)
        
        # Determine category
        question_lower = question.lower()
        if 'compare' in question_lower or 'versus' in question_lower or 'vs' in question_lower:
            category = 'comparative'
        elif 'mechanism' in question_lower or 'how does' in question_lower:
            category = 'mechanistic'
        elif 'and' in question_lower and ('treatment' in question_lower or 'therapy' in question_lower):
            category = 'multi-factor'
        elif 'risk' in question_lower or 'benefit' in question_lower:
            category = 'risk-benefit'
        else:
            category = 'multi-step'
        
        return {
            'is_complex': is_complex,
            'category': category,
            'question': question
        }
    
    def reason_with_evidence(
        self,
        question: str,
        evidence_chunks: List[str],
        source_metadata: Optional[List[dict]] = None
    ) -> Dict:
        """
        Public method for workflow integration - applies CoT reasoning.
        
        Returns:
            Dict with reasoning results including final_answer, reasoning_steps, etc.
        """
        # Call the main reason method
        result = self.reason(question, evidence_chunks, force_cot=True)
        
        # Convert to dict format expected by workflow
        reasoning_steps = []
        for path in result.reasoning_paths:
            reasoning_steps.extend(path.reasoning_steps)
        
        return {
            'final_answer': result.refined_answer,
            'reasoning_steps': reasoning_steps,
            'num_paths': len(result.reasoning_paths),
            'consensus_reached': result.consensus_confidence > 0.6,
            'consensus_decision': result.consensus_decision,
            'consensus_confidence': result.consensus_confidence,
            'used_ensemble': result.used_ensemble,
            'paths': [
                {
                    'decision': p.decision,
                    'confidence': p.confidence,
                    'steps': p.reasoning_steps
                }
                for p in result.reasoning_paths
            ]
        }
        
    def _is_complex_question(self, question: str) -> bool:
        """
        Determine if a question requires Chain-of-Thought reasoning.
        
        Complex questions typically involve:
        - Multiple conditions/factors
        - Comparative analysis
        - Multi-step reasoning
        - Mechanism understanding
        """
        question_lower = question.lower()
        
        # HIGH-WEIGHT indicators (1 match = complex)
        high_weight_indicators = [
            # Comparative questions - always complex
            r'\b(compare|comparison|versus|vs\.?)\b',
            r'\b(better|worse|superior|inferior)\s+(than|to)\b',
            # Multi-condition treatment
            r'\b(with|and)\s+\w+\s+(with|and)\s+\w+',  # "hypertension with diabetes and..."
            # Mechanism questions
            r'\b(mechanism|pathophysiology|pathway)\b',
            # Drug interactions
            r'\b(interact|interaction|contraindication)\b',
            # Risk-benefit analysis
            r'\b(risk|risks)\s+(and|or)\s+(benefit|benefits)\b',
            r'\b(benefit|benefits)\s+(and|or)\s+(risk|risks)\b',
            r'\b(advantages?\s+and\s+disadvantages?)\b',
        ]
        
        # Check high-weight first
        for pattern in high_weight_indicators:
            if re.search(pattern, question_lower):
                return True
        
        # LOW-WEIGHT indicators (need 2+ matches)
        low_weight_indicators = [
            # Multi-factor questions
            r'\b(and|with|combined|alongside|together)\b.*\b(treatment|therapy|effect)',
            # Conditional questions  
            r'\b(if|when|given|assuming|in case of)\b',
            # Risk/benefit (individual mentions)
            r'\b(risk|benefit|advantage|disadvantage|trade-?off)\b',
            # Diagnosis questions with multiple symptoms
            r'\b(differential|diagnosis|present with|symptoms)\b',
            # How/why questions
            r'\b(how|why)\s+(does|do|is|are|can)\b',
        ]
        
        matches = sum(1 for pattern in low_weight_indicators 
                     if re.search(pattern, question_lower))
        
        # Consider complex if 2+ low-weight indicators present
        return matches >= 2
    
    def _create_cot_prompt(
        self,
        question: str,
        evidence: List[str],
        path_index: int = 0
    ) -> str:
        """
        Create Chain-of-Thought prompt for medical reasoning.
        
        Inspired by Med-PaLM 2's prompting strategy:
        - Explicit step-by-step reasoning
        - Evidence integration at each step
        - Medical domain expertise framing
        """
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}]: {chunk}"
            for i, chunk in enumerate(evidence[:5])  # Limit to top 5
        ])
        
        # Vary temperature/approach for different paths
        approach_variants = [
            "systematic analysis",
            "differential diagnosis approach", 
            "evidence-based synthesis"
        ]
        approach = approach_variants[path_index % len(approach_variants)]
        
        prompt = f"""You are an expert medical professional. Analyze this medical question using {approach}.

QUESTION: {question}

RELEVANT EVIDENCE:
{evidence_text}

Think through this step-by-step:

STEP 1 - UNDERSTAND THE QUESTION:
What exactly is being asked? What are the key medical concepts involved?

STEP 2 - ANALYZE THE EVIDENCE:
What does each piece of evidence tell us? Are there any conflicts or gaps?

STEP 3 - MEDICAL REASONING:
Based on established medical knowledge and the evidence:
- What are the key factors to consider?
- What mechanisms or relationships are relevant?
- Are there any important caveats or conditions?

STEP 4 - SYNTHESIZE CONCLUSION:
Combining all the above, what is the most supported conclusion?

STEP 5 - FINAL ANSWER:
State your conclusion clearly. For yes/no questions, explicitly state "Final Answer: yes/no/maybe"

Now provide your complete analysis:"""

        return prompt
    
    def _extract_decision_from_path(self, reasoning: str) -> Tuple[str, float]:
        """Extract yes/no/maybe decision and confidence from a reasoning path."""
        reasoning_lower = reasoning.lower()
        
        # Look for explicit final answer
        patterns = [
            r'final\s+answer[:\s]+\b(yes|no|maybe)\b',
            r'conclusion[:\s]+\b(yes|no|maybe)\b',
            r'\b(yes|no|maybe)\b\s*[.!]?\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reasoning_lower)
            if match:
                decision = match.group(1)
                # Estimate confidence based on language
                if 'strongly' in reasoning_lower or 'clearly' in reasoning_lower:
                    confidence = 0.9
                elif 'likely' in reasoning_lower or 'probably' in reasoning_lower:
                    confidence = 0.7
                elif 'may' in reasoning_lower or 'possibly' in reasoning_lower:
                    confidence = 0.5
                else:
                    confidence = 0.6
                return decision, confidence
        
        # Fallback: count positive vs negative indicators
        yes_indicators = len(re.findall(r'\b(supports?|confirms?|effective|significant)\b', reasoning_lower))
        no_indicators = len(re.findall(r'\b(not|no|fails?|ineffective|insignificant)\b', reasoning_lower))
        
        if yes_indicators > no_indicators:
            return 'yes', 0.5
        elif no_indicators > yes_indicators:
            return 'no', 0.5
        else:
            return 'maybe', 0.4
    
    def _extract_reasoning_steps(self, reasoning: str) -> List[str]:
        """Extract individual reasoning steps from the response."""
        steps = []
        
        # Look for STEP markers
        step_pattern = r'STEP\s*\d+[:\s-]+([^\n]+(?:\n(?!STEP)[^\n]+)*)'
        matches = re.findall(step_pattern, reasoning, re.IGNORECASE)
        
        if matches:
            steps = [m.strip() for m in matches if m.strip()]
        else:
            # Fallback: split by numbered points or paragraphs
            lines = reasoning.split('\n')
            current_step = []
            for line in lines:
                if re.match(r'^\d+[.\)]\s+', line) or re.match(r'^[-•]\s+', line):
                    if current_step:
                        steps.append(' '.join(current_step))
                    current_step = [line]
                elif line.strip():
                    current_step.append(line.strip())
            if current_step:
                steps.append(' '.join(current_step))
        
        return steps[:5]  # Limit to 5 steps
    
    def generate_reasoning_paths(
        self,
        question: str,
        evidence: List[str]
    ) -> List[ReasoningPath]:
        """
        Generate multiple diverse reasoning paths.
        
        Each path approaches the problem from a slightly different angle
        to enable self-consistency voting.
        """
        if self.llm is None:
            raise ValueError("LLM not set. Call set_llm() first.")
        
        paths = []
        
        for i in range(self.num_paths):
            # Create prompt with variation
            prompt = self._create_cot_prompt(question, evidence, path_index=i)
            
            # Generate reasoning
            try:
                reasoning = self.llm.generate(prompt)
            except Exception as e:
                print(f"  CoT path {i+1} failed: {e}")
                continue
            
            # Extract components
            steps = self._extract_reasoning_steps(reasoning)
            decision, confidence = self._extract_decision_from_path(reasoning)
            
            path = ReasoningPath(
                reasoning_steps=steps,
                intermediate_conclusions=steps[-2:] if len(steps) >= 2 else steps,
                final_answer=reasoning,
                confidence=confidence,
                decision=decision
            )
            paths.append(path)
        
        return paths
    
    def _self_consistency_vote(self, paths: List[ReasoningPath]) -> Tuple[str, float]:
        """
        Apply self-consistency voting across reasoning paths.
        
        The most common answer across paths is selected as the consensus.
        Confidence is based on agreement level.
        """
        if not paths:
            return 'maybe', 0.0
        
        decisions = [p.decision for p in paths]
        decision_counts = Counter(decisions)
        
        # Get most common decision
        consensus_decision, count = decision_counts.most_common(1)[0]
        
        # Calculate confidence based on agreement
        agreement_ratio = count / len(paths)
        
        # Weight by individual path confidences
        matching_confidences = [p.confidence for p in paths if p.decision == consensus_decision]
        avg_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0.5
        
        # Final confidence combines agreement and individual confidences
        consensus_confidence = (agreement_ratio + avg_confidence) / 2
        
        return consensus_decision, consensus_confidence
    
    def _ensemble_refinement(
        self,
        question: str,
        evidence: List[str],
        paths: List[ReasoningPath],
        consensus_decision: str
    ) -> str:
        """
        Ensemble Refinement: Synthesize final answer from multiple reasoning paths.
        
        Inspired by Med-PaLM 2's ensemble refinement approach:
        "The model is prompted to synthesize the reasoning from multiple
        chain-of-thought outputs into a single refined response."
        """
        if self.llm is None:
            return paths[0].final_answer if paths else ""
        
        # Summarize each path's key reasoning
        path_summaries = []
        for i, path in enumerate(paths[:3]):  # Use top 3 paths
            key_points = path.reasoning_steps[-2:] if len(path.reasoning_steps) >= 2 else path.reasoning_steps
            summary = f"Path {i+1} ({path.decision}, confidence={path.confidence:.2f}): {' '.join(key_points)}"
            path_summaries.append(summary)
        
        evidence_text = "\n".join([f"- {e[:200]}..." for e in evidence[:3]])
        
        prompt = f"""You are synthesizing multiple medical reasoning analyses into a final expert response.

QUESTION: {question}

KEY EVIDENCE:
{evidence_text}

MULTIPLE REASONING PATHS:
{chr(10).join(path_summaries)}

CONSENSUS: The majority of reasoning paths conclude "{consensus_decision}"

TASK: Synthesize these analyses into a single, coherent, expert-level response that:
1. Incorporates the strongest reasoning from each path
2. Addresses any conflicts between paths
3. Provides clear evidence-based conclusions
4. Cites sources when possible (e.g., [Source: Document Name])
5. Ends with "Final Answer: {consensus_decision}" (or adjust if evidence strongly suggests otherwise)

REFINED RESPONSE:"""

        try:
            refined = self.llm.generate(prompt)
            return refined
        except Exception as e:
            print(f"  Ensemble refinement failed: {e}")
            # Fallback to best individual path
            best_path = max(paths, key=lambda p: p.confidence)
            return best_path.final_answer
    
    def reason(
        self,
        question: str,
        evidence: List[str],
        force_cot: bool = False
    ) -> CoTResult:
        """
        Apply Chain-of-Thought reasoning to a medical question.
        
        Args:
            question: The medical question to answer
            evidence: Retrieved evidence chunks
            force_cot: Force CoT even for simple questions
            
        Returns:
            CoTResult with reasoning paths and final answer
        """
        # Check if CoT is enabled and question is complex enough
        use_cot = force_cot or (ENABLE_CHAIN_OF_THOUGHT and self._is_complex_question(question))
        
        if not use_cot:
            # Return empty result - caller should use standard generation
            return CoTResult(
                question=question,
                reasoning_paths=[],
                consensus_decision='',
                consensus_confidence=0.0,
                refined_answer='',
                used_ensemble=False
            )
        
        # Generate multiple reasoning paths
        paths = self.generate_reasoning_paths(question, evidence)
        
        if not paths:
            return CoTResult(
                question=question,
                reasoning_paths=[],
                consensus_decision='maybe',
                consensus_confidence=0.0,
                refined_answer='Unable to generate reasoning paths.',
                used_ensemble=False
            )
        
        # Self-consistency voting
        if self.use_self_consistency and len(paths) > 1:
            consensus_decision, consensus_confidence = self._self_consistency_vote(paths)
        else:
            # Use first path if only one
            consensus_decision = paths[0].decision
            consensus_confidence = paths[0].confidence
        
        # Ensemble refinement
        if self.use_ensemble_refinement and len(paths) > 1:
            refined_answer = self._ensemble_refinement(
                question, evidence, paths, consensus_decision
            )
            used_ensemble = True
        else:
            refined_answer = paths[0].final_answer
            used_ensemble = False
        
        return CoTResult(
            question=question,
            reasoning_paths=paths,
            consensus_decision=consensus_decision,
            consensus_confidence=consensus_confidence,
            refined_answer=refined_answer,
            used_ensemble=used_ensemble
        )


# =============================================================================
# Integration helper for workflow
# =============================================================================

def should_use_cot(question: str) -> bool:
    """Quick check if question warrants Chain-of-Thought reasoning."""
    if not ENABLE_CHAIN_OF_THOUGHT:
        return False
    
    cot = MedicalChainOfThought()
    return cot._is_complex_question(question)


def format_reasoning_for_display(cot_result: CoTResult, verbose: bool = True) -> str:
    """
    Format Chain-of-Thought reasoning for user display.
    
    Shows the thinking process when CoT is enabled.
    """
    if not cot_result or not cot_result.reasoning_paths:
        return ""
    
    lines = []
    lines.append("\n" + "="*60)
    lines.append("🧠 CHAIN-OF-THOUGHT REASONING")
    lines.append("="*60)
    
    lines.append(f"\n📊 Generated {len(cot_result.reasoning_paths)} reasoning paths")
    lines.append(f"📊 Consensus: {cot_result.consensus_decision.upper()} ({cot_result.consensus_confidence:.0%} confidence)")
    lines.append(f"📊 Ensemble Refinement: {'Yes' if cot_result.used_ensemble else 'No'}")
    
    if verbose:
        for i, path in enumerate(cot_result.reasoning_paths, 1):
            lines.append(f"\n--- Reasoning Path {i} ({path.decision}, {path.confidence:.0%}) ---")
            for j, step in enumerate(path.reasoning_steps, 1):
                # Truncate long steps
                step_text = step[:200] + "..." if len(step) > 200 else step
                lines.append(f"  Step {j}: {step_text}")
    
    lines.append("\n" + "="*60)
    
    return "\n".join(lines)


# Alias for backward compatibility and cleaner naming
MedicalCoTReasoner = MedicalChainOfThought


if __name__ == "__main__":
    # Test complexity detection
    test_questions = [
        "What is the first-line treatment for hypertension?",  # Simple
        "How does ACE inhibitor combined with calcium channel blocker compare to monotherapy for hypertension in diabetic patients?",  # Complex
        "Does aspirin reduce cardiovascular risk?",  # Simple
        "What is the mechanism by which statins reduce LDL cholesterol and what are the risks of myopathy when combined with fibrates?",  # Complex
    ]
    
    cot = MedicalChainOfThought()
    
    for q in test_questions:
        is_complex = cot._is_complex_question(q)
        print(f"Complex: {is_complex} | {q[:80]}...")
