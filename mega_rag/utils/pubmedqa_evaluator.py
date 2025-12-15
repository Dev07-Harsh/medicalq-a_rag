"""
PubMedQA Evaluation Module for MEGA-RAG
Evaluates yes/no/maybe classification accuracy and answer quality.
"""
import re
import json
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from mega_rag.config import BASE_DIR, DEFAULT_EVAL_SAMPLES
from mega_rag.utils.data_loader import DatasetManager, QASample


@dataclass
class PubMedQAResult:
    """Result from a single PubMedQA evaluation."""
    pubid: str
    question: str
    generated_answer: str
    predicted_decision: str  # yes/no/maybe
    ground_truth_decision: str  # yes/no/maybe
    ground_truth_long_answer: str
    is_correct: bool
    faithfulness: float
    answer_similarity: float
    retrieval_contexts: List[str] = field(default_factory=list)
    workflow_iterations: int = 0


@dataclass
class PubMedQAMetrics:
    """Aggregated metrics for PubMedQA evaluation."""
    total_samples: int
    accuracy: float  # Exact match on yes/no/maybe
    macro_f1: float  # F1 averaged across classes
    per_class_accuracy: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: Dict[str, Dict[str, int]]
    avg_faithfulness: float
    avg_answer_similarity: float
    hallucination_rate: float
    avg_iterations: float


class PubMedQAEvaluator:
    """
    Evaluator for PubMedQA dataset.
    Measures classification accuracy (yes/no/maybe) and answer quality.
    """

    def __init__(self, workflow=None):
        self.workflow = workflow
        self.data_manager = DatasetManager()
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy load embedding model for similarity computation."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        return self._embedding_model

    def _llm_extract_decision(self, answer: str) -> str:
        """
        Use LLM to extract yes/no/maybe decision when heuristics fail.
        More accurate but slower than pattern matching.
        """
        try:
            from mega_rag.core.llm import create_llm
            llm = create_llm()
            
            prompt = f"""Analyze the following medical answer and determine if the conclusion is YES, NO, or MAYBE.

ANSWER:
{answer[:1000]}

Based on the answer above, what is the final conclusion?
- Reply YES if the answer affirms the question/hypothesis
- Reply NO if the answer negates or rejects the question/hypothesis  
- Reply MAYBE if the answer is uncertain, inconclusive, or mixed

Reply with ONLY one word: YES, NO, or MAYBE"""

            response = llm.generate(prompt).strip().upper()
            
            if 'YES' in response:
                return 'yes'
            elif 'NO' in response:
                return 'no'
            elif 'MAYBE' in response:
                return 'maybe'
            else:
                return 'maybe'  # Default to uncertain if LLM response unclear
                
        except Exception as e:
            print(f"  LLM decision extraction failed: {e}")
            return 'maybe'  # Default to uncertain on error

    def extract_decision(self, answer: str, use_llm_fallback: bool = True, debug: bool = False) -> str:
        """
        Extract yes/no/maybe decision from LLM generated answer.
        Uses multiple heuristics to infer the decision.
        Prioritizes explicit "Final Answer:" patterns.
        
        Args:
            answer: The generated answer text
            use_llm_fallback: If True, use LLM when heuristics fail (more accurate but slower)
            debug: If True, print detailed extraction steps
        
        Returns:
            'yes', 'no', or 'maybe'
        """
        if debug:
            print("\n" + "="*60)
            print("[DEBUG EXTRACT_DECISION] Starting decision extraction")
            print(f"[DEBUG EXTRACT_DECISION] Answer length: {len(answer)}")
            print(f"[DEBUG EXTRACT_DECISION] Answer preview: {answer[:300]}...")
            print(f"[DEBUG EXTRACT_DECISION] Answer end: ...{answer[-200:]}")
            print("-"*60)
        
        answer_lower = answer.lower()

        # Pattern 1: HIGH PRIORITY - Explicit "Final Answer: yes/no/maybe" (at end of answer)
        # Check the last 200 characters first for the final answer pattern
        last_part = answer_lower[-300:] if len(answer_lower) > 300 else answer_lower
        final_answer_patterns = [
            r'final\s*answer[:\s]*\b(yes|no|maybe)\b',
            r'answer[:\s]*\b(yes|no|maybe)\b\s*[.!]?\s*$',
            r'\b(yes|no|maybe)\b\s*[.!]?\s*$',  # Very end of answer
        ]

        for pattern in final_answer_patterns:
            match = re.search(pattern, last_part)
            if match:
                if debug:
                    print(f"[DEBUG EXTRACT_DECISION] ✓ Pattern 1 matched: '{pattern}' -> {match.group(1)}")
                    print("="*60 + "\n")
                return match.group(1)
        
        if debug:
            print("[DEBUG EXTRACT_DECISION] Pattern 1 (Final Answer at end): No match")

        # Pattern 2: Explicit patterns anywhere in the answer
        explicit_patterns = [
            r'final\s+answer[:\s]+\b(yes|no|maybe)\b',
            r'conclusion[:\s]+\b(yes|no|maybe)\b',
            r'the\s+answer\s+is[:\s]+\b(yes|no|maybe)\b',
            r'in\s+conclusion[,:\s]+\b(yes|no|maybe)\b',
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                if debug:
                    print(f"[DEBUG EXTRACT_DECISION] ✓ Pattern 2 matched: '{pattern}' -> {match.group(1)}")
                    print("="*60 + "\n")
                return match.group(1)
        
        if debug:
            print("[DEBUG EXTRACT_DECISION] Pattern 2 (Explicit patterns): No match")

        # Pattern 3: Strong affirmative/negative indicators
        strong_yes = [
            r'\byes\b[,.]?\s*(the|this|it|there)',
            r'^yes\b',  # Answer starts with yes
            r'evidence\s+(strongly\s+)?(supports?|confirms?|shows?)',
            r'results?\s+(strongly\s+)?(support|confirm|show|demonstrate|indicate)',
            r'significantly\s+(increased?|improved?|higher|greater|reduced|decreased)',
            r'(is|are|was|were)\s+(highly\s+)?effective',
            r'positive\s+(correlation|association|effect|results?)',
            r'statistically\s+significant',
            r'does\s+(indeed\s+)?',  # "does indeed" or "does ameliorate"
        ]

        strong_no = [
            r'\bno\b[,.]?\s*(the|this|it|there)',
            r'^no\b',  # Answer starts with no
            r'evidence\s+(does\s+not|doesn\'t)\s+support',
            r'no\s+significant\s+(difference|effect|association|improvement)',
            r'(is|are|was|were)\s+not\s+(significantly\s+)?effective',
            r'failed\s+to\s+(show|demonstrate|find)',
            r'did\s+not\s+(show|demonstrate|find|improve|reduce)',
            r'no\s+(evidence|indication|support)',
            r'not\s+(associated|correlated|effective)',
        ]

        strong_maybe = [
            r'\bmaybe\b',
            r'inconclusive',
            r'mixed\s+(results?|evidence|findings)',
            r'conflicting\s+(results?|evidence)',
            r'insufficient\s+evidence',
        ]

        # Weak maybe indicators (lower weight)
        weak_maybe = [
            r'further\s+(research|study|investigation)\s+(is\s+)?needed',
            r'limited\s+evidence',
            r'\bunclear\b',
            r'\buncertain\b',
        ]

        # Count matches for each category
        yes_count = sum(2 for p in strong_yes if re.search(p, answer_lower))
        no_count = sum(2 for p in strong_no if re.search(p, answer_lower))
        maybe_count = sum(2 for p in strong_maybe if re.search(p, answer_lower))
        maybe_count += sum(1 for p in weak_maybe if re.search(p, answer_lower))

        # Pattern 4: First word check (high weight)
        first_words = answer_lower.split()[:5]
        if first_words:
            if first_words[0] in ['yes', 'yes,', 'yes.']:
                yes_count += 3
            elif first_words[0] in ['no', 'no,', 'no.']:
                no_count += 3
            elif first_words[0] in ['maybe', 'maybe,', 'perhaps', 'possibly']:
                maybe_count += 3

        # Pattern 5: Look for negation that might flip meaning
        # "does not" patterns increase no_count
        if re.search(r'does\s+not|did\s+not|do\s+not|cannot|can\'t', answer_lower):
            no_count += 1

        if debug:
            print(f"[DEBUG EXTRACT_DECISION] Pattern 3-5 scores: yes={yes_count}, no={no_count}, maybe={maybe_count}")

        # Determine decision based on counts
        # Only return "maybe" if it clearly dominates OR both yes/no are low
        if yes_count > no_count and yes_count >= maybe_count:
            if debug:
                print(f"[DEBUG EXTRACT_DECISION] ✓ Decision by score: 'yes' (yes={yes_count} > no={no_count})")
                print("="*60 + "\n")
            return 'yes'
        elif no_count > yes_count and no_count >= maybe_count:
            if debug:
                print(f"[DEBUG EXTRACT_DECISION] ✓ Decision by score: 'no' (no={no_count} > yes={yes_count})")
                print("="*60 + "\n")
            return 'no'
        elif maybe_count >= 2:  # Only return maybe if strong indicators
            if debug:
                print(f"[DEBUG EXTRACT_DECISION] ✓ Decision by score: 'maybe' (maybe={maybe_count} >= 2)")
                print("="*60 + "\n")
            return 'maybe'

        if debug:
            print("[DEBUG EXTRACT_DECISION] Score-based decision inconclusive, trying fallbacks...")

        # Fallback: Look at first 200 characters for any occurrence
        first_part = answer_lower[:200]
        if 'yes' in first_part and 'no' not in first_part:
            if debug:
                print("[DEBUG EXTRACT_DECISION] ✓ Fallback: 'yes' found in first 200 chars")
                print("="*60 + "\n")
            return 'yes'
        elif 'no' in first_part and 'yes' not in first_part:
            if debug:
                print("[DEBUG EXTRACT_DECISION] ✓ Fallback: 'no' found in first 200 chars")
                print("="*60 + "\n")
            return 'no'

        # Check if answer indicates uncertainty
        if 'cannot provide' in answer_lower or 'unable to' in answer_lower:
            if debug:
                print("[DEBUG EXTRACT_DECISION] ✓ Fallback: uncertainty phrase found -> 'maybe'")
                print("="*60 + "\n")
            return 'maybe'

        # Look at last sentence for hints
        sentences = answer_lower.split('.')
        if sentences:
            last_sentence = sentences[-2] if len(sentences) > 1 else sentences[-1]
            if 'yes' in last_sentence:
                if debug:
                    print("[DEBUG EXTRACT_DECISION] ✓ Fallback: 'yes' in last sentence")
                    print("="*60 + "\n")
                return 'yes'
            elif 'no' in last_sentence:
                if debug:
                    print("[DEBUG EXTRACT_DECISION] ✓ Fallback: 'no' in last sentence")
                    print("="*60 + "\n")
                return 'no'

        # =================================================================
        # LLM Fallback: Use LLM when heuristics fail
        # =================================================================
        # Instead of defaulting to 'yes' (which introduces bias), we use
        # the LLM to make a more informed decision when patterns are unclear
        if use_llm_fallback:
            if debug:
                print("[DEBUG EXTRACT_DECISION] All heuristics failed, using LLM fallback...")
            decision = self._llm_extract_decision(answer)
            if debug:
                print(f"[DEBUG EXTRACT_DECISION] LLM fallback returned: '{decision}'")
                print("="*60 + "\n")
            return decision
        
        # If LLM fallback disabled, default to 'maybe' (unbiased)
        if debug:
            print("[DEBUG EXTRACT_DECISION] ✗ All methods failed, defaulting to 'maybe'")
            print("="*60 + "\n")
        return 'maybe'

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not text1 or not text2:
            return 0.0

        embeddings = self.embedding_model.encode([text1, text2], normalize_embeddings=True)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Compute faithfulness score (how well answer aligns with context)."""
        if not contexts:
            return 0.0

        combined_context = " ".join(contexts)
        return self.compute_similarity(answer, combined_context)

    def evaluate_single(
        self,
        sample: QASample,
        verbose: bool = False,
        debug: bool = False
    ) -> PubMedQAResult:
        """Evaluate a single PubMedQA sample."""
        question = sample.question
        ground_truth_decision = getattr(sample, 'final_decision', 'maybe')
        ground_truth_long = getattr(sample, 'long_answer', '')
        # Support both pubid and pubmed_id attribute names
        pubid = getattr(sample, 'pubid', '') or getattr(sample, 'pubmed_id', '')
        original_contexts = getattr(sample, 'contexts', [])

        if debug:
            print("\n" + "#"*70)
            print(f"[DEBUG EVALUATE_SINGLE] Evaluating sample: {pubid}")
            print(f"[DEBUG EVALUATE_SINGLE] Question: {question[:150]}...")
            print(f"[DEBUG EVALUATE_SINGLE] Ground truth: {ground_truth_decision}")
            print("#"*70)

        # Run workflow
        try:
            result = self.workflow.run(question)
            generated_answer = result.get('answer', '')
            retrieval_contexts = result.get('context', [])
            iterations = result.get('iterations', 0)
            
            if debug:
                print(f"[DEBUG EVALUATE_SINGLE] Workflow completed, iterations={iterations}")
                print(f"[DEBUG EVALUATE_SINGLE] Answer length: {len(generated_answer)}")
                print(f"[DEBUG EVALUATE_SINGLE] Retrieved {len(retrieval_contexts)} context chunks")
                
        except Exception as e:
            if verbose or debug:
                print(f"Error running workflow: {e}")
            generated_answer = f"Error: {str(e)}"
            retrieval_contexts = []
            iterations = 0

        # Extract decision from generated answer
        predicted_decision = self.extract_decision(generated_answer, debug=debug)

        # Check correctness
        is_correct = predicted_decision == ground_truth_decision
        
        if debug:
            print(f"\n[DEBUG EVALUATE_SINGLE] ===== RESULT =====")
            print(f"[DEBUG EVALUATE_SINGLE] Ground Truth: {ground_truth_decision}")
            print(f"[DEBUG EVALUATE_SINGLE] Predicted:    {predicted_decision}")
            print(f"[DEBUG EVALUATE_SINGLE] Correct:      {'✓ YES' if is_correct else '✗ NO'}")
            print("#"*70 + "\n")

        # Compute metrics
        faithfulness = self.compute_faithfulness(generated_answer, retrieval_contexts)
        answer_similarity = self.compute_similarity(generated_answer, ground_truth_long)

        return PubMedQAResult(
            pubid=pubid,
            question=question,
            generated_answer=generated_answer,
            predicted_decision=predicted_decision,
            ground_truth_decision=ground_truth_decision,
            ground_truth_long_answer=ground_truth_long,
            is_correct=is_correct,
            faithfulness=faithfulness,
            answer_similarity=answer_similarity,
            retrieval_contexts=retrieval_contexts,
            workflow_iterations=iterations
        )

    def compute_f1(self, predictions: List[str], ground_truths: List[str], label: str) -> float:
        """Compute F1 score for a specific label."""
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def compute_metrics(self, results: List[PubMedQAResult]) -> PubMedQAMetrics:
        """Compute aggregated metrics from evaluation results."""
        if not results:
            return PubMedQAMetrics(
                total_samples=0, accuracy=0, macro_f1=0,
                per_class_accuracy={}, per_class_f1={},
                confusion_matrix={}, avg_faithfulness=0,
                avg_answer_similarity=0, hallucination_rate=0, avg_iterations=0
            )

        predictions = [r.predicted_decision for r in results]
        ground_truths = [r.ground_truth_decision for r in results]
        labels = ['yes', 'no', 'maybe']

        # Accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results)

        # Per-class accuracy and F1
        per_class_accuracy = {}
        per_class_f1 = {}
        for label in labels:
            label_samples = [r for r in results if r.ground_truth_decision == label]
            if label_samples:
                per_class_accuracy[label] = sum(1 for r in label_samples if r.is_correct) / len(label_samples)
            else:
                per_class_accuracy[label] = 0.0
            per_class_f1[label] = self.compute_f1(predictions, ground_truths, label)

        # Macro F1
        macro_f1 = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0

        # Confusion matrix
        confusion_matrix = {label: {l: 0 for l in labels} for label in labels}
        for r in results:
            gt = r.ground_truth_decision
            pred = r.predicted_decision
            if gt in confusion_matrix and pred in confusion_matrix[gt]:
                confusion_matrix[gt][pred] += 1

        # Average metrics
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_answer_similarity = sum(r.answer_similarity for r in results) / len(results)
        avg_iterations = sum(r.workflow_iterations for r in results) / len(results)

        # Hallucination rate (faithfulness < 0.5)
        hallucination_count = sum(1 for r in results if r.faithfulness < 0.5)
        hallucination_rate = hallucination_count / len(results)

        return PubMedQAMetrics(
            total_samples=len(results),
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class_accuracy=per_class_accuracy,
            per_class_f1=per_class_f1,
            confusion_matrix=confusion_matrix,
            avg_faithfulness=avg_faithfulness,
            avg_answer_similarity=avg_answer_similarity,
            hallucination_rate=hallucination_rate,
            avg_iterations=avg_iterations
        )

    def evaluate_dataset(
        self,
        dataset_name: str = 'pubmedqa_official_test',  # ⭐ Use official test split
        sample_size: int = DEFAULT_EVAL_SAMPLES,
        output_dir: Optional[Path] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run full evaluation on PubMedQA dataset.

        Args:
            dataset_name: Name of dataset to evaluate. Recommended: 'pubmedqa_official_test'
                          Available: pubmedqa_official_test (199 samples, expert-labeled)
                                    pubmedqa_official_dev (99 samples, for tuning)
                                    pubmedqa_csv_labeled (legacy)
            sample_size: Number of samples to evaluate (None = all samples in split)
            output_dir: Directory to save results
            verbose: Print progress details

        Returns:
            Dict with metrics, results, and report paths
        """
        if self.workflow is None:
            raise ValueError("Workflow not initialized. Pass workflow to constructor.")

        # Load dataset
        print(f"Loading {dataset_name} dataset (sample_size={sample_size})...")
        samples = self.data_manager.load_dataset(dataset_name, sample_size=sample_size)
        print(f"Loaded {len(samples)} samples")

        # Run evaluation
        results = []
        for sample in tqdm(samples, desc="Evaluating"):
            result = self.evaluate_single(sample, verbose=verbose)
            results.append(result)

            if verbose:
                status = "✓" if result.is_correct else "✗"
                print(f"  {status} Pred: {result.predicted_decision}, GT: {result.ground_truth_decision}")

        # Compute metrics
        metrics = self.compute_metrics(results)

        # Prepare output
        output = {
            'metrics': {
                'total_samples': metrics.total_samples,
                'accuracy': metrics.accuracy,
                'macro_f1': metrics.macro_f1,
                'per_class_accuracy': metrics.per_class_accuracy,
                'per_class_f1': metrics.per_class_f1,
                'confusion_matrix': metrics.confusion_matrix,
                'avg_faithfulness': metrics.avg_faithfulness,
                'avg_answer_similarity': metrics.avg_answer_similarity,
                'hallucination_rate': metrics.hallucination_rate,
                'avg_iterations': metrics.avg_iterations
            },
            'results': [
                {
                    'pubid': r.pubid,
                    'question': r.question,
                    'generated_answer': r.generated_answer[:500],
                    'predicted_decision': r.predicted_decision,
                    'ground_truth_decision': r.ground_truth_decision,
                    'is_correct': r.is_correct,
                    'faithfulness': r.faithfulness,
                    'answer_similarity': r.answer_similarity
                }
                for r in results
            ],
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'sample_size': sample_size
        }

        # Save outputs
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save JSON
            json_path = output_dir / f"pubmedqa_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to: {json_path}")

            # Generate HTML report
            html_path = output_dir / f"pubmedqa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            self._generate_html_report(output, html_path)
            print(f"HTML report saved to: {html_path}")

            output['json_path'] = str(json_path)
            output['html_path'] = str(html_path)

        return output

    def _generate_html_report(self, output: Dict, path: Path):
        """Generate HTML evaluation report."""
        metrics = output['metrics']
        results = output['results']

        # Create confusion matrix HTML
        labels = ['yes', 'no', 'maybe']
        cm = metrics['confusion_matrix']
        cm_rows = ""
        for gt in labels:
            cells = "".join(f"<td>{cm.get(gt, {}).get(pred, 0)}</td>" for pred in labels)
            cm_rows += f"<tr><th>{gt}</th>{cells}</tr>"

        # Create results table rows
        result_rows = ""
        for r in results[:50]:  # Limit to 50 for readability
            status = "correct" if r['is_correct'] else "incorrect"
            result_rows += f"""
            <tr class="{status}">
                <td>{r['pubid']}</td>
                <td>{r['question'][:100]}...</td>
                <td>{r['predicted_decision']}</td>
                <td>{r['ground_truth_decision']}</td>
                <td>{r['faithfulness']:.3f}</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PubMedQA Evaluation Report - MEGA-RAG</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .correct {{ background: #d4edda; }}
        .incorrect {{ background: #f8d7da; }}
        .confusion-matrix {{ width: auto; margin: 20px auto; }}
        .confusion-matrix th, .confusion-matrix td {{ text-align: center; width: 80px; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>PubMedQA Evaluation Report</h1>
        <p class="timestamp">Generated: {output['timestamp']} | Dataset: {output['dataset']} | Samples: {output['sample_size']}</p>

        <h2>Overall Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['macro_f1']:.3f}</div>
                <div class="metric-label">Macro F1</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['avg_faithfulness']:.3f}</div>
                <div class="metric-label">Avg Faithfulness</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['hallucination_rate']:.1%}</div>
                <div class="metric-label">Hallucination Rate</div>
            </div>
        </div>

        <h2>Per-Class Performance</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
            </tr>
            <tr>
                <td>Yes</td>
                <td>{metrics['per_class_accuracy'].get('yes', 0):.1%}</td>
                <td>{metrics['per_class_f1'].get('yes', 0):.3f}</td>
            </tr>
            <tr>
                <td>No</td>
                <td>{metrics['per_class_accuracy'].get('no', 0):.1%}</td>
                <td>{metrics['per_class_f1'].get('no', 0):.3f}</td>
            </tr>
            <tr>
                <td>Maybe</td>
                <td>{metrics['per_class_accuracy'].get('maybe', 0):.1%}</td>
                <td>{metrics['per_class_f1'].get('maybe', 0):.3f}</td>
            </tr>
        </table>

        <h2>Confusion Matrix</h2>
        <table class="confusion-matrix">
            <tr>
                <th></th>
                <th colspan="3">Predicted</th>
            </tr>
            <tr>
                <th>Actual</th>
                <th>Yes</th>
                <th>No</th>
                <th>Maybe</th>
            </tr>
            {cm_rows}
        </table>

        <h2>Sample Results (First 50)</h2>
        <table>
            <tr>
                <th>PubID</th>
                <th>Question</th>
                <th>Predicted</th>
                <th>Ground Truth</th>
                <th>Faithfulness</th>
            </tr>
            {result_rows}
        </table>
    </div>
</body>
</html>
"""
        with open(path, 'w') as f:
            f.write(html)

    def print_summary(self, metrics: PubMedQAMetrics):
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print("PUBMEDQA EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Samples: {metrics.total_samples}")
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:      {metrics.accuracy:.1%}")
        print(f"  Macro F1:      {metrics.macro_f1:.3f}")
        print(f"\nPer-Class Accuracy:")
        for label, acc in metrics.per_class_accuracy.items():
            print(f"  {label:8s}: {acc:.1%}")
        print(f"\nAnswer Quality Metrics:")
        print(f"  Faithfulness:       {metrics.avg_faithfulness:.3f}")
        print(f"  Answer Similarity:  {metrics.avg_answer_similarity:.3f}")
        print(f"  Hallucination Rate: {metrics.hallucination_rate:.1%}")
        print(f"  Avg Iterations:     {metrics.avg_iterations:.2f}")
        print("=" * 60)


def quick_pubmedqa_test(workflow, num_samples: int = 5):
    """Quick test on a few PubMedQA samples."""
    evaluator = PubMedQAEvaluator(workflow=workflow)

    print(f"\nQuick PubMedQA Test ({num_samples} samples)")
    print("-" * 40)

    samples = evaluator.data_manager.load_dataset('pubmedqa_csv_labeled', sample_size=num_samples)

    results = []
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{num_samples}] {sample.question[:80]}...")
        result = evaluator.evaluate_single(sample, verbose=True)
        results.append(result)

        status = "✓" if result.is_correct else "✗"
        print(f"  Predicted: {result.predicted_decision}, Ground Truth: {result.ground_truth_decision} {status}")
        print(f"  Faithfulness: {result.faithfulness:.3f}")

    metrics = evaluator.compute_metrics(results)
    evaluator.print_summary(metrics)

    return results, metrics


if __name__ == "__main__":
    # Test decision extraction
    print("Testing decision extraction...")

    test_answers = [
        "Yes, the evidence supports this conclusion.",
        "No, there is no significant association found.",
        "The results are inconclusive and further research is needed.",
        "Based on the evidence, the answer is yes.",
        "No significant difference was observed between groups.",
    ]

    evaluator = PubMedQAEvaluator()
    for answer in test_answers:
        decision = evaluator.extract_decision(answer)
        print(f"  '{answer[:50]}...' -> {decision}")

    print("\nTesting data loading...")
    samples = evaluator.data_manager.load_dataset('pubmedqa_csv_labeled', sample_size=3)
    for s in samples:
        print(f"  Q: {s.question[:60]}...")
        print(f"  GT: {getattr(s, 'final_decision', 'N/A')}")
