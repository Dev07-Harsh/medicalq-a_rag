"""
RAGAS Evaluation Pipeline for MEGA-RAG
Evaluates system performance using Faithfulness, Answer Relevancy, and other metrics.
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

from mega_rag.config import BASE_DIR, PQA_ARTIFICIAL_PATH, PQA_LABELED_PATH, DEFAULT_EVAL_SAMPLES


@dataclass
class EvaluationResult:
    """Result from evaluation."""
    question: str
    ground_truth: str
    generated_answer: str
    context: List[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    hallucination_detected: bool
    overall_score: float


class RAGASEvaluator:
    """
    Evaluation using RAGAS metrics.
    Falls back to custom metrics if RAGAS is not available.
    """

    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas
        self._ragas_available = False

        if use_ragas:
            try:
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
                self._ragas_available = True
                self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
                print("RAGAS evaluation enabled")
            except ImportError:
                print("RAGAS not available, using custom metrics")
                self._ragas_available = False

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single QA pair.

        Args:
            question: The question
            answer: Generated answer
            context: Retrieved context chunks
            ground_truth: Optional ground truth answer

        Returns:
            Dict of metric scores
        """
        if self._ragas_available and ground_truth:
            return self._ragas_evaluate(question, answer, context, ground_truth)
        else:
            return self._custom_evaluate(question, answer, context, ground_truth)

    def _ragas_evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """Evaluate using RAGAS library."""
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset

        # Create dataset
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [context],
            'ground_truth': [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        # Run evaluation
        try:
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )

            return {
                'faithfulness': result['faithfulness'],
                'answer_relevancy': result['answer_relevancy'],
                'context_precision': result['context_precision'],
                'context_recall': result['context_recall']
            }
        except Exception as e:
            print(f"RAGAS evaluation error: {e}")
            return self._custom_evaluate(question, answer, context, ground_truth)

    def _custom_evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Custom evaluation metrics (fallback)."""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Embed everything
        answer_emb = model.encode([answer], normalize_embeddings=True)
        question_emb = model.encode([question], normalize_embeddings=True)

        context_text = " ".join(context)
        context_emb = model.encode([context_text], normalize_embeddings=True)

        # Faithfulness: answer similarity to context
        faithfulness = float(cosine_similarity(answer_emb, context_emb)[0][0])

        # Answer relevancy: answer relevance to question
        answer_relevancy = float(cosine_similarity(answer_emb, question_emb)[0][0])

        # Context precision/recall (simplified)
        if ground_truth:
            gt_emb = model.encode([ground_truth], normalize_embeddings=True)
            context_precision = float(cosine_similarity(context_emb, gt_emb)[0][0])
            context_recall = float(cosine_similarity(answer_emb, gt_emb)[0][0])
        else:
            context_precision = faithfulness
            context_recall = answer_relevancy

        return {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        }


class DatasetEvaluator:
    """
    Evaluate MEGA-RAG on the PQA datasets.
    """

    def __init__(self, workflow=None):
        self.workflow = workflow
        self.evaluator = RAGASEvaluator()

    def load_pqa_dataset(
        self,
        dataset_path: Optional[Path] = None,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Load PQA dataset from parquet file."""
        path = dataset_path or PQA_LABELED_PATH

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_parquet(path)

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        return df

    def evaluate_dataset(
        self,
        dataset_path: Optional[Path] = None,
        sample_size: int = 100,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate MEGA-RAG on a dataset.

        Args:
            dataset_path: Path to dataset (default: pqa_labeled)
            sample_size: Number of samples to evaluate
            output_path: Optional path to save results

        Returns:
            Dict with aggregated metrics and per-sample results
        """
        if self.workflow is None:
            raise ValueError("Workflow not initialized. Pass workflow to constructor.")

        # Load dataset
        df = self.load_pqa_dataset(dataset_path, sample_size)
        print(f"Evaluating on {len(df)} samples...")

        results = []
        metrics_sum = {
            'faithfulness': 0,
            'answer_relevancy': 0,
            'context_precision': 0,
            'context_recall': 0,
            'hallucination_count': 0
        }

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            question = row['Question']
            ground_truth = row['Ground Truth']

            # Run workflow
            try:
                workflow_result = self.workflow.run(question)
                answer = workflow_result['answer']
                context = workflow_result['context']

                # Evaluate
                metrics = self.evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    context=context,
                    ground_truth=ground_truth
                )

                # Check for hallucination
                is_hallucinated = metrics['faithfulness'] < 0.5

                result = {
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': answer,
                    'faithfulness': metrics['faithfulness'],
                    'answer_relevancy': metrics['answer_relevancy'],
                    'context_precision': metrics['context_precision'],
                    'context_recall': metrics['context_recall'],
                    'hallucination_detected': is_hallucinated,
                    'iterations': workflow_result.get('iterations', 0)
                }

                results.append(result)

                # Accumulate metrics
                for key in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    metrics_sum[key] += metrics[key]
                if is_hallucinated:
                    metrics_sum['hallucination_count'] += 1

            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                continue

        # Compute averages
        n = len(results)
        avg_metrics = {
            'faithfulness': metrics_sum['faithfulness'] / n if n > 0 else 0,
            'answer_relevancy': metrics_sum['answer_relevancy'] / n if n > 0 else 0,
            'context_precision': metrics_sum['context_precision'] / n if n > 0 else 0,
            'context_recall': metrics_sum['context_recall'] / n if n > 0 else 0,
            'hallucination_rate': metrics_sum['hallucination_count'] / n if n > 0 else 0,
            'samples_evaluated': n
        }

        evaluation_output = {
            'aggregate_metrics': avg_metrics,
            'per_sample_results': results
        }

        # Save results
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(evaluation_output, f, indent=2)
            print(f"Results saved to {output_path}")

        return evaluation_output

    def compare_with_baseline(
        self,
        baseline_results: Dict,
        mega_rag_results: Dict
    ) -> pd.DataFrame:
        """Compare MEGA-RAG results with a baseline."""
        comparison = []

        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'hallucination_rate']:
            baseline_val = baseline_results['aggregate_metrics'].get(metric, 0)
            mega_rag_val = mega_rag_results['aggregate_metrics'].get(metric, 0)
            improvement = mega_rag_val - baseline_val

            comparison.append({
                'metric': metric,
                'baseline': baseline_val,
                'mega_rag': mega_rag_val,
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_val * 100) if baseline_val != 0 else 0
            })

        return pd.DataFrame(comparison)


def quick_evaluate(workflow, questions: List[str], ground_truths: Optional[List[str]] = None):
    """Quick evaluation on a list of questions."""
    evaluator = RAGASEvaluator(use_ragas=False)

    results = []
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question[:50]}...")

        # Run workflow
        result = workflow.run(question)

        # Evaluate
        gt = ground_truths[i] if ground_truths else None
        metrics = evaluator.evaluate_single(
            question=question,
            answer=result['answer'],
            context=result['context'],
            ground_truth=gt
        )

        print(f"  Faithfulness: {metrics['faithfulness']:.3f}")
        print(f"  Answer Relevancy: {metrics['answer_relevancy']:.3f}")
        print(f"  Answer preview: {result['answer'][:100]}...")

        results.append({
            'question': question,
            'answer': result['answer'],
            'metrics': metrics
        })

    return results


def evaluate_pubmedqa(
    workflow,
    sample_size: int = DEFAULT_EVAL_SAMPLES,
    output_dir: Optional[Path] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on PubMedQA dataset.

    Args:
        workflow: MEGA-RAG workflow instance
        sample_size: Number of samples to evaluate
        output_dir: Directory to save results (optional)
        verbose: Print detailed progress

    Returns:
        Dict with metrics and results
    """
    from mega_rag.utils.pubmedqa_evaluator import PubMedQAEvaluator
    from mega_rag.utils.data_loader import DatasetManager

    evaluator = PubMedQAEvaluator(workflow=workflow)
    output_dir = output_dir or (BASE_DIR / "evaluation_results")

    # Prefer indexed test dataset if available
    data_manager = DatasetManager()
    available = data_manager.get_available_datasets()

    if 'pubmedqa_indexed' in available:
        dataset_name = 'pubmedqa_indexed'
        print("Using indexed test dataset (ensures retrieval matches test samples).")
    else:
        dataset_name = 'pubmedqa_csv_labeled'
        print("Note: Run 'python run.py --index --include-pubmedqa' for better evaluation accuracy.")

    results = evaluator.evaluate_dataset(
        dataset_name=dataset_name,
        sample_size=sample_size,
        output_dir=output_dir,
        verbose=verbose
    )

    # Print summary
    from mega_rag.utils.pubmedqa_evaluator import PubMedQAMetrics
    metrics = PubMedQAMetrics(
        total_samples=results['metrics']['total_samples'],
        accuracy=results['metrics']['accuracy'],
        macro_f1=results['metrics']['macro_f1'],
        per_class_accuracy=results['metrics']['per_class_accuracy'],
        per_class_f1=results['metrics']['per_class_f1'],
        confusion_matrix=results['metrics']['confusion_matrix'],
        avg_faithfulness=results['metrics']['avg_faithfulness'],
        avg_answer_similarity=results['metrics']['avg_answer_similarity'],
        hallucination_rate=results['metrics']['hallucination_rate'],
        avg_iterations=results['metrics']['avg_iterations']
    )
    evaluator.print_summary(metrics)

    return results


if __name__ == "__main__":
    # Test dataset loading
    try:
        evaluator = DatasetEvaluator()
        df = evaluator.load_pqa_dataset(sample_size=5)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample question: {df['Question'].iloc[0][:100]}...")
    except Exception as e:
        print(f"Error: {e}")
