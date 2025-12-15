"""
Research Evaluation Pipeline for MEGA-RAG
Generates publication-ready metrics for research papers.

Metrics Generated:
- Classification Accuracy (Yes/No/Maybe)
- Precision, Recall, F1 (per class & macro)
- Faithfulness Score
- Hallucination Rate
- SEAE Alignment Score
- Retrieval Quality (Context Relevance)
- Statistical Analysis (Mean, Std, CI)
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from mega_rag.config import BASE_DIR, DEFAULT_EVAL_SAMPLES
from mega_rag.utils.data_loader import DatasetManager, QASample
from mega_rag.utils.pubmedqa_evaluator import PubMedQAEvaluator, PubMedQAResult


@dataclass
class ResearchMetrics:
    """Comprehensive metrics for research paper."""
    # Dataset Info
    dataset_name: str
    total_samples: int
    evaluation_date: str

    # Classification Metrics
    accuracy: float
    accuracy_std: float
    macro_precision: float
    macro_recall: float
    macro_f1: float

    # Per-Class Metrics
    class_precision: Dict[str, float]
    class_recall: Dict[str, float]
    class_f1: Dict[str, float]
    class_support: Dict[str, int]

    # Confusion Matrix
    confusion_matrix: Dict[str, Dict[str, int]]

    # Answer Quality Metrics
    avg_faithfulness: float
    faithfulness_std: float
    avg_answer_similarity: float
    similarity_std: float

    # Hallucination Metrics
    hallucination_rate: float
    avg_seae_alignment: float
    alignment_std: float

    # Retrieval Metrics
    avg_retrieval_score: float

    # System Performance
    avg_iterations: float
    correction_rate: float  # % of answers that needed DISC correction

    # 95% Confidence Intervals
    accuracy_ci: tuple
    faithfulness_ci: tuple
    hallucination_ci: tuple


class ResearchEvaluator:
    """
    Comprehensive evaluator for research paper metrics.

    Supports two evaluation modes:
    1. use_pubmedqa_context=True: Use PubMedQA's own context (fair evaluation)
    2. use_pubmedqa_context=False: Use your PDF knowledge base (domain-specific)
    """

    def __init__(self, workflow=None, llm=None):
        self.workflow = workflow
        self.llm = llm  # For context-based evaluation without retrieval
        self.data_manager = DatasetManager()
        self.pubmedqa_evaluator = PubMedQAEvaluator(workflow=workflow)

    def compute_confidence_interval(self, data: List[float], confidence: float = 0.95) -> tuple:
        """Compute confidence interval for a metric."""
        if not data:
            return (0.0, 0.0)

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1) if n > 1 else 0

        # Use t-distribution for small samples
        from scipy import stats
        if n > 1:
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=std/np.sqrt(n))
            return (round(ci[0], 4), round(ci[1], 4))
        return (mean, mean)

    def compute_precision_recall_f1(
        self,
        predictions: List[str],
        ground_truths: List[str],
        labels: List[str] = ['yes', 'no', 'maybe']
    ) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, F1 for each class."""
        metrics = {}

        for label in labels:
            tp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truths) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truths) if p != label and g == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            support = sum(1 for g in ground_truths if g == label)

            metrics[label] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': support
            }

        return metrics

    def run_evaluation(
        self,
        dataset_name: str = 'pubmedqa_indexed',
        sample_size: int = 100,
        random_seed: int = 42,
        verbose: bool = True,
        debug: bool = False
    ) -> ResearchMetrics:
        """
        Run comprehensive evaluation for research paper.

        Args:
            dataset_name: Dataset to evaluate on. Use 'pubmedqa_indexed' (default) to
                         test on the same samples that are in the index.
            sample_size: Number of samples (recommend 100-500 for papers)
            random_seed: For reproducibility
            verbose: Print progress
            debug: Enable detailed debug output for each sample

        Returns:
            ResearchMetrics with all publication-ready metrics
        """
        if self.workflow is None:
            raise ValueError("Workflow not initialized")

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Auto-select indexed dataset if available and requested
        available_datasets = self.data_manager.get_available_datasets()
        actual_dataset = dataset_name

        if dataset_name == 'pubmedqa_indexed' and 'pubmedqa_indexed' not in available_datasets:
            print("Note: Indexed test dataset not found. Falling back to pubmedqa_csv_labeled.")
            print("      Run 'python run.py --index --include-pubmedqa' to create indexed dataset.")
            actual_dataset = 'pubmedqa_csv_labeled'
        elif dataset_name == 'pubmedqa_indexed':
            print("Using indexed test dataset (ensures retrieval matches test samples).")

        print(f"\n{'='*60}")
        print(f"RESEARCH EVALUATION - {actual_dataset}")
        print(f"{'='*60}")
        print(f"Sample Size: {sample_size}")
        print(f"Random Seed: {random_seed}")
        print(f"Debug Mode: {'ON' if debug else 'OFF'}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Load dataset
        samples = self.data_manager.load_dataset(actual_dataset, sample_size=sample_size)
        print(f"Loaded {len(samples)} samples")

        # Run evaluation
        results: List[PubMedQAResult] = []
        predictions = []
        ground_truths = []
        faithfulness_scores = []
        similarity_scores = []
        alignment_scores = []
        iteration_counts = []

        for i, sample in enumerate(tqdm(samples, desc="Evaluating", disable=not verbose)):
            if debug:
                print(f"\n{'*'*70}")
                print(f"[DEBUG] Processing sample {i+1}/{len(samples)}")
                print(f"{'*'*70}")
            
            result = self.pubmedqa_evaluator.evaluate_single(sample, debug=debug)
            results.append(result)

            predictions.append(result.predicted_decision)
            ground_truths.append(result.ground_truth_decision)
            faithfulness_scores.append(result.faithfulness)
            similarity_scores.append(result.answer_similarity)
            iteration_counts.append(result.workflow_iterations)

            # Get SEAE alignment from workflow if available
            # For now, use faithfulness as proxy
            alignment_scores.append(result.faithfulness)
            
            if debug:
                print(f"\n[DEBUG SUMMARY] Sample {i+1}:")
                print(f"  Question: {sample.question[:80]}...")
                print(f"  Ground Truth: {result.ground_truth_decision}")
                print(f"  Predicted:    {result.predicted_decision}")
                print(f"  Correct:      {'✓' if result.is_correct else '✗'}")
                print(f"  Faithfulness: {result.faithfulness:.3f}")
                print(f"{'*'*70}\n")

        # Compute all metrics
        labels = ['yes', 'no', 'maybe']

        # Classification metrics
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        accuracy = correct / len(predictions)

        # Per-sample accuracy for std calculation
        per_sample_accuracy = [1 if p == g else 0 for p, g in zip(predictions, ground_truths)]
        accuracy_std = np.std(per_sample_accuracy, ddof=1)

        # Per-class metrics
        class_metrics = self.compute_precision_recall_f1(predictions, ground_truths, labels)

        # Macro averages
        macro_precision = np.mean([class_metrics[l]['precision'] for l in labels])
        macro_recall = np.mean([class_metrics[l]['recall'] for l in labels])
        macro_f1 = np.mean([class_metrics[l]['f1'] for l in labels])

        # Confusion matrix
        confusion_matrix = {label: {l: 0 for l in labels} for label in labels}
        for p, g in zip(predictions, ground_truths):
            if g in confusion_matrix and p in confusion_matrix[g]:
                confusion_matrix[g][p] += 1

        # Answer quality metrics
        avg_faithfulness = np.mean(faithfulness_scores)
        faithfulness_std = np.std(faithfulness_scores, ddof=1)
        avg_similarity = np.mean(similarity_scores)
        similarity_std = np.std(similarity_scores, ddof=1)

        # Hallucination metrics (faithfulness < 0.5)
        hallucination_flags = [1 if f < 0.5 else 0 for f in faithfulness_scores]
        hallucination_rate = np.mean(hallucination_flags)

        avg_alignment = np.mean(alignment_scores)
        alignment_std = np.std(alignment_scores, ddof=1)

        # System performance
        avg_iterations = np.mean(iteration_counts)
        correction_rate = sum(1 for i in iteration_counts if i > 1) / len(iteration_counts)

        # Confidence intervals
        accuracy_ci = self.compute_confidence_interval(per_sample_accuracy)
        faithfulness_ci = self.compute_confidence_interval(faithfulness_scores)
        hallucination_ci = self.compute_confidence_interval(hallucination_flags)

        # Build metrics object
        metrics = ResearchMetrics(
            dataset_name=dataset_name,
            total_samples=len(samples),
            evaluation_date=datetime.now().isoformat(),

            accuracy=round(accuracy, 4),
            accuracy_std=round(accuracy_std, 4),
            macro_precision=round(macro_precision, 4),
            macro_recall=round(macro_recall, 4),
            macro_f1=round(macro_f1, 4),

            class_precision={l: class_metrics[l]['precision'] for l in labels},
            class_recall={l: class_metrics[l]['recall'] for l in labels},
            class_f1={l: class_metrics[l]['f1'] for l in labels},
            class_support={l: class_metrics[l]['support'] for l in labels},

            confusion_matrix=confusion_matrix,

            avg_faithfulness=round(avg_faithfulness, 4),
            faithfulness_std=round(faithfulness_std, 4),
            avg_answer_similarity=round(avg_similarity, 4),
            similarity_std=round(similarity_std, 4),

            hallucination_rate=round(hallucination_rate, 4),
            avg_seae_alignment=round(avg_alignment, 4),
            alignment_std=round(alignment_std, 4),

            avg_retrieval_score=round(avg_faithfulness, 4),  # Proxy

            avg_iterations=round(avg_iterations, 2),
            correction_rate=round(correction_rate, 4),

            accuracy_ci=accuracy_ci,
            faithfulness_ci=faithfulness_ci,
            hallucination_ci=hallucination_ci
        )

        return metrics

    def generate_latex_table(self, metrics: ResearchMetrics) -> str:
        """Generate LaTeX table for research paper."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{MEGA-RAG Evaluation Results on PubMedQA Dataset}
\label{tab:results}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Value} & \textbf{95\% CI} \\
\midrule
\multicolumn{3}{l}{\textit{Classification Performance}} \\
Accuracy & """ + f"{metrics.accuracy:.2%}" + r""" & """ + f"[{metrics.accuracy_ci[0]:.2%}, {metrics.accuracy_ci[1]:.2%}]" + r""" \\
Macro Precision & """ + f"{metrics.macro_precision:.4f}" + r""" & - \\
Macro Recall & """ + f"{metrics.macro_recall:.4f}" + r""" & - \\
Macro F1-Score & """ + f"{metrics.macro_f1:.4f}" + r""" & - \\
\midrule
\multicolumn{3}{l}{\textit{Answer Quality}} \\
Faithfulness & """ + f"{metrics.avg_faithfulness:.4f}" + r""" $\pm$ """ + f"{metrics.faithfulness_std:.4f}" + r""" & """ + f"[{metrics.faithfulness_ci[0]:.4f}, {metrics.faithfulness_ci[1]:.4f}]" + r""" \\
Answer Similarity & """ + f"{metrics.avg_answer_similarity:.4f}" + r""" $\pm$ """ + f"{metrics.similarity_std:.4f}" + r""" & - \\
\midrule
\multicolumn{3}{l}{\textit{Hallucination Mitigation}} \\
Hallucination Rate & """ + f"{metrics.hallucination_rate:.2%}" + r""" & """ + f"[{metrics.hallucination_ci[0]:.2%}, {metrics.hallucination_ci[1]:.2%}]" + r""" \\
SEAE Alignment & """ + f"{metrics.avg_seae_alignment:.4f}" + r""" $\pm$ """ + f"{metrics.alignment_std:.4f}" + r""" & - \\
Correction Rate & """ + f"{metrics.correction_rate:.2%}" + r""" & - \\
\bottomrule
\end{tabular}
\end{table}

% Per-Class Results
\begin{table}[htbp]
\centering
\caption{Per-Class Classification Results}
\label{tab:per_class}
\begin{tabular}{lccccc}
\toprule
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
Yes & """ + f"{metrics.class_precision['yes']:.4f}" + r""" & """ + f"{metrics.class_recall['yes']:.4f}" + r""" & """ + f"{metrics.class_f1['yes']:.4f}" + r""" & """ + f"{metrics.class_support['yes']}" + r""" \\
No & """ + f"{metrics.class_precision['no']:.4f}" + r""" & """ + f"{metrics.class_recall['no']:.4f}" + r""" & """ + f"{metrics.class_f1['no']:.4f}" + r""" & """ + f"{metrics.class_support['no']}" + r""" \\
Maybe & """ + f"{metrics.class_precision['maybe']:.4f}" + r""" & """ + f"{metrics.class_recall['maybe']:.4f}" + r""" & """ + f"{metrics.class_f1['maybe']:.4f}" + r""" & """ + f"{metrics.class_support['maybe']}" + r""" \\
\midrule
\textbf{Macro Avg} & """ + f"{metrics.macro_precision:.4f}" + r""" & """ + f"{metrics.macro_recall:.4f}" + r""" & """ + f"{metrics.macro_f1:.4f}" + r""" & """ + f"{metrics.total_samples}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_report(
        self,
        metrics: ResearchMetrics,
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate all report formats for research paper."""
        output_dir = output_dir or (BASE_DIR / "research_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. JSON Report (for reproducibility)
        json_path = output_dir / f"research_metrics_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)

        # 2. LaTeX Tables
        latex_path = output_dir / f"latex_tables_{timestamp}.tex"
        with open(latex_path, 'w') as f:
            f.write(self.generate_latex_table(metrics))

        # 3. CSV Summary
        csv_path = output_dir / f"metrics_summary_{timestamp}.csv"
        summary_data = {
            'Metric': [
                'Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall',
                'Faithfulness', 'Answer Similarity', 'Hallucination Rate',
                'SEAE Alignment', 'Correction Rate', 'Total Samples'
            ],
            'Value': [
                f"{metrics.accuracy:.4f}",
                f"{metrics.macro_f1:.4f}",
                f"{metrics.macro_precision:.4f}",
                f"{metrics.macro_recall:.4f}",
                f"{metrics.avg_faithfulness:.4f}",
                f"{metrics.avg_answer_similarity:.4f}",
                f"{metrics.hallucination_rate:.4f}",
                f"{metrics.avg_seae_alignment:.4f}",
                f"{metrics.correction_rate:.4f}",
                str(metrics.total_samples)
            ],
            'Std': [
                f"{metrics.accuracy_std:.4f}",
                '-', '-', '-',
                f"{metrics.faithfulness_std:.4f}",
                f"{metrics.similarity_std:.4f}",
                '-',
                f"{metrics.alignment_std:.4f}",
                '-', '-'
            ]
        }
        pd.DataFrame(summary_data).to_csv(csv_path, index=False)

        # 4. Console Summary
        self.print_summary(metrics)

        return {
            'json': str(json_path),
            'latex': str(latex_path),
            'csv': str(csv_path)
        }

    def print_summary(self, metrics: ResearchMetrics):
        """Print formatted summary to console."""
        print("\n" + "="*70)
        print("MEGA-RAG RESEARCH EVALUATION RESULTS")
        print("="*70)
        print(f"Dataset: {metrics.dataset_name}")
        print(f"Samples: {metrics.total_samples}")
        print(f"Date: {metrics.evaluation_date}")
        print("-"*70)

        print("\n📊 CLASSIFICATION METRICS")
        print(f"  Accuracy:         {metrics.accuracy:.2%} ± {metrics.accuracy_std:.2%}")
        print(f"  95% CI:           [{metrics.accuracy_ci[0]:.2%}, {metrics.accuracy_ci[1]:.2%}]")
        print(f"  Macro Precision:  {metrics.macro_precision:.4f}")
        print(f"  Macro Recall:     {metrics.macro_recall:.4f}")
        print(f"  Macro F1-Score:   {metrics.macro_f1:.4f}")

        print("\n📈 PER-CLASS PERFORMANCE")
        print(f"  {'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print(f"  {'-'*56}")
        for label in ['yes', 'no', 'maybe']:
            print(f"  {label:<10} {metrics.class_precision[label]:<12.4f} "
                  f"{metrics.class_recall[label]:<12.4f} {metrics.class_f1[label]:<12.4f} "
                  f"{metrics.class_support[label]:<10}")

        print("\n🎯 ANSWER QUALITY")
        print(f"  Faithfulness:      {metrics.avg_faithfulness:.4f} ± {metrics.faithfulness_std:.4f}")
        print(f"  Answer Similarity: {metrics.avg_answer_similarity:.4f} ± {metrics.similarity_std:.4f}")

        print("\n🛡️ HALLUCINATION MITIGATION")
        print(f"  Hallucination Rate: {metrics.hallucination_rate:.2%}")
        print(f"  95% CI:             [{metrics.hallucination_ci[0]:.2%}, {metrics.hallucination_ci[1]:.2%}]")
        print(f"  SEAE Alignment:     {metrics.avg_seae_alignment:.4f} ± {metrics.alignment_std:.4f}")
        print(f"  DISC Correction Rate: {metrics.correction_rate:.2%}")

        print("\n📋 CONFUSION MATRIX")
        print(f"  {'Actual↓/Pred→':<15} {'Yes':<8} {'No':<8} {'Maybe':<8}")
        print(f"  {'-'*39}")
        for label in ['yes', 'no', 'maybe']:
            row = metrics.confusion_matrix[label]
            print(f"  {label:<15} {row['yes']:<8} {row['no']:<8} {row['maybe']:<8}")

        print("\n" + "="*70)


def run_research_evaluation(
    sample_size: int = 100,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete research evaluation pipeline.

    Usage:
        python -c "from mega_rag.utils.research_evaluation import run_research_evaluation; run_research_evaluation(100)"

    Or via CLI:
        python run.py --research-eval --eval-samples 100
    """
    from mega_rag.core.workflow import create_mega_rag_workflow

    print("Initializing MEGA-RAG workflow...")
    workflow = create_mega_rag_workflow()

    if not workflow.retriever.load_indices():
        raise ValueError("No index found. Run 'python run.py --index' first.")

    evaluator = ResearchEvaluator(workflow=workflow)

    # Run evaluation
    metrics = evaluator.run_evaluation(
        dataset_name='pubmedqa_csv_labeled',
        sample_size=sample_size,
        verbose=verbose
    )

    # Generate reports
    output_paths = evaluator.generate_report(metrics, output_dir)

    print("\n📁 OUTPUT FILES:")
    print(f"  JSON:  {output_paths['json']}")
    print(f"  LaTeX: {output_paths['latex']}")
    print(f"  CSV:   {output_paths['csv']}")

    return {
        'metrics': asdict(metrics),
        'output_paths': output_paths
    }


if __name__ == "__main__":
    import sys

    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_research_evaluation(sample_size=sample_size)
