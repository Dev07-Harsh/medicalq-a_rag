#!/usr/bin/env python3
"""
MEGA-RAG Balanced Evaluation Script

This script ensures proper evaluation with:
1. Balanced class distribution (yes/no/maybe)
2. Sufficient sample size
3. Proper metrics (Macro F1, per-class metrics)
4. Ablation study support
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import random
from datetime import datetime
from typing import List, Dict, Tuple
from collections import Counter

from mega_rag.utils.data_loader import DatasetManager


def load_balanced_test_set(
    target_per_class: int = 50,
    classes: List[str] = ['yes', 'no', 'maybe']
) -> List[Dict]:
    """
    Load a balanced test set with equal representation of each class.
    
    Args:
        target_per_class: Number of samples per class
        classes: The classes to balance
        
    Returns:
        List of balanced samples
    """
    dm = DatasetManager()
    
    # Load all labeled samples
    all_samples = dm.load_pubmedqa_labeled(sample_size=1000)
    
    # Group by final_decision
    by_class = {c: [] for c in classes}
    
    for sample in all_samples:
        decision = getattr(sample, 'final_decision', 'unknown')
        if decision and decision.lower() in classes:
            by_class[decision.lower()].append(sample)
    
    print("\n📊 Dataset Distribution:")
    for c, samples in by_class.items():
        print(f"  {c}: {len(samples)} samples")
    
    # Sample equally from each class
    balanced_samples = []
    for c in classes:
        available = by_class[c]
        if len(available) < target_per_class:
            print(f"  ⚠️ Only {len(available)} samples for '{c}', using all")
            balanced_samples.extend(available)
        else:
            random.seed(42)  # Reproducibility
            balanced_samples.extend(random.sample(available, target_per_class))
    
    print(f"\n✅ Balanced test set: {len(balanced_samples)} samples")
    
    # Verify balance
    final_counts = Counter(
        getattr(s, 'final_decision', 'unknown').lower() 
        for s in balanced_samples
    )
    print(f"  Final distribution: {dict(final_counts)}")
    
    return balanced_samples


def run_ablation_study(samples: List) -> Dict:
    """
    Run ablation study to measure contribution of each component.
    
    Tests:
    1. Vector only
    2. Vector + BM25
    3. Vector + BM25 + Graph
    4. Full system (+ SEAE + DISC)
    5. Full system + CoT
    """
    print("\n🔬 Starting Ablation Study...")
    print("=" * 60)
    
    # This is a placeholder - in real implementation, you'd:
    # 1. Modify config for each ablation
    # 2. Re-initialize workflow
    # 3. Run evaluation
    # 4. Collect metrics
    
    results = {
        'ablations': [
            {'name': 'Vector Only', 'accuracy': None, 'faithfulness': None},
            {'name': 'Vector + BM25', 'accuracy': None, 'faithfulness': None},
            {'name': 'Tri-Brid (+ Graph)', 'accuracy': None, 'faithfulness': None},
            {'name': '+ SEAE Audit', 'accuracy': None, 'faithfulness': None},
            {'name': '+ DISC Correction', 'accuracy': None, 'faithfulness': None},
            {'name': '+ Chain-of-Thought', 'accuracy': None, 'faithfulness': None},
        ],
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(samples)
    }
    
    print("\n⚠️ Ablation study requires manual configuration changes.")
    print("To run properly:")
    print("  1. Set GRAPH_TOP_K=0, disable SEAE in config")
    print("  2. Run evaluation")
    print("  3. Enable each component one by one")
    print("  4. Record results")
    
    return results


def analyze_errors(results: List[Dict]) -> Dict:
    """
    Analyze prediction errors to understand failure modes.
    
    Categories:
    1. Retrieval Failure - relevant docs not retrieved
    2. Generation Failure - wrong answer despite good evidence
    3. Grounding Failure - hallucinated claims
    4. Classification Failure - correct answer, wrong yes/no/maybe
    """
    errors = {
        'retrieval_failure': [],
        'generation_failure': [],
        'grounding_failure': [],
        'classification_failure': [],
        'correct': []
    }
    
    for r in results:
        if r.get('is_correct', False):
            errors['correct'].append(r)
        else:
            # Analyze error type
            faithfulness = r.get('faithfulness', 0)
            if faithfulness < 0.3:
                errors['retrieval_failure'].append(r)
            elif faithfulness < 0.6:
                errors['grounding_failure'].append(r)
            else:
                # Good grounding but wrong answer
                if r.get('answer_similarity', 0) > 0.7:
                    errors['classification_failure'].append(r)
                else:
                    errors['generation_failure'].append(r)
    
    summary = {
        category: len(samples)
        for category, samples in errors.items()
    }
    
    print("\n📉 Error Analysis:")
    total_errors = sum(v for k, v in summary.items() if k != 'correct')
    for category, count in summary.items():
        if category == 'correct':
            print(f"  ✅ Correct: {count}")
        else:
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            print(f"  ❌ {category}: {count} ({pct:.1f}% of errors)")
    
    return {
        'summary': summary,
        'examples': {
            category: [
                {
                    'question': e.get('question', '')[:100],
                    'predicted': e.get('predicted_decision', ''),
                    'ground_truth': e.get('ground_truth_decision', ''),
                    'faithfulness': e.get('faithfulness', 0)
                }
                for e in samples[:3]  # Top 3 examples
            ]
            for category, samples in errors.items()
        }
    }


def main():
    print("=" * 60)
    print("MEGA-RAG Research Evaluation Tool")
    print("=" * 60)
    
    # Load balanced test set
    samples = load_balanced_test_set(target_per_class=50)
    
    # Show ablation study instructions
    ablation_results = run_ablation_study(samples)
    
    print("\n" + "=" * 60)
    print("Next Steps for Publication:")
    print("=" * 60)
    print("""
1. Run full evaluation with balanced 150-sample test set
2. Complete ablation study (enable/disable components)
3. Run error analysis on failures
4. Compare against baselines:
   - GPT-4 zero-shot
   - Simple RAG (vector only)
   - BM25 only
5. Calculate statistical significance (bootstrap CI)
""")


if __name__ == "__main__":
    main()
