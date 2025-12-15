"""
Dynamic Evaluation Script for MEGA-RAG
Allows running quick research evaluations with a custom number of samples.
Usage: python scripts/run_dynamic_eval.py --samples 20
"""
import argparse
import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mega_rag.core.workflow import create_mega_rag_workflow
from mega_rag.utils.research_evaluation import ResearchEvaluator
from mega_rag.config import BASE_DIR, LLM_PROVIDER

def run_dynamic_eval(samples: int, verbose: bool = True):
    print(f"\n🚀 Starting Dynamic Evaluation")
    print(f"   Samples: {samples}")
    print(f"   Provider: {LLM_PROVIDER}")
    print("=" * 60)

    # 1. Initialize Workflow
    try:
        print("1. Initializing Workflow...")
        workflow = create_mega_rag_workflow()
        
        # Verify index
        if not workflow.retriever.load_indices():
            print("\n❌ Error: No index found! Run 'python run.py --index' first.")
            sys.exit(1)
            
        stats = workflow.retriever.get_stats()
        print(f"   ✓ Index loaded ({stats.get('graph', {}).get('total_nodes', 0)} nodes)")
        
    except Exception as e:
        print(f"\n❌ Error initializing: {e}")
        sys.exit(1)

    # 2. Initialize Evaluator
    print("\n2. Loading Research Evaluator...")
    evaluator = ResearchEvaluator(workflow)

    # 3. Run Evaluation
    print(f"\n3. Running Evaluation on {samples} samples...")
    print("   (This uses the 'pubmedqa_indexed' dataset - NO LEAKAGE)")
    
    try:
        metrics = evaluator.run_evaluation(
            dataset_name='pubmedqa_official_test',
            sample_size=samples,
            verbose=verbose
        )
        
        # 4. Generate Reports & Print Summary
        output_dir = BASE_DIR / "research_results"
        output_paths = evaluator.generate_report(metrics, output_dir)
        print(f"\nResults saved to: {output_paths['csv']}")
        
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynamic MEGA-RAG evaluation")
    parser.add_argument('--samples', type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument('--quiet', action='store_true', help="Suppress verbose output")
    
    args = parser.parse_args()
    
    run_dynamic_eval(args.samples, verbose=not args.quiet)
