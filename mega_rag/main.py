"""
MEGA-RAG: Medical Evidence-Guided Augmented Retrieval-Augmented Generation
Main application entry point.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, List

from mega_rag.config import (
    KNOWLEDGE_BASE_PDFS,
    BASE_DIR,
    GEMINI_API_KEY,
    DEFAULT_EVAL_SAMPLES,
    LLM_PROVIDER
)


def setup_environment():
    """Setup and verify environment."""
    print("=" * 60)
    print("MEGA-RAG: Medical Evidence-Guided Augmentation")
    print("=" * 60)

    # Check LLM provider
    print(f"LLM Provider: {LLM_PROVIDER}")

    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            print("⚠️ WARNING: GEMINI_API_KEY not set!")
            print("Set it via: export GEMINI_API_KEY='your-key'")
            print("Or create a .env file with GEMINI_API_KEY=your-key")
            print("Or switch to Ollama: set LLM_PROVIDER='ollama' in config.py")
            return False
        print("✓ Gemini API key configured")
    elif LLM_PROVIDER == "ollama":
        print("✓ Using local Ollama model")

    return True


def index_documents(
    pdf_paths: Optional[List[str]] = None,
    force: bool = False,
    include_pubmedqa: bool = False,
    pubmedqa_samples: Optional[int] = None
):
    """Index PDF documents and optionally PubMedQA contexts into the retrieval system."""
    from mega_rag.retrieval.hybrid_retriever import HybridRetriever
    from mega_rag.core.document_processor import DocumentProcessor

    print("\n--- Document Indexing ---")

    # Initialize retriever
    retriever = HybridRetriever(use_reranker=False)  # Disable reranker for indexing

    # Check if already indexed
    if not force and retriever.load_indices():
        stats = retriever.get_stats()
        print(f"Existing index found: {stats}")
        response = input("Re-index? (y/N): ").strip().lower()
        if response != 'y':
            return retriever

    all_documents = []

    # Get PDF paths
    if pdf_paths:
        paths = [Path(p) for p in pdf_paths]
    else:
        paths = [p for p in KNOWLEDGE_BASE_PDFS if p.exists()]

    if paths:
        print(f"\n[1/2] Indexing {len(paths)} PDF files...")
        for p in paths:
            print(f"  - {p.name}")

        # Process PDFs
        processor = DocumentProcessor()
        pdf_documents = processor.process_multiple_pdfs(paths)
        all_documents.extend(pdf_documents)
        print(f"  Created {len(pdf_documents)} chunks from PDFs")
    else:
        print("No PDF files found to index.")

    # Index PubMedQA contexts if requested
    if include_pubmedqa:
        from mega_rag.utils.data_loader import OfficialPubMedQALoader
        
        print(f"\n[2/2] Indexing Official PubMedQA-L contexts...")
        
        try:
            # Use the official PubMedQA-L loader (1000 expert-labeled samples)
            official_loader = OfficialPubMedQALoader()
            
            # Load pre-prepared indexing documents (contexts only, NO ground truth)
            pubmedqa_docs = official_loader.load_indexing_documents()
            all_documents.extend(pubmedqa_docs)
            
            # Show class distribution for reference
            test_dist = official_loader.get_class_distribution('test')
            print(f"  Test set distribution: {test_dist}")
            print(f"  ⚠️  Note: Ground truth answers are NOT indexed (data leakage prevention)")

        except FileNotFoundError as e:
            print(f"  Warning: Official PubMedQA splits not found: {e}")
            print("  Run: python scripts/prepare_pubmedqa_split.py first")
            print("  Continuing without PubMedQA contexts...")

    if not all_documents:
        print("\nNo documents found to index!")
        return None

    print(f"\n--- Total: {len(all_documents)} document chunks ---")

    retriever.index_documents(all_documents)
    retriever.save_indices()

    return retriever


def interactive_mode(workflow):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("Interactive Mode - Ask medical questions!")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'verbose' to toggle detailed output")
    print("=" * 60)

    verbose = False

    while True:
        try:
            question = input("\n🔍 Question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if question.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue

            # Run workflow
            print("\n⏳ Processing...")
            result = workflow.run(question)

            # Check reliability - THIS IS THE CORE FEATURE
            is_reliable = result.get('is_reliable', False)

            # Display answer with reliability indicator
            print("\n" + "-" * 40)
            if is_reliable:
                print("✅ ANSWER (Evidence-Grounded):")
            else:
                print("⚠️  ANSWER REFUSED (Low Alignment):")
            print("-" * 40)
            print(result['answer'])

            # Display sources only if answer is reliable
            sources = result.get('sources', [])
            if sources and is_reliable:
                print("\n" + "-" * 40)
                print("📚 SOURCES USED:")
                print("-" * 40)
                for i, source in enumerate(sources, 1):
                    source_name = source.replace('.pdf', '').replace('_', ' ')
                    print(f"  [{i}] {source_name}")

            # Show metadata and timing only in verbose mode
            if verbose:
                print("\n" + "-" * 40)
                print("📊 METADATA:")
                print("-" * 40)
                print(f"Iterations: {result.get('iterations', 0)}")

                seae = result.get('seae_result', {})
                if seae:
                    print(f"Alignment Score: {seae.get('alignment_score', 'N/A'):.3f}")
                    print(f"Evidence Coverage: {seae.get('evidence_coverage', 'N/A'):.3f}")

                # Display timing information
                timing = result.get('timing', {})
                if timing:
                    print("\n⏱️  TIMING:")
                    for key, value in sorted(timing.items()):
                        if key != 'total':
                            print(f"  {key}: {value:.2f}s")
                    if 'total' in timing:
                        print(f"  ─────────────────")
                        print(f"  Total: {timing['total']:.2f}s")

                print("\nWorkflow Trace:")
                for step in result.get('workflow_trace', []):
                    print(f"  {step}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def run_evaluation(workflow, sample_size: int = 50):
    """Run evaluation on PQA dataset."""
    from mega_rag.utils.evaluation import DatasetEvaluator

    print("\n--- Running Evaluation ---")

    evaluator = DatasetEvaluator(workflow=workflow)
    results = evaluator.evaluate_dataset(
        sample_size=sample_size,
        output_path=BASE_DIR / "evaluation_results.json"
    )

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)

    metrics = results['aggregate_metrics']
    print(f"Samples Evaluated: {metrics['samples_evaluated']}")
    print(f"Faithfulness:      {metrics['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {metrics['answer_relevancy']:.3f}")
    print(f"Context Precision: {metrics['context_precision']:.3f}")
    print(f"Context Recall:    {metrics['context_recall']:.3f}")
    print(f"Hallucination Rate: {metrics['hallucination_rate']:.1%}")

    return results


def run_pubmedqa_evaluation(workflow, sample_size: int = DEFAULT_EVAL_SAMPLES, verbose: bool = False):
    """Run evaluation on PubMedQA dataset."""
    from mega_rag.utils.evaluation import evaluate_pubmedqa

    print("\n--- Running PubMedQA Evaluation ---")
    print(f"Sample size: {sample_size}")

    output_dir = BASE_DIR / "evaluation_results"
    results = evaluate_pubmedqa(
        workflow=workflow,
        sample_size=sample_size,
        output_dir=output_dir,
        verbose=verbose
    )

    # Print paths to output files
    if 'json_path' in results:
        print(f"\nResults saved to:")
        print(f"  JSON: {results['json_path']}")
    if 'html_path' in results:
        print(f"  HTML: {results['html_path']}")

    return results


def run_tests():
    """Run test suite."""
    import subprocess

    print("\n--- Running Test Suite ---")

    test_dir = BASE_DIR / "tests"
    if not test_dir.exists():
        print("Test directory not found. Creating tests...")
        test_dir.mkdir(exist_ok=True)

    # Try pytest first
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", str(test_dir), "-v", "--tb=short"],
            capture_output=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest")
        return False


def run_research_evaluation(workflow, sample_size: int = DEFAULT_EVAL_SAMPLES, verbose: bool = True):
    """Run comprehensive research evaluation for paper publication."""
    from mega_rag.utils.research_evaluation import ResearchEvaluator

    print("\n" + "="*60)
    print("RESEARCH PAPER EVALUATION")
    print("="*60)

    output_dir = BASE_DIR / "research_results"
    evaluator = ResearchEvaluator(workflow=workflow)

    # Run evaluation on INDEXED data only (User verified)
    metrics = evaluator.run_evaluation(
        dataset_name='pubmedqa_indexed',
        sample_size=sample_size,
        verbose=verbose
    )

    # Generate all reports
    output_paths = evaluator.generate_report(metrics, output_dir)

    print("\n" + "="*60)
    print("OUTPUT FILES FOR RESEARCH PAPER:")
    print("="*60)
    print(f"  JSON (raw data):    {output_paths['json']}")
    print(f"  LaTeX (tables):     {output_paths['latex']}")
    print(f"  CSV (summary):      {output_paths['csv']}")
    print("\nCopy the LaTeX tables directly into your paper!")

    return metrics

    return metrics


def single_query(workflow, question: str, verbose: bool = False):
    """Process a single question."""
    print(f"\nQuestion: {question}")
    print("\nProcessing...")

    result = workflow.run(question)

    # Check reliability - THIS IS THE CORE FEATURE
    is_reliable = result.get('is_reliable', False)
    seae = result.get('seae_result', {})
    alignment_score = seae.get('alignment_score', 0) if seae else 0

    print("\n" + "=" * 40)
    if is_reliable:
        print("✅ ANSWER (Evidence-Grounded):")
    else:
        print("⚠️  ANSWER REFUSED (Low Alignment):")
    print("=" * 40)
    print(result['answer'])

    # Display sources only if answer is reliable
    sources = result.get('sources', [])
    if sources and is_reliable:
        print("\n" + "-" * 40)
        print("📚 SOURCES USED:")
        print("-" * 40)
        for i, source in enumerate(sources, 1):
            source_name = source.replace('.pdf', '').replace('_', ' ')
            print(f"  [{i}] {source_name}")

    print("\n" + "-" * 40)
    print("📊 Metadata:")
    print(f"  Reliable: {'Yes' if is_reliable else 'No'}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Alignment Score: {alignment_score:.3f}")

    # Display timing and trace only in verbose mode
    if verbose:
        timing = result.get('timing', {})
        if timing:
            print("\n⏱️  TIMING:")
            for key, value in sorted(timing.items()):
                if key != 'total':
                    print(f"  {key}: {value:.2f}s")
            if 'total' in timing:
                print(f"  ─────────────────")
                print(f"  Total: {timing['total']:.2f}s")

        trace = result.get('workflow_trace', [])
        if trace:
            print("\n📋 Workflow Trace:")
            for step in trace:
                print(f"  {step}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MEGA-RAG: Medical Evidence-Guided Augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --index                      # Index PDF documents only
  python run.py --index --include-pubmedqa   # Index PDFs + PubMedQA contexts
  python run.py --index --include-pubmedqa --pubmedqa-samples 500  # Limit PubMedQA samples
  python run.py --interactive                # Interactive Q&A mode
  python run.py --query "What is hypertension?"  # Single query
  python run.py --evaluate                   # Evaluate on PQA dataset
  python run.py --evaluate-pubmedqa          # Evaluate on PubMedQA dataset
  python run.py --research-eval --eval-samples 200  # Research paper evaluation
  python run.py --test                       # Run test suite
        """
    )
    parser.add_argument(
        '--index', '-i',
        action='store_true',
        help='Index documents (run before first query)'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single question to answer'
    )
    parser.add_argument(
        '--evaluate', '-e',
        action='store_true',
        help='Run evaluation on PQA dataset'
    )
    parser.add_argument(
        '--evaluate-pubmedqa',
        action='store_true',
        help='Run evaluation on PubMedQA dataset (yes/no/maybe classification)'
    )
    parser.add_argument(
        '--research-eval',
        action='store_true',
        help='Run comprehensive research evaluation using INDEXED samples (generates LaTeX tables, JSON, CSV)'
    )
    parser.add_argument(
        '--eval-samples',
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help=f'Number of samples for evaluation (default: {DEFAULT_EVAL_SAMPLES})'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test suite'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output for evaluation'
    )
    parser.add_argument(
        '--pdfs',
        nargs='+',
        help='Custom PDF paths to index'
    )
    parser.add_argument(
        '--force-reindex',
        action='store_true',
        help='Force re-indexing of documents'
    )
    parser.add_argument(
        '--include-pubmedqa',
        action='store_true',
        help='Include PubMedQA contexts in the knowledge base (expands coverage)'
    )
    parser.add_argument(
        '--pubmedqa-samples',
        type=int,
        default=None,
        help='Number of PubMedQA samples to index (default: all labeled + 1000 artificial)'
    )

    args = parser.parse_args()

    # Setup
    if not setup_environment():
        if not args.index:  # Allow indexing without API key
            sys.exit(1)

    # Test mode (doesn't need workflow)
    if args.test:
        run_tests()
        return

    # Index mode
    if args.index:
        index_documents(
            pdf_paths=args.pdfs,
            force=args.force_reindex,
            include_pubmedqa=args.include_pubmedqa,
            pubmedqa_samples=args.pubmedqa_samples
        )
        if not args.query and not args.evaluate and not args.evaluate_pubmedqa and not args.interactive:
            print("\n✓ Indexing complete. Use --query, --evaluate, --evaluate-pubmedqa, or --interactive to query.")
            return

    # Initialize workflow
    from mega_rag.core.workflow import create_mega_rag_workflow

    print("\nInitializing MEGA-RAG workflow...")
    try:
        workflow = create_mega_rag_workflow()
    except Exception as e:
        print(f"Error initializing workflow: {e}")
        print("Make sure to run with --index first to build the index.")
        sys.exit(1)

    # Load indices
    if not workflow.retriever.load_indices():
        print("\n⚠️  No index found. Run with --index first.")
        sys.exit(1)

    print(f"Index loaded: {workflow.retriever.get_stats()}")

    # Mode selection
    if args.query:
        single_query(workflow, args.query, verbose=args.verbose)
    elif args.evaluate:
        run_evaluation(workflow, args.eval_samples)
    elif args.evaluate_pubmedqa:
        run_pubmedqa_evaluation(workflow, args.eval_samples, args.verbose)
    elif args.research_eval:
        run_research_evaluation(workflow, args.eval_samples, args.verbose)
    else:
        # Default to interactive mode
        interactive_mode(workflow)


if __name__ == "__main__":
    main()
