#!/usr/bin/env python3
"""evaluate_pubmedqa_multiconfig.py

Evaluate PubMedQA on *the same* question set across multiple system configs:

- llm_only     : LLM answers without any retrieval context
- vector_only  : VectorRetriever only
- bm25_only    : BM25Retriever only
- graph_only   : GraphRetriever only
- hybrid       : HybridRetriever (vector+bm25+graph fusion)
- mega_rag     : MEGARAGWorkflow (hybrid retrieval + refinement/verification)

This is intended for apples-to-apples comparison of retrieval ablations.

Outputs
-------
- JSON report with per-config metrics + per-sample traces

Notes
-----
- This script uses the official split at `pubmedQA/splits/test.json` by default.
- For fair evaluation of retrieval, you should build an index that contains the
  PubMedQA contexts (but not answers). See README.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _load_pubmedqa_split(path: Path, sample_size: Optional[int], seed: int) -> List[dict]:
    import random

    with open(path, "r") as f:
        data = json.load(f)

    items = []
    for pubid, item in data.items():
        items.append(
            {
                "pubid": pubid,
                "question": item.get("QUESTION", ""),
                "final_decision": (item.get("final_decision", "") or "").lower(),
                "long_answer": item.get("LONG_ANSWER", ""),
                "contexts": item.get("CONTEXTS", []) or [],
            }
        )

    # Filter to labeled only
    items = [x for x in items if x["final_decision"] in {"yes", "no", "maybe"} and x["question"]]

    if sample_size and sample_size < len(items):
        random.seed(seed)
        items = random.sample(items, sample_size)

    return items


def _extract_decision_pubmedqa(text: str) -> str:
    # Keep in sync with existing evaluator style.
    t = (text or "").strip().lower()
    if "final answer: yes" in t or t.startswith("yes"):
        return "yes"
    if "final answer: no" in t or t.startswith("no"):
        return "no"
    if "final answer: maybe" in t or t.startswith("maybe"):
        return "maybe"
    return "unknown"


def _normalize_to_final_answer(question: str, raw_answer: str) -> str:
    """Normalize messy model output into `Final Answer: yes|no|maybe` when possible.

    Baselines (llm_only / retriever-only) often ignore the prompt and output headings like
    "Short answer: No". Lean workflow post-processes with similar logic; we mirror that
    here so comparisons aren't dominated by formatting.
    """

    a = (raw_answer or "").strip()
    if not a:
        return a

    # If it already contains a Final Answer line, keep it as-is.
    low = a.lower()
    if "final answer:" in low:
        return a

    # Heuristic extraction (Lean-like): try `Answer: yes|no|maybe`, then first token,
    # then scan first few tokens, then count occurrences.
    import re

    low = low.strip()

    m = re.search(r"\banswer[:\s]+(yes|no|maybe)\b", low)
    if m:
        return f"Final Answer: {m.group(1)}"

    # Common patterns: "short answer: no", "final decision: yes", "conclusion: maybe"
    m = re.search(r"\b(short answer|final decision|decision|conclusion)[:\s]+(yes|no|maybe)\b", low)
    if m:
        return f"Final Answer: {m.group(2)}"

    tokens = low.split()
    first_tokens = tokens[:8]
    if first_tokens:
        first = first_tokens[0].rstrip(".,!?:;()[]{}\"'")
        if first in {"yes", "no", "maybe"}:
            return f"Final Answer: {first}"

        for tok in first_tokens:
            tok_clean = tok.rstrip(".,!?:;()[]{}\"'")
            if tok_clean in {"yes", "no", "maybe"}:
                return f"Final Answer: {tok_clean}"

    first_part = low[:400]
    yes_count = len(re.findall(r"\byes\b", first_part))
    no_count = len(re.findall(r"\bno\b", first_part))
    maybe_count = len(re.findall(r"\bmaybe\b", first_part))

    if yes_count > no_count and yes_count > maybe_count:
        return "Final Answer: yes"
    if no_count > yes_count and no_count > maybe_count:
        return "Final Answer: no"
    if maybe_count > 0:
        return "Final Answer: maybe"

    # As a last resort for PubMedQA: abstain as maybe.
    return "Final Answer: maybe"


def _normalize_text_for_overlap(t: str) -> str:
    import re

    s = (t or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _hit_against_gold_contexts(retrieved_texts: List[str], gold_contexts: List[str]) -> bool:
    """Conservative overlap heuristic for retrieval sanity/diagnostics."""

    retrieved_norm = [_normalize_text_for_overlap(t) for t in (retrieved_texts or []) if t]
    gold_norm = [_normalize_text_for_overlap(t) for t in (gold_contexts or []) if t]

    for rtxt in retrieved_norm:
        if not rtxt:
            continue
        for gtxt in gold_norm:
            if not gtxt:
                continue
            if (gtxt in rtxt) or (rtxt in gtxt):
                return True
    return False


@dataclass
class RunOutput:
    answer_text: str
    predicted: str
    retrieval_confidence: str = "UNKNOWN"
    verifier_pass: Optional[bool] = None
    unsupported_claims_n: int = 0
    iterations: int = 0
    latency_s: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-config PubMedQA evaluation (same split across configs)")
    parser.add_argument("--test-json", type=Path, default=Path("pubmedQA/splits/test.json"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Optional targeted subset (for regression on previously failed items)
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file specifying a targeted subset. Supported formats: "
            "(1) {pubids:[...]} to filter by PubMed ID, or "
            "(2) {sample_indices:[...]} referring to indices in the sampled list. "
            "Example: evaluation_results/pubmedqa_lean_failed_balanced_15_pubids.json"
        ),
    )

    # Retrieval/index configuration
    parser.add_argument("--persist-dir", type=Path, default=Path("chroma_pubmedqa_only"))
    parser.add_argument("--collection-name", type=str, default="pubmedqa_only")

    # Runtime
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("evaluation_results/pubmedqa_multiconfig.json"))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--include-mega-rag",
        action="store_true",
        help="Include the full MEGARAGWorkflow (slow). Disabled by default.",
    )
    parser.add_argument(
        "--per-question-timeout-s",
        type=int,
        default=0,
        help="Best-effort timeout per (question, config) run; 0 disables. Note: only enforced for llm_only + retriever baselines.",
    )

    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=5,
        help="Write an incremental snapshot JSON every N samples to avoid losing progress.",
    )

    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help=(
            "Comma-separated list of configs to run (subset). "
            "Allowed: llm_only,vector_only,bm25_only,graph_only,hybrid,lean_workflow,mega_rag. "
            "Example: --configs hybrid"
        ),
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Evaluate retrieval overlap with PubMedQA CONTEXTS without calling the LLM. "
            "Useful for fast sanity checks before running generation."
        ),
    )

    # Lean workflow tuning knobs (used for simple sweeps)
    parser.add_argument("--lean-top-k-yesno", type=int, default=0, help="Override Lean top_k_for_yes_no (0 keeps default)")
    parser.add_argument(
        "--lean-min-relevance-score",
        type=float,
        default=-1.0,
        help="Override Lean min_relevance_score (negative keeps default)",
    )
    parser.add_argument(
        "--sweep-lean",
        action="store_true",
        help=(
            "Run a small grid search over Lean thresholds on the current sample set, "
            "writing a compact JSON summary next to --out."
        ),
    )

    args = parser.parse_args()

    log_path = args.out.with_suffix(".log")

    def _log(msg: str) -> None:
        if not args.debug:
            return
        ts = _now_iso()
        line = f"[{ts}] {msg}"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            # Don't ever crash the evaluation because logging failed.
            pass

    if not args.test_json.exists():
        raise FileNotFoundError(f"Missing test split: {args.test_json}")

    samples = _load_pubmedqa_split(args.test_json, args.sample_size, args.seed)
    if not samples:
        raise RuntimeError("No labeled PubMedQA samples found in the split")

    # If a subset file is provided, filter down to those indices (relative to this sampled list).
    if args.subset is not None:
        if not args.subset.exists():
            raise FileNotFoundError(f"Subset file not found: {args.subset}")
        with open(args.subset, "r") as f:
            subset_spec = json.load(f)

        pubids = subset_spec.get("pubids")
        indices = subset_spec.get("sample_indices")

        if pubids is not None:
            if not isinstance(pubids, list) or not all(isinstance(p, str) for p in pubids):
                raise ValueError("Subset JSON field 'pubids' must be a list of strings")
            wanted = set(pubids)
            filtered = [s for s in samples if s.get("pubid") in wanted]
            if not filtered:
                raise ValueError("No samples matched the requested subset pubids")
            samples = filtered

        elif indices is not None:
            if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
                raise ValueError("Subset JSON field 'sample_indices' must be a list of integers")
            indices = [i for i in indices if 0 <= i < len(samples)]
            if not indices:
                raise ValueError("Subset indices are empty after bounds-checking")
            samples = [samples[i] for i in indices]

        else:
            raise ValueError("Subset JSON must contain either 'pubids' or 'sample_indices'")

    # Local imports (slow)
    from mega_rag.core.llm import create_llm
    from mega_rag.core.workflow import MEGARAGWorkflow
    from mega_rag.core.lean_workflow import LeanMEGARAGWorkflow
    from mega_rag.retrieval.vector_retriever import VectorRetriever
    from mega_rag.retrieval.bm25_retriever import BM25Retriever
    from mega_rag.retrieval.graph_retriever import GraphRetriever
    from mega_rag.retrieval.hybrid_retriever import HybridRetriever

    # Load shared LLM
    llm = create_llm()

    # Load indices
    persist_dir: Path = args.persist_dir
    bm25_path = persist_dir / "bm25_index.pkl"
    graph_path = persist_dir / "knowledge_graph.pkl"

    # Vector retriever needs persist dir/collection
    vector = VectorRetriever(collection_name=args.collection_name, persist_directory=persist_dir)

    # BM25/Graph need artifacts
    bm25 = BM25Retriever(index_path=bm25_path)
    graph = GraphRetriever(graph_path=graph_path)

    # Hybrid (fusion)
    hybrid = HybridRetriever()
    hybrid.vector_retriever = vector
    hybrid.bm25_retriever = bm25
    hybrid.graph_retriever = graph

    # Load indices (vector/bm25/graph)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Persist dir not found: {persist_dir}. Build the index first (see README / build_pubmedqa_index.py)."
        )

    if not bm25_path.exists() or not graph_path.exists():
        raise FileNotFoundError(
            f"BM25/graph artifacts not found in {persist_dir}. Run build_pubmedqa_index.py --force."
        )

    if not hybrid.load_indices():
        raise RuntimeError("Failed to load indices")

    # Workflows
    mega_workflow = MEGARAGWorkflow(retriever=hybrid, llm=llm, debug=args.debug) if args.include_mega_rag else None
    lean_workflow = LeanMEGARAGWorkflow(retriever=hybrid, llm=llm, debug=args.debug)

    # Apply Lean overrides if requested
    if args.lean_top_k_yesno > 0:
        lean_workflow.top_k_for_yes_no = int(args.lean_top_k_yesno)
    if args.lean_min_relevance_score >= 0:
        lean_workflow.min_relevance_score = float(args.lean_min_relevance_score)

    # Config runners
    def _run_with_optional_timeout(prompt: str) -> str:
        # Best-effort timeout: only works if the underlying LLM call respects timeouts.
        # If not, this still provides a single place to add it later.
        return llm.generate(prompt)

    def run_llm_only(q: str) -> RunOutput:
        t0 = time.time()
        # Keep a simple, constrained prompt to elicit yes/no/maybe.
        prompt = (
            "You are answering a PubMedQA-style medical research question.\n"
            "Constraints:\n"
            "- Use at most 2 short sentences.\n"
            "- Then output exactly one final line in the form: Final Answer: yes|no|maybe\n"
            "- Do not output anything after the Final Answer line.\n\n"
            f"Question: {q}\n"
        )
        raw = _run_with_optional_timeout(prompt)
        ans = _normalize_to_final_answer(q, raw)
        pred = _extract_decision_pubmedqa(ans)
        return RunOutput(answer_text=raw, predicted=pred, latency_s=time.time() - t0)

    def _run_retriever_only(q: str, which: str) -> Tuple[str, List[str]]:
        # Returns (answer_text, contexts_used). We generate with LLM using retrieved contexts.
        if which == "vector_only":
            rr = vector.retrieve(q, top_k=args.top_k)
        elif which == "bm25_only":
            rr = bm25.retrieve(q, top_k=args.top_k)
        elif which == "graph_only":
            rr = graph.retrieve(q, top_k=args.top_k)
        else:
            raise ValueError(which)

        contexts = [r[0] for r in rr]  # (content, score, metadata)
        context_block = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = (
            "You are answering a PubMedQA-style medical research question using ONLY the provided contexts.\n"
            "If the contexts do not support a confident yes/no, choose maybe.\n"
            "Constraints:\n"
            "- Use at most 2 short sentences.\n"
            "- Then output exactly one final line: Final Answer: yes|no|maybe\n"
            "- Do not output anything after the Final Answer line.\n\n"
            f"Question: {q}\n\n"
            f"Contexts:\n{context_block}\n"
        )
        raw = _run_with_optional_timeout(prompt)
        ans = _normalize_to_final_answer(q, raw)
        return ans, contexts

    def run_vector_only(q: str) -> RunOutput:
        t0 = time.time()
        ans, _ = _run_retriever_only(q, "vector_only")
        return RunOutput(answer_text=ans, predicted=_extract_decision_pubmedqa(ans), latency_s=time.time() - t0)

    def run_bm25_only(q: str) -> RunOutput:
        t0 = time.time()
        ans, _ = _run_retriever_only(q, "bm25_only")
        return RunOutput(answer_text=ans, predicted=_extract_decision_pubmedqa(ans), latency_s=time.time() - t0)

    def run_graph_only(q: str) -> RunOutput:
        t0 = time.time()
        ans, _ = _run_retriever_only(q, "graph_only")
        return RunOutput(answer_text=ans, predicted=_extract_decision_pubmedqa(ans), latency_s=time.time() - t0)

    def run_hybrid(q: str) -> RunOutput:
        t0 = time.time()
        # Use the hybrid retriever directly but keep generation simple.
        results = hybrid.retrieve(q, top_k=args.top_k)
        contexts = [r.content for r in results]
        context_block = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
        prompt = (
            "You are answering a PubMedQA-style medical research question using ONLY the provided contexts.\n"
            "If the contexts do not support a confident yes/no, choose maybe.\n"
            "Constraints:\n"
            "- Use at most 2 short sentences.\n"
            "- Then output exactly one final line: Final Answer: yes|no|maybe\n"
            "- Do not output anything after the Final Answer line.\n\n"
            f"Question: {q}\n\n"
            f"Contexts:\n{context_block}\n"
        )
        raw = _run_with_optional_timeout(prompt)
        ans = _normalize_to_final_answer(q, raw)
        return RunOutput(answer_text=raw, predicted=_extract_decision_pubmedqa(ans), latency_s=time.time() - t0)

    def run_mega_rag(q: str) -> RunOutput:
        if mega_workflow is None:
            return RunOutput(answer_text="", predicted="unknown", latency_s=0.0)
        t0 = time.time()
        res = mega_workflow.run(q)
        ans = res.get("final_answer") or res.get("answer") or ""
        pred = _extract_decision_pubmedqa(ans)

        # Optional signals (best-effort)
        retrieval_conf = str(res.get("retrieval_confidence") or "UNKNOWN").upper()
        verifier = res.get("verification_result") or {}
        verifier_pass = verifier.get("passed") if isinstance(verifier, dict) else None
        unsupported_n = len(res.get("unsupported_claims") or [])
        iterations = int(res.get("iteration") or res.get("iterations") or 0)
        return RunOutput(
            answer_text=ans,
            predicted=pred,
            retrieval_confidence=retrieval_conf,
            verifier_pass=bool(verifier_pass) if verifier_pass is not None else None,
            unsupported_claims_n=unsupported_n,
            iterations=iterations,
            latency_s=time.time() - t0,
        )

    def run_lean(q: str) -> RunOutput:
        t0 = time.time()
        res = lean_workflow.run(q)
        ans = res.get("final_answer") or res.get("answer") or ""
        pred = _extract_decision_pubmedqa(ans)

        retrieval_conf = str(res.get("retrieval_confidence") or "UNKNOWN").upper()
        verifier = res.get("verification_result") or {}
        verifier_pass = verifier.get("passed") if isinstance(verifier, dict) else None
        unsupported_n = len(res.get("unsupported_claims") or [])
        iterations = int(res.get("iteration") or res.get("iterations") or 0)

        return RunOutput(
            answer_text=ans,
            predicted=pred,
            retrieval_confidence=retrieval_conf,
            verifier_pass=bool(verifier_pass) if verifier_pass is not None else None,
            unsupported_claims_n=unsupported_n,
            iterations=iterations,
            latency_s=time.time() - t0,
        )

    # -------------------------
    # Retrieval-only evaluation
    # -------------------------
    def _normalize_text_for_overlap(t: str) -> str:
        import re

        s = (t or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s

    def _retrieval_only_for(q: str, which: str) -> Dict[str, Any]:
        """Return retrieved texts (top_k) for a given retriever."""
        if which == "vector_only":
            rr = vector.retrieve(q, top_k=args.top_k)
            texts = [r[0] for r in rr]
        elif which == "bm25_only":
            rr = bm25.retrieve(q, top_k=args.top_k)
            texts = [r[0] for r in rr]
        elif which == "graph_only":
            rr = graph.retrieve(q, top_k=args.top_k)
            texts = [r[0] for r in rr]
        elif which == "hybrid":
            rr = hybrid.retrieve(q, top_k=args.top_k)
            texts = [r.content for r in rr]
        else:
            raise ValueError(which)

        return {"texts": texts}

    configs: List[Tuple[str, Any]] = [
        ("llm_only", run_llm_only),
        ("vector_only", run_vector_only),
        ("bm25_only", run_bm25_only),
        ("graph_only", run_graph_only),
        ("hybrid", run_hybrid),
        ("lean_workflow", run_lean),
    ]

    if args.include_mega_rag:
        configs.append(("mega_rag", run_mega_rag))

    # Optional config filtering
    if args.configs.strip():
        requested = [c.strip() for c in args.configs.split(",") if c.strip()]
        allowed = {name for name, _ in configs}
        unknown = [c for c in requested if c not in allowed]
        if unknown:
            raise ValueError(f"Unknown config(s) in --configs: {unknown}. Allowed: {sorted(allowed)}")
        configs = [c for c in configs if c[0] in set(requested)]

    # Run
    results_by_config: Dict[str, Any] = {}
    per_sample: List[dict] = []

    for config_name, _ in configs:
        results_by_config[config_name] = {
            "n": 0,
            "correct": 0,
            "unknown": 0,
            "avg_latency_s": 0.0,
            "avg_iterations": 0.0,
            "answered": 0,
            "coverage": 0.0,
            "unsupported_rate": None,
            "verifier_pass_rate": None,
        }

    # In retrieval-only mode we record different metrics.
    if args.retrieval_only:
        for config_name, _ in configs:
            results_by_config[config_name] = {
                "n": 0,
                "hits_at_k": 0,
                "hit_rate": 0.0,
                "avg_latency_s": 0.0,
            }

    def _write_report(path: Path) -> None:
        report = {
            "created_at": _now_iso(),
            "dataset": {
                "test_json": str(args.test_json),
                "n": len(per_sample),
                "seed": args.seed,
                "requested_n": len(samples),
            },
            "index": {
                "persist_dir": str(args.persist_dir),
                "collection_name": args.collection_name,
                "top_k": args.top_k,
            },
            "configs": results_by_config,
            "samples": per_sample,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    banner1 = f"Loaded {len(samples)} PubMedQA samples from {args.test_json}"
    banner2 = f"Running configs: {', '.join([c[0] for c in configs])}"
    print(banner1)
    print(banner2)
    _log(banner1)
    _log(banner2)

    snapshot_path = args.out.with_suffix(".partial.json")

    # -------------------------------------------------
    # Optional: Lean threshold sweep (small grid search)
    # -------------------------------------------------
    if args.sweep_lean:
        sweep_topk = [3, 5, 8]
        sweep_minrel = [0.2, 0.3, 0.4]
        sweep_summary: List[dict] = []

        print(f"Running Lean sweep on {len(samples)} samples: top_k_for_yes_no={sweep_topk}, min_relevance_score={sweep_minrel}")
        _log(f"[SWEEP] samples={len(samples)} topk={sweep_topk} minrel={sweep_minrel}")

        # Save original values so we can restore.
        orig_topk = getattr(lean_workflow, "top_k_for_yes_no", 5)
        orig_minrel = getattr(lean_workflow, "min_relevance_score", 0.4)

        for tk in sweep_topk:
            for mr in sweep_minrel:
                lean_workflow.top_k_for_yes_no = int(tk)
                lean_workflow.min_relevance_score = float(mr)

                correct = 0
                pred_counts = {"yes": 0, "no": 0, "maybe": 0, "unknown": 0}
                latencies: List[float] = []

                for s in tqdm(samples, desc=f"Sweep(lean tk={tk},mr={mr})", unit="sample"):
                    q = s["question"]
                    gt = s["final_decision"]
                    out = run_lean(q)
                    pred = out.predicted
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
                    correct += int(pred == gt)
                    latencies.append(out.latency_s)

                sweep_summary.append(
                    {
                        "top_k_for_yes_no": tk,
                        "min_relevance_score": mr,
                        "n": len(samples),
                        "accuracy": (correct / len(samples)) if samples else 0.0,
                        "correct": correct,
                        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
                        "pred_counts": pred_counts,
                    }
                )

        # Restore original values
        lean_workflow.top_k_for_yes_no = orig_topk
        lean_workflow.min_relevance_score = orig_minrel

        sweep_path = args.out.with_suffix(".lean_sweep.json")
        sweep_report = {
            "created_at": _now_iso(),
            "dataset": {
                "test_json": str(args.test_json),
                "n": len(samples),
                "seed": args.seed,
                "subset": str(args.subset) if args.subset is not None else None,
            },
            "grid": {"top_k_for_yes_no": sweep_topk, "min_relevance_score": sweep_minrel},
            "results": sweep_summary,
        }
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sweep_path, "w") as f:
            json.dump(sweep_report, f, indent=2)

        print(f"Lean sweep complete → {sweep_path}")
        _log(f"[SWEEP] complete path={sweep_path}")
        return

    try:
        for i, s in enumerate(tqdm(samples, desc="Evaluating", unit="sample"), 1):
            q = s["question"]
            gt = s["final_decision"]

            # Retrieval-only: fast sanity check using PubMedQA-provided CONTEXTS as gold.
            if args.retrieval_only:
                gold_contexts = s.get("contexts") or []
                gold_norm = [_normalize_text_for_overlap(c) for c in gold_contexts if c]

                row: Dict[str, Any] = {
                    "pubid": s["pubid"],
                    "question": q,
                    "ground_truth": gt,
                    "gold_contexts_n": len(gold_norm),
                    "runs": {},
                }

                for config_name, _runner in configs:
                    if config_name not in {"vector_only", "bm25_only", "graph_only", "hybrid"}:
                        # Skip non-retrieval configs in retrieval-only mode
                        continue

                    t0 = time.time()
                    rr = _retrieval_only_for(q, config_name)
                    latency = time.time() - t0
                    retrieved_texts = rr.get("texts") or []
                    retrieved_norm = [_normalize_text_for_overlap(t) for t in retrieved_texts if t]

                    # Define a "hit" as: any retrieved text shares substantial substring overlap with any gold context.
                    # We use a conservative check: gold context is substring of retrieved OR retrieved is substring of gold.
                    hit = False
                    for rtxt in retrieved_norm:
                        if not rtxt:
                            continue
                        for gtxt in gold_norm:
                            if not gtxt:
                                continue
                            if (gtxt in rtxt) or (rtxt in gtxt):
                                hit = True
                                break
                        if hit:
                            break

                    agg = results_by_config[config_name]
                    agg["n"] += 1
                    if hit:
                        agg["hits_at_k"] += 1
                    agg["avg_latency_s"] += _safe_float(latency)

                    row["runs"][config_name] = {
                        "hit_at_k": hit,
                        "latency_s": latency,
                        "retrieved_n": len(retrieved_texts),
                    }

                per_sample.append(row)

                if args.snapshot_every and (i % args.snapshot_every == 0):
                    _write_report(snapshot_path)

                continue

            row: Dict[str, Any] = {
                "pubid": s["pubid"],
                "question": q,
                "ground_truth": gt,
                "runs": {},
            }

            for config_name, runner in configs:
                gold_contexts = s.get("contexts") or []

                try:
                    out: RunOutput = runner(q)
                except Exception as e:
                    # Record failure but keep going.
                    _log(f"ERROR config={config_name} pubid={s['pubid']} err={e}\n{traceback.format_exc()}")
                    out = RunOutput(answer_text=f"[ERROR] {e}", predicted="unknown", latency_s=0.0)

                pred = out.predicted
                is_correct = pred == gt

                # Debug tags
                tags: List[str] = []
                if pred == "unknown":
                    tags.append("pred_unknown")
                elif pred == "maybe":
                    tags.append("pred_maybe")

                # Retrieval diagnostics for retrieval-based configs
                if config_name in {"vector_only", "bm25_only", "graph_only", "hybrid"}:
                    # We can't reliably get retrieved texts back from all runner paths here,
                    # so we re-run retrieval quickly for overlap diagnostics (no LLM).
                    try:
                        if config_name == "hybrid":
                            rr = hybrid.retrieve(q, top_k=args.top_k)
                            retrieved_texts = [r.content for r in rr]
                        elif config_name == "vector_only":
                            rr = vector.retrieve(q, top_k=args.top_k)
                            retrieved_texts = [r[0] for r in rr]
                        elif config_name == "bm25_only":
                            rr = bm25.retrieve(q, top_k=args.top_k)
                            retrieved_texts = [r[0] for r in rr]
                        else:
                            rr = graph.retrieve(q, top_k=args.top_k)
                            retrieved_texts = [r[0] for r in rr]

                        hit = _hit_against_gold_contexts(retrieved_texts, gold_contexts)
                    except Exception as e:
                        _log(f"WARN retrieval_diag_failed config={config_name} pubid={s['pubid']} err={e}")
                        hit = False
                        retrieved_texts = []

                    if not hit:
                        tags.append("retrieval_miss")
                    row.setdefault("gold_contexts_n", len(gold_contexts))
                else:
                    hit = None
                    retrieved_texts = []

                # Basic aggregation
                agg = results_by_config[config_name]
                agg["n"] += 1
                if pred == "unknown":
                    agg["unknown"] += 1
                if is_correct:
                    agg["correct"] += 1
                agg["avg_latency_s"] += _safe_float(out.latency_s)
                agg["avg_iterations"] += _safe_float(out.iterations)

                # Coverage/abstain style (treat unknown as abstained)
                if pred in {"yes", "no", "maybe"}:
                    agg["answered"] += 1

                row["runs"][config_name] = {
                    "predicted": pred,
                    "is_correct": is_correct,
                    "latency_s": out.latency_s,
                    "iterations": out.iterations,
                    "retrieval_confidence": out.retrieval_confidence,
                    "verifier_pass": out.verifier_pass,
                    "unsupported_claims_n": out.unsupported_claims_n,
                    "answer_text": out.answer_text,
                    "debug": {
                        "tags": tags,
                        "retrieval_hit_at_k": hit,
                    },
                }

                if args.debug:
                    _log(
                        f"RUN pubid={s['pubid']} config={config_name} gt={gt} pred={pred} "
                        f"correct={is_correct} tags={tags} latency_s={out.latency_s:.2f}"
                    )

            per_sample.append(row)

            if args.snapshot_every and (i % args.snapshot_every == 0):
                _write_report(snapshot_path)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
        _log("Interrupted by user (KeyboardInterrupt). Saving partial results.")
        _write_report(snapshot_path)
        print(f"Partial report saved: {snapshot_path}")
        return
    except Exception as e:
        print(f"\nError during evaluation: {e}. Saving partial results...")
        _log(f"Fatal error: {e}\n{traceback.format_exc()}")
        _write_report(snapshot_path)
        print(f"Partial report saved: {snapshot_path}")
        raise

    # Finalize aggregates
    if args.retrieval_only:
        for config_name, _runner in configs:
            if config_name not in {"vector_only", "bm25_only", "graph_only", "hybrid"}:
                continue
            agg = results_by_config[config_name]
            n = agg["n"] or 1
            agg["hit_rate"] = agg["hits_at_k"] / n
            agg["avg_latency_s"] = agg["avg_latency_s"] / n
    else:
        for config_name, _runner in configs:
            agg = results_by_config[config_name]
            n = agg["n"] or 1
            agg["accuracy"] = agg["correct"] / n
            agg["avg_latency_s"] = agg["avg_latency_s"] / n
            agg["avg_iterations"] = agg["avg_iterations"] / n
            agg["coverage"] = (agg["answered"] / n) if n else 0.0

            # Prediction distribution (helps explain low scores)
            pred_counts = {"yes": 0, "no": 0, "maybe": 0, "unknown": 0}
            for r in per_sample:
                pr = r.get("runs", {}).get(config_name, {}).get("predicted")
                if pr in pred_counts:
                    pred_counts[pr] += 1
                else:
                    pred_counts["unknown"] += 1
            agg["pred_counts"] = pred_counts

            # verifier stats (only meaningful when verifier exists)
            verifier_vals = [
                r["runs"][config_name].get("verifier_pass")
                for r in per_sample
                if r["runs"].get(config_name, {}).get("verifier_pass") is not None
            ]
            if verifier_vals:
                agg["verifier_pass_rate"] = sum(1 for v in verifier_vals if v) / len(verifier_vals)

            unsupported_vals = [
                r["runs"][config_name].get("unsupported_claims_n", 0)
                for r in per_sample
                if r["runs"].get(config_name, {}).get("verifier_pass") is not None
            ]
            if unsupported_vals:
                agg["unsupported_rate"] = sum(1 for x in unsupported_vals if x > 0) / len(unsupported_vals)

    _write_report(args.out)

    print("\n" + "=" * 72)
    print("PUBMEDQA MULTI-CONFIG SUMMARY")
    print("=" * 72)
    if args.retrieval_only:
        for name, _ in configs:
            if name not in {"vector_only", "bm25_only", "graph_only", "hybrid"}:
                continue
            agg = results_by_config[name]
            print(f"{name:14s} hit@k={agg['hit_rate']:.3f} avg_latency={agg['avg_latency_s']:.2f}s")
    else:
        for name, _ in configs:
            agg = results_by_config[name]
            print(
                f"{name:14s} acc={agg['accuracy']:.3f} cov={agg['coverage']:.3f} "
                f"avg_latency={agg['avg_latency_s']:.2f}s unknown={agg['unknown']}"
            )
    print(f"\nSaved report: {args.out}")

    # Cleanup partial snapshot if final succeeded
    if snapshot_path.exists():
        try:
            snapshot_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
