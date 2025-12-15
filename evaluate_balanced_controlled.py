#!/usr/bin/env python3
"""evaluate_balanced_controlled.py

Balanced PubMedQA evaluation that uses a *controlled PubMedQA-only* index and
reports abstention + hallucination metrics.

Key metrics
-----------
- coverage: fraction of questions answered (not abstained)
- accuracy_answered: accuracy computed only on answered questions
- unsupported_rate: fraction of answered questions with >=1 unsupported claim
  (from LEAN workflow verifier output)

This evaluation loads questions from `pubmedQA/splits/test.json`.

It expects a controlled index created by `build_pubmedqa_index.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple


def load_balanced_samples(test_json: Path, n_per_class: int = 20) -> List[dict]:
    with open(test_json, "r") as f:
        data = json.load(f)

    by_label = {"yes": [], "no": [], "maybe": []}
    for pubid, item in data.items():
        label = (item.get("final_decision", "") or "").lower()
        if label not in by_label:
            continue

        by_label[label].append(
            {
                "pubid": pubid,
                "question": item.get("QUESTION", ""),
                "ground_truth": label,
            }
        )

    samples: List[dict] = []
    for label in ["yes", "no", "maybe"]:
        pool = by_label[label]
        k = min(n_per_class, len(pool))
        samples.extend(random.sample(pool, k))

    random.shuffle(samples)
    return samples


def parse_label(text: str) -> str:
    t = (text or "").strip().lower()
    if "final answer: yes" in t or t.startswith("yes"):
        return "yes"
    if "final answer: no" in t or t.startswith("no"):
        return "no"
    if "final answer: maybe" in t or t.startswith("maybe"):
        return "maybe"
    return "unknown"


def _safe_upper(x: object) -> str:
    return (str(x) if x is not None else "").upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced evaluation using controlled PubMedQA-only index")
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-json", type=Path, default=Path("pubmedQA/splits/test.json"))

    parser.add_argument("--persist-dir", type=Path, default=Path("chroma_pubmedqa_only"))
    parser.add_argument("--collection-name", type=str, default="pubmedqa_only")

    # Abstention knobs
    parser.add_argument(
        "--abstain-on",
        choices=["never", "low_retrieval", "low_or_verifier"],
        default="low_or_verifier",
        help="When to abstain instead of forcing yes/no/maybe",
    )

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)

    if not args.test_json.exists():
        raise FileNotFoundError(f"Missing test split: {args.test_json}")

    samples = load_balanced_samples(args.test_json, n_per_class=args.n_per_class)
    print(f"Loaded {len(samples)} balanced samples ({args.n_per_class}/class)")

    # Local imports (slow)
    from mega_rag.core.lean_workflow import LeanMEGARAGWorkflow
    from mega_rag.core.llm import OllamaLLM
    from mega_rag.retrieval.vector_retriever import VectorRetriever
    from mega_rag.retrieval.bm25_retriever import BM25Retriever
    from mega_rag.retrieval.graph_retriever import GraphRetriever
    from mega_rag.retrieval.hybrid_retriever import HybridRetriever

    persist_dir: Path = args.persist_dir
    bm25_path = persist_dir / "bm25_index.pkl"
    graph_path = persist_dir / "knowledge_graph.pkl"

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Controlled persist dir not found: {persist_dir}\n"
            "Run: python build_pubmedqa_index.py"
        )
    if not bm25_path.exists() or not graph_path.exists():
        raise FileNotFoundError(
            f"Controlled BM25/Graph artifacts not found in: {persist_dir}\n"
            "Run: python build_pubmedqa_index.py --force"
        )

    retriever = HybridRetriever()
    retriever.vector_retriever = VectorRetriever(
        collection_name=args.collection_name,
        persist_directory=persist_dir,
    )
    retriever.bm25_retriever = BM25Retriever(index_path=bm25_path)
    retriever.graph_retriever = GraphRetriever(graph_path=graph_path)

    print("Loading controlled indices...")
    if not retriever.load_indices():
        raise RuntimeError("Failed to load controlled indices")

    llm = OllamaLLM(model_name="meditron")
    workflow = LeanMEGARAGWorkflow(retriever=retriever, llm=llm, debug=args.debug)

    totals = {"yes": 0, "no": 0, "maybe": 0}
    correct = {"yes": 0, "no": 0, "maybe": 0}

    answered = 0
    correct_answered = 0
    abstained = 0
    unsupported_answered = 0

    t0 = time.time()

    for i, s in enumerate(samples, 1):
        q = s["question"]
        gt = s["ground_truth"]
        totals[gt] += 1

        res = workflow.run(q)
        pred_text = res.get("final_answer", res.get("answer", ""))
        pred = parse_label(pred_text)

        retrieval_conf = _safe_upper(res.get("retrieval_confidence", ""))
        if not retrieval_conf:
            retrieval_conf = "UNKNOWN"
        ver = res.get("verification_result") or {}
        verifier_pass = bool(ver.get("passed", False))
        unsupported_claims = res.get("unsupported_claims") or []

        do_abstain = False
        if args.abstain_on == "low_retrieval":
            do_abstain = retrieval_conf == "LOW"
        elif args.abstain_on == "low_or_verifier":
            do_abstain = (retrieval_conf == "LOW") or (not verifier_pass)

        if do_abstain:
            abstained += 1
            status = "ABSTAIN"
        else:
            answered += 1
            is_correct = pred == gt
            if is_correct:
                correct[gt] += 1
                correct_answered += 1
            if unsupported_claims:
                unsupported_answered += 1
            status = "✓" if is_correct else "✗"

        print(
            f"[{i:03d}/{len(samples)}] GT={gt:5s} pred={pred:7s} "
            f"conf={retrieval_conf:7s} verifier={'PASS' if verifier_pass else 'FAIL'} -> {status}"
        )

    elapsed = time.time() - t0

    total_n = sum(totals.values())
    overall_acc = sum(correct.values()) / total_n if total_n else 0.0
    coverage = answered / total_n if total_n else 0.0
    acc_answered = correct_answered / answered if answered else 0.0
    unsupported_rate = unsupported_answered / answered if answered else 0.0

    print("\n" + "=" * 72)
    print("CONTROLLED BALANCED EVAL SUMMARY")
    print("=" * 72)
    print(f"Samples            : {total_n}")
    print(f"Overall accuracy   : {overall_acc:.3f}")
    print(f"Coverage           : {coverage:.3f}  (answered={answered}, abstained={abstained})")
    print(f"Accuracy@answered  : {acc_answered:.3f}")
    print(f"Unsupported rate   : {unsupported_rate:.3f}")
    print(f"Time               : {elapsed:.1f}s ({elapsed/total_n:.1f}s/sample)")

    print("\nPer-class accuracy (on answered only):")
    for label in ["yes", "no", "maybe"]:
        denom = totals[label]
        acc = (correct[label] / denom) if denom else 0.0
        print(f"  {label.upper():5s}: {correct[label]}/{denom} = {acc:.3f}")


if __name__ == "__main__":
    main()
