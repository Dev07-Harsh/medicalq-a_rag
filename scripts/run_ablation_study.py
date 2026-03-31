#!/usr/bin/env python3
"""
MEGA-RAG Ablation Study

Runs systematic evaluation across multiple configurations to measure
each component's contribution. Generates a comparison table for the paper.

Usage:
  # Full ablation (all configs, 500 test samples — takes ~6-10 hours)
  python scripts/run_ablation_study.py

  # Quick ablation (subset of configs, 50 samples — for debugging)
  python scripts/run_ablation_study.py --quick

  # Specific configs only
  python scripts/run_ablation_study.py --configs llm_only,oracle_context,lean_workflow

Output:
  evaluation_results/ablation_study.json   — Full results
  evaluation_results/ablation_summary.csv  — Summary table
  evaluation_results/ablation_latex.tex    — LaTeX table for paper
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


def run_eval(
    configs: str,
    test_json: str,
    sample_size: int,
    output_path: str,
    extra_args: list = None,
) -> dict:
    """Run evaluate_pubmedqa_multiconfig.py with given configs."""
    cmd = [
        sys.executable,
        "evaluate_pubmedqa_multiconfig.py",
        "--configs", configs,
        "--test-json", test_json,
        "--sample-size", str(sample_size),
        "--out", output_path,
        "--seed", "42",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {configs}")
    print(f"  Output: {output_path}")
    print(f"  Samples: {sample_size}")
    print(f"{'='*60}")

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, capture_output=False)

    if result.returncode != 0:
        print(f"  WARNING: {configs} exited with code {result.returncode}")
        return {}

    # Load results
    try:
        with open(output_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR reading results: {e}")
        return {}


def generate_latex_table(results: list) -> str:
    """Generate a LaTeX table from ablation results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Ablation Study: Component Contributions on PubMedQA (Official 500-sample test set)}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Accuracy & Macro-F1 & Avg Latency (s) & Coverage \\",
        r"\midrule",
    ]

    for r in results:
        name = r["config"].replace("_", r"\_")
        acc = f"{r['accuracy']:.1%}"
        f1 = f"{r['macro_f1']:.3f}" if r.get("macro_f1") else "—"
        latency = f"{r['avg_latency']:.1f}"
        coverage = f"{r['coverage']:.1%}"
        lines.append(f"  {name} & {acc} & {f1} & {latency} & {coverage} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_csv(results: list) -> str:
    """Generate CSV summary."""
    lines = ["config,accuracy,macro_f1,avg_latency_s,coverage,correct,total,pred_yes,pred_no,pred_maybe"]
    for r in results:
        preds = r.get("pred_counts", {})
        lines.append(
            f"{r['config']},{r['accuracy']:.4f},{r.get('macro_f1', 0):.4f},"
            f"{r['avg_latency']:.2f},{r['coverage']:.4f},"
            f"{r['correct']},{r['total']},"
            f"{preds.get('yes', 0)},{preds.get('no', 0)},{preds.get('maybe', 0)}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="MEGA-RAG Ablation Study")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 50 samples, fewer configs")
    parser.add_argument("--sample-size", type=int, default=None, help="Override sample size")
    parser.add_argument("--test-json", type=str, default="pubmedQA/splits/test.json")
    parser.add_argument("--configs", type=str, default=None, help="Comma-separated configs to run")
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation_results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Define ablation configurations
    if args.configs:
        config_list = args.configs.split(",")
    elif args.quick:
        config_list = ["llm_only", "oracle_context", "hybrid", "lean_workflow"]
    else:
        config_list = [
            "llm_only",
            "oracle_context",
            "vector_only",
            "bm25_only",
            "graph_only",
            "hybrid",
            "lean_workflow",
        ]

    sample_size = args.sample_size or (50 if args.quick else 500)

    print("=" * 60)
    print("MEGA-RAG ABLATION STUDY")
    print("=" * 60)
    print(f"Configs: {config_list}")
    print(f"Test set: {args.test_json}")
    print(f"Sample size: {sample_size}")
    print(f"Started: {datetime.now().isoformat()}")

    # Run each config
    all_results = []

    for config_name in config_list:
        out_path = args.out_dir / f"ablation_{config_name}.json"
        extra = []
        if config_name == "mega_rag":
            extra = ["--include-mega-rag"]

        data = run_eval(
            configs=config_name,
            test_json=args.test_json,
            sample_size=sample_size,
            output_path=str(out_path),
            extra_args=extra,
        )

        if data and "configs" in data:
            cfg_data = data["configs"].get(config_name, {})
            all_results.append({
                "config": config_name,
                "accuracy": cfg_data.get("accuracy", 0),
                "macro_f1": cfg_data.get("macro_f1", 0),
                "avg_latency": cfg_data.get("avg_latency_s", 0),
                "coverage": cfg_data.get("coverage", 0),
                "correct": cfg_data.get("correct", 0),
                "total": cfg_data.get("n", 0),
                "pred_counts": cfg_data.get("pred_counts", {}),
                "per_class_f1": cfg_data.get("per_class_f1", {}),
            })

    # Sort by accuracy
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # Save combined results
    combined = {
        "created_at": datetime.now().isoformat(),
        "test_json": args.test_json,
        "sample_size": sample_size,
        "results": all_results,
    }
    combined_path = args.out_dir / "ablation_study.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Generate summary outputs
    csv_text = generate_csv(all_results)
    csv_path = args.out_dir / "ablation_summary.csv"
    with open(csv_path, "w") as f:
        f.write(csv_text)

    latex_text = generate_latex_table(all_results)
    latex_path = args.out_dir / "ablation_latex.tex"
    with open(latex_path, "w") as f:
        f.write(latex_text)

    # Print summary
    print("\n" + "=" * 72)
    print("ABLATION STUDY RESULTS")
    print("=" * 72)
    print(f"{'Config':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Latency':>10} {'Coverage':>10}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['config']:<20} {r['accuracy']:>9.1%} {r.get('macro_f1', 0):>10.3f} "
              f"{r['avg_latency']:>9.1f}s {r['coverage']:>9.1%}")
    print("=" * 72)
    print(f"\nSaved: {combined_path}")
    print(f"       {csv_path}")
    print(f"       {latex_path}")


if __name__ == "__main__":
    main()
