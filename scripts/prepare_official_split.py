#!/usr/bin/env python3
"""
PubMedQA Official Split Preparation

Creates splits that match the official PubMedQA benchmark protocol:
  - Test:  500 samples (FIXED — from official test_ground_truth.json)
  - Train: 450 samples (remaining, for fine-tuning)
  - Dev:   50 samples  (remaining, for hyperparameter tuning)

Also creates:
  - train_balanced.json: Class-balanced training set (for fine-tuning without bias)
  - Indexing documents: Contexts only (no answers) for RAG retrieval

References:
  - Jin et al. 2019: https://arxiv.org/abs/1909.06146
  - Official leaderboard: https://pubmedqa.github.io/
  - Standard metrics: Accuracy + Macro-F1 on 500 test samples

IMPORTANT: Published PubMedQA results use the provided context (oracle).
For RAG evaluation, report BOTH oracle-context and retrieved-context results.
"""

import json
import random
from pathlib import Path
from collections import Counter
from datetime import datetime


RANDOM_SEED = 42
DEV_SIZE = 50


def main():
    project_root = Path(__file__).parent.parent

    # Paths
    full_data_path = project_root / "pubmedQA" / "official" / "ori_pqal.json"
    test_gt_path = project_root / "pubmedQA" / "official_split" / "test_ground_truth.json"
    output_dir = project_root / "pubmedQA" / "splits"

    # Verify files exist
    if not full_data_path.exists():
        raise FileNotFoundError(
            f"Missing: {full_data_path}\n"
            "Download from: https://github.com/pubmedqa/pubmedqa"
        )
    if not test_gt_path.exists():
        raise FileNotFoundError(
            f"Missing: {test_gt_path}\n"
            "Run: curl -sL https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json "
            f"-o {test_gt_path}"
        )

    # Load data
    with open(full_data_path) as f:
        full_data = json.load(f)
    with open(test_gt_path) as f:
        test_gt = json.load(f)

    print(f"Full dataset: {len(full_data)} samples")
    print(f"Official test PMIDs: {len(test_gt)}")

    test_pmids = set(test_gt.keys())
    remaining_pmids = sorted(set(full_data.keys()) - test_pmids)

    # Shuffle remaining and split into train/dev
    random.seed(RANDOM_SEED)
    random.shuffle(remaining_pmids)
    dev_pmids = set(remaining_pmids[:DEV_SIZE])
    train_pmids = set(remaining_pmids[DEV_SIZE:])

    # Build split dicts
    test_data = {pmid: full_data[pmid] for pmid in test_pmids}
    dev_data = {pmid: full_data[pmid] for pmid in dev_pmids}
    train_data = {pmid: full_data[pmid] for pmid in train_pmids}

    # Print distributions
    for name, data in [("Train", train_data), ("Dev", dev_data), ("Test", test_data)]:
        dist = Counter((v.get("final_decision", "") or "").lower() for v in data.values())
        total = sum(dist.values())
        parts = " | ".join(f"{k}={v} ({v/total*100:.1f}%)" for k, v in sorted(dist.items()))
        print(f"  {name} ({total}): {parts}")

    # =========================================================================
    # Create BALANCED training set (for fine-tuning)
    # =========================================================================
    # Undersample majority classes to match minority class count
    by_class = {"yes": [], "no": [], "maybe": []}
    for pmid, sample in train_data.items():
        d = (sample.get("final_decision", "") or "").lower()
        if d in by_class:
            by_class[d].append(pmid)

    min_class_count = min(len(v) for v in by_class.values())
    print(f"\nBalanced training set: {min_class_count} per class = {min_class_count * 3} total")

    random.seed(RANDOM_SEED)
    balanced_pmids = []
    for cls, pmids in by_class.items():
        random.shuffle(pmids)
        balanced_pmids.extend(pmids[:min_class_count])
        print(f"  {cls}: {len(pmids)} → {min_class_count} (sampled)")

    random.shuffle(balanced_pmids)
    train_balanced = {pmid: train_data[pmid] for pmid in balanced_pmids}

    # Also create an OVERSAMPLED balanced set (repeat minority classes)
    max_class_count = max(len(v) for v in by_class.values())
    oversampled_pmids = []
    for cls, pmids in by_class.items():
        # Repeat + sample to reach max_class_count
        expanded = pmids.copy()
        while len(expanded) < max_class_count:
            expanded.extend(pmids)
        random.shuffle(expanded)
        oversampled_pmids.extend(expanded[:max_class_count])

    random.shuffle(oversampled_pmids)
    train_oversampled = {pmid: train_data[pmid] for pmid in oversampled_pmids}
    print(f"  Oversampled set: {len(train_oversampled)} total ({max_class_count}/class)")

    # =========================================================================
    # Save all splits
    # =========================================================================
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("train", train_data),
        ("dev", dev_data),
        ("test", test_data),
        ("train_balanced", train_balanced),
        ("train_oversampled", train_oversampled),
    ]:
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path.name}: {len(data)} samples")

    # =========================================================================
    # Indexing documents (contexts only, NO answers)
    # =========================================================================
    indexing_docs = []
    for pmid, sample in full_data.items():
        contexts = sample.get("CONTEXTS", [])
        labels = sample.get("LABELS", [])
        if contexts:
            if labels and len(labels) == len(contexts):
                text = "\n\n".join(f"[{l}] {c}" for l, c in zip(labels, contexts))
            else:
                text = "\n\n".join(contexts)
            indexing_docs.append({
                "pubid": pmid,
                "content": text,
                "question": sample.get("QUESTION", ""),
                "source": "pubmedqa",
                "type": "abstract",
            })

    idx_path = output_dir / "indexing_documents.json"
    with open(idx_path, "w") as f:
        json.dump(indexing_docs, f, indent=2)
    print(f"Saved indexing_documents.json: {len(indexing_docs)} docs")

    # =========================================================================
    # Save metadata
    # =========================================================================
    metadata = {
        "created": datetime.now().isoformat(),
        "protocol": "Official PubMedQA benchmark (Jin et al. 2019)",
        "source": "ori_pqal.json + test_ground_truth.json",
        "total_samples": 1000,
        "splits": {
            "train": len(train_data),
            "train_balanced": len(train_balanced),
            "train_oversampled": len(train_oversampled),
            "dev": len(dev_data),
            "test": len(test_data),
        },
        "test_note": "Official fixed 500-sample test set from pubmedqa.github.io",
        "balanced_note": f"Undersampled to {min_class_count}/class for unbiased fine-tuning",
        "oversampled_note": f"Oversampled minority classes to {max_class_count}/class",
        "metrics": "Accuracy + Macro-F1 (both required for publication)",
        "evaluation_modes": {
            "oracle_context": "Use provided CONTEXTS directly (comparable to leaderboard)",
            "rag_retrieval": "Retrieve contexts via RAG pipeline (harder, your contribution)",
        },
        "random_seed": RANDOM_SEED,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata.json")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("OFFICIAL SPLIT PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
Files created in {output_dir}/:
  train.json              — {len(train_data)} samples (natural distribution)
  train_balanced.json     — {len(train_balanced)} samples ({min_class_count}/class, for fine-tuning)
  train_oversampled.json  — {len(train_oversampled)} samples ({max_class_count}/class, alternative)
  dev.json                — {len(dev_data)} samples (hyperparameter tuning)
  test.json               — {len(test_data)} samples (OFFICIAL, never train on this!)
  indexing_documents.json — {len(indexing_docs)} docs (contexts only, for RAG index)

Evaluation protocol:
  1. Report Accuracy AND Macro-F1 on the 500-sample test set
  2. Report both oracle-context and RAG-retrieval results
  3. For fine-tuning, use train_balanced.json to avoid class bias
  4. Use dev.json for threshold tuning, NEVER test.json
""")


if __name__ == "__main__":
    main()
