#!/usr/bin/env python3
"""
PubMedQA Dataset Preparation Script

This script prepares the official PubMedQA-L dataset for MEGA-RAG:
1. Loads the official 1000 expert-labeled samples (yes/no/maybe)
2. Creates stratified train/test splits
3. Extracts contexts for indexing (WITHOUT answers)
4. Saves test set separately for fair evaluation

Key Insight:
- ALL contexts are indexed (model needs evidence to answer)
- Answers (long_answer, final_decision) are NEVER indexed
- Test evaluation checks if model can reason from evidence to correct answer
"""
import json
import os
import random
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple
import csv

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 200  # Hold out 200 for testing (balanced across classes)
DEV_SIZE = 100   # Development set for tuning
TRAIN_SIZE = 700  # Rest for training/indexing


def load_official_pubmedqa(filepath: str) -> Dict:
    """Load the official PubMedQA-L JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples from official PubMedQA-L")
    return data


def analyze_distribution(data: Dict) -> Dict[str, int]:
    """Analyze class distribution."""
    decisions = [v.get('final_decision', 'unknown') for v in data.values()]
    dist = Counter(decisions)
    print("\nClass Distribution:")
    for cls, count in sorted(dist.items()):
        pct = count / len(data) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    return dict(dist)


def stratified_split(
    data: Dict,
    test_size: int = 200,
    dev_size: int = 100,
    seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Create stratified train/dev/test splits.
    
    Ensures each split has proportional representation of yes/no/maybe.
    """
    random.seed(seed)
    
    # Group by class
    by_class = {'yes': [], 'no': [], 'maybe': []}
    for pubid, sample in data.items():
        decision = sample.get('final_decision', 'unknown')
        if decision in by_class:
            by_class[decision].append((pubid, sample))
    
    # Calculate proportions
    total = len(data)
    test_samples = {}
    dev_samples = {}
    train_samples = {}
    
    for cls, samples in by_class.items():
        random.shuffle(samples)
        n_total = len(samples)
        
        # Proportional split
        n_test = max(1, int(n_total / total * test_size))
        n_dev = max(1, int(n_total / total * dev_size))
        
        # Assign to splits
        for i, (pubid, sample) in enumerate(samples):
            if i < n_test:
                test_samples[pubid] = sample
            elif i < n_test + n_dev:
                dev_samples[pubid] = sample
            else:
                train_samples[pubid] = sample
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Dev:   {len(dev_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")
    
    return train_samples, dev_samples, test_samples


def extract_indexing_documents(data: Dict) -> List[Dict]:
    """
    Extract documents for indexing from PubMedQA samples.
    
    CRITICAL: Only extracts CONTEXTS (abstracts), NOT answers!
    This ensures fair evaluation - model must reason from evidence.
    """
    documents = []
    
    for pubid, sample in data.items():
        # Get contexts (list of abstract paragraphs)
        contexts = sample.get('CONTEXTS', [])
        labels = sample.get('LABELS', [])
        question = sample.get('QUESTION', '')
        
        # Combine contexts into a single document
        # Each context paragraph becomes part of the document
        if contexts:
            # Create a rich document with structure
            context_text = "\n\n".join(contexts)
            
            # Add section labels if available
            if labels and len(labels) == len(contexts):
                labeled_sections = []
                for label, text in zip(labels, contexts):
                    labeled_sections.append(f"[{label}] {text}")
                context_text = "\n\n".join(labeled_sections)
            
            doc = {
                'pubid': pubid,
                'content': context_text,
                'question': question,  # Store question for reference (but don't index for retrieval)
                'source': 'pubmedqa',
                'type': 'abstract'
            }
            documents.append(doc)
    
    print(f"✓ Extracted {len(documents)} documents for indexing")
    return documents


def save_splits(
    train_data: Dict,
    dev_data: Dict,
    test_data: Dict,
    output_dir: Path
) -> None:
    """Save all splits to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (preserves full structure)
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"✓ Saved train.json ({len(train_data)} samples)")
    
    with open(output_dir / 'dev.json', 'w') as f:
        json.dump(dev_data, f, indent=2)
    print(f"✓ Saved dev.json ({len(dev_data)} samples)")
    
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"✓ Saved test.json ({len(test_data)} samples)")
    
    # Also save as CSV for easy viewing
    for name, data in [('train', train_data), ('dev', dev_data), ('test', test_data)]:
        csv_path = output_dir / f'{name}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pubid', 'question', 'final_decision', 'long_answer', 'num_contexts'])
            for pubid, sample in data.items():
                writer.writerow([
                    pubid,
                    sample.get('QUESTION', ''),
                    sample.get('final_decision', ''),
                    sample.get('LONG_ANSWER', '')[:200] + '...' if len(sample.get('LONG_ANSWER', '')) > 200 else sample.get('LONG_ANSWER', ''),
                    len(sample.get('CONTEXTS', []))
                ])
        print(f"✓ Saved {name}.csv")
    
    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'source': 'Official PubMedQA-L (ori_pqal.json)',
        'total_samples': len(train_data) + len(dev_data) + len(test_data),
        'splits': {
            'train': len(train_data),
            'dev': len(dev_data),
            'test': len(test_data)
        },
        'class_distribution': {
            'train': dict(Counter(s.get('final_decision') for s in train_data.values())),
            'dev': dict(Counter(s.get('final_decision') for s in dev_data.values())),
            'test': dict(Counter(s.get('final_decision') for s in test_data.values()))
        },
        'random_seed': RANDOM_SEED,
        'notes': [
            'All contexts are indexed for retrieval',
            'Answers (long_answer, final_decision) are NOT indexed',
            'Test set is held out for fair evaluation',
            'Dev set is for hyperparameter tuning'
        ]
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata.json")


def save_indexing_documents(documents: List[Dict], output_path: Path) -> None:
    """Save documents in format ready for indexing."""
    with open(output_path, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"✓ Saved {len(documents)} indexing documents to {output_path}")


def main():
    print("=" * 60)
    print("PubMedQA Dataset Preparation for MEGA-RAG")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'pubmedQA' / 'official' / 'ori_pqal.json'
    output_dir = project_root / 'pubmedQA' / 'splits'
    
    # Check if official dataset exists
    if not input_path.exists():
        print(f"❌ Official dataset not found at: {input_path}")
        print("Please download from: https://github.com/pubmedqa/pubmedqa")
        return
    
    # Load data
    data = load_official_pubmedqa(input_path)
    
    # Analyze distribution
    analyze_distribution(data)
    
    # Create stratified splits
    train_data, dev_data, test_data = stratified_split(
        data,
        test_size=TEST_SIZE,
        dev_size=DEV_SIZE,
        seed=RANDOM_SEED
    )
    
    # Verify split distributions
    print("\nVerifying split distributions:")
    for name, split_data in [('Train', train_data), ('Dev', dev_data), ('Test', test_data)]:
        dist = Counter(s.get('final_decision') for s in split_data.values())
        print(f"  {name}: {dict(dist)}")
    
    # Save splits
    print("\nSaving splits...")
    save_splits(train_data, dev_data, test_data, output_dir)
    
    # Extract and save indexing documents
    # Index ALL samples' contexts (train + dev + test)
    # This is correct because we only index CONTEXTS, not ANSWERS
    print("\nPreparing indexing documents...")
    all_data = {**train_data, **dev_data, **test_data}
    indexing_docs = extract_indexing_documents(all_data)
    save_indexing_documents(indexing_docs, output_dir / 'indexing_documents.json')
    
    print("\n" + "=" * 60)
    print("✅ PREPARATION COMPLETE")
    print("=" * 60)
    print(f"""
Next Steps:
1. Run indexing: python run.py --index
   - This will index WHO PDF + PubMedQA contexts
   
2. Run evaluation: python run.py --evaluate
   - Uses test.json (200 samples) for fair evaluation
   
3. For development/tuning: use dev.json (100 samples)

Key Files Created:
  {output_dir}/train.json  - {len(train_data)} samples for training
  {output_dir}/dev.json    - {len(dev_data)} samples for development
  {output_dir}/test.json   - {len(test_data)} samples for testing
  {output_dir}/indexing_documents.json - Contexts for ChromaDB indexing
""")


if __name__ == "__main__":
    main()
