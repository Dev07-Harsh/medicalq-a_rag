"""
Data Loader for various medical QA datasets.
Supports PQA, PubMedQA, and other formats.
Includes official PubMedQA-L splits (expert-labeled with yes/no/maybe).
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass

from mega_rag.config import (
    BASE_DIR, PQA_ARTIFICIAL_PATH, PQA_LABELED_PATH,
    PUBMEDQA_DIR, PUBMEDQA_LABELED_PATH, PUBMEDQA_ARTIFICIAL_PATH, PUBMEDQA_UNLABELED_PATH,
    PUBMEDQA_SPLITS_DIR
)

# Import Document for creating indexable documents
from mega_rag.core.document_processor import Document


@dataclass
class QASample:
    """Represents a single QA sample."""
    question: str
    ground_truth: str
    knowledge: Optional[str] = None
    difficulty: Optional[str] = None
    hallucinated_answer: Optional[str] = None
    hallucination_category: Optional[str] = None
    source: str = "unknown"
    # Additional fields for PubMedQA official format
    pubmed_id: Optional[str] = None
    final_decision: Optional[str] = None  # yes/no/maybe
    long_answer: Optional[str] = None
    contexts: Optional[List[str]] = None
    mesh_terms: Optional[List[str]] = None
    labels: Optional[List[str]] = None  # BACKGROUND, METHODS, RESULTS, CONCLUSIONS


class PQALoader:
    """Load PQA (PharmacoQA) datasets."""

    def __init__(self):
        self.artificial_path = PQA_ARTIFICIAL_PATH
        self.labeled_path = PQA_LABELED_PATH

    def load_artificial(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load artificial PQA dataset (9000 samples)."""
        return self._load_parquet(self.artificial_path, sample_size, "pqa_artificial")

    def load_labeled(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load labeled PQA dataset (1000 samples)."""
        return self._load_parquet(self.labeled_path, sample_size, "pqa_labeled")

    def _load_parquet(
        self,
        path: Path,
        sample_size: Optional[int],
        source: str
    ) -> List[QASample]:
        """Load samples from parquet file."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_parquet(path)

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)

        samples = []
        for _, row in df.iterrows():
            sample = QASample(
                question=row.get('Question', ''),
                ground_truth=row.get('Ground Truth', ''),
                knowledge=row.get('Knowledge', None),
                difficulty=row.get('Difficulty Level', None),
                hallucinated_answer=row.get('Hallucinated Answer', None),
                hallucination_category=row.get('Category of Hallucination', None),
                source=source
            )
            samples.append(sample)

        return samples


class PubMedQALoader:
    """
    Load PubMedQA dataset.
    Download from: https://github.com/pubmedqa/pubmedqa
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (BASE_DIR / "pubmedqa")

    def load_pqal(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PQA-Labeled subset."""
        path = self.data_dir / "data" / "pqal" / "pqal_fold0" / "test_set.json"
        return self._load_json(path, sample_size, "pubmedqa_labeled")

    def load_pqau(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PQA-Unlabeled subset."""
        path = self.data_dir / "data" / "pqau" / "pqau.json"
        return self._load_json(path, sample_size, "pubmedqa_unlabeled")

    def load_pqaa(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PQA-Artificial subset."""
        path = self.data_dir / "data" / "pqaa" / "pqaa.json"
        return self._load_json(path, sample_size, "pubmedqa_artificial")

    def _load_json(
        self,
        path: Path,
        sample_size: Optional[int],
        source: str
    ) -> List[QASample]:
        """Load samples from PubMedQA JSON format."""
        if not path.exists():
            print(f"PubMedQA file not found: {path}")
            print("Download from: https://github.com/pubmedqa/pubmedqa")
            return []

        with open(path, 'r') as f:
            data = json.load(f)

        samples = []
        items = list(data.items())

        if sample_size:
            import random
            random.seed(42)
            items = random.sample(items, min(sample_size, len(items)))

        for pmid, item in items:
            # Combine context from abstracts
            contexts = item.get('CONTEXTS', [])
            knowledge = " ".join(contexts) if contexts else None

            # Get the question
            question = item.get('QUESTION', '')

            # Get answer/label
            final_decision = item.get('final_decision', '')
            long_answer = item.get('LONG_ANSWER', '')
            ground_truth = long_answer if long_answer else final_decision

            sample = QASample(
                question=question,
                ground_truth=ground_truth,
                knowledge=knowledge,
                source=source
            )
            samples.append(sample)

        return samples


class PubMedQACSVLoader:
    """
    Load PubMedQA dataset from CSV files.
    Handles the CSV format with JSON context field.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or PUBMEDQA_DIR

    def load_labeled(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PubMedQA labeled CSV (has yes/no/maybe labels)."""
        path = self.data_dir / "pubmed_pqa_labeled.csv"
        return self._load_csv(path, sample_size, "pubmedqa_csv_labeled", has_label=True)

    def load_artificial(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PubMedQA artificial CSV (has yes/no labels)."""
        path = self.data_dir / "pubmed_pqa_artificial.csv"
        return self._load_csv(path, sample_size, "pubmedqa_csv_artificial", has_label=True)

    def load_unlabeled(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load PubMedQA unlabeled CSV (no labels)."""
        path = self.data_dir / "pqa_unlabeled.csv"
        return self._load_csv(path, sample_size, "pubmedqa_csv_unlabeled", has_label=False)

    def _parse_context_field(self, context_str: str) -> List[str]:
        """Parse the context field which contains a JSON-like dict with contexts array."""
        import ast
        import re

        if not context_str or context_str == 'nan':
            return []

        try:
            # The context field is a string representation of a dict
            # e.g., "{'contexts': array(['text1', 'text2', ...], dtype=object)}"
            # We need to handle the numpy array notation

            # Remove 'array(' and handle the closing
            cleaned = context_str.replace("array(", "")
            # Remove dtype=... and trailing )
            cleaned = re.sub(r",\s*dtype=\w+\)", "", cleaned)
            # Remove any remaining trailing )
            cleaned = re.sub(r"\)\s*}$", "}", cleaned)

            context_dict = ast.literal_eval(cleaned)
            contexts = context_dict.get('contexts', [])
            if isinstance(contexts, (list, tuple)):
                return [str(c) for c in contexts]
            return [str(contexts)]
        except Exception:
            # Second fallback: try to extract text between quotes
            try:
                # Extract strings that look like context paragraphs
                matches = re.findall(r"'([^']{50,})'", context_str)
                if matches:
                    return matches
            except Exception:
                pass
            # Final fallback: return as single context if it looks like text
            if context_str and len(context_str) > 100 and '{' not in context_str[:20]:
                return [context_str]
            return []

    def _load_csv(
        self,
        path: Path,
        sample_size: Optional[int],
        source: str,
        has_label: bool = True
    ) -> List[QASample]:
        """Load samples from PubMedQA CSV format."""
        if not path.exists():
            raise FileNotFoundError(f"PubMedQA CSV not found: {path}")

        # Read CSV with sampling
        if sample_size:
            # For large files, read in chunks and sample
            df = pd.read_csv(path, nrows=sample_size * 10)  # Read more to ensure variety
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        else:
            df = pd.read_csv(path)

        samples = []
        for _, row in df.iterrows():
            # Parse context field
            context_str = row.get('context', '')
            contexts = self._parse_context_field(str(context_str))
            knowledge = " ".join(contexts) if contexts else None

            # Get question
            question = str(row.get('question', ''))

            # Get ground truth (long_answer + final_decision if available)
            long_answer = str(row.get('long_answer', ''))
            final_decision = str(row.get('final_decision', '')) if has_label else None

            # Combine for ground truth
            if final_decision and final_decision != 'nan':
                ground_truth = f"{long_answer}\n\nFinal Answer: {final_decision}"
            else:
                ground_truth = long_answer

            sample = QASample(
                question=question,
                ground_truth=ground_truth,
                knowledge=knowledge,
                source=source
            )
            # Store additional metadata
            sample.final_decision = final_decision if has_label else None
            sample.pubid = str(row.get('pubid', ''))
            sample.long_answer = long_answer
            sample.contexts = contexts

            samples.append(sample)

        return samples

    def get_sample_with_context(self, sample_size: int = 10) -> List[Dict]:
        """Get samples with full context for testing retrieval."""
        samples = self.load_labeled(sample_size)
        return [
            {
                'pubid': getattr(s, 'pubid', ''),
                'question': s.question,
                'contexts': getattr(s, 'contexts', []),
                'long_answer': getattr(s, 'long_answer', ''),
                'final_decision': getattr(s, 'final_decision', ''),
                'ground_truth': s.ground_truth
            }
            for s in samples
        ]

    def extract_contexts_as_documents(
        self,
        sample_size: Optional[int] = None,
        dataset_type: str = 'labeled'
    ) -> List[Document]:
        """
        Extract PubMedQA contexts as indexable Document objects.

        This allows expanding the knowledge base with PubMedQA abstracts,
        enabling the system to answer questions on diverse medical topics.

        Args:
            sample_size: Number of samples to extract (None = all)
            dataset_type: 'labeled', 'artificial', or 'unlabeled'

        Returns:
            List of Document objects ready for indexing
            
        Note:
            Ground truth (long_answer) is NOT indexed by default to prevent data leakage.
            Set INDEX_GROUND_TRUTH=True in config.py only for experimental purposes.
        """
        from mega_rag.config import INDEX_GROUND_TRUTH
        
        # Load the appropriate dataset
        if dataset_type == 'labeled':
            samples = self.load_labeled(sample_size)
        elif dataset_type == 'artificial':
            samples = self.load_artificial(sample_size)
        elif dataset_type == 'unlabeled':
            samples = self.load_unlabeled(sample_size)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        documents = []

        for sample in samples:
            pubid = getattr(sample, 'pubid', 'unknown')
            contexts = getattr(sample, 'contexts', [])
            # NOTE: question is NOT stored in metadata to prevent retrieval shortcuts
            long_answer = getattr(sample, 'long_answer', '')

            # Each context (abstract section) becomes a separate document
            for ctx_idx, context in enumerate(contexts):
                if not context or len(context.strip()) < 50:
                    continue  # Skip empty or very short contexts

                doc = Document(
                    content=context,
                    metadata={
                        "source": f"PubMedQA:{pubid}",
                        "filename": f"pubmedqa_{pubid}.txt",
                        "chunk_index": ctx_idx,
                        "total_chunks": len(contexts),
                        "pubid": pubid,
                        # "question" field REMOVED to prevent retrieval shortcuts
                        "dataset": f"pubmedqa_{dataset_type}",
                        "page": 1  # PubMedQA contexts don't have pages
                    },
                    doc_id=f"pubmedqa_{pubid}",
                    chunk_id=ctx_idx
                )
                documents.append(doc)

            # =================================================================
            # CRITICAL: Ground Truth Indexing Control
            # =================================================================
            # By default, long_answer (ground truth) is NOT indexed to prevent
            # data leakage during evaluation. The system must reason from 
            # contexts only, not from answer keys.
            # 
            # Set INDEX_GROUND_TRUTH=True in config.py ONLY for:
            # - Debugging purposes
            # - Ablation studies comparing with/without ground truth
            # =================================================================
            if INDEX_GROUND_TRUTH and long_answer and len(long_answer.strip()) >= 50:
                print(f"  ⚠️  WARNING: Indexing ground truth for {pubid} (DATA LEAKAGE RISK)")
                doc = Document(
                    content=f"Summary: {long_answer}",
                    metadata={
                        "source": f"PubMedQA:{pubid}:summary",
                        "filename": f"pubmedqa_{pubid}_summary.txt",
                        "chunk_index": len(contexts),
                        "total_chunks": len(contexts) + 1,
                        "pubid": pubid,
                        "dataset": f"pubmedqa_{dataset_type}",
                        "is_summary": True,
                        "is_ground_truth": True,  # Flag for identification
                        "page": 1
                    },
                    doc_id=f"pubmedqa_{pubid}_summary",
                    chunk_id=len(contexts)
                )
                documents.append(doc)

        return documents


def load_indexed_test_dataset(sample_size: Optional[int] = None) -> List[QASample]:
    """
    Load the PubMedQA test dataset that was saved during indexing.
    This ensures testing uses the same samples that are in the index.

    Args:
        sample_size: Number of samples to return (random subset). None = all samples.

    Returns:
        List of QASample objects from the indexed dataset
    """
    import json
    import random

    test_dataset_path = BASE_DIR / "pubmedqa_test_dataset.json"

    if not test_dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_dataset_path}\n"
            "Run indexing with --include-pubmedqa first to create this file."
        )

    with open(test_dataset_path, 'r') as f:
        samples_data = json.load(f)

    print(f"Loaded {len(samples_data)} samples from indexed test dataset")

    # Random sample if requested
    if sample_size and sample_size < len(samples_data):
        random.seed(42)  # For reproducibility
        samples_data = random.sample(samples_data, sample_size)

    # Convert to QASample objects
    samples = []
    for data in samples_data:
        sample = QASample(
            question=data['question'],
            ground_truth=data['ground_truth'],
            knowledge=" ".join(data.get('contexts', [])),
            source='pubmedqa_indexed'
        )
        # Store additional attributes
        sample.pubid = data.get('pubid', '')
        sample.contexts = data.get('contexts', [])
        sample.long_answer = data.get('long_answer', '')
        sample.final_decision = data.get('final_decision', '')

        samples.append(sample)

    return samples


class OfficialPubMedQALoader:
    """
    Load official PubMedQA-L (expert-labeled) dataset from prepared splits.
    
    This loader uses the official ori_pqal.json from PubMedQA, which contains
    1000 expert-labeled samples with yes/no/maybe classes.
    
    The splits are prepared by scripts/prepare_pubmedqa_split.py:
    - train.json: 702 samples (70%) - for training/fine-tuning (if needed)
    - dev.json: 99 samples (10%) - for hyperparameter tuning
    - test.json: 199 samples (20%) - for final evaluation (NEVER peek during development)
    - indexing_documents.json: Context documents for ChromaDB indexing
    
    CRITICAL: Test set answers are NEVER indexed. Only contexts are indexed.
    """
    
    def __init__(self, splits_dir: Optional[Path] = None):
        self.splits_dir = splits_dir or PUBMEDQA_SPLITS_DIR
        
    def _validate_splits_exist(self):
        """Check if split files exist."""
        required_files = ['train.json', 'dev.json', 'test.json', 'indexing_documents.json']
        missing = [f for f in required_files if not (self.splits_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing split files: {missing}\n"
                f"Run: python scripts/prepare_pubmedqa_split.py to create splits."
            )
    
    def load_train(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load training split (702 samples)."""
        return self._load_split('train.json', sample_size, 'pubmedqa_official_train')
    
    def load_dev(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load development split (99 samples) - for hyperparameter tuning."""
        return self._load_split('dev.json', sample_size, 'pubmedqa_official_dev')
    
    def load_test(self, sample_size: Optional[int] = None) -> List[QASample]:
        """Load test split (199 samples) - for final evaluation ONLY."""
        return self._load_split('test.json', sample_size, 'pubmedqa_official_test')
    
    def _load_split(
        self,
        filename: str,
        sample_size: Optional[int],
        source: str
    ) -> List[QASample]:
        """Load samples from a split file."""
        filepath = self.splits_dir / filename
        
        if not filepath.exists():
            self._validate_splits_exist()  # Will raise with helpful message
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        items = list(data.items())
        
        if sample_size:
            import random
            random.seed(42)
            items = random.sample(items, min(sample_size, len(items)))
        
        for pmid, item in items:
            # Combine contexts
            contexts = item.get('CONTEXTS', [])
            labels = item.get('LABELS', [])
            knowledge = " ".join(contexts) if contexts else None
            
            # Get question and answers
            question = item.get('QUESTION', '')
            final_decision = item.get('final_decision', '')
            long_answer = item.get('LONG_ANSWER', '')
            mesh_terms = item.get('MESHES', [])
            
            # Ground truth includes both the explanation and the decision
            if long_answer:
                ground_truth = f"{long_answer}\n\nFinal Answer: {final_decision}"
            else:
                ground_truth = final_decision
            
            sample = QASample(
                question=question,
                ground_truth=ground_truth,
                knowledge=knowledge,
                source=source,
                pubmed_id=pmid,
                final_decision=final_decision,
                long_answer=long_answer,
                contexts=contexts,
                mesh_terms=mesh_terms,
                labels=labels
            )
            samples.append(sample)
        
        return samples
    
    def load_indexing_documents(self) -> List['Document']:
        """
        Load pre-prepared indexing documents for ChromaDB.
        
        These are CONTEXT-ONLY documents extracted from all 1000 PubMedQA samples.
        Ground truth answers are NOT included.
        
        Returns:
            List of Document objects ready for indexing
        """
        from mega_rag.core.document_processor import Document
        
        filepath = self.splits_dir / 'indexing_documents.json'
        
        if not filepath.exists():
            self._validate_splits_exist()
        
        with open(filepath, 'r') as f:
            docs_data = json.load(f)
        
        documents = []
        for idx, doc_data in enumerate(docs_data):
            # Handle the format from prepare_pubmedqa_split.py
            pubid = doc_data.get('pubid', f'unknown_{idx}')
            content = doc_data.get('content', '')
            question = doc_data.get('question', '')
            source = doc_data.get('source', 'pubmedqa')
            doc_type = doc_data.get('type', 'abstract')
            
            if not content or len(content.strip()) < 50:
                continue  # Skip empty or very short content
            
            doc = Document(
                content=content,
                metadata={
                    "source": f"PubMedQA:{pubid}",
                    "filename": f"pubmedqa_{pubid}.txt",
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "pubid": pubid,
                    "dataset": "pubmedqa_official",
                    "doc_type": doc_type,
                    "page": 1
                    # NOTE: question is NOT included to prevent retrieval shortcuts
                },
                doc_id=f"pubmedqa_{pubid}",
                chunk_id=0
            )
            documents.append(doc)
        
        print(f"📚 Loaded {len(documents)} indexing documents from official PubMedQA-L")
        return documents
    
    def get_class_distribution(self, split: str = 'test') -> Dict[str, int]:
        """Get class distribution for a split."""
        if split == 'train':
            samples = self.load_train()
        elif split == 'dev':
            samples = self.load_dev()
        elif split == 'test':
            samples = self.load_test()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        distribution = {'yes': 0, 'no': 0, 'maybe': 0}
        for sample in samples:
            decision = sample.final_decision.lower() if sample.final_decision else ''
            if decision in distribution:
                distribution[decision] += 1
        
        return distribution


class DatasetManager:
    """
    Unified interface for loading various medical QA datasets.
    """

    def __init__(self):
        self.pqa_loader = PQALoader()
        self.pubmedqa_loader = PubMedQALoader()
        self.pubmedqa_csv_loader = PubMedQACSVLoader()
        self.official_pubmedqa_loader = OfficialPubMedQALoader()

    def load_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None
    ) -> List[QASample]:
        """
        Load a dataset by name.

        Available datasets:
        - pqa_artificial: PQA artificial (9000 samples)
        - pqa_labeled: PQA labeled (1000 samples)
        - pubmedqa_labeled: PubMedQA labeled subset (JSON)
        - pubmedqa_unlabeled: PubMedQA unlabeled subset (JSON)
        - pubmedqa_artificial: PubMedQA artificial subset (JSON)
        - pubmedqa_csv_labeled: PubMedQA labeled CSV (~1.8M samples)
        - pubmedqa_csv_artificial: PubMedQA artificial CSV (~1.8M samples)
        - pubmedqa_csv_unlabeled: PubMedQA unlabeled CSV (~521K samples)
        - pubmedqa_indexed: PubMedQA samples saved during indexing (recommended for testing)
        - pubmedqa_official_train: Official PubMedQA-L train split (702 samples)
        - pubmedqa_official_dev: Official PubMedQA-L dev split (99 samples)
        - pubmedqa_official_test: Official PubMedQA-L test split (199 samples) ⭐ RECOMMENDED
        """
        # Special case for indexed dataset
        if dataset_name == 'pubmedqa_indexed':
            return load_indexed_test_dataset(sample_size)

        loaders = {
            'pqa_artificial': self.pqa_loader.load_artificial,
            'pqa_labeled': self.pqa_loader.load_labeled,
            'pubmedqa_labeled': self.pubmedqa_loader.load_pqal,
            'pubmedqa_unlabeled': self.pubmedqa_loader.load_pqau,
            'pubmedqa_artificial': self.pubmedqa_loader.load_pqaa,
            'pubmedqa_csv_labeled': self.pubmedqa_csv_loader.load_labeled,
            'pubmedqa_csv_artificial': self.pubmedqa_csv_loader.load_artificial,
            'pubmedqa_csv_unlabeled': self.pubmedqa_csv_loader.load_unlabeled,
            # Official PubMedQA-L splits (expert-labeled with yes/no/maybe)
            'pubmedqa_official_train': self.official_pubmedqa_loader.load_train,
            'pubmedqa_official_dev': self.official_pubmedqa_loader.load_dev,
            'pubmedqa_official_test': self.official_pubmedqa_loader.load_test,  # ⭐ RECOMMENDED
        }

        if dataset_name not in loaders:
            available = ', '.join(loaders.keys()) + ', pubmedqa_indexed'
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

        return loaders[dataset_name](sample_size)

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        available = []

        # Check PQA (Parquet)
        if PQA_ARTIFICIAL_PATH.exists():
            available.append('pqa_artificial')
        if PQA_LABELED_PATH.exists():
            available.append('pqa_labeled')

        # Check PubMedQA (JSON format)
        pubmedqa_dir = BASE_DIR / "pubmedqa" / "data"
        if pubmedqa_dir.exists():
            if (pubmedqa_dir / "pqal").exists():
                available.append('pubmedqa_labeled')
            if (pubmedqa_dir / "pqau").exists():
                available.append('pubmedqa_unlabeled')
            if (pubmedqa_dir / "pqaa").exists():
                available.append('pubmedqa_artificial')

        # Check PubMedQA CSV files
        if PUBMEDQA_LABELED_PATH.exists():
            available.append('pubmedqa_csv_labeled')
        if PUBMEDQA_ARTIFICIAL_PATH.exists():
            available.append('pubmedqa_csv_artificial')
        if PUBMEDQA_UNLABELED_PATH.exists():
            available.append('pubmedqa_csv_unlabeled')

        # Check for indexed test dataset
        test_dataset_path = BASE_DIR / "pubmedqa_test_dataset.json"
        if test_dataset_path.exists():
            available.append('pubmedqa_indexed')
        
        # Check for official PubMedQA-L splits (⭐ RECOMMENDED)
        if PUBMEDQA_SPLITS_DIR.exists():
            if (PUBMEDQA_SPLITS_DIR / 'train.json').exists():
                available.append('pubmedqa_official_train')
            if (PUBMEDQA_SPLITS_DIR / 'dev.json').exists():
                available.append('pubmedqa_official_dev')
            if (PUBMEDQA_SPLITS_DIR / 'test.json').exists():
                available.append('pubmedqa_official_test')  # ⭐ RECOMMENDED

        return available

    def get_combined_dataset(
        self,
        datasets: List[str],
        samples_per_dataset: Optional[int] = None
    ) -> List[QASample]:
        """Load and combine multiple datasets."""
        combined = []

        for dataset_name in datasets:
            try:
                samples = self.load_dataset(dataset_name, samples_per_dataset)
                combined.extend(samples)
                print(f"Loaded {len(samples)} samples from {dataset_name}")
            except Exception as e:
                print(f"Could not load {dataset_name}: {e}")

        return combined


def download_pubmedqa():
    """Download PubMedQA dataset."""
    import subprocess

    pubmedqa_dir = BASE_DIR / "pubmedqa"

    if pubmedqa_dir.exists():
        print(f"PubMedQA directory already exists: {pubmedqa_dir}")
        return

    print("Downloading PubMedQA dataset...")
    try:
        subprocess.run([
            "git", "clone",
            "https://github.com/pubmedqa/pubmedqa.git",
            str(pubmedqa_dir)
        ], check=True)
        print(f"Downloaded to: {pubmedqa_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        print("Manual download: git clone https://github.com/pubmedqa/pubmedqa.git")


if __name__ == "__main__":
    # Test data loading
    manager = DatasetManager()

    print("Available datasets:", manager.get_available_datasets())

    # Load PQA
    try:
        samples = manager.load_dataset('pqa_labeled', sample_size=5)
        print(f"\nLoaded {len(samples)} PQA samples")
        if samples:
            print(f"Sample question: {samples[0].question[:100]}...")
    except Exception as e:
        print(f"PQA loading error: {e}")

    # Load PubMedQA CSV
    try:
        samples = manager.load_dataset('pubmedqa_csv_labeled', sample_size=5)
        print(f"\nLoaded {len(samples)} PubMedQA CSV samples")
        if samples:
            print(f"Sample question: {samples[0].question[:100]}...")
            print(f"Final decision: {getattr(samples[0], 'final_decision', 'N/A')}")
            print(f"Contexts count: {len(getattr(samples[0], 'contexts', []))}")
    except Exception as e:
        print(f"PubMedQA CSV loading error: {e}")

    # Check for PubMedQA JSON
    if 'pubmedqa_labeled' not in manager.get_available_datasets():
        print("\nPubMedQA JSON not found. To download:")
        print("  python -c \"from mega_rag.utils.data_loader import download_pubmedqa; download_pubmedqa()\"")
