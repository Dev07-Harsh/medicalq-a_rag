"""
MEGA-RAG Configuration Settings
Enhanced with Med-PaLM 2 inspired improvements
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_AUTO_FALLBACK = True

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  

# Ollama Configuration (Local Models)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron")  # Medical LLM: 74.9% PubMedQA baseline

# Ollama generation controls
# NOTE: Some local models (including Meditron) may become less faithful when allowed to generate
# very long completions. Keep this reasonably small and increase only if needed.
# Default lowered to speed up evaluation runs and reduce rambling outputs.
# You can override per-run: export OLLAMA_MAX_TOKENS=800
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "192"))

# =============================================================================
# DATA INTEGRITY SETTINGS (Critical for valid evaluation)
# =============================================================================
# Set to False to prevent data leakage during evaluation
# When False, ground truth answers (long_answer) are NOT indexed
INDEX_GROUND_TRUTH = False  # CRITICAL: Keep False for valid benchmarks

# Embedding Configuration - Using Medical Domain-Specific Model
# S-PubMedBert-MS-MARCO: Fine-tuned on PubMed + MS MARCO for medical retrieval
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768  # PubMedBERT uses 768 dimensions

# GPU Configuration
# Set USE_GPU=True to enable GPU acceleration (faster but uses more memory)
# Set to False if you encounter out-of-memory errors
USE_GPU = True
GPU_BATCH_SIZE = 2  

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SEMANTIC_THRESHOLD_PERCENTILE = 95  

# Retrieval Configuration
VECTOR_TOP_K = 30  
BM25_TOP_K = 30    
GRAPH_TOP_K = 5    # Graph ENABLED (Reduced from 8 to reduce noise)
RERANK_TOP_K = 15   # Increased to 15 to capture relevant docs at rank 10+

# Cross-Encoder Configuration
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Refinement Loop Configuration
MAX_REFINEMENT_ITERATIONS = 4  # Increased from 3 to give more correction attempts
SEAE_THRESHOLD = 0.35 # Lowered to 0.35 to reduce 'Maybe' bias
DISC_MAX_CORRECTIONS = 4  

# Re-retrieval Configuration
RE_RETRIEVAL_THRESHOLD_OFFSET = 0.15  # Trigger re-retrieval if score < SEAE_THRESHOLD - this

# Query Expansion Configuration
ENABLE_QUERY_EXPANSION = True  

# =============================================================================
# CHAIN-OF-THOUGHT CONFIGURATION (Med-PaLM 2 Inspired)
# =============================================================================
# Enable ensemble refinement with self-consistency for complex medical questions
ENABLE_CHAIN_OF_THOUGHT = True
COT_NUM_REASONING_PATHS = 3  # Number of diverse reasoning paths to generate
COT_SELF_CONSISTENCY = True  # Use self-consistency voting across paths
COT_ENSEMBLE_REFINEMENT = True  # Refine answer using ensemble of paths

# =============================================================================
# SELF-CONSISTENCY VOTING (Hallucination Reduction)
# =============================================================================
# Key technique from Self-RAG and Medprompt papers
ENABLE_SELF_CONSISTENCY = True  # Enable self-consistency voting for yes/no questions
SELF_CONSISTENCY_NUM_PATHS = 3  # Number of reasoning paths (higher = more robust, slower)
SELF_CONSISTENCY_MIN_AGREEMENT = 0.5  # Minimum agreement for confident answer

# Citation Verification
ENABLE_CITATION_VERIFICATION = True  # Verify citations against evidence post-generation

# Graph Configuration
GRAPH_SIMILARITY_THRESHOLD = 0.60  
SPACY_MODEL = "en_core_web_sm"  # Standard model for entity extraction (Neural Linking)

# Collection name for ChromaDB
COLLECTION_NAME = "mega_rag_medical"

# Dataset paths - PQA (Parquet)
PQA_ARTIFICIAL_PATH = BASE_DIR / "pqa_artificial.parquet"
PQA_LABELED_PATH = BASE_DIR / "pqa_labeled.parquet"

# PubMedQA Dataset paths (CSV)
PUBMEDQA_DIR = BASE_DIR / "pubmedQA"
PUBMEDQA_LABELED_PATH = PUBMEDQA_DIR / "pubmed_pqa_labeled.csv"
PUBMEDQA_ARTIFICIAL_PATH = PUBMEDQA_DIR / "pubmed_pqa_artificial.csv"
PUBMEDQA_UNLABELED_PATH = PUBMEDQA_DIR / "pqa_unlabeled.csv"

# Official PubMedQA-L splits (expert-labeled with yes/no/maybe)
# Created by scripts/prepare_pubmedqa_split.py from ori_pqal.json
PUBMEDQA_SPLITS_DIR = PUBMEDQA_DIR / "splits"
PUBMEDQA_OFFICIAL_TRAIN = PUBMEDQA_SPLITS_DIR / "train.json"     # 702 samples (70%)
PUBMEDQA_OFFICIAL_DEV = PUBMEDQA_SPLITS_DIR / "dev.json"         # 99 samples (10%)
PUBMEDQA_OFFICIAL_TEST = PUBMEDQA_SPLITS_DIR / "test.json"       # 199 samples (20%)
PUBMEDQA_INDEXING_DOCS = PUBMEDQA_SPLITS_DIR / "indexing_documents.json"  # 1000 contexts

# Evaluation Configuration
DEFAULT_EVAL_SAMPLES = 100  # Default sample size for evaluation

# PDF Knowledge base paths (can be extended)
KNOWLEDGE_BASE_PDFS = [
    BASE_DIR / "Guideline for the pharmacological treatment of hypertension in adults.pdf",
    BASE_DIR / "web annex A- summary of evidence.pdf",
    BASE_DIR / "who_web_annex_b.pdf",
]
