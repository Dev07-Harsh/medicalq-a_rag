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

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================
# Provider priority: Set LLM_PROVIDER to control the primary model.
# Supported: "gemini", "groq", "ollama", "auto"
#   - "gemini": Google Gemini API (free tier: 1,500 req/day)
#   - "groq":   Groq Cloud API (free tier: 14,400 req/day, ultra-fast)
#   - "ollama": Local models via Ollama (no API needed)
#   - "auto":   Smart fallback chain: Gemini → Groq → Ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")

# Auto-fallback: When primary hits rate limits or errors, try next provider
LLM_AUTO_FALLBACK = os.getenv("LLM_AUTO_FALLBACK", "true").lower() == "true"

# Fallback chain order (comma-separated). Only used when LLM_AUTO_FALLBACK=True
# Providers are tried in order; failed/unavailable ones are skipped.
LLM_FALLBACK_CHAIN = os.getenv("LLM_FALLBACK_CHAIN", "gemini,groq,ollama").split(",")

# =============================================================================
# GEMINI CONFIGURATION (Google AI Studio - Free Tier)
# =============================================================================
# Free tier: 15 RPM, 1,500 RPD, 1M TPM
# Get API key: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# =============================================================================
# GROQ CONFIGURATION (GroqCloud - Free Tier)
# =============================================================================
# Free tier: 30 RPM, 14,400 RPD, 6,000 TPM
# Get API key: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Default 8B: 500K TPD. Set to llama-3.3-70b-versatile for quality
# Available free models:
#   - "llama-3.3-70b-versatile"       (best quality, 6K TPM)
#   - "deepseek-r1-distill-llama-70b" (strong reasoning)
#   - "llama-3.1-8b-instant"          (fastest, 6K TPM)
#   - "gemma2-9b-it"                  (good balance, 15K TPM)
#   - "mixtral-8x7b-32768"            (32K context, 5K TPM)
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "512"))  # Reduced to conserve daily token quota

# =============================================================================
# OLLAMA CONFIGURATION (Local Models)
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "meditron")
# Recommended local models (sorted by quality, all fit 8GB RAM with Q4):
#   - "meditron"       (7B, medical-specific, 74.9% PubMedQA baseline)
#   - "biomistral"     (7B, PubMed-trained)
#   - "llama3.1:8b"    (8B, best general reasoning)
#   - "phi4-mini"      (3.8B, great for limited RAM)
#   - "qwen2.5:7b"     (7B, strong multilingual)
#   - "gemma2:2b"      (2.6B, ultra-lightweight)
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))

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

# =============================================================================
# CONTEXTUAL RETRIEVAL (Anthropic-inspired)
# =============================================================================
# Prepend a short LLM-generated context to each chunk before embedding.
# This makes chunks self-contained and improves retrieval by up to 67%.
# The context is used only for embedding — the raw chunk is shown to the LLM.
# Ref: https://www.anthropic.com/news/contextual-retrieval
ENABLE_CONTEXTUAL_CHUNKING = os.getenv("ENABLE_CONTEXTUAL_CHUNKING", "true").lower() == "true"

# Retrieval Configuration
VECTOR_TOP_K = 30  
BM25_TOP_K = 30    
GRAPH_TOP_K = 5    # Graph ENABLED (Reduced from 8 to reduce noise)
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "6"))  # Chunks sent to LLM after reranking (lower = less tokens)

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
# NOTE: Each path = 1 extra LLM call. With Groq 28 RPM, 3 paths means ~3x slower.
# Set to False or reduce paths when running large evaluations with rate-limited APIs.
ENABLE_SELF_CONSISTENCY = os.getenv("ENABLE_SELF_CONSISTENCY", "true").lower() == "true"
SELF_CONSISTENCY_NUM_PATHS = int(os.getenv("SELF_CONSISTENCY_NUM_PATHS", "3"))
SELF_CONSISTENCY_MIN_AGREEMENT = 0.5  # Minimum agreement for confident answer

# Citation Verification
ENABLE_CITATION_VERIFICATION = os.getenv("ENABLE_CITATION_VERIFICATION", "true").lower() == "true"

# Graph Configuration
GRAPH_SIMILARITY_THRESHOLD = 0.60  
SPACY_MODEL = "en_core_web_sm"  # Standard model for entity extraction (Neural Linking)
# Use scispaCy for biomedical NER when available (better medical entity extraction)
# Install: pip install scispacy && pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
# Falls back to en_core_web_sm if not installed
SPACY_MODEL_BIOMEDICAL = os.getenv("SPACY_MODEL_BIOMEDICAL", "en_core_sci_lg")
ENABLE_BIOMEDICAL_NER = os.getenv("ENABLE_BIOMEDICAL_NER", "true").lower() == "true"

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
