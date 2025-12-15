# MEGA-RAG: Medical Evidence-Guided Augmentation

A hallucination-mitigation system for Medical QA using Tri-Brid Retrieval and Self-Correction.

## Quick Start

### 1. Activate Virtual Environment
```bash
cd "/Users/harsh/Downloads/Minor project"
source venv/bin/activate
```

### 2. Index Documents (First Time Only)
This processes PDFs and builds the retrieval indices:
```bash
python run.py --index
```
Note: This may take 5-10 minutes as it downloads the BGE-M3 embedding model (~2GB) and processes all PDFs.

### 3. Run the System

**Interactive Mode (Recommended):**
```bash
python run.py --interactive
```

**Single Query:**
```bash
python run.py --query "What is first-line treatment for hypertension?"
```

**Run Evaluation:**
```bash
python run.py --evaluate --eval-samples 50
```

## All Available Commands

| Command | Description |
|---------|-------------|
| `python run.py --index` | Index PDF documents (run first time) |
| `python run.py --index --include-pubmedqa` | Index PDFs + PubMedQA contexts (expanded coverage) |
| `python run.py --interactive` | Interactive Q&A mode |
| `python run.py --query "your question"` | Answer a single question |
| `python run.py --evaluate` | Run evaluation on PQA dataset |
| `python run.py --evaluate-pubmedqa` | Run evaluation on PubMedQA dataset |
| `python run.py --research-eval` | Run research paper evaluation (LaTeX, JSON, CSV) |
| `python run.py --test` | Run test suite |
| `python run.py --force-reindex` | Force re-index all documents |

### Expanding Knowledge Base with PubMedQA

By default, the knowledge base only includes WHO Hypertension PDFs. To expand coverage to answer diverse medical questions:

```bash
# Index PDFs + all PubMedQA labeled contexts
python run.py --index --include-pubmedqa --force-reindex

# Index with limited PubMedQA samples (faster)
python run.py --index --include-pubmedqa --pubmedqa-samples 500 --force-reindex
```

This adds PubMedQA abstracts to the knowledge base, enabling the system to answer questions on any medical topic in the PubMedQA dataset.

### Evaluation Options

```bash
# PQA Dataset evaluation
python run.py --evaluate --eval-samples 100

# PubMedQA Dataset evaluation (yes/no/maybe classification)
python run.py --evaluate-pubmedqa --eval-samples 100 --verbose

# Research paper evaluation (generates LaTeX tables)
python run.py --research-eval --eval-samples 200 --verbose

# Run test suite
python run.py --test
```

Results are saved to `evaluation_results/` folder as JSON and HTML reports.
Research results are saved to `research_results/` folder with LaTeX tables for papers.

## Controlled PubMedQA Benchmarking (recommended for research/evals)

The default index can include PDFs and other sources. For *retrieval benchmarking* (and to avoid
accidentally attributing errors to retrieval contamination), use the **controlled PubMedQA-only index**.

### 1. Build a PubMedQA-only index (contexts only)

This builds a separate Chroma persist directory plus BM25/graph artifacts. It indexes **only**
`pubmedQA/splits/indexing_documents.json` (context-only documents; no ground-truth answers).

```bash
python build_pubmedqa_index.py --persist-dir chroma_pubmedqa_only --collection-name pubmedqa_only --force
```

### 2. Run a balanced evaluation on the official test split

This evaluation reports standard accuracy *plus* abstention-style reliability metrics:

- **Coverage**: fraction of questions answered (not abstained)
- **Accuracy@answered**: accuracy on answered questions only
- **Unsupported rate**: fraction of answered questions where the verifier flagged unsupported claims

```bash
python evaluate_balanced_controlled.py --n-per-class 20 --persist-dir chroma_pubmedqa_only --collection-name pubmedqa_only
```

### “Industry-style” reporting tip

When comparing systems, report deltas against baselines rather than comparing directly with
fine-tuned leaderboard numbers (e.g., published PubMedQA results for *fine-tuned* models).

A good minimal report is:

- No-context baseline (LLM only)
- Gold-context baseline (LLM with provided PubMedQA contexts)
- RAG (retrieved contexts)

This repo includes `benchmark_fair.py` for baseline-style comparisons.

## Project Structure

```
mega_rag/
├── config.py              # Configuration (API keys, model settings)
├── main.py                # CLI entry point
├── core/
│   ├── document_processor.py  # Semantic chunking
│   ├── llm.py                 # LLM integration (Gemini + Ollama with auto-fallback)
│   └── workflow.py            # LangGraph orchestration
├── retrieval/
│   ├── vector_retriever.py    # ChromaDB + BGE-M3
│   ├── bm25_retriever.py      # BM25 keyword search
│   ├── graph_retriever.py     # NetworkX entity graph
│   └── hybrid_retriever.py    # Tri-Brid fusion
├── refinement/
│   ├── seae.py                # Hallucination auditor
│   └── disc.py                # Self-correction module
└── utils/
    ├── data_loader.py         # Dataset loaders (PQA + PubMedQA)
    ├── evaluation.py          # RAGAS evaluation
    ├── pubmedqa_evaluator.py  # PubMedQA evaluation metrics
    └── research_evaluation.py # Research paper metrics (LaTeX/CSV)

pubmedQA/                      # PubMedQA dataset (CSV)
├── pubmed_pqa_labeled.csv     # Labeled (yes/no/maybe)
├── pubmed_pqa_artificial.csv  # Auto-labeled
└── pqa_unlabeled.csv          # Unlabeled

tests/                         # Test suite
├── test_data_loader.py
├── test_pubmedqa_evaluator.py
└── test_integration.py
```

## Configuration

### LLM Provider (Gemini or Ollama)

The system supports two LLM providers with **auto-fallback**:

| Provider | Type | Setup |
|----------|------|-------|
| Gemini (default) | Cloud API | Set `GEMINI_API_KEY` in `.env` |
| Ollama (Mistral) | Local | Install Ollama + pull model |

**To switch providers**, edit `mega_rag/config.py`:
```python
LLM_PROVIDER = "gemini"  # Options: "gemini", "ollama"
```

**Auto-Fallback**: When Gemini hits rate limits (429), automatically switches to local Ollama.

### Setting up Ollama (Local Model)

1. **Install Ollama**: https://ollama.ai
2. **Pull Mistral model**:
   ```bash
   ollama pull mistral
   ```
3. **Start Ollama server** (runs automatically on macOS):
   ```bash
   ollama serve
   ```

Now the system will use Mistral locally when Gemini is unavailable.

### Environment Variables

Edit `.env` file:
```
GEMINI_API_KEY=your-api-key-here
LLM_PROVIDER=gemini          # or "ollama" for local
OLLAMA_MODEL=mistral         # or "llama3.1", "phi3"
```

Edit `mega_rag/config.py` to change:
- Model settings (GEMINI_MODEL, OLLAMA_MODEL)
- Embedding model (EMBEDDING_MODEL)
- Retrieval parameters (VECTOR_TOP_K, BM25_TOP_K, etc.)

## Troubleshooting

**API Rate Limit (429 Error):**
- System will auto-fallback to local Ollama if configured
- Or wait a few minutes and try again
- Or switch to Ollama: set `LLM_PROVIDER = "ollama"` in config.py

**Module Not Found:**
- Make sure virtual environment is activated: `source venv/bin/activate`

**Index Not Found:**
- Run `python run.py --index` first

## Author
HARSH SINGH
