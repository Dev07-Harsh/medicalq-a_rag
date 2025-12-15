# MEGA-RAG: Medical Evidence-Guided Augmentation

## Project Overview

**MEGA-RAG** (Medical Evidence-Guided Augmentation for Retrieval-Augmented Generation) is an advanced hallucination-mitigation system designed specifically for Medical Question Answering (QA). The system combines state-of-the-art retrieval techniques with self-correction mechanisms to ensure factually accurate and evidence-grounded medical responses.

### Author
**HARSH SINGH**

### Project Type
Minor Project - Medical AI / Natural Language Processing

---

## Problem Statement

### The Challenge of Medical AI Hallucinations

Large Language Models (LLMs) have shown remarkable capabilities in generating human-like text, but they suffer from a critical flaw: **hallucination** - generating plausible-sounding but factually incorrect information. In the medical domain, this problem is particularly dangerous because:

1. **Patient Safety**: Incorrect medical information can lead to harmful decisions
2. **Trust Deficit**: Healthcare professionals cannot rely on AI that generates unverified claims
3. **Liability Issues**: Medical misinformation can have legal consequences
4. **Knowledge Gaps**: LLMs may not have access to the latest medical guidelines

### Why Existing Solutions Fall Short

| Approach | Limitation |
|----------|------------|
| Basic RAG | Single retrieval method misses relevant information |
| Fine-tuning | Expensive, requires constant updates, still hallucinates |
| Prompt Engineering | Cannot verify factual accuracy |
| Simple Fact-Checking | Doesn't integrate with generation process |

---

## Solution: MEGA-RAG Architecture

MEGA-RAG addresses these challenges through a **three-pillar approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MEGA-RAG PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   TRI-BRID   │───▶│   GENERATE   │───▶│    REFINE    │      │
│  │  RETRIEVAL   │    │   (Gemini)   │    │  (SEAE+DISC) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐        │
│  │  Vector    │      │  Initial   │      │  Verified  │        │
│  │  + BM25    │      │  Response  │      │  Response  │        │
│  │  + Graph   │      │            │      │            │        │
│  └────────────┘      └────────────┘      └────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Tri-Brid Retrieval System

The retrieval system combines three complementary approaches to maximize recall and precision:

#### 1.1 Vector Retrieval (Dense)
- **Model**: BGE-M3 (BAAI/bge-m3)
- **Database**: ChromaDB
- **Embedding Dimension**: 1024
- **Similarity Metric**: Cosine Similarity

```
Query → BGE-M3 Encoder → Query Embedding → ChromaDB Search → Top-K Documents
```

**Advantages**:
- Captures semantic meaning beyond keywords
- Handles synonyms and paraphrases
- Multilingual support for medical terminology

#### 1.2 BM25 Retrieval (Sparse)
- **Algorithm**: Best Matching 25 (BM25)
- **Library**: rank_bm25
- **Tokenization**: Word-level with medical term preservation

```
Query → Tokenize → BM25 Score Calculation → Top-K Documents
```

**Advantages**:
- Excellent for exact term matching
- Handles rare medical terms effectively
- Computationally efficient

#### 1.3 Graph Retrieval (Entity-Based)
- **Framework**: NetworkX
- **Entity Extraction**: spaCy NER
- **Relationship**: Co-occurrence based

```
Query → Entity Extraction → Graph Traversal → Related Chunks
```

**Advantages**:
- Captures relationships between medical concepts
- Enables multi-hop reasoning
- Connects related symptoms, diseases, treatments

#### 1.4 Hybrid Fusion with Cross-Encoder Reranking

```python
# Fusion Process
1. Get results from Vector (top 10)
2. Get results from BM25 (top 10)
3. Get results from Graph (top 5)
4. Merge and deduplicate
5. Rerank with Cross-Encoder (ms-marco-MiniLM-L-6-v2)
6. Return top 5 most relevant chunks
```

---

### 2. Semantic Chunking

Unlike fixed-size chunking, MEGA-RAG uses **semantic chunking** based on meaning shifts:

```
┌─────────────────────────────────────────────────────────────┐
│                   SEMANTIC CHUNKING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Document Text                                              │
│       │                                                     │
│       ▼                                                     │
│  Split into Sentences                                       │
│       │                                                     │
│       ▼                                                     │
│  Compute Sentence Embeddings (BGE-M3)                       │
│       │                                                     │
│       ▼                                                     │
│  Calculate Cosine Distance Between Consecutive Sentences    │
│       │                                                     │
│       ▼                                                     │
│  Find Breakpoints (95th Percentile Threshold)               │
│       │                                                     │
│       ▼                                                     │
│  Create Semantic Chunks                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Preserves semantic coherence within chunks
- Better retrieval accuracy
- Maintains context for medical explanations

---

### 3. SEAE: Semantic-Evidential Alignment Evaluation

SEAE is the **hallucination auditor** that verifies generated responses against retrieved evidence.

```
┌─────────────────────────────────────────────────────────────┐
│                      SEAE MODULE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Generated Response                                         │
│       │                                                     │
│       ▼                                                     │
│  Extract Individual Claims                                  │
│       │                                                     │
│       ▼                                                     │
│  For Each Claim:                                            │
│    ├── Compare with Retrieved Evidence                      │
│    ├── Compute Semantic Similarity                          │
│    └── Determine: SUPPORTED / UNSUPPORTED / PARTIAL         │
│       │                                                     │
│       ▼                                                     │
│  Calculate Overall Alignment Score (0.0 - 1.0)              │
│       │                                                     │
│       ▼                                                     │
│  If Score < 0.7 → Flag for Correction                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Evaluation Criteria**:
| Score | Interpretation |
|-------|----------------|
| 0.9 - 1.0 | Fully Supported |
| 0.7 - 0.9 | Mostly Supported |
| 0.5 - 0.7 | Partially Supported |
| < 0.5 | Unsupported (Hallucination) |

---

### 4. DISC: Discrepancy-Identified Self-Clarification

DISC is the **self-correction module** that refines responses based on SEAE feedback.

```
┌─────────────────────────────────────────────────────────────┐
│                      DISC MODULE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Original Response + SEAE Feedback                   │
│       │                                                     │
│       ▼                                                     │
│  Identify Unsupported Claims                                │
│       │                                                     │
│       ▼                                                     │
│  Generate Correction Prompt:                                │
│    "The following claims lack evidence support:             │
│     [list of unsupported claims]                            │
│     Please revise using only the provided evidence."        │
│       │                                                     │
│       ▼                                                     │
│  LLM Generates Corrected Response                           │
│       │                                                     │
│       ▼                                                     │
│  Re-evaluate with SEAE                                      │
│       │                                                     │
│       ▼                                                     │
│  Repeat if needed (max 2 corrections)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. LangGraph Workflow Orchestration

The entire pipeline is orchestrated using **LangGraph** for stateful, cyclic workflows:

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  RETRIEVE   │ ◄─────────────┐
                    └──────┬──────┘               │
                           │                      │
                           ▼                      │
                    ┌─────────────┐               │
                    │  GENERATE   │               │
                    └──────┬──────┘               │
                           │                      │
                           ▼                      │
                    ┌─────────────┐               │
                    │    AUDIT    │               │
                    │   (SEAE)    │               │
                    └──────┬──────┘               │
                           │                      │
                    ┌──────┴──────┐               │
                    │             │               │
               Score ≥ 0.7   Score < 0.7          │
                    │             │               │
                    ▼             ▼               │
             ┌─────────┐   ┌─────────────┐        │
             │   END   │   │   CORRECT   │────────┘
             └─────────┘   │   (DISC)    │   (max 3 iterations)
                           └─────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| LLM | Gemini 2.5 Flash | Response generation |
| Embeddings | BGE-M3 | Semantic embeddings |
| Vector Store | ChromaDB | Dense retrieval |
| Keyword Search | BM25 (rank_bm25) | Sparse retrieval |
| Graph | NetworkX | Entity relationships |
| Reranking | Cross-Encoder (ms-marco) | Result refinement |
| NER | spaCy | Entity extraction |
| Workflow | LangGraph | Pipeline orchestration |
| PDF Processing | pypdf | Document extraction |

---

## Advantages of MEGA-RAG

### 1. Reduced Hallucinations
- **Tri-Brid retrieval** ensures comprehensive evidence coverage
- **SEAE auditing** detects unsupported claims
- **DISC correction** removes or revises hallucinated content

### 2. Evidence-Grounded Responses
- Every claim is traceable to source documents
- Responses include confidence scores
- Users can verify information against original sources

### 3. Domain Adaptability
- Works with any medical PDF documents
- Can be extended to other specialized domains
- Supports multilingual medical terminology (BGE-M3)

### 4. Memory Efficient
- Optimized for systems with limited RAM (8GB)
- CPU-friendly processing mode
- Batch processing for large documents

### 5. Transparent Decision Making
- Provides retrieval scores for each source
- Shows alignment scores for claims
- Logs correction iterations

### 6. Cost Effective
- Uses Gemini 2.5 Flash (free tier available)
- Local embeddings (no API costs for retrieval)
- Efficient chunking reduces token usage

---

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |
| Python | 3.10+ | 3.13 |
| GPU | Not required | Optional (CUDA) |

---

## Configuration Parameters

```python
# Embedding Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

# Chunking Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
SEMANTIC_THRESHOLD_PERCENTILE = 95

# Retrieval Configuration
VECTOR_TOP_K = 10
BM25_TOP_K = 10
GRAPH_TOP_K = 5
RERANK_TOP_K = 5

# Refinement Configuration
MAX_REFINEMENT_ITERATIONS = 3
SEAE_THRESHOLD = 0.7
DISC_MAX_CORRECTIONS = 2
```

---

## Evaluation Metrics

MEGA-RAG is evaluated using the **RAGAS** (Retrieval Augmented Generation Assessment) framework:

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | How well the answer is grounded in context | > 0.8 |
| Answer Relevancy | How relevant the answer is to the question | > 0.8 |
| Context Precision | Relevance of retrieved context | > 0.7 |
| Context Recall | Coverage of required information | > 0.7 |

---

## Dataset

The system is evaluated on the **PQA (Pharmaceutical Question Answering)** dataset:

- **PQA Artificial**: Synthetically generated medical QA pairs
- **PQA Labeled**: Human-annotated medical questions with verified answers

---

## Usage Examples

### Interactive Mode
```bash
python run.py --interactive
```

```
MEGA-RAG Interactive Mode
========================
Enter your medical question (or 'quit' to exit):

> What is the first-line treatment for hypertension?

Retrieving relevant evidence...
Generating response...
Auditing for hallucinations...

Answer:
According to WHO guidelines, the first-line treatment for hypertension
includes thiazide diuretics, ACE inhibitors, ARBs, and calcium channel
blockers. The choice depends on patient-specific factors including age,
ethnicity, and comorbidities.

Confidence Score: 0.92
Sources: WHO Hypertension Guidelines (2021)
```

### Single Query
```bash
python run.py --query "What are the side effects of ACE inhibitors?"
```

### Evaluation Mode
```bash
python run.py --evaluate --eval-samples 50
```

---

## Project Structure

```
mega_rag/
├── config.py                 # Configuration settings
├── main.py                   # CLI entry point
├── core/
│   ├── document_processor.py # Semantic chunking
│   ├── llm.py                # Gemini API integration
│   └── workflow.py           # LangGraph orchestration
├── retrieval/
│   ├── vector_retriever.py   # ChromaDB + BGE-M3
│   ├── bm25_retriever.py     # BM25 keyword search
│   ├── graph_retriever.py    # NetworkX entity graph
│   └── hybrid_retriever.py   # Tri-Brid fusion
├── refinement/
│   ├── seae.py               # Hallucination auditor
│   └── disc.py               # Self-correction module
└── utils/
    ├── data_loader.py        # Dataset loaders
    └── evaluation.py         # RAGAS evaluation
```

---

## Comparison with Other Approaches

| Feature | Basic RAG | MEGA-RAG |
|---------|-----------|----------|
| Retrieval Methods | Single (Vector) | Triple (Vector + BM25 + Graph) |
| Chunking | Fixed-size | Semantic |
| Hallucination Check | None | SEAE Auditor |
| Self-Correction | None | DISC Module |
| Reranking | None | Cross-Encoder |
| Domain Optimization | Generic | Medical-focused |

---

## Limitations and Future Work

### Current Limitations
1. Requires pre-indexed documents (no real-time web search)
2. Limited to text-based PDFs (no image/table extraction)
3. English-focused (though BGE-M3 supports multiple languages)

### Future Enhancements
1. **Multi-modal support**: Process medical images and charts
2. **Real-time updates**: Integration with medical databases
3. **Fine-tuned models**: Domain-specific embedding models
4. **User feedback loop**: Learn from corrections over time
5. **Confidence calibration**: Better uncertainty quantification

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. WHO (2021). "Guideline for the pharmacological treatment of hypertension in adults"
3. BGE-M3: "Multilingual, Multi-Functionality, Multi-Granularity Text Embeddings"
4. RAGAS: "Evaluation framework for Retrieval Augmented Generation"
5. LangGraph: "Building stateful, multi-actor applications with LLMs"

---

## License

This project is developed for educational purposes as part of a Minor Project.

---

## Contact

For questions or collaboration:
- **Author**: HARSH SINGH
- **Project**: Minor Project - MEGA-RAG
