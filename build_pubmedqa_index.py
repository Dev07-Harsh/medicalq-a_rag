#!/usr/bin/env python3
"""build_pubmedqa_index.py

Build a *controlled* retrieval index that contains ONLY PubMedQA contexts.

Why this exists
--------------
The default index in this repo can include PDFs and other sources, which makes it
hard to attribute errors to retrieval vs. generation. This script creates a
clean, isolated index for benchmarking retrieval quality.

What it builds
--------------
- Chroma vector store in a dedicated persist directory
- BM25 index pickle in a dedicated path
- Graph index pickle in a dedicated path

It indexes only context documents from the official PubMedQA-L split artifacts
created by `scripts/prepare_pubmedqa_split.py`.

Usage (example)
--------------
  python build_pubmedqa_index.py --persist-dir chroma_pubmedqa_only \
    --collection-name pubmedqa_only --force

Notes
-----
- This intentionally indexes CONTEXTS only (no LONG_ANSWER).
- For fair evaluation, keep this index separate from the default `chroma_db/`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _rm_if_exists(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PubMedQA-only retrieval index")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("chroma_pubmedqa_only"),
        help="Chroma persist directory for the controlled index",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="pubmedqa_only",
        help="Chroma collection name for the controlled index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index artifacts before rebuilding",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of context documents to index (debug/dev)",
    )

    args = parser.parse_args()

    # Local imports (slow)
    from mega_rag.utils.data_loader import OfficialPubMedQALoader
    from mega_rag.retrieval.vector_retriever import VectorRetriever
    from mega_rag.retrieval.bm25_retriever import BM25Retriever
    from mega_rag.retrieval.graph_retriever import GraphRetriever
    from mega_rag.retrieval.hybrid_retriever import HybridRetriever

    persist_dir: Path = args.persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = persist_dir / "bm25_index.pkl"
    graph_path = persist_dir / "knowledge_graph.pkl"

    if args.force:
        print("[INDEX] --force enabled: deleting existing controlled index artifacts")
        _rm_if_exists(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

    # Load official indexing docs (context-only, all 1000 samples)
    loader = OfficialPubMedQALoader()
    documents = loader.load_indexing_documents()
    if args.max_docs is not None:
        documents = documents[: max(0, args.max_docs)]

    print(f"[INDEX] Documents to index: {len(documents)}")
    print(f"[INDEX] Chroma persist dir: {persist_dir}")
    print(f"[INDEX] Chroma collection : {args.collection_name}")
    print(f"[INDEX] BM25 path         : {bm25_path}")
    print(f"[INDEX] Graph path        : {graph_path}")

    # Build controlled HybridRetriever with isolated backing stores
    retriever = HybridRetriever()

    # Swap out the vector store to use isolated persist dir + collection
    retriever.vector_retriever = VectorRetriever(
        collection_name=args.collection_name,
        persist_directory=persist_dir,
    )

    # Swap BM25 + graph to use isolated file paths
    retriever.bm25_retriever = BM25Retriever(index_path=bm25_path)
    retriever.graph_retriever = GraphRetriever(graph_path=graph_path)

    retriever.index_documents(documents)
    retriever.save_indices()

    stats = retriever.get_stats()
    print(f"\n✓ Controlled PubMedQA-only index built: {stats}")


if __name__ == "__main__":
    main()
