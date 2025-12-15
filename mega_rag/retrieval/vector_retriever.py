"""
Vector Retriever using ChromaDB and BGE-M3 embeddings.
Implements dense retrieval for the Tri-Brid system.
Supports GPU acceleration with memory management.
"""
from __future__ import annotations
import gc
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
from tqdm import tqdm

from mega_rag.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_TOP_K,
    USE_GPU,
    GPU_BATCH_SIZE
)


def get_device():
    """Get the best available device (GPU/MPS/CPU)."""
    if not USE_GPU:
        return "cpu"

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

if TYPE_CHECKING:
    from mega_rag.core.document_processor import Document


class VectorRetriever:
    """
    Dense vector retrieval using ChromaDB and BGE-M3 embeddings.
    Supports GPU acceleration with memory management.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        persist_directory: Optional[Path] = None
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory or CHROMA_DIR

        # Initialize embedding model with GPU support
        print(f"Loading embedding model: {embedding_model}")
        self.device = get_device()
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        print(f"  Using {self.device.upper()} for embeddings")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with GPU memory management."""
        if not documents:
            return

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_unique_id = f"{doc.doc_id}_{doc.chunk_id}"
            ids.append(doc_unique_id)
            texts.append(doc.content)
            metadatas.append(doc.metadata)

        # Generate embeddings in batches to avoid OOM
        print(f"Generating embeddings for {len(texts)} documents...")
        embed_batch_size = GPU_BATCH_SIZE * 2 if USE_GPU else 32  # Smaller batches for MPS
        all_embeddings = []

        try:
            for i in tqdm(range(0, len(texts), embed_batch_size), desc="Embedding"):
                batch_texts = texts[i:i + embed_batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ).tolist()
                all_embeddings.extend(batch_embeddings)

                # Clear GPU memory after each batch
                clear_gpu_memory()
        except RuntimeError as e:
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                print(f"\n  GPU OOM detected, falling back to CPU...")
                self.embedding_model = self.embedding_model.to("cpu")
                self.device = "cpu"
                all_embeddings = []
                for i in tqdm(range(0, len(texts), 32), desc="Embedding (CPU)"):
                    batch_texts = texts[i:i + 32]
                    batch_embeddings = self.embedding_model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    ).tolist()
                    all_embeddings.extend(batch_embeddings)
                    gc.collect()
            else:
                raise

        # Add to collection in batches
        chroma_batch_size = 100
        for i in range(0, len(ids), chroma_batch_size):
            end_idx = min(i + chroma_batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=all_embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        print(f"Added {len(documents)} documents to vector store")

    def retrieve(
        self,
        query: str,
        top_k: int = VECTOR_TOP_K
    ) -> List[Tuple[str, float, dict]]:
        """
        Retrieve top-k similar documents for a query.

        Returns:
            List of (document_text, similarity_score, metadata) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        # Process results
        retrieved = []
        if results['documents'] and results['documents'][0]:
            docs = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]

            for doc, dist, meta in zip(docs, distances, metadatas):
                # Convert distance to similarity (cosine distance to similarity)
                similarity = 1 - dist
                retrieved.append((doc, similarity, meta))

        return retrieved

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    # Test vector retriever
    retriever = VectorRetriever()
    print(f"Collection count: {retriever.get_collection_count()}")

    # Test with a sample query
    if retriever.get_collection_count() > 0:
        results = retriever.retrieve("What is hypertension?")
        print("\nTop results:")
        for doc, score, meta in results[:3]:
            print(f"Score: {score:.4f}")
            print(f"Content: {doc[:200]}...")
            print()
