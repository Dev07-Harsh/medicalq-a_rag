"""
Document Processor with Semantic Chunking
Processes PDFs and applies semantic chunking based on meaning shifts.
Uses BGE-M3 for both chunking and retrieval with memory optimization.
Supports GPU acceleration with memory management.
"""
import os
import gc
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from pypdf import PdfReader
from tqdm import tqdm

from mega_rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEMANTIC_THRESHOLD_PERCENTILE,
    EMBEDDING_MODEL,
    USE_GPU,
    GPU_BATCH_SIZE
)


def get_device():
    """Get the best available device (GPU/MPS/CPU) with memory management."""
    if not USE_GPU:
        return "cpu"

    import torch
    if torch.cuda.is_available():
        # Clear CUDA cache before starting
        torch.cuda.empty_cache()
        print("  Using CUDA GPU for embeddings")
        return "cuda"
    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon GPU)
        print("  Using MPS (Apple GPU) for embeddings")
        return "mps"
    else:
        print("  GPU not available, using CPU")
        return "cpu"


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: dict
    doc_id: str
    chunk_id: int


# Global model instance to avoid loading multiple times
_EMBEDDING_MODEL = None
_DEVICE = None


def get_embedding_model():
    """Get or create the shared embedding model instance with GPU support."""
    global _EMBEDDING_MODEL, _DEVICE
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _DEVICE = get_device()
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL, device=_DEVICE)
    return _EMBEDDING_MODEL


def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing, but gc.collect helps
            pass
    except ImportError:
        pass


class SemanticChunker:
    """
    Semantic Chunking based on meaning shifts.
    Uses BGE-M3 with GPU support and memory-optimized batch processing.
    """

    def __init__(
        self,
        threshold_percentile: int = SEMANTIC_THRESHOLD_PERCENTILE,
        min_chunk_size: int = 100,
        max_chunk_size: int = CHUNK_SIZE,
        batch_size: int = None  # Auto-set based on GPU/CPU
    ):
        # Use shared BGE-M3 model
        self.embedding_model = get_embedding_model()
        self.threshold_percentile = threshold_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        # Use GPU_BATCH_SIZE if GPU enabled, otherwise small batches for CPU
        self.batch_size = batch_size or (GPU_BATCH_SIZE if USE_GPU else 2)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _compute_breakpoints(self, sentences: List[str]) -> List[int]:
        """
        Compute semantic breakpoints using cosine distance.
        Memory-optimized: processes in batches with GPU memory management.
        Auto-fallback to CPU if GPU OOM occurs.
        """
        import torch

        if len(sentences) < 2:
            return []

        all_embeddings = []

        # Process in batches with memory management
        try:
            with torch.no_grad():
                for i in range(0, len(sentences), self.batch_size):
                    batch = sentences[i:i + self.batch_size]

                    batch_embeddings = self.embedding_model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )

                    all_embeddings.extend(batch_embeddings)

                    # Clear GPU/CPU memory after each batch to prevent OOM
                    clear_gpu_memory()
        except RuntimeError as e:
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                print(f"\n  GPU OOM detected, falling back to CPU...")
                # Fallback to CPU
                self.embedding_model = self.embedding_model.to("cpu")
                all_embeddings = []
                with torch.no_grad():
                    for i in range(0, len(sentences), self.batch_size):
                        batch = sentences[i:i + self.batch_size]
                        batch_embeddings = self.embedding_model.encode(
                            batch,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        all_embeddings.extend(batch_embeddings)
                        gc.collect()
            else:
                raise

        embeddings = np.array(all_embeddings)

        # Compute cosine distances between consecutive sentences
        distances = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            distance = 1 - sim
            distances.append(distance)

        # Cleanup
        del all_embeddings, embeddings
        gc.collect()

        if not distances:
            return []

        # Find threshold at the specified percentile
        threshold = np.percentile(distances, self.threshold_percentile)

        # Find breakpoints where distance exceeds threshold
        breakpoints = [i + 1 for i, d in enumerate(distances) if d > threshold]

        return breakpoints

    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks based on meaning shifts.
        Falls back to size-based chunking if needed.
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        if len(sentences) < 3:
            return [text] if len(text) >= self.min_chunk_size else []

        print(f"    Processing {len(sentences)} sentences...")
        breakpoints = self._compute_breakpoints(sentences)

        # Create chunks based on breakpoints
        chunks = []
        start = 0

        for bp in breakpoints + [len(sentences)]:
            chunk_sentences = sentences[start:bp]
            chunk_text = ' '.join(chunk_sentences)

            # If chunk is too large, split by size
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_by_size(chunk_text)
                chunks.extend(sub_chunks)
            elif len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

            start = bp

        return chunks

    def _split_by_size(self, text: str) -> List[str]:
        """Fallback size-based chunking with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # Try to break at a sentence boundary
            if end < len(text):
                search_start = max(end - 100, start)
                last_period = text.rfind('.', search_start, end)
                if last_period > start:
                    end = last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - CHUNK_OVERLAP

        return chunks


class DocumentProcessor:
    """
    Process documents (PDFs) and create semantic chunks.
    """

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker or SemanticChunker()

    def load_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        reader = PdfReader(pdf_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return '\n'.join(text_parts)

    def load_pdf_with_pages(self, pdf_path: Path) -> List[tuple]:
        """Extract text from PDF with page numbers."""
        reader = PdfReader(pdf_path)
        pages_with_text = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                pages_with_text.append((page_num, text))

        return pages_with_text

    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF into semantic chunks with page tracking."""
        pdf_path = Path(pdf_path)
        doc_id = pdf_path.stem

        print(f"Processing: {pdf_path.name}")

        # Load PDF with page numbers
        pages_with_text = self.load_pdf_with_pages(pdf_path)

        # Build page boundaries for tracking which page each character belongs to
        page_boundaries = []  # List of (start_char, end_char, page_num)
        full_text_parts = []
        current_pos = 0

        for page_num, text in pages_with_text:
            start_pos = current_pos
            full_text_parts.append(text)
            current_pos += len(text) + 1  # +1 for newline
            page_boundaries.append((start_pos, current_pos, page_num))

        full_text = '\n'.join(full_text_parts)

        # Chunk the full text
        chunks = self.chunker.chunk(full_text)

        # Find which page each chunk primarily belongs to
        def find_page_for_chunk(chunk_text: str) -> int:
            """Find the page number where this chunk starts."""
            chunk_start = full_text.find(chunk_text)
            if chunk_start == -1:
                return 1  # Default to page 1 if not found

            for start, end, page_num in page_boundaries:
                if start <= chunk_start < end:
                    return page_num
            return 1

        documents = []
        for i, chunk in enumerate(chunks):
            page_num = find_page_for_chunk(chunk)

            doc = Document(
                content=chunk,
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "page": page_num
                },
                doc_id=doc_id,
                chunk_id=i
            )
            documents.append(doc)

        print(f"  Created {len(documents)} chunks from {len(pages_with_text)} pages")

        # Cleanup after each PDF
        gc.collect()

        return documents

    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> List[Document]:
        """Process multiple PDFs into semantic chunks."""
        all_documents = []

        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            documents = self.process_pdf(pdf_path)
            all_documents.extend(documents)

            # Force garbage collection between PDFs
            gc.collect()

        return all_documents

    def process_text(self, text: str, doc_id: str = "text_doc") -> List[Document]:
        """Process raw text into semantic chunks."""
        chunks = self.chunker.chunk(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={
                    "source": "text_input",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                doc_id=doc_id,
                chunk_id=i
            )
            documents.append(doc)

        return documents


if __name__ == "__main__":
    # Test the document processor
    from mega_rag.config import KNOWLEDGE_BASE_PDFS

    processor = DocumentProcessor()

    for pdf_path in KNOWLEDGE_BASE_PDFS:
        if pdf_path.exists():
            docs = processor.process_pdf(pdf_path)
            print(f"\nSample chunk from {pdf_path.name}:")
            if docs:
                print(docs[0].content[:300] + "...")
