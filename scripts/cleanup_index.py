#!/usr/bin/env python3
"""
MEGA-RAG Index Cleanup Script

This script cleans up old ChromaDB indices after major changes like:
- Embedding model change (BAAI/bge-m3 → PubMedBERT)
- Data leakage fix (removing ground truth from index)
- Collection schema changes

Usage:
    python scripts/cleanup_index.py [--force]
    
Options:
    --force     Skip confirmation prompt
"""
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mega_rag.config import CHROMA_DIR, COLLECTION_NAME


def get_index_size(path: Path) -> str:
    """Get human-readable size of the index directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    # Convert to human-readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.2f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.2f} TB"


def cleanup_chroma_index(force: bool = False):
    """Clean up ChromaDB index directory."""
    chroma_path = Path(CHROMA_DIR)
    
    print("=" * 60)
    print("MEGA-RAG Index Cleanup Tool")
    print("=" * 60)
    print()
    
    if not chroma_path.exists():
        print(f"✓ No index found at: {chroma_path}")
        print("  Nothing to clean up.")
        return
    
    # Show index info
    index_size = get_index_size(chroma_path)
    print(f"Index Location: {chroma_path}")
    print(f"Index Size: {index_size}")
    print(f"Collection: {COLLECTION_NAME}")
    print()
    
    # Reasons for cleanup
    print("IMPORTANT: You should clean up the index after:")
    print("  1. Changing the embedding model")
    print("  2. Fixing data leakage (removing ground truth)")
    print("  3. Adding/removing documents from the corpus")
    print()
    
    if not force:
        print("⚠️  WARNING: This will DELETE all indexed data!")
        print("   You will need to re-run indexing after cleanup.")
        print()
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
    
    print()
    print("Cleaning up index...")
    
    try:
        # Remove the entire chroma directory
        shutil.rmtree(chroma_path)
        print(f"✓ Removed: {chroma_path}")
        
        # Recreate empty directory
        chroma_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created fresh directory: {chroma_path}")
        
        print()
        print("=" * 60)
        print("CLEANUP COMPLETE")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Run indexing: python run.py --index")
        print("  2. Or use the data loader: python -m mega_rag.data.data_loader")
        print()
        print("The new index will use:")
        print("  - Embedding model: pritamdeka/S-PubMedBert-MS-MARCO")
        print("  - Data leakage fix: Ground truth NOT indexed")
        print()
        
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")
        sys.exit(1)


def main():
    force = "--force" in sys.argv
    cleanup_chroma_index(force=force)


if __name__ == "__main__":
    main()
