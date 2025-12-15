
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mega_rag.core.workflow import create_mega_rag_workflow
from mega_rag.config import RERANK_TOP_K

def debug_retrieval(question: str, expected_pubid: str):
    print(f"\n🔍 DEBUGGING RETRIEVAL")
    print(f"Question: {question}")
    print(f"Target PubID: {expected_pubid}")
    print("-" * 60)

    # Initialize workflow
    print("Initializing workflow (loading indices)...")
    workflow = create_mega_rag_workflow()
    
    # Run Retrieval ONLY
    print(f"Running retrieval (Topic K={RERANK_TOP_K})...")
    results = workflow.retriever.retrieve(question, top_k=10) # Get top 10 to see if it's buried
    
    print(f"\nFound {len(results)} chunks.")
    
    found_target = False
    for i, res in enumerate(results):
        # Check source metadata
        source = res.metadata.get('pubid', 'unknown') or res.metadata.get('filename', 'unknown')
        is_target = str(expected_pubid) in str(source) or str(expected_pubid) in res.content
        
        marker = "✅ TARGET FOUND" if is_target else ""
        if is_target: found_target = True
        
        print(f"\n[{i+1}] Score: {res.fusion_score:.4f} (Vec: {res.vector_score:.2f}, BM25: {res.bm25_score:.2f}) {marker}")
        print(f"Source: {source}")
        print(f"Content Preview: {res.content[:200]}...")
        
    print("-" * 60)
    if found_target:
        print("✅ SUCCESS: Target document was retrieved.")
    else:
        print("❌ FAILURE: Target document NOT found in top 10.")

if __name__ == "__main__":
    # Test Case: PubID 23972333
    Q = "Has the prevalence of health care services use increased over the last decade (2001-2009) in elderly people?"
    ID = "23972333"
    
    debug_retrieval(Q, ID)
