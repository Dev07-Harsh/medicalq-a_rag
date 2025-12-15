
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mega_rag.refinement.seae import SEAE
from mega_rag.config import SEAE_THRESHOLD

def debug_seae():
    print("🔍 DEBUGGING SEAE SCORING")
    print("-" * 60)
    
    seae = SEAE()
    
    # Test Case: PubID 23972333
    question = "Has the prevalence of health care services use increased over the last decade (2001-2009) in elderly people?"
    
    # Valid "Yes" Answer
    answer = "Yes, the usage of health care services significantly increased. The number of general practitioner visits among women and men significantly increased from 2001 to 2009."
    
    # Ground Truth Context
    context_chunk = """[RESULTS] The total number of subjects was 24,349. Women had higher prevalence of general practitioner visits than men in all surveys. The number of general practitioner visits among women and men significantly increased from 2001 to 2009 (women: OR 1.43, 1.27-1.61; men: OR 1.71, 1.49-1.97)."""
    
    print(f"Question: {question}")
    print(f"Test Answer: {answer}")
    print(f"Context Snippet: ...{context_chunk[-150:]}")
    print("-" * 60)
    
    # Run Evaluation
    result = seae.evaluate(question, answer, [context_chunk])
    
    print(f"\n📈 SEAE RESULT:")
    print(f"  Alignment Score: {result.alignment_score:.4f}")
    print(f"  Threshold:       {SEAE_THRESHOLD}")
    print(f"  Is Aligned:      {result.is_aligned}")
    print(f"  Evidence Coverage: {result.evidence_coverage:.2f}")
    
    if result.misaligned_claims:
        print(f"\n⚠️ Misaligned Claims:")
        for claim in result.misaligned_claims:
            print(f"  - {claim}")
            
    # Check claim scores specifically
    print("\n🔍 Detailed Claim Scores:")
    for claim, score in result.claim_scores:
        print(f"  Claim: '{claim}' -> Score: {score:.4f}")

if __name__ == "__main__":
    debug_seae()
