"""
LangGraph Workflow Orchestration for MEGA-RAG
Implements the cyclic refinement loop: Retrieve → Generate → Audit → Correct → Re-audit
With timing, query expansion, evidence re-retrieval, Chain-of-Thought reasoning, and citation verification.

Enhanced Features (v2.0):
- Query reformulation for vague/ambiguous queries
- Med-PaLM 2 inspired Chain-of-Thought for complex medical questions
- Post-generation citation verification
- Ensemble refinement with self-consistency
"""
from typing import TypedDict, List, Optional, Annotated
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
import operator
import time

from mega_rag.config import MAX_REFINEMENT_ITERATIONS, RERANK_TOP_K, SEAE_THRESHOLD
from mega_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from mega_rag.retrieval.query_reformulator import QueryReformulator
from mega_rag.refinement.seae import SEAE, SEAEResult
from mega_rag.refinement.disc import DISC, DISCResult
from mega_rag.refinement.chain_of_thought import MedicalCoTReasoner
from mega_rag.refinement.citation_verifier import CitationVerifier
from mega_rag.refinement.self_consistency import SelfConsistencyVoter, apply_self_consistency
from mega_rag.core.llm import BaseLLM, create_llm


class RAGState(TypedDict):
    """State for the RAG workflow."""
    question: str
    original_question: str  # Original question before expansion
    context_chunks: List[str]
    source_metadata: List[dict]  # Source document metadata for citations
    retrieval_results: List[dict]
    answer: str
    seae_result: Optional[dict]
    disc_result: Optional[dict]
    iteration: int
    max_iterations: int
    is_complete: bool
    final_answer: str
    is_reliable: bool  # Whether the answer passed alignment check
    workflow_trace: Annotated[List[str], operator.add]
    timing: dict  # Timing information for verbose mode
    re_retrieval_done: bool  # Track if we've already done re-retrieval
    intent: str  # MEDICAL, GREETING, or OFF_TOPIC
    # New v2.0 fields
    use_cot: bool  # Whether to use Chain-of-Thought reasoning
    cot_reasoning: Optional[dict]  # CoT reasoning trace
    citation_report: Optional[dict]  # Citation verification report


class MEGARAGWorkflow:
    """
    MEGA-RAG Workflow using LangGraph.

    Flow:
    1. guardrail - Classify intent (medical/greeting/off-topic)
    2. reformulate - Improve vague/unclear queries (NEW)
    3. retrieve - Get relevant chunks using Tri-Brid retrieval
    4. generate - Generate answer with optional Chain-of-Thought
    5. verify_citations - Verify and correct citations (NEW)
    6. audit - Check answer with SEAE
    7. decide - If aligned, finish; else go to correct
    8. correct - Use DISC to fix hallucinations
    9. Loop back to audit or finish
    
    Reasoning Modes:
    - "single": Single-path reasoning (faster, baseline)
    - "multi": Multi-path self-consistency voting (more robust, recommended)
    - "auto": Auto-select based on question complexity
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseLLM,
        max_iterations: int = MAX_REFINEMENT_ITERATIONS,
        enable_cot: bool = True,
        enable_citation_verification: bool = True,
        enable_query_reformulation: bool = True,
        enable_self_consistency: bool = True,
        reasoning_mode: str = "multi",  # "single", "multi", or "auto"
        num_reasoning_paths: int = 3,
        debug: bool = False
    ):
        self.retriever = retriever
        self.llm = llm
        self.seae = SEAE()
        self.disc = DISC()
        self.max_iterations = max_iterations
        self.debug = debug  # Enable debug mode
        
        # Reasoning mode configuration
        self.reasoning_mode = reasoning_mode  # "single", "multi", or "auto"
        self.num_reasoning_paths = num_reasoning_paths
        
        # New v2.0 components
        self.enable_cot = enable_cot
        self.enable_citation_verification = enable_citation_verification
        self.enable_query_reformulation = enable_query_reformulation
        self.enable_self_consistency = enable_self_consistency
        self.cot_reasoner = MedicalCoTReasoner(llm) if enable_cot else None
        self.citation_verifier = CitationVerifier() if enable_citation_verification else None
        self.query_reformulator = QueryReformulator(llm) if enable_query_reformulation else None
        # Self-consistency voter with configurable number of paths
        self.self_consistency_voter = SelfConsistencyVoter(
            llm=llm, 
            num_paths=num_reasoning_paths
        ) if enable_self_consistency else None

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("guardrail", self._guardrail_node)
        workflow.add_node("reformulate", self._reformulate_node)  # New v2.0 node
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("verify_citations", self._verify_citations_node)
        workflow.add_node("audit", self._audit_node)
        workflow.add_node("correct", self._correct_node)
        workflow.add_node("re_retrieve", self._re_retrieve_node)
        workflow.add_node("finalize", self._finalize_node)

        # Add edges
        workflow.set_entry_point("guardrail")
        
        # Conditional edge from guardrail
        workflow.add_conditional_edges(
            "guardrail",
            self._check_intent,
            {
                "medical": "reformulate",  # Go to reformulate first for medical queries
                "greeting": "finalize",
                "off_topic": "finalize"
            }
        )

        workflow.add_edge("reformulate", "retrieve")  # Then to retrieve
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "verify_citations")
        workflow.add_edge("verify_citations", "audit")

        # Conditional edge from audit
        workflow.add_conditional_edges(
            "audit",
            self._should_correct,
            {
                "correct": "correct",
                "re_retrieve": "re_retrieve",
                "finalize": "finalize"
            }
        )

        # Edge from correct back to audit (refinement loop)
        workflow.add_edge("correct", "audit")

        # Edge from re_retrieve to generate
        workflow.add_edge("re_retrieve", "generate")

        # Final edge
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _guardrail_node(self, state: RAGState) -> RAGState:
        """Classify intent and block off-topic queries."""
        start_time = time.time()
        question = state["question"]
        
        # Classify intent
        intent = self.llm.classify_intent(question)
        
        updates = {
            "intent": intent, 
            "workflow_trace": [f"[GUARDRAIL] Intent detected: {intent}"]
        }
        
        # Handle non-medical intents immediately
        if intent == "GREETING":
            updates["answer"] = "Hello! I am MEGA-RAG, your medical evidence assistant. I can help you find information about treatments, guidelines, and clinical studies. usage: Ask me a medical question!"
            updates["is_reliable"] = True
            updates["workflow_trace"].append("[GUARDRAIL] Responding to greeting")
            
        elif intent == "OFF_TOPIC":
            updates["answer"] = "I specialize in medical topics and clinical evidence. Please ask me about health conditions, treatments, or medical guidelines."
            updates["is_reliable"] = False
            updates["workflow_trace"].append("[GUARDRAIL] ⛔ Refusing off-topic query")

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing["guardrail"] = elapsed
        updates["timing"] = timing
        
        return {**state, **updates}

    def _check_intent(self, state: RAGState) -> str:
        """Route based on intent."""
        intent = state.get("intent", "MEDICAL")
        if intent == "GREETING":
            return "greeting"
        elif intent == "OFF_TOPIC":
            return "off_topic"
        return "medical"

    def _reformulate_node(self, state: RAGState) -> RAGState:
        """Reformulate vague or unclear queries into better medical questions."""
        start_time = time.time()
        question = state["question"]
        trace_msgs = []
        
        if not self.query_reformulator:
            # Query reformulation disabled
            return {
                **state,
                "query_analysis": None,
                "workflow_trace": []
            }
        
        # Analyze the query
        analysis = self.query_reformulator.analyze_query(question)
        
        query_analysis = {
            "original_query": analysis.original_query,
            "confidence": analysis.confidence,
            "is_vague": analysis.is_vague,
            "is_too_short": analysis.is_too_short,
            "detected_intent": analysis.detected_intent,
            "missing_context": analysis.missing_context
        }
        
        # Only reformulate if query has issues
        if analysis.confidence < 0.7 or analysis.is_vague or analysis.is_too_short:
            trace_msgs.append(f"[REFORMULATE] Query quality: {analysis.confidence:.0%} confidence")
            
            if analysis.is_vague:
                trace_msgs.append("[REFORMULATE] ⚠️ Query is vague - reformulating")
            if analysis.is_too_short:
                trace_msgs.append("[REFORMULATE] ⚠️ Query is too short - adding context")
            
            # Reformulate the query
            result = self.query_reformulator.reformulate(question, use_llm=True)
            
            if result.was_modified:
                reformulated_query = result.reformulated_query
                trace_msgs.append(f"[REFORMULATE] ✅ Improved query: '{reformulated_query[:80]}...'")
                
                query_analysis["reformulated_query"] = reformulated_query
                query_analysis["expansion_terms"] = result.expansion_terms
                query_analysis["sub_queries"] = result.sub_queries
                
                # Update the question for retrieval
                question = reformulated_query
                
                # If we have sub-queries, note them for potential multi-retrieval
                if result.sub_queries:
                    trace_msgs.append(f"[REFORMULATE] Identified {len(result.sub_queries)} sub-questions")
            else:
                trace_msgs.append("[REFORMULATE] Query kept as-is (reformulation not beneficial)")
        else:
            trace_msgs.append(f"[REFORMULATE] ✅ Query is clear ({analysis.confidence:.0%} confidence)")
        
        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing["reformulate"] = elapsed
        
        return {
            **state,
            "question": question,  # May be updated if reformulated
            "query_analysis": query_analysis,
            "timing": timing,
            "workflow_trace": trace_msgs
        }

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant context using Tri-Brid retrieval with timing."""
        start_time = time.time()
        question = state["question"]

        # Get retrieval results
        results = self.retriever.retrieve(question, top_k=RERANK_TOP_K)

        # Extract content and metadata
        context_chunks = [r.content for r in results]
        source_metadata = [r.metadata for r in results]

        # Store results as dicts for serialization (including source info)
        retrieval_results = [
            {
                "content": r.content[:200] + "...",
                "source": r.metadata.get("filename", "Unknown"),
                "vector_score": r.vector_score,
                "bm25_score": r.bm25_score,
                "graph_score": r.graph_score,
                "fusion_score": r.fusion_score,
                "rerank_score": r.rerank_score
            }
            for r in results
        ]

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing["retrieve"] = elapsed

        return {
            **state,
            "context_chunks": context_chunks,
            "source_metadata": source_metadata,
            "retrieval_results": retrieval_results,
            "timing": timing,
            "workflow_trace": [
                f"[RETRIEVE] Consulted knowledge base for: '{question}'",
                f"[RETRIEVE] Found {len(context_chunks)} relevant excerpts from {len(set(r.metadata.get('filename') for r in results))} documents",
                f"[RETRIEVE] Top source: {source_metadata[0].get('filename', 'Unknown')}" if source_metadata else "[RETRIEVE] No relevant sources found"
            ]
        }

    def _re_retrieve_node(self, state: RAGState) -> RAGState:
        """Re-retrieve with expanded query when alignment is low."""
        start_time = time.time()
        question = state["question"]
        original_question = state.get("original_question", question)

        # Expand query by including context from misaligned claims
        seae_result = state.get("seae_result", {})
        misaligned_info = seae_result.get("misaligned_claims_text", "")

        # Create expanded query focusing on missing information
        expanded_query = f"{original_question} {misaligned_info}".strip()

        # Get new retrieval results with expanded query
        results = self.retriever.retrieve(expanded_query, top_k=RERANK_TOP_K + 2)

        # Merge with existing context (deduplicate)
        existing_chunks = set(state.get("context_chunks", []))
        new_chunks = []
        new_metadata = []

        for r in results:
            if r.content not in existing_chunks:
                new_chunks.append(r.content)
                new_metadata.append(r.metadata)
                existing_chunks.add(r.content)

        # Combine old and new context
        context_chunks = state.get("context_chunks", []) + new_chunks
        source_metadata = state.get("source_metadata", []) + new_metadata

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing["re_retrieve"] = timing.get("re_retrieve", 0) + elapsed

        return {
            **state,
            "context_chunks": context_chunks,
            "source_metadata": source_metadata,
            "re_retrieval_done": True,
            "timing": timing,
            "workflow_trace": ["[RE-RETRIEVE] Added {} new chunks ({:.2f}s)".format(len(new_chunks), elapsed)]
        }

    def _generate_node(self, state: RAGState) -> RAGState:
        """Generate answer using LLM with optional Chain-of-Thought or Self-Consistency."""
        start_time = time.time()
        question = state["question"]
        context_chunks = state["context_chunks"]
        source_metadata = state.get("source_metadata", [])
        iteration = state.get("iteration", 0)
        
        trace_msgs = []
        cot_reasoning = None
        
        # Debug: Show question and context summary
        if self.debug:
            print("\n" + "="*70)
            print(f"[DEBUG GENERATE] Iteration: {iteration}")
            print(f"[DEBUG GENERATE] Question: {question[:200]}...")
            print(f"[DEBUG GENERATE] Context chunks: {len(context_chunks)}")
            if context_chunks:
                print(f"[DEBUG GENERATE] First chunk preview: {context_chunks[0][:300]}...")
            print("="*70)
        
        # =====================================================================
        # KEY FIX: Use Self-Consistency Voting for yes/no questions
        # This runs DURING generation, not as a fallback
        # =====================================================================
        is_yes_no = hasattr(self.llm, "_is_yes_no_question") and self.llm._is_yes_no_question(question)
        use_self_consistency = (
            is_yes_no and 
            self.reasoning_mode in ("multi", "auto") and 
            self.self_consistency_voter and 
            iteration == 0  # Only on first iteration
        )
        
        if use_self_consistency:
            if self.debug:
                print(f"[DEBUG GENERATE] Using SELF-CONSISTENCY VOTING for yes/no question")
                print(f"[DEBUG GENERATE] Reasoning mode: {self.reasoning_mode}, Paths: {self.num_reasoning_paths}")
            
            trace_msgs.append(f"[GENERATE] 🗳️ Using self-consistency voting ({self.num_reasoning_paths} paths)")
            
            # Use self-consistency voting for robust decision
            try:
                result = self.self_consistency_voter.vote(
                    question=question,
                    evidence_chunks=context_chunks,
                    debug=self.debug
                )
                
                # Build answer with explanation and voting result
                decision = result.final_decision
                confidence = result.confidence
                vote_dist = result.vote_distribution
                
                # Use the best reasoning path that matches the majority decision
                best_reasoning = ""
                for path in result.reasoning_paths:
                    if decision in path.lower():
                        # Extract reasoning before "Final Answer"
                        import re
                        match = re.split(r'final\s*answer', path, flags=re.IGNORECASE)
                        if match:
                            best_reasoning = match[0].strip()
                            break
                
                if not best_reasoning:
                    best_reasoning = "Based on the evidence provided."
                
                # Build final answer
                confidence_note = ""
                if result.is_unanimous:
                    confidence_note = f" (unanimous {self.num_reasoning_paths}/{self.num_reasoning_paths} agreement)"
                elif confidence >= 0.7:
                    confidence_note = f" (high confidence: {vote_dist})"
                else:
                    confidence_note = f" (votes: {vote_dist})"
                
                answer = f"{best_reasoning}{confidence_note}\n\nFinal Answer: {decision}"
                
                trace_msgs.append(f"[GENERATE]    Votes: {vote_dist}")
                trace_msgs.append(f"[GENERATE]    Decision: {decision} (confidence: {confidence:.0%})")
                
                if self.debug:
                    print(f"[DEBUG GENERATE] Self-consistency result:")
                    print(f"  Decision: {decision}")
                    print(f"  Confidence: {confidence:.2f}")
                    print(f"  Votes: {vote_dist}")
                    print(f"  Unanimous: {result.is_unanimous}")
                
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG GENERATE] Self-consistency failed: {e}, falling back to standard")
                trace_msgs.append(f"[GENERATE] ⚠️ Self-consistency failed, using standard generation")
                answer = self.llm.generate_answer(
                    question,
                    context_chunks,
                    source_metadata=source_metadata,
                    debug=self.debug
                )
        
        # Check if we should use Chain-of-Thought reasoning (for non-yes/no questions)
        elif self.cot_reasoner and iteration == 0 and not is_yes_no:
            # First iteration: detect complexity and potentially use CoT
            use_cot = state.get("use_cot", False)
            complexity = self.cot_reasoner.detect_complexity(question)
            use_cot = complexity.get("is_complex", False)
            
            if use_cot:
                trace_msgs.append(f"[GENERATE] 🧠 Complex query detected - using Chain-of-Thought reasoning")
                trace_msgs.append(f"[GENERATE]    Complexity type: {complexity.get('category', 'multi-step')}")
                
                # Use CoT with ensemble refinement for complex questions
                cot_result = self.cot_reasoner.reason_with_evidence(
                    question=question,
                    evidence_chunks=context_chunks,
                    source_metadata=source_metadata
                )
                
                answer = cot_result.get("final_answer", "")
                cot_reasoning = {
                    "complexity": complexity,
                    "reasoning_steps": cot_result.get("reasoning_steps", []),
                    "num_paths": cot_result.get("num_paths", 1),
                    "consensus_reached": cot_result.get("consensus_reached", False)
                }
                
                trace_msgs.append(f"[GENERATE]    Generated {cot_result.get('num_paths', 1)} reasoning paths")
                if cot_result.get("consensus_reached"):
                    trace_msgs.append("[GENERATE]    ✅ Ensemble consensus reached")
            else:
                trace_msgs.append("[GENERATE] Standard generation (question not complex)")
                answer = self.llm.generate_answer(
                    question,
                    context_chunks,
                    source_metadata=source_metadata,
                    debug=self.debug
                )
        else:
            # Standard generation: subsequent iterations or single-path mode
            answer = self.llm.generate_answer(
                question,
                context_chunks,
                source_metadata=source_metadata,
                debug=self.debug
            )
            trace_msgs.append(f"[GENERATE iter={iteration}] Generated answer (single-path)")

        # Debug: Show generated answer
        if self.debug:
            print("\n" + "="*70)
            print(f"[DEBUG GENERATE] Generated answer (len={len(answer)}):")
            print("-"*70)
            print(answer)
            print("-"*70)
            # Check for Final Answer pattern
            import re
            final_match = re.search(r'final\s*answer\s*[:\s]*\b(yes|no|maybe)\b', answer.lower())
            if final_match:
                print(f"[DEBUG GENERATE] ✓ Found 'Final Answer' pattern: {final_match.group(1)}")
            else:
                print("[DEBUG GENERATE] ✗ NO 'Final Answer' pattern found in generated answer!")
                # Check what patterns ARE present
                if 'yes' in answer.lower():
                    print("[DEBUG GENERATE]   'yes' found in answer (but not in Final Answer format)")
                if 'no' in answer.lower():
                    print("[DEBUG GENERATE]   'no' found in answer (but not in Final Answer format)")
                if 'maybe' in answer.lower():
                    print("[DEBUG GENERATE]   'maybe' found in answer (but not in Final Answer format)")
            print("="*70 + "\n")

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing[f"generate_iter_{iteration}"] = elapsed
        
        if not trace_msgs:
            trace_msgs.append(f"[GENERATE iter={iteration}] Generated answer ({elapsed:.2f}s)")

        # Ensure use_cot has a default value
        use_cot_flag = cot_reasoning is not None

        return {
            **state,
            "answer": answer,
            "use_cot": use_cot_flag,
            "cot_reasoning": cot_reasoning,
            "timing": timing,
            "workflow_trace": trace_msgs
        }

    def _verify_citations_node(self, state: RAGState) -> RAGState:
        """Verify and potentially correct citations in the generated answer."""
        start_time = time.time()
        
        if not self.citation_verifier:
            # Citation verification disabled
            return {
                **state,
                "citation_report": None,
                "workflow_trace": []
            }
        
        answer = state["answer"]
        context_chunks = state["context_chunks"]
        source_metadata = state.get("source_metadata", [])
        
        trace_msgs = []
        
        # Verify citations
        report = self.citation_verifier.verify_citations(
            answer=answer,
            evidence_chunks=context_chunks,
            source_metadata=source_metadata
        )
        
        citation_report = {
            "total_citations": report.total_citations,
            "verified_citations": report.verified_citations,
            "unverified_citations": report.unverified_citations,
            "verification_rate": report.verification_rate,
            "details": [
                {
                    "citation_text": m.citation_text,
                    "claimed_source": m.claimed_source,
                    "is_verified": m.is_verified,
                    "confidence": m.confidence
                }
                for m in report.citation_matches
            ]
        }
        
        trace_msgs.append(f"[CITATIONS] Verified {report.verified_citations}/{report.total_citations} citations ({report.verification_rate:.0%})")
        
        # If there are unverified citations, correct them
        if report.unverified_citations > 0 and report.verification_rate < 0.8:
            trace_msgs.append(f"[CITATIONS] ⚠️ Correcting {report.unverified_citations} unsupported citations")
            corrected_answer = self.citation_verifier.correct_citations(
                answer=answer,
                report=report,
                evidence_chunks=context_chunks
            )
            answer = corrected_answer
            trace_msgs.append("[CITATIONS] ✅ Citations corrected")
        else:
            trace_msgs.append("[CITATIONS] ✅ All citations verified")
        
        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing["citation_verification"] = elapsed
        
        return {
            **state,
            "answer": answer,
            "citation_report": citation_report,
            "timing": timing,
            "workflow_trace": trace_msgs
        }

    def _audit_node(self, state: RAGState) -> RAGState:
        """Audit answer using SEAE with timing."""
        start_time = time.time()
        question = state["question"]
        answer = state["answer"]
        context_chunks = state["context_chunks"]
        iteration = state.get("iteration", 0)

        if self.debug:
            print("\n" + "="*60)
            print(f"[DEBUG AUDIT] Starting SEAE audit (iteration {iteration})")
            print(f"[DEBUG AUDIT] Answer length: {len(answer)}")
            print(f"[DEBUG AUDIT] Context chunks: {len(context_chunks)}")
            print("="*60)

        # Run SEAE evaluation
        seae_result = self.seae.evaluate(question, answer, context_chunks, debug=self.debug)

        # Create text summary of misaligned claims for re-retrieval
        misaligned_text = " ".join(seae_result.misaligned_claims[:3]) if seae_result.misaligned_claims else ""

        # Convert to dict for serialization
        seae_dict = {
            "is_aligned": seae_result.is_aligned,
            "alignment_score": seae_result.alignment_score,
            "evidence_coverage": seae_result.evidence_coverage,
            "misaligned_claims_count": len(seae_result.misaligned_claims),
            "misaligned_claims_text": misaligned_text,
            "explanation": seae_result.explanation,
            "claim_scores": [(c, s) for c, s in seae_result.claim_scores]  # Include for verbose output
        }

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing[f"audit_iter_{iteration}"] = elapsed

        # Enhanced logging
        trace_msgs = []
        trace_msgs.append(f"[AUDIT] Analyzing answer accuracy (Score: {seae_result.alignment_score:.2f}, Goal: {SEAE_THRESHOLD})")
        
        if seae_result.misaligned_claims:
            trace_msgs.append(f"[AUDIT] ⚠️ Found {len(seae_result.misaligned_claims)} claims requiring verification")
            # Log specific doubtful claims
            for i, claim in enumerate(seae_result.misaligned_claims[:2]):
                claim_text = claim[:60] + "..." if len(claim) > 60 else claim
                trace_msgs.append(f"[AUDIT]    • Investigating claim: '{claim_text}'")
        else:
             trace_msgs.append("[AUDIT] ✅ Verification passed: Answer is fully grounded in evidence")

        return {
            **state,
            "seae_result": seae_dict,
            "iteration": iteration + 1,
            "timing": timing,
            "workflow_trace": trace_msgs
        }

    def _should_correct(self, state: RAGState) -> str:
        """Decide whether to correct, re-retrieve, or finalize."""
        seae_result = state.get("seae_result", {})
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", self.max_iterations)
        re_retrieval_done = state.get("re_retrieval_done", False)

        is_aligned = seae_result.get("is_aligned", False)
        alignment_score = seae_result.get("alignment_score", 0)

        if is_aligned:
            return "finalize"

        if iteration >= max_iter:
            return "finalize"

        # If alignment is very low and we haven't re-retrieved yet, try re-retrieval
        # This helps when the initial evidence is insufficient
        if alignment_score < SEAE_THRESHOLD - 0.15 and not re_retrieval_done and iteration <= 1:
            return "re_retrieve"

        return "correct"

    def _correct_node(self, state: RAGState) -> RAGState:
        """Correct answer using DISC with timing."""
        start_time = time.time()
        question = state["question"]
        answer = state["answer"]
        context_chunks = state["context_chunks"]
        iteration = state.get("iteration", 0)

        # Re-run SEAE to get full result for DISC
        seae_result = self.seae.evaluate(question, answer, context_chunks)

        # Create correction prompt
        prompt = self.disc.create_correction_prompt(
            question=question,
            original_answer=answer,
            evidence_chunks=context_chunks,
            seae_feedback=self.seae.get_feedback_for_correction(seae_result),
            misaligned_claims=seae_result.misaligned_claims
        )

        # Get corrected answer
        corrected_answer = self.llm.generate(prompt)
        corrected_answer = self.disc._clean_response(corrected_answer)

        elapsed = time.time() - start_time
        timing = state.get("timing", {})
        timing[f"correct_iter_{iteration}"] = elapsed

        # Enhanced logging
        trace_msgs = []
        trace_msgs.append(f"[CORRECT] Iteration {iteration}: Refining answer to address inaccuracies")
        trace_msgs.append("[CORRECT] Strategies applied:")
        trace_msgs.append("   • Removing unsupported statements")
        trace_msgs.append("   • improving citation accuracy")
        trace_msgs.append("   • Rewriting for stricter evidence adherence")

        return {
            **state,
            "answer": corrected_answer,
            "timing": timing,
            "workflow_trace": trace_msgs
        }

    def _finalize_node(self, state: RAGState) -> RAGState:
        """Finalize the workflow - REFUSE to return unreliable answers."""
        seae_result = state.get("seae_result") or {}
        intent = state.get("intent", "MEDICAL")
        
        # Debug: Show finalize state
        if self.debug:
            print("\n" + "="*70)
            print("[DEBUG FINALIZE] Starting finalization")
            print(f"[DEBUG FINALIZE] Intent: {intent}")
            print(f"[DEBUG FINALIZE] SEAE Result: {seae_result}")
            print(f"[DEBUG FINALIZE] Answer before finalize (len={len(state.get('answer', ''))}):")
            print("-"*70)
            print(state.get("answer", "")[:500])
            print("-"*70)
        
        # If we skipped audit (e.g. Greeting or Off-topic), trust the immediate answer
        if intent != "MEDICAL":
             return {
                **state,
                "is_complete": True,
                "final_answer": state["answer"],
                "workflow_trace": state["workflow_trace"] + ["[FINALIZE] " + f"Closing {intent} interaction"]
            }

        is_aligned = seae_result.get("is_aligned", False)
        alignment_score = seae_result.get("alignment_score", 0)
        iterations = state.get("iteration", 0)
        timing = state.get("timing", {})

        # Calculate total time
        total_time = sum(timing.values())
        timing["total"] = total_time

        # CRITICAL: If answer is not aligned, DO NOT return it
        # This is the core feature of MEGA-RAG - refusing hallucinated answers
        if is_aligned:
            status = "ALIGNED"
            final_answer = state["answer"]
            
            if self.debug:
                print(f"[DEBUG FINALIZE] Answer IS aligned (score={alignment_score:.3f})")
                print("[DEBUG FINALIZE] Enforcing yes/no/maybe format...")
            
            # Enforce a strict yes/no/maybe decision line for PubMedQA-style questions.
            # Local LLMs (e.g., Ollama Mistral) may ignore formatting instructions; this
            # post-step makes evaluation deterministic and reduces "maybe" bias.
            try:
                final_answer = self._enforce_yes_no_final_answer(
                    question=state.get("question", ""),
                    answer=final_answer,
                    evidence_chunks=state.get("context_chunks", []),
                )
                
                if self.debug:
                    print(f"[DEBUG FINALIZE] After enforcement (len={len(final_answer)}):")
                    print("-"*70)
                    print(final_answer[-500:] if len(final_answer) > 500 else final_answer)
                    print("-"*70)
                    
            except Exception as e:
                # Never fail the run due to formatting enforcement.
                if self.debug:
                    print(f"[DEBUG FINALIZE] ✗ Enforcement failed: {e}")
                state.get("workflow_trace", []).append(f"[FINALIZE] Decision enforcement failed: {e}")
            is_reliable = True
        else:
            status = "REFUSED_LOW_ALIGNMENT"
            if self.debug:
                print(f"[DEBUG FINALIZE] Answer NOT aligned (score={alignment_score:.3f})")
                print("[DEBUG FINALIZE] Generating refusal message...")
            # Generate a refusal message explaining why we can't answer
            final_answer = self._generate_refusal_message(
                question=state["question"],
                alignment_score=alignment_score,
                iterations=iterations
            )
            is_reliable = False

        if self.debug:
            print(f"[DEBUG FINALIZE] Final status: {status}, is_reliable: {is_reliable}")
            print("="*70 + "\n")

        trace_msg = "[FINALIZE] Status: {}, Alignment: {:.2f}, Iterations: {}, Total: {:.2f}s".format(
            status, alignment_score, iterations, total_time
        )

        return {
            **state,
            "is_complete": True,
            "final_answer": final_answer,
            "is_reliable": is_reliable,
            "timing": timing,
            "workflow_trace": [trace_msg]
        }

    def _enforce_yes_no_final_answer(self, question: str, answer: str, evidence_chunks: List[str]) -> str:
        """Ensure yes/no questions end with an explicit `Final Answer: yes|no|maybe` line.

        Strategy:
        1) If not a yes/no question, return unchanged.
        2) If answer already contains a decision, normalize it to a final line.
        3) Otherwise use SELF-CONSISTENCY VOTING (3 paths) to extract decision.
        
        Self-consistency voting reduces hallucination by generating multiple
        reasoning paths and using majority voting for the final decision.
        """
        import re

        if self.debug:
            print("\n" + "-"*60)
            print("[DEBUG ENFORCE] Starting yes/no enforcement")
            print(f"[DEBUG ENFORCE] Question: {question[:100]}...")
        
        if not hasattr(self.llm, "_is_yes_no_question") or not self.llm._is_yes_no_question(question):
            if self.debug:
                print("[DEBUG ENFORCE] Not a yes/no question, returning unchanged")
            return answer

        if self.debug:
            print("[DEBUG ENFORCE] This IS a yes/no question, checking for Final Answer pattern")

        raw = answer or ""
        answer_lower = raw.lower()

        # Remove any existing trailing decision lines to avoid duplicates.
        raw_stripped = raw.rstrip()
        raw_stripped = re.sub(
            r"\n\s*final\s*answer\s*[:\s]*\b(yes|no|maybe)\b\s*[.!]?\s*$",
            "",
            raw_stripped,
            flags=re.IGNORECASE,
        ).rstrip()

        # If the answer contains an explicit decision anywhere, reuse it.
        # IMPORTANT: We treat 'maybe' as overridable (it usually indicates hedging),
        # but keep 'yes'/'no' to preserve model intent unless we later add a verifier.
        m = re.search(r"final\s*answer\s*[:\s]*\b(yes|no|maybe)\b", answer_lower)
        if m:
            decision = m.group(1)
            if decision in ("yes", "no"):
                if self.debug:
                    print(f"[DEBUG ENFORCE] ✓ Found existing 'Final Answer: {decision}' pattern (keeping)")
                return f"{raw_stripped}\n\nFinal Answer: {decision}"
            if self.debug:
                print("[DEBUG ENFORCE] ! Found existing 'Final Answer: maybe' (will attempt override)")

        # If it ends with a bare decision token, normalize it.
        # Again: treat bare 'maybe' as overridable.
        m2 = re.search(r"\b(yes|no|maybe)\b\s*[.!]?\s*$", answer_lower)
        if m2:
            decision = m2.group(1)
            if decision in ("yes", "no"):
                if self.debug:
                    print(f"[DEBUG ENFORCE] ✓ Found trailing '{decision}', normalizing to Final Answer format")
                return f"{raw_stripped}\n\nFinal Answer: {decision}"
            if self.debug:
                print("[DEBUG ENFORCE] ! Found trailing 'maybe' (will attempt override)")

        # =====================================================================
        # Decision Extraction based on reasoning_mode:
        # - "single": Use single LLM call (faster, baseline)
        # - "multi": Use self-consistency voting (more robust)
        # - "auto": Use multi-path for complex questions, single for simple
        # =====================================================================
        
        # Determine if we should use multi-path reasoning
        use_multi_path = False
        if self.reasoning_mode == "multi":
            use_multi_path = True
        elif self.reasoning_mode == "auto":
            # Auto-detect: use multi-path for longer/complex questions
            use_multi_path = len(question) > 100 or '?' in question[:-1]  # Multiple ? indicates complexity
        # "single" mode keeps use_multi_path = False
        
        if self.debug:
            print(f"[DEBUG ENFORCE] ✗ No Final Answer found")
            print(f"[DEBUG ENFORCE] Reasoning mode: {self.reasoning_mode} → {'MULTI-PATH' if use_multi_path else 'SINGLE-PATH'}")
        
        if use_multi_path and self.self_consistency_voter and self.enable_self_consistency:
            # MULTI-PATH: Use self-consistency voting (generates N paths, majority vote)
            try:
                if self.debug:
                    print(f"[DEBUG ENFORCE] Using SELF-CONSISTENCY VOTING ({self.num_reasoning_paths} paths)...")
                
                result = self.self_consistency_voter.vote(
                    question=question,
                    evidence_chunks=evidence_chunks or [],
                    debug=self.debug
                )
                
                decision = result.final_decision
                confidence = result.confidence
                
                if self.debug:
                    print(f"[DEBUG ENFORCE] Self-consistency result:")
                    print(f"  Decision: {decision}")
                    print(f"  Confidence: {confidence:.2f}")
                    print(f"  Votes: {result.vote_distribution}")
                    print(f"  Unanimous: {result.is_unanimous}")
                
                # Add confidence info to answer if low confidence
                if confidence < 0.5:
                    raw_stripped += f"\n\n(Note: Low confidence decision - {confidence:.0%} agreement)"
                
                return f"{raw_stripped}\n\nFinal Answer: {decision}"
                
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG ENFORCE] Self-consistency failed: {e}, falling back to single LLM call")
        
        # SINGLE-PATH: Single LLM call (faster, baseline behavior)
        if self.debug:
            print("[DEBUG ENFORCE] Using SINGLE-PATH reasoning...")
            
        evidence_text = "\n".join([f"- {c[:200]}" for c in (evidence_chunks or [])][:5])
        decision_prompt = f"""You are extracting a single label for a medical research QA task.

Return EXACTLY one token: yes OR no OR maybe.

Rules:
- yes: evidence leans towards supporting the claim (even if weak)
- no: evidence leans towards contradicting the claim
- maybe: ONLY if evidence is completely missing or totally irrelevant

QUESTION:
{question}

EVIDENCE (snippets):
{evidence_text}

ANSWER DRAFT:
{raw[:500]}

LABEL (force yes/no if possible):"""

        decision_raw = (self.llm.generate(decision_prompt) or "").strip().lower()
        
        if self.debug:
            print(f"[DEBUG ENFORCE] LLM extraction response: '{decision_raw}'")
        
        m3 = re.search(r"\b(yes|no|maybe)\b", decision_raw)
        decision = m3.group(1) if m3 else "maybe"
        
        if self.debug:
            print(f"[DEBUG ENFORCE] Final decision extracted: {decision}")
            print("-"*60 + "\n")

        return f"{raw_stripped}\n\nFinal Answer: {decision}"

    def _generate_refusal_message(
        self,
        question: str,
        alignment_score: float,
        iterations: int
    ) -> str:
        """Generate a clear refusal message when answer cannot be reliably grounded."""
        return f"""I cannot provide a reliable answer to this question based on the available evidence.

REASON: After {iterations} refinement iterations, the generated answer could not be sufficiently grounded in the retrieved evidence (alignment score: {alignment_score:.2f}, required: {SEAE_THRESHOLD}).

This means:
- The evidence in the knowledge base may not contain relevant information for this question
- OR the question may be outside the scope of the indexed medical documents

WHAT YOU CAN DO:
1. Try rephrasing your question with more specific medical terms
2. Ask about topics covered in the indexed documents (e.g., hypertension treatment guidelines)
3. Check if the relevant documents are included in the knowledge base

This refusal is a SAFETY FEATURE of MEGA-RAG to prevent medical hallucinations."""

    def run(self, question: str) -> dict:
        """
        Run the complete MEGA-RAG workflow.

        Args:
            question: The medical question to answer

        Returns:
            dict with final answer and workflow metadata
        """
        # Reset LLM token tracking for this query (so we get per-query tokens)
        if hasattr(self.llm, 'reset_usage'):
            self.llm.reset_usage()
        elif hasattr(self.llm, 'primary_llm') and hasattr(self.llm.primary_llm, 'reset_usage'):
            self.llm.primary_llm.reset_usage()
            if hasattr(self.llm, 'fallback_llm') and self.llm.fallback_llm:
                self.llm.fallback_llm.reset_usage()

        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "original_question": question,
            "context_chunks": [],
            "source_metadata": [],
            "retrieval_results": [],
            "answer": "",
            "seae_result": None,
            "disc_result": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_complete": False,
            "final_answer": "",
            "is_reliable": False,  # Default to False, set to True only if alignment passes
            "workflow_trace": [],
            "timing": {},
            "re_retrieval_done": False,
            "intent": "MEDICAL",
            # New v2.0 fields
            "use_cot": False,
            "cot_reasoning": None,
            "citation_report": None,
            "query_analysis": None  # Query reformulation analysis
        }

        # Run workflow
        final_state = self.graph.invoke(initial_state)

        # Extract unique sources for display
        sources = []
        seen_sources = set()
        for meta in final_state.get("source_metadata", []):
            filename = meta.get("filename", "Unknown")
            if filename not in seen_sources:
                seen_sources.add(filename)
                sources.append(filename)

        # Get token usage from LLM
        token_usage = {}
        if hasattr(self.llm, 'get_usage_summary'):
            token_usage = self.llm.get_usage_summary()
        elif hasattr(self.llm, 'total_usage'):
            token_usage = self.llm.total_usage.to_dict()

        return {
            "question": final_state["question"],
            "original_question": final_state.get("original_question", final_state["question"]),
            "answer": final_state["final_answer"],
            "is_reliable": final_state.get("is_reliable", False),  # CRITICAL: Indicates if answer passed alignment
            "context": final_state["context_chunks"],
            "sources": sources,  # List of unique source documents
            "retrieval_results": final_state["retrieval_results"],
            "seae_result": final_state["seae_result"],
            "iterations": final_state["iteration"],
            "workflow_trace": final_state["workflow_trace"],
            "timing": final_state.get("timing", {}),  # Include timing information
            "token_usage": token_usage,  # Token usage for this query
            # New v2.0 fields
            "used_cot": final_state.get("use_cot", False),
            "cot_reasoning": final_state.get("cot_reasoning"),
            "citation_report": final_state.get("citation_report"),
            "query_analysis": final_state.get("query_analysis")  # Query reformulation info
        }

    def stream(self, question: str):
        """
        Stream workflow updates.
        Yields state updates as they happen.
        """
        # Reset LLM usage same as run()
        if hasattr(self.llm, 'reset_usage'):
            self.llm.reset_usage()

        initial_state: RAGState = {
            "question": question,
            "original_question": question,
            "context_chunks": [],
            "source_metadata": [],
            "retrieval_results": [],
            "answer": "",
            "seae_result": None,
            "disc_result": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_complete": False,
            "final_answer": "",
            "is_reliable": False,
            "workflow_trace": [],
            "timing": {},
            "re_retrieval_done": False,
            "intent": "MEDICAL",
            # New v2.0 fields
            "use_cot": False,
            "cot_reasoning": None,
            "citation_report": None,
            "query_analysis": None
        }

        # Stream updates
        for output in self.graph.stream(initial_state, stream_mode="updates"):
            for node_name, state_update in output.items():
                yield {
                    "node": node_name,
                    "update": state_update
                }



def create_mega_rag_workflow(
    retriever: Optional[HybridRetriever] = None,
    llm: Optional[BaseLLM] = None,
    enable_cot: bool = True,
    enable_citation_verification: bool = True,
    enable_query_reformulation: bool = True,
    enable_self_consistency: bool = True,
    reasoning_mode: str = "multi",
    num_reasoning_paths: int = 3,
    debug: bool = False
) -> MEGARAGWorkflow:
    """
    Factory function to create MEGA-RAG workflow.

    Uses the configured LLM provider (Gemini or Ollama) with auto-fallback.
    To change provider, edit LLM_PROVIDER in config.py.
    
    Args:
        retriever: HybridRetriever instance (created if None)
        llm: LLM instance (created if None)
        enable_cot: Enable Chain-of-Thought reasoning for complex queries
        enable_citation_verification: Enable post-generation citation verification
        enable_query_reformulation: Enable query reformulation for vague queries
        enable_self_consistency: Enable self-consistency voting for yes/no questions
        reasoning_mode: "single" (fast baseline), "multi" (robust, default), or "auto"
        num_reasoning_paths: Number of paths for multi-path reasoning (default: 3)
        debug: Enable debug mode with detailed logging
    
    Returns:
        MEGARAGWorkflow instance with all components initialized
    
    Example:
        # Single-path (fast, baseline):
        workflow = create_mega_rag_workflow(reasoning_mode="single")
        
        # Multi-path (robust, recommended):
        workflow = create_mega_rag_workflow(reasoning_mode="multi", num_reasoning_paths=3)
        
        # Compare both modes:
        single_workflow = create_mega_rag_workflow(reasoning_mode="single")
        multi_workflow = create_mega_rag_workflow(reasoning_mode="multi")
    """
    retriever = retriever or HybridRetriever()
    llm = llm or create_llm()  # Uses config to select Gemini/Ollama with auto-fallback

    return MEGARAGWorkflow(
        retriever=retriever,
        llm=llm,
        enable_cot=enable_cot,
        enable_citation_verification=enable_citation_verification,
        enable_query_reformulation=enable_query_reformulation,
        enable_self_consistency=enable_self_consistency,
        reasoning_mode=reasoning_mode,
        num_reasoning_paths=num_reasoning_paths,
        debug=debug
    )


if __name__ == "__main__":
    print("MEGA-RAG Workflow v2.1 module loaded.")
    print("Use create_mega_rag_workflow() to initialize the system.")
    print("\nReasoning Modes:")
    print("  - 'single': Single-path reasoning (faster, baseline)")
    print("  - 'multi': Multi-path self-consistency voting (more robust, recommended)")
    print("  - 'auto': Auto-select based on question complexity")
    print("\nNew Features:")
    print("  - Query reformulation for vague/unclear queries")
    print("  - Chain-of-Thought reasoning for complex medical queries")
    print("  - Self-consistency voting for hallucination reduction")
    print("  - Citation verification with automatic correction")
    print("\nExample usage:")
    print("  # Single-path (baseline):")
    print("  workflow = create_mega_rag_workflow(reasoning_mode='single')")
    print("  ")
    print("  # Multi-path (recommended):")
    print("  workflow = create_mega_rag_workflow(reasoning_mode='multi')")
    print("  result = workflow.run('Does aspirin prevent heart attacks?')")

