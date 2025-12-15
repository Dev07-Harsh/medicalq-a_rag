"""
Query Reformulation Module for MEGA-RAG

Handles vague, ambiguous, or poorly-formed medical queries by:
1. Detecting query quality issues (too short, vague, missing context)
2. Using LLM to reformulate into clearer medical questions
3. Expanding with medical terminology and context
4. Generating sub-queries for complex multi-part questions

Inspired by:
- Query2Doc (Wang et al., 2023)
- Step-Back Prompting (Zheng et al., 2023)
- HyDE - Hypothetical Document Embeddings (Gao et al., 2022)
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class QueryAnalysis:
    """Analysis of query quality and characteristics."""
    original_query: str
    is_vague: bool
    is_too_short: bool
    is_ambiguous: bool
    is_complex: bool
    missing_context: List[str]
    detected_intent: str  # symptom_query, treatment_query, diagnosis_query, etc.
    confidence: float
    suggestions: List[str]


@dataclass
class ReformulatedQuery:
    """Result of query reformulation."""
    original_query: str
    reformulated_query: str
    sub_queries: List[str]
    expansion_terms: List[str]
    hypothetical_answer: Optional[str]  # For HyDE-style retrieval
    reformulation_reason: str
    was_modified: bool


class QueryReformulator:
    """
    Reformulates vague or poorly-formed medical queries into clearer questions.
    
    Uses multiple strategies:
    1. Clarity enhancement - adds specificity to vague queries
    2. Medical terminology - adds proper medical terms
    3. Query decomposition - breaks complex queries into sub-questions
    4. HyDE - generates hypothetical answer for better retrieval
    """
    
    # Minimum query length to be considered non-vague
    MIN_QUERY_LENGTH = 15
    MIN_WORD_COUNT = 3
    
    # Vague/ambiguous patterns
    VAGUE_PATTERNS = [
        r'^what\s+(is|are)\s+\w+\??$',  # "what is X?" without context
        r'^tell\s+me\s+about',
        r'^explain\s+\w+\??$',
        r'^how\s+to\s+\w+\??$',  # "how to treat?" without specifics
        r'^(help|info|information)\s*(about|on|for)?',
        r'^\w+\??$',  # Single word queries
    ]
    
    # Medical question type patterns
    INTENT_PATTERNS = {
        'treatment_query': [
            r'treat(ment|ing)?', r'therap(y|ies)', r'medication', r'drug',
            r'first[- ]line', r'manage(ment)?', r'cure', r'prescribe'
        ],
        'diagnosis_query': [
            r'diagnos(e|is|tic)', r'test(s|ing)?', r'screen(ing)?',
            r'detect(ion)?', r'identify', r'confirm'
        ],
        'symptom_query': [
            r'symptom', r'sign(s)?', r'present(ation|ing)?', r'manifest',
            r'feel(ing)?', r'experience', r'suffer'
        ],
        'prognosis_query': [
            r'prognos(is|tic)', r'outcome', r'survival', r'mortality',
            r'life expectancy', r'recover(y)?', r'long[- ]term'
        ],
        'mechanism_query': [
            r'mechanism', r'how\s+(does|do)', r'why\s+(does|do)',
            r'pathophysiology', r'cause(s)?', r'etiology'
        ],
        'prevention_query': [
            r'prevent(ion)?', r'avoid', r'reduce\s+risk', r'prophylax'
        ],
        'comparison_query': [
            r'(vs|versus|compared?\s+to)', r'difference\s+between',
            r'better\s+than', r'which\s+(is\s+)?better'
        ]
    }
    
    # Medical domain keywords that indicate the query is already specific
    SPECIFIC_MEDICAL_TERMS = [
        'hypertension', 'diabetes', 'mellitus', 'carcinoma', 'syndrome',
        'therapy', 'inhibitor', 'receptor', 'enzyme', 'antibody',
        'mg', 'dosage', 'contraindication', 'adverse', 'efficacy',
        'randomized', 'clinical trial', 'meta-analysis', 'guideline'
    ]
    
    def __init__(self, llm=None):
        """
        Initialize the query reformulator.
        
        Args:
            llm: Optional LLM instance for advanced reformulation
        """
        self.llm = llm
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query quality and detect issues.
        
        Args:
            query: The user's query
            
        Returns:
            QueryAnalysis with detected issues and suggestions
        """
        query = query.strip()
        query_lower = query.lower()
        words = query.split()
        
        # Check basic length
        is_too_short = len(query) < self.MIN_QUERY_LENGTH or len(words) < self.MIN_WORD_COUNT
        
        # Check for vague patterns
        is_vague = False
        for pattern in self.VAGUE_PATTERNS:
            if re.search(pattern, query_lower):
                is_vague = True
                break
        
        # Check if query has specific medical terms (less likely to be vague)
        has_specific_terms = any(term in query_lower for term in self.SPECIFIC_MEDICAL_TERMS)
        if has_specific_terms:
            is_vague = False  # Override if specific terms present
        
        # Detect ambiguity (multiple possible interpretations)
        is_ambiguous = self._detect_ambiguity(query_lower)
        
        # Detect complex multi-part questions
        is_complex = self._detect_complexity(query_lower)
        
        # Detect missing context
        missing_context = self._detect_missing_context(query_lower)
        
        # Detect query intent
        detected_intent = self._detect_intent(query_lower)
        
        # Calculate confidence
        confidence = 1.0
        if is_too_short:
            confidence -= 0.3
        if is_vague:
            confidence -= 0.3
        if is_ambiguous:
            confidence -= 0.2
        if missing_context:
            confidence -= 0.1 * len(missing_context)
        confidence = max(0.0, confidence)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            query, is_vague, is_too_short, missing_context, detected_intent
        )
        
        return QueryAnalysis(
            original_query=query,
            is_vague=is_vague,
            is_too_short=is_too_short,
            is_ambiguous=is_ambiguous,
            is_complex=is_complex,
            missing_context=missing_context,
            detected_intent=detected_intent,
            confidence=confidence,
            suggestions=suggestions
        )
    
    def _detect_ambiguity(self, query: str) -> bool:
        """Detect if query has multiple possible interpretations."""
        ambiguous_terms = [
            'it', 'this', 'that', 'they', 'them',  # Pronouns without referent
            'the condition', 'the disease', 'the problem',  # Vague references
            'something', 'anything', 'whatever'
        ]
        return any(term in query for term in ambiguous_terms)
    
    def _detect_complexity(self, query: str) -> bool:
        """Detect if query contains multiple sub-questions."""
        complexity_indicators = [
            ' and ', ' or ', ' also ', ' plus ',
            'first,', 'second,', 'additionally',
            '?', ';'  # Multiple question marks or semicolons
        ]
        indicator_count = sum(1 for ind in complexity_indicators if ind in query)
        return indicator_count >= 2 or query.count('?') > 1
    
    def _detect_missing_context(self, query: str) -> List[str]:
        """Detect what context might be missing from the query."""
        missing = []
        
        # Treatment queries should specify condition
        if any(re.search(p, query) for p in self.INTENT_PATTERNS['treatment_query']):
            if not any(term in query for term in ['for', 'of', 'in patients with']):
                missing.append('condition/disease')
        
        # Symptom queries should specify context
        if 'symptom' in query:
            if 'of' not in query and 'from' not in query:
                missing.append('disease context')
        
        # Dosage queries should specify patient population
        if 'dosage' in query or 'dose' in query:
            if 'adult' not in query and 'pediatric' not in query and 'child' not in query:
                missing.append('patient population')
        
        return missing
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        return 'general_query'
    
    def _generate_suggestions(
        self,
        query: str,
        is_vague: bool,
        is_too_short: bool,
        missing_context: List[str],
        intent: str
    ) -> List[str]:
        """Generate suggestions for improving the query."""
        suggestions = []
        
        if is_too_short:
            suggestions.append("Add more details about your specific medical question")
        
        if is_vague:
            suggestions.append("Specify the condition, treatment, or symptom you're asking about")
        
        if 'condition/disease' in missing_context:
            suggestions.append("Mention the specific disease or condition (e.g., 'treatment for hypertension')")
        
        if 'patient population' in missing_context:
            suggestions.append("Specify the patient group (adult, pediatric, elderly, pregnant)")
        
        if intent == 'treatment_query':
            suggestions.append("Consider specifying: first-line vs alternative, contraindications, patient factors")
        
        return suggestions
    
    def reformulate(self, query: str, use_llm: bool = True) -> ReformulatedQuery:
        """
        Reformulate a query into a clearer medical question.
        
        Args:
            query: The user's original query
            use_llm: Whether to use LLM for advanced reformulation
            
        Returns:
            ReformulatedQuery with improved query and sub-queries
        """
        # Analyze the query first
        analysis = self.analyze_query(query)
        
        # If query is already good quality, still valid roughly, but we WANT expansion for retrieval recall.
        # So we skip the "return as-is" block to force expansion generation.
        # if analysis.confidence >= 0.7 and not analysis.is_complex: ... (REMOVED)
        
        # Apply reformulation strategies
        reformulated = query
        sub_queries = []
        expansion_terms = []
        hypothetical_answer = None
        reasons = []
        
        # Strategy 1: Rule-based reformulation for simple cases
        if analysis.is_too_short or analysis.is_vague:
            reformulated, rule_expansions = self._rule_based_reformulation(query, analysis)
            expansion_terms.extend(rule_expansions)
            reasons.append("Added specificity to vague query")
        
        # Strategy 2: Decompose complex queries
        if analysis.is_complex:
            sub_queries = self._decompose_query(query)
            reasons.append(f"Decomposed into {len(sub_queries)} sub-queries")
        
        # Strategy 3: LLM-based reformulation for difficult/medical cases
        # ALWAYS try LLM reformulation if LLM is available, to richer terms
        if use_llm and self.llm: 
            llm_result = self._llm_reformulation(query, analysis)
            if llm_result:
                # Only replace query if LLM explicitly suggests checking specific things
                # But mostly we want the expansion terms or subqueries
                if analysis.confidence < 0.7:
                     reformulated = llm_result.get('reformulated', reformulated)
                     
                if llm_result.get('sub_queries'):
                    sub_queries.extend(llm_result['sub_queries'])
                hypothetical_answer = llm_result.get('hypothetical_answer')
                reasons.append("LLM-enhanced reformulation")
        
        # Strategy 4: Add medical expansion terms
        expansion_terms.extend(self._get_expansion_terms(reformulated, analysis.detected_intent))
        
        # CRITICAL FIX: Append expansion terms to the query so retrieval actually uses them
        unique_expansions = list(set(expansion_terms))
        if unique_expansions:
            reformulated = f"{reformulated} {' '.join(unique_expansions)}"
        
        was_modified = reformulated != query or len(sub_queries) > 0
        
        return ReformulatedQuery(
            original_query=query,
            reformulated_query=reformulated,
            sub_queries=sub_queries,
            expansion_terms=unique_expansions,
            hypothetical_answer=hypothetical_answer,
            reformulation_reason="; ".join(reasons) if reasons else "Expanded with search terms",
            was_modified=was_modified
        )
    
    def _rule_based_reformulation(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> Tuple[str, List[str]]:
        """Apply rule-based reformulation for common patterns."""
        reformulated = query
        expansions = []
        
        # Pattern: "what is X" -> "What is X, including its causes, symptoms, and treatment?"
        if re.match(r'^what\s+(is|are)\s+(\w+)\??$', query.lower()):
            match = re.match(r'^what\s+(is|are)\s+(\w+)\??$', query.lower())
            if match:
                term = match.group(2)
                reformulated = f"What is {term}, including its definition, causes, symptoms, diagnosis, and treatment options?"
                expansions.extend(['definition', 'etiology', 'clinical features', 'management'])
        
        # Pattern: "treatment for X" -> add context
        if analysis.detected_intent == 'treatment_query':
            if 'first-line' not in query.lower() and 'guideline' not in query.lower():
                reformulated = reformulated.rstrip('?') + " according to current clinical guidelines?"
                expansions.extend(['first-line', 'guideline', 'recommended'])
        
        # Pattern: Single word queries
        if len(query.split()) == 1:
            term = query.strip('?')
            reformulated = f"What is {term} in the medical context, including its clinical significance?"
            expansions.extend(['medical', 'clinical', 'definition'])
        
        return reformulated, expansions
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into sub-queries."""
        sub_queries = []
        
        # Split on "and" for compound questions
        if ' and ' in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Only meaningful parts
                    # Ensure it's a complete question
                    if not part.endswith('?'):
                        part = part + '?'
                    if not part[0].isupper():
                        part = part[0].upper() + part[1:]
                    sub_queries.append(part)
        
        # Split on multiple question marks
        if query.count('?') > 1:
            parts = query.split('?')
            for part in parts:
                part = part.strip()
                if len(part) > 10:
                    sub_queries.append(part + '?')
        
        return sub_queries[:3]  # Max 3 sub-queries
    
    def _llm_reformulation(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> Optional[Dict]:
        """Use LLM to reformulate the query."""
        if not self.llm:
            return None
        
        prompt = f"""You are a medical query reformulation expert. Improve the following query to be clearer and more specific for searching medical literature.

Original Query: {query}

Issues Detected:
- Too short: {analysis.is_too_short}
- Vague: {analysis.is_vague}
- Ambiguous: {analysis.is_ambiguous}
- Missing context: {', '.join(analysis.missing_context) if analysis.missing_context else 'None'}
- Detected intent: {analysis.detected_intent}

Instructions:
1. Reformulate into a clear, specific medical question
2. Add relevant medical terminology
3. If the query has multiple parts, list them as separate sub-questions
4. Generate a brief hypothetical answer that would help find relevant documents

Respond in this exact format:
REFORMULATED: [Your improved query]
SUB_QUERIES: [Comma-separated sub-queries, or "None"]
HYPOTHETICAL: [A 1-2 sentence hypothetical answer]
"""
        
        try:
            response = self.llm.generate(prompt)
            
            # Parse the response
            result = {}
            
            reformulated_match = re.search(r'REFORMULATED:\s*(.+?)(?=SUB_QUERIES:|$)', response, re.DOTALL)
            if reformulated_match:
                result['reformulated'] = reformulated_match.group(1).strip()
            
            subq_match = re.search(r'SUB_QUERIES:\s*(.+?)(?=HYPOTHETICAL:|$)', response, re.DOTALL)
            if subq_match:
                subq_text = subq_match.group(1).strip()
                if subq_text.lower() != 'none':
                    result['sub_queries'] = [q.strip() for q in subq_text.split(',') if q.strip()]
            
            hypo_match = re.search(r'HYPOTHETICAL:\s*(.+?)$', response, re.DOTALL)
            if hypo_match:
                result['hypothetical_answer'] = hypo_match.group(1).strip()
            
            return result if result else None
            
        except Exception as e:
            print(f"LLM reformulation failed: {e}")
            return None
    
    def _get_expansion_terms(self, query: str, intent: str) -> List[str]:
        """Get medical expansion terms based on query and intent."""
        expansions = []
        query_lower = query.lower()
        
        # Intent-based expansions
        intent_expansions = {
            'treatment_query': ['therapy', 'management', 'medication', 'intervention'],
            'diagnosis_query': ['diagnostic', 'testing', 'screening', 'assessment'],
            'symptom_query': ['clinical features', 'presentation', 'manifestations'],
            'prognosis_query': ['outcome', 'survival', 'mortality', 'long-term'],
            'mechanism_query': ['pathophysiology', 'etiology', 'mechanism of action'],
            'prevention_query': ['prophylaxis', 'preventive', 'risk reduction'],
            'comparison_query': ['comparative', 'efficacy', 'superiority']
        }
        
        if intent in intent_expansions:
            expansions.extend(intent_expansions[intent])
        
        # Common medical term expansions
        term_expansions = {
            'blood pressure': ['hypertension', 'antihypertensive'],
            'sugar': ['glucose', 'diabetes', 'glycemic'],
            'heart': ['cardiac', 'cardiovascular', 'coronary'],
            'kidney': ['renal', 'nephro'],
            'liver': ['hepatic', 'hepato'],
            'cancer': ['carcinoma', 'malignancy', 'oncology', 'tumor'],
            'stroke': ['cerebrovascular', 'CVA', 'ischemic stroke'],
            'infection': ['infectious', 'antimicrobial', 'antibiotic']
        }
        
        for term, synonyms in term_expansions.items():
            if term in query_lower:
                expansions.extend(synonyms[:2])
        
        return list(set(expansions))[:5]  # Max 5 expansion terms
    
    def generate_hyde_document(self, query: str) -> Optional[str]:
        """
        Generate a hypothetical document for HyDE-style retrieval.
        
        HyDE (Hypothetical Document Embeddings) generates a fake but plausible
        answer, then uses its embedding to find similar real documents.
        
        Args:
            query: The user's query
            
        Returns:
            A hypothetical answer document or None
        """
        if not self.llm:
            return None
        
        prompt = f"""Generate a brief, factual medical paragraph that would answer this question. 
Write as if it's from a medical textbook or clinical guideline.
Be specific but don't make up statistics or citations.

Question: {query}

Medical Paragraph:"""
        
        try:
            response = self.llm.generate(prompt)
            # Clean up the response
            response = response.strip()
            if len(response) > 50:  # Only return if substantial
                return response
            return None
        except Exception:
            return None


# Convenience function for quick reformulation
def reformulate_query(query: str, llm=None) -> ReformulatedQuery:
    """
    Convenience function to reformulate a query.
    
    Args:
        query: The user's query
        llm: Optional LLM instance
        
    Returns:
        ReformulatedQuery result
    """
    reformulator = QueryReformulator(llm=llm)
    return reformulator.reformulate(query)


if __name__ == "__main__":
    # Test the reformulator
    print("Query Reformulator Test")
    print("=" * 50)
    
    test_queries = [
        "diabetes",
        "what is hypertension",
        "treatment",
        "how to cure cancer",
        "What is the first-line treatment for hypertension in adults with diabetes?",
        "symptoms and treatment of flu and when to see doctor?",
        "blood pressure medication side effects and interactions with other drugs",
    ]
    
    reformulator = QueryReformulator()
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        analysis = reformulator.analyze_query(query)
        print(f"  Confidence: {analysis.confidence:.2f}")
        print(f"  Vague: {analysis.is_vague}, Short: {analysis.is_too_short}")
        print(f"  Intent: {analysis.detected_intent}")
        
        result = reformulator.reformulate(query, use_llm=False)
        if result.was_modified:
            print(f"  Reformulated: {result.reformulated_query}")
            if result.sub_queries:
                print(f"  Sub-queries: {result.sub_queries}")
            if result.expansion_terms:
                print(f"  Expansions: {result.expansion_terms}")
        else:
            print("  ✓ No reformulation needed")
