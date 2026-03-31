"""
LLM Integration for MEGA-RAG
Supports both Gemini (cloud) and Ollama (local) with auto-fallback.

To switch between models, change LLM_PROVIDER in config.py:
  - "gemini": Use Google Gemini API (cloud)
  - "ollama": Use local Ollama server with Mistral/Llama

Auto-fallback: When Gemini hits rate limits, automatically switches to Ollama.
"""
import os
import requests
from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# =============================================================================
# Token Usage Tracking
# =============================================================================

@dataclass
class TokenUsage:
    """Track token usage for a single call or cumulative."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

    def to_dict(self) -> dict:
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text (rough approximation).
    Uses ~4 characters per token as a reasonable estimate.
    """
    return max(1, len(text) // 4)

from mega_rag.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_AUTO_FALLBACK,
    LLM_FALLBACK_CHAIN,
)


# =============================================================================
# Base LLM Interface
# =============================================================================

class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self):
        # Token tracking
        self._last_usage = TokenUsage()
        self._cumulative_usage = TokenUsage()
        self._call_count = 0

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from the LLM."""
        pass

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None
    ) -> str:
        """Generate answer for a medical question using retrieved context."""
        pass

    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of the user query.
        Returns: 'MEDICAL', 'GREETING', or 'OFF_TOPIC'
        
        NOTE: Made very permissive to avoid false positives with medical models like Meditron.
        """
        # Quick check for obvious greetings (without using LLM)
        query_lower = query.lower().strip()
        greeting_only_patterns = [
            "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
            "how are you", "what's up", "greetings"
        ]
        if query_lower in greeting_only_patterns or len(query_lower) < 10:
            return "GREETING"
        
        # For PubMedQA-style questions, assume MEDICAL (bypass LLM classification)
        # These are research questions that should always be treated as medical
        if any(word in query_lower for word in [
            "?", "does", "is ", "are ", "can ", "do ", "should", "would",
            "patient", "treatment", "diagnosis", "study", "clinical", "therapy",
            "disease", "disorder", "syndrome", "symptom", "medical", "health",
            "drug", "medicine", "surgery", "hospital", "doctor", "physician"
        ]):
            return "MEDICAL"
        
        # Default to MEDICAL for anything that looks like a question
        return "MEDICAL"

    def __call__(self, prompt: str) -> str:
        """Allow using instance as callable."""
        return self.generate(prompt)

    def _track_usage(self, prompt_tokens: int, completion_tokens: int):
        """Track token usage for this call."""
        self._last_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        self._cumulative_usage = self._cumulative_usage + self._last_usage
        self._call_count += 1

    @property
    def last_usage(self) -> TokenUsage:
        """Get token usage from the last call."""
        return self._last_usage

    @property
    def total_usage(self) -> TokenUsage:
        """Get cumulative token usage across all calls."""
        return self._cumulative_usage

    @property
    def call_count(self) -> int:
        """Get total number of LLM calls."""
        return self._call_count

    def reset_usage(self):
        """Reset cumulative token tracking."""
        self._cumulative_usage = TokenUsage()
        self._call_count = 0

    def get_usage_summary(self) -> dict:
        """Get a summary of token usage."""
        return {
            'last_call': self._last_usage.to_dict(),
            'cumulative': self._cumulative_usage.to_dict(),
            'call_count': self._call_count,
            'avg_tokens_per_call': (
                self._cumulative_usage.total_tokens / self._call_count
                if self._call_count > 0 else 0
            )
        }

    def _is_yes_no_question(self, question: str) -> bool:
        """
        Detect if question expects a yes/no/maybe answer.
        
        Handles multiple PubMedQA formats:
        1. Direct questions: "Does X cause Y?"
        2. Colon format: "Topic description: is it effective?"
        3. Embedded questions: "Study of X: can it improve Y?"
        """
        import re
        question_lower = question.lower().strip()
        
        # Pattern 1: Questions starting with yes/no words
        yes_no_starters = [
            'does ', 'do ', 'is ', 'are ', 'can ', 'will ', 'would ', 'should ',
            'could ', 'has ', 'have ', 'was ', 'were ', 'did '
        ]
        if any(question_lower.startswith(starter) for starter in yes_no_starters):
            return True
        
        # Pattern 2: "Topic: is/does/can X?" format (common in PubMedQA)
        # Match patterns like ": is it", ": does it", ": can it", etc.
        colon_pattern = r':\s*(is|does|do|are|can|will|would|should|could|has|have|was|were|did)\s+'
        if re.search(colon_pattern, question_lower):
            return True
        
        # Pattern 3: Questions ending with "?" that contain yes/no indicators
        # Look for yes/no words anywhere in the question
        if question_lower.endswith('?'):
            yes_no_words = ['is it', 'does it', 'can it', 'are they', 'do they', 
                           'should it', 'will it', 'could it', 'has it', 'have they']
            if any(word in question_lower for word in yes_no_words):
                return True
        
        return False

    def _build_medical_prompt(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None,
        max_context_chars: int = 12000  # ~3000 tokens for context
    ) -> str:
        """Build the prompt for medical QA with citations.
        
        Args:
            question: The medical question
            context_chunks: Retrieved evidence chunks
            source_metadata: Source metadata for each chunk
            system_instruction: Optional custom system instruction
            max_context_chars: Maximum characters for context (default ~3000 tokens)
        """
        # TOKEN TRACKING: Estimate tokens for debugging
        total_context_chars = sum(len(c) for c in context_chunks)
        
        # OPTIMIZATION: Truncate or limit chunks if context is too large
        # This prevents overwhelming the model and improves response quality
        truncated_chunks = []
        current_chars = 0
        
        for i, chunk in enumerate(context_chunks):
            chunk_chars = len(chunk)
            if current_chars + chunk_chars > max_context_chars:
                # Truncate the last chunk to fit
                remaining = max_context_chars - current_chars
                if remaining > 200:  # Only include if meaningful content
                    truncated_chunks.append(chunk[:remaining] + "...")
                break
            truncated_chunks.append(chunk)
            current_chars += chunk_chars
        
        # Use truncated chunks
        context_chunks = truncated_chunks
        
        # Format context with detailed source names
        context_parts = []
        source_list = []  # Track unique sources for reference section

        for i, chunk in enumerate(context_chunks):
            # Build detailed source label
            source_name = "Unknown Source"
            page_info = ""

            if source_metadata and i < len(source_metadata):
                meta = source_metadata[i]
                source_name = meta.get('filename', 'Unknown Source')
                source_name = source_name.replace('.pdf', '').replace('_', ' ')

                # Add page number if available
                page_num = meta.get('page', meta.get('page_number', None))
                if page_num is not None:
                    page_info = f", Page {page_num}"

            full_source = f"{source_name}{page_info}"
            source_list.append(full_source)

            context_parts.append(f"[EVIDENCE {i+1}]\nSource: {full_source}\nContent: {chunk}")

        context = "\n\n" + "\n\n---\n\n".join(context_parts)

        # Build source reference list for the prompt
        unique_sources = list(dict.fromkeys(source_list))  # Preserve order, remove duplicates
        source_reference = "\n".join([f"  [{i+1}] {src}" for i, src in enumerate(unique_sources)])

        # Check if this is a yes/no question (like PubMedQA)
        is_yes_no = self._is_yes_no_question(question)

        # Default system instruction for medical QA - simplified for better local model compatibility
        if system_instruction is None:
            if is_yes_no:
                # Special prompt for yes/no/maybe questions (PubMedQA style)
                # CRITICAL: Very explicit format with strict instructions
                system_instruction = """You are a medical expert answering a yes/no/maybe research question.

TASK: Based ONLY on the evidence provided, determine if the answer is yes, no, or maybe.

IMPORTANT INSTRUCTIONS:
1. Read the QUESTION carefully - it asks whether something is true/effective/valid
2. Check the EVIDENCE for support or contradiction
3. Write 1-2 sentences explaining your reasoning with citations
4. YOU MUST end your response with EXACTLY this format on its own line:

Final Answer: yes

OR

Final Answer: no

OR

Final Answer: maybe

DECISION RULES (BE DECISIVE - avoid "maybe" unless truly uncertain):
- "yes" = The evidence SUPPORTS or CONFIRMS what the question asks
- "no" = The evidence CONTRADICTS or REFUTES what the question asks  
- "maybe" = ONLY use if evidence is truly MISSING or COMPLETELY INCONCLUSIVE

CRITICAL: Your response MUST end with "Final Answer: yes" or "Final Answer: no" or "Final Answer: maybe" on a separate line. This is required for evaluation."""
            else:
                system_instruction = """You are a medical assistant. Answer ONLY using the provided evidence.

RULES:
1. Use ONLY information from the evidence below
2. Cite sources after each fact: [Source: Document Name]
3. If evidence is missing, say "The evidence does not cover this"
4. Do NOT add information not in the evidence"""

        if is_yes_no:
            prompt = f"""{system_instruction}

EVIDENCE:
{context}

QUESTION: {question}

Analyze the evidence and provide your answer. You MUST end with "Final Answer: yes" or "Final Answer: no" or "Final Answer: maybe".

ANSWER:"""
        else:
            # Non yes/no mode: make the instructions explicit and short-output oriented.
            # This improves faithfulness for local medical models (e.g., Meditron) when max tokens is high.
            prompt = f"""{system_instruction}

SOURCES (use these exact names in citations):
{source_reference}

EVIDENCE:
{context}

QUESTION:
{question}

RESPONSE REQUIREMENTS:
1. Use ONLY information from EVIDENCE. If not supported, say: "The evidence does not cover this."
2. Be concise and clinical. Prefer 5-12 sentences.
3. Every sentence that states a medical fact MUST end with a citation in this format: [Source: Document Name]
4. Do NOT add outside knowledge, mechanisms, dosages, or guidelines unless explicitly present in EVIDENCE.
5. If evidence conflicts, state the conflict and cite both sources.

ANSWER:"""

        return prompt


# =============================================================================
# Gemini LLM (Cloud)
# =============================================================================

class GeminiLLM(BaseLLM):
    """
    Gemini LLM integration for medical QA generation.
    Uses Google's Gemini API (cloud-based).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GEMINI_MODEL,
        temperature: float = 0.3,
        max_output_tokens: int = 4096
    ):
        super().__init__()  # Initialize token tracking
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold

        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Safety settings - adjusted for medical content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=0.95,
            top_k=40
        )

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

        self.model_name = model_name
        self._genai = genai  # Store for token counting
        print(f"✓ Gemini LLM initialized: {model_name}")

    def generate(self, prompt: str) -> str:
        """Generate response from Gemini with token tracking."""
        try:
            response = self.model.generate_content(prompt)

            # Track token usage from response metadata
            prompt_tokens = estimate_tokens(prompt)  # Estimate for prompt
            completion_tokens = estimate_tokens(response.text) if response.text else 0

            # Try to get actual token counts from response if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                if hasattr(usage, 'prompt_token_count'):
                    prompt_tokens = usage.prompt_token_count
                if hasattr(usage, 'candidates_token_count'):
                    completion_tokens = usage.candidates_token_count

            self._track_usage(prompt_tokens, completion_tokens)
            return response.text
        except Exception as e:
            error_str = str(e)
            # Check for rate limit error
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit: {e}")
            print(f"Gemini generation error: {e}")
            return f"Error generating response: {error_str}"

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False
    ) -> str:
        """Generate answer for a medical question using retrieved context."""
        prompt = self._build_medical_prompt(
            question, context_chunks, source_metadata, system_instruction
        )
        
        if debug:
            # Token analysis
            prompt_tokens = estimate_tokens(prompt)
            context_tokens = sum(estimate_tokens(c) for c in context_chunks)
            print("\n" + "="*70)
            print("[DEBUG GEMINI] TOKEN ANALYSIS:")
            print("="*70)
            print(f"  📊 Total prompt tokens (estimated): {prompt_tokens:,}")
            print(f"  📊 Context tokens (estimated): {context_tokens:,}")
            print(f"  📊 Context chunks: {len(context_chunks)}")
            print(f"  📊 Prompt length (chars): {len(prompt):,}")
            print(f"  📊 Model limit: 1,000,000 tokens (Gemini)")
            print(f"  📊 Recommended: <100,000 tokens for best quality")
            if prompt_tokens > 100000:
                print(f"  ⚠️  WARNING: Prompt may be too long for optimal results!")
            print("="*70)
            print("[DEBUG GEMINI] PROMPT SENT:")
            print("="*70)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("="*70)
        
        response = self.generate(prompt)
        
        if debug:
            print("\n" + "="*70)
            print("[DEBUG GEMINI] RAW RESPONSE:")
            print("="*60)
            print(response)
            print("="*60)
            import re
            final_match = re.search(r'final\s*answer\s*[:\s]*\b(yes|no|maybe)\b', response.lower())
            if final_match:
                print(f"[DEBUG GEMINI] ✓ Found 'Final Answer': {final_match.group(1)}")
            else:
                print("[DEBUG GEMINI] ✗ NO 'Final Answer' pattern found!")
            print("="*60 + "\n")
        
        return response


# =============================================================================
# Ollama LLM (Local)
# =============================================================================

class OllamaLLM(BaseLLM):
    """
    Ollama LLM integration for local inference.
    Supports Mistral, Llama 3.1, Phi-3, and other Ollama models.

    Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull mistral
    3. Start server: ollama serve (runs on localhost:11434)
    """

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.25,  # Moderate temp for better maybe detection
        # Default kept conservative for faithfulness; override via OLLAMA_MAX_TOKENS env var.
        max_tokens: int = OLLAMA_MAX_TOKENS
    ):
        super().__init__()  # Initialize token tracking
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Verify Ollama is running
        if not self._check_ollama_status():
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is installed and running:\n"
                "  1. Install: https://ollama.ai\n"
                "  2. Pull model: ollama pull mistral\n"
                "  3. Start: ollama serve"
            )

        print(f"✓ Ollama LLM initialized: {model_name} (local)")

    def _check_ollama_status(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        """Generate response from Ollama with token tracking."""
        try:
            # Use raw=True to bypass Ollama's built-in chat template
            # This is critical for Meditron which has a Vicuna template that interferes
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "raw": True,  # CRITICAL: Bypass built-in chat template
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "stop": ["\n\nUSER:", "\nUSER:", "USER:", "\n\n\n"]  # Stop at next turn
                    }
                },
                timeout=120  # 2 minute timeout for generation
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "")

                # Ollama provides token counts in response
                prompt_tokens = data.get("prompt_eval_count", estimate_tokens(prompt))
                completion_tokens = data.get("eval_count", estimate_tokens(result))

                self._track_usage(prompt_tokens, completion_tokens)
                return result
            else:
                return f"Ollama error: {response.status_code} - {response.text}"

        except requests.exceptions.Timeout:
            return "Error: Ollama request timed out. The model may be loading."
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False
    ) -> str:
        """Generate answer for a medical question using retrieved context."""
        # For Ollama/Mistral, use smaller context limit due to 32K context window
        prompt = self._build_medical_prompt(
            question, context_chunks, source_metadata, system_instruction,
            max_context_chars=8000  # ~2000 tokens, safer for Mistral's 32K limit
        )
        
        if debug:
            # Token analysis - CRITICAL for Ollama/Mistral
            prompt_tokens = estimate_tokens(prompt)
            context_tokens = sum(estimate_tokens(c) for c in context_chunks)
            print("\n" + "="*70)
            print("[DEBUG OLLAMA] TOKEN ANALYSIS:")
            print("="*70)
            print(f"  📊 Total prompt tokens (estimated): {prompt_tokens:,}")
            print(f"  📊 Context tokens (estimated): {context_tokens:,}")
            print(f"  📊 Context chunks used: {len(context_chunks)}")
            print(f"  📊 Prompt length (chars): {len(prompt):,}")
            print(f"  📊 Model limit: ~32,000 tokens (Mistral)")
            print(f"  📊 Recommended: <16,000 tokens for best quality")
            if prompt_tokens > 16000:
                print(f"  ⚠️  WARNING: Prompt exceeds recommended limit for Mistral!")
                print(f"  ⚠️  This may cause truncation or poor responses!")
            if prompt_tokens > 28000:
                print(f"  🚨 CRITICAL: Prompt near/exceeds Mistral context limit!")
            print("="*70)
            print("[DEBUG OLLAMA] PROMPT SENT:")
            print("="*70)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("="*70)
        
        response = self.generate(prompt)
        
        if debug:
            print("\n" + "="*70)
            print("[DEBUG OLLAMA] RAW RESPONSE:")
            print("="*70)
            print(response)
            print("="*70)
            import re
            final_match = re.search(r'final\s*answer\s*[:\s]*\b(yes|no|maybe)\b', response.lower())
            if final_match:
                print(f"[DEBUG OLLAMA] ✓ Found 'Final Answer': {final_match.group(1)}")
            else:
                print("[DEBUG OLLAMA] ✗ NO 'Final Answer' pattern found!")
            print("="*70 + "\n")
        
        return response


# =============================================================================
# Groq LLM (Cloud - Free Tier)
# =============================================================================

class GroqLLM(BaseLLM):
    """
    Groq Cloud LLM integration with built-in rate limiting.
    Free tier: 30 RPM, 14,400 RPD — ultra-fast inference on custom LPU hardware.

    Rate limiter ensures we stay under 28 RPM (safe margin below 30 RPM limit).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = GROQ_MODEL,
        temperature: float = 0.3,
        max_tokens: int = GROQ_MAX_TOKENS,
        max_rpm: int = 28,  # Stay below 30 RPM limit
    ):
        super().__init__()
        self.api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in .env or environment.\n"
                "Get a free key at: https://console.groq.com/keys"
            )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

        # Rate limiter: track call timestamps in a sliding window
        import collections
        self._call_times = collections.deque()
        self._max_rpm = max_rpm
        self._min_interval = 60.0 / max_rpm  # ~2.14s between calls at 28 RPM

        print(f"✓ Groq LLM initialized: {model_name} (rate limit: {max_rpm} RPM)")

    def _wait_for_rate_limit(self):
        """Block until we're safe to make the next API call."""
        import time as _time
        from mega_rag.utils.logger import get_logger
        log = get_logger("groq")

        now = _time.time()

        # Remove calls older than 60 seconds
        while self._call_times and now - self._call_times[0] > 60:
            self._call_times.popleft()

        # If we've hit the RPM limit, wait until the oldest call expires
        if len(self._call_times) >= self._max_rpm:
            wait_until = self._call_times[0] + 60.0
            sleep_time = wait_until - now + 0.1
            if sleep_time > 0:
                log.info(f"[Rate limit] {len(self._call_times)}/{self._max_rpm} RPM — waiting {sleep_time:.1f}s")
                _time.sleep(sleep_time)

        # Also enforce minimum interval between consecutive calls
        if self._call_times:
            elapsed = now - self._call_times[-1]
            if elapsed < self._min_interval:
                _time.sleep(self._min_interval - elapsed)

        self._call_times.append(_time.time())

    def generate(self, prompt: str) -> str:
        """Generate response from Groq API with rate limiting and logging."""
        from mega_rag.utils.logger import get_logger
        log = get_logger("groq")

        self._wait_for_rate_limit()
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=60,
            )

            if response.status_code == 429:
                raise RateLimitError(f"Groq rate limit hit: {response.text}")

            if response.status_code != 200:
                return f"Groq error: {response.status_code} - {response.text}"

            data = response.json()
            result = data["choices"][0]["message"]["content"]

            # Track token usage from response
            usage = data.get("usage", {})
            prompt_tok = usage.get("prompt_tokens", estimate_tokens(prompt))
            comp_tok = usage.get("completion_tokens", estimate_tokens(result))
            self._track_usage(prompt_tok, comp_tok)

            cumulative = self._cumulative_usage.total_tokens
            log.debug(f"Call #{self._call_count}: {prompt_tok}+{comp_tok}={prompt_tok+comp_tok} tok | cumulative: {cumulative} tok | model: {self.model_name}")
            return result

        except RateLimitError as e:
            log.error(f"RATE LIMIT: {e}")
            raise
        except requests.exceptions.Timeout:
            log.error("TIMEOUT: Groq request timed out")
            return "Error: Groq request timed out."
        except requests.exceptions.RequestException as e:
            log.error(f"CONNECTION ERROR: {e}")
            return f"Error connecting to Groq: {str(e)}"

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
    ) -> str:
        """Generate answer using Groq with medical context."""
        # Use smaller context to conserve daily token quota
        # 70B model: 100K TPD limit → ~4000 chars/call
        # 8B model: 500K TPD limit → ~8000 chars/call
        max_chars = 4000 if "70b" in self.model_name else 8000
        prompt = self._build_medical_prompt(
            question, context_chunks, source_metadata, system_instruction,
            max_context_chars=max_chars,
        )

        if debug:
            prompt_tokens = estimate_tokens(prompt)
            print(f"\n[DEBUG GROQ] Model: {self.model_name}")
            print(f"[DEBUG GROQ] Prompt tokens (est): {prompt_tokens:,}")
            print(f"[DEBUG GROQ] Context chunks: {len(context_chunks)}")

        response = self.generate(prompt)

        if debug:
            print(f"[DEBUG GROQ] Response length: {len(response)} chars")
            import re
            final_match = re.search(r'final\s*answer\s*[:\s]*\b(yes|no|maybe)\b', response.lower())
            if final_match:
                print(f"[DEBUG GROQ] ✓ Found 'Final Answer': {final_match.group(1)}")
            else:
                print("[DEBUG GROQ] ✗ NO 'Final Answer' pattern found!")

        return response


# =============================================================================
# Rate Limit Error
# =============================================================================

class RateLimitError(Exception):
    """Raised when API rate limit is hit."""
    pass


def retry_with_backoff(func, max_retries=3, base_delay=5):
    """Retry a function with exponential backoff on rate limit errors.

    Waits base_delay * 2^attempt seconds between retries.
    Used during evaluation to handle per-minute API rate limits gracefully.
    """
    import time as _time
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
            _time.sleep(delay)


# =============================================================================
# Unified LLM with Auto-Fallback
# =============================================================================

class UnifiedLLM(BaseLLM):
    """
    Multi-provider LLM with automatic fallback chain.

    Supports any combination of: Gemini, Groq, Ollama.
    When one provider fails (rate limit, error, unavailable), automatically
    tries the next one in the chain.

    Fallback chain is configurable via:
      - LLM_FALLBACK_CHAIN env var (e.g., "gemini,groq,ollama")
      - Or pass fallback_chain parameter directly

    Examples:
      LLM_PROVIDER=auto   → tries gemini → groq → ollama
      LLM_PROVIDER=gemini → uses gemini, falls back per LLM_FALLBACK_CHAIN
      LLM_PROVIDER=groq   → uses groq only (no fallback unless LLM_AUTO_FALLBACK=true)
    """

    # Maps provider name → class
    PROVIDER_CLASSES = {
        "gemini": GeminiLLM,
        "groq": GroqLLM,
        "ollama": OllamaLLM,
    }

    def __init__(
        self,
        primary_provider: str = LLM_PROVIDER,
        auto_fallback: bool = LLM_AUTO_FALLBACK,
        fallback_chain: Optional[List[str]] = None,
    ):
        super().__init__()
        self.auto_fallback = auto_fallback
        self._active_index = 0

        # Build the provider chain
        if primary_provider == "auto":
            chain = fallback_chain or LLM_FALLBACK_CHAIN
        elif auto_fallback:
            # Primary first, then the rest of the fallback chain (without duplicates)
            rest = [p for p in (fallback_chain or LLM_FALLBACK_CHAIN) if p != primary_provider]
            chain = [primary_provider] + rest
        else:
            chain = [primary_provider]

        # Initialize each provider in the chain (skip failures)
        self._providers: List[tuple[str, BaseLLM]] = []
        for name in chain:
            name = name.strip()
            cls = self.PROVIDER_CLASSES.get(name)
            if cls is None:
                print(f"  ⚠ Unknown provider '{name}', skipping")
                continue
            try:
                llm = cls()
                self._providers.append((name, llm))
            except (ValueError, ConnectionError, Exception) as e:
                print(f"  ⚠ {name} unavailable: {e}")

        if not self._providers:
            raise ValueError(
                "No LLM providers available. Set at least one of:\n"
                "  - GEMINI_API_KEY (free: https://aistudio.google.com/apikey)\n"
                "  - GROQ_API_KEY   (free: https://console.groq.com/keys)\n"
                "  - Ollama running (https://ollama.ai)"
            )

        active_name, _ = self._providers[0]
        fallback_names = [n for n, _ in self._providers[1:]]
        print(f"  Active: {active_name}" + (f" | Fallbacks: {', '.join(fallback_names)}" if fallback_names else ""))

    def _get_active(self) -> tuple[str, BaseLLM]:
        """Get the currently active provider."""
        return self._providers[self._active_index]

    def _try_next_fallback(self, error_msg: str) -> bool:
        """Try to switch to the next provider in the chain. Returns True if switched."""
        if not self.auto_fallback:
            return False
        next_idx = self._active_index + 1
        if next_idx >= len(self._providers):
            return False
        self._active_index = next_idx
        name, _ = self._providers[next_idx]
        print(f"\n  ⚠ {error_msg}")
        print(f"  → Falling back to: {name}")
        return True

    @property
    def last_usage(self) -> TokenUsage:
        _, llm = self._get_active()
        return llm.last_usage

    @property
    def total_usage(self) -> TokenUsage:
        _, llm = self._get_active()
        return llm.total_usage

    def get_usage_summary(self) -> dict:
        summaries = {}
        for name, llm in self._providers:
            summaries[name] = llm.get_usage_summary()
        active_name, _ = self._get_active()
        return {
            'providers': summaries,
            'active_provider': active_name,
            'active_index': self._active_index,
            'chain': [n for n, _ in self._providers],
        }

    def generate(self, prompt: str) -> str:
        """Generate with retry + automatic fallback through the provider chain.

        On rate limit: retries 3 times with exponential backoff (5s, 10s, 20s).
        If still failing, falls back to the next provider in the chain.
        """
        while True:
            name, llm = self._get_active()
            try:
                return retry_with_backoff(lambda: llm.generate(prompt))
            except RateLimitError as e:
                if not self._try_next_fallback(str(e)):
                    return f"All providers exhausted. Last error: {e}"
            except Exception as e:
                if not self._try_next_fallback(str(e)):
                    return f"All providers exhausted. Last error: {e}"

    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_metadata: Optional[List[dict]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Generate answer with retry + automatic fallback through the provider chain."""
        while True:
            name, llm = self._get_active()
            try:
                return retry_with_backoff(
                    lambda: llm.generate_answer(
                        question, context_chunks, source_metadata, system_instruction
                    )
                )
            except RateLimitError as e:
                if not self._try_next_fallback(str(e)):
                    return f"All providers exhausted. Last error: {e}"
            except Exception as e:
                if not self._try_next_fallback(str(e)):
                    return f"All providers exhausted. Last error: {e}"

    def reset_fallback(self):
        """Reset to the first provider in the chain."""
        self._active_index = 0
        name, _ = self._get_active()
        print(f"  → Reset to primary provider: {name}")

    @property
    def active_provider(self) -> str:
        name, _ = self._get_active()
        if self._active_index > 0:
            return f"{name} (fallback)"
        return name


# =============================================================================
# Factory Function
# =============================================================================

def create_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Factory function to create the appropriate LLM.

    Args:
        provider: "gemini", "groq", "ollama", "auto", or None (uses config).

    Returns:
        BaseLLM instance

    Examples:
        create_llm()           → uses LLM_PROVIDER from config (default: "auto")
        create_llm("gemini")   → Gemini with auto-fallback chain
        create_llm("groq")     → Groq with auto-fallback chain
        create_llm("ollama")   → Local Ollama only (no fallback)
    """
    provider = provider or LLM_PROVIDER

    # Direct provider without fallback (when auto_fallback is off)
    if not LLM_AUTO_FALLBACK and provider != "auto":
        cls = UnifiedLLM.PROVIDER_CLASSES.get(provider)
        if cls is None:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return cls()

    # Use UnifiedLLM for fallback chain support
    return UnifiedLLM(primary_provider=provider)


# =============================================================================
# Legacy alias for backward compatibility
# =============================================================================

# Keep GeminiLLM as the default export for existing code
# But now workflow will use create_llm() which handles fallback


class GeminiRAGChain:
    """
    Simple RAG chain using LLM for answer generation.
    Combines retrieval and generation in a single interface.
    """

    def __init__(
        self,
        retriever,
        llm: Optional[BaseLLM] = None
    ):
        self.retriever = retriever
        self.llm = llm or create_llm()

    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Execute RAG query: retrieve context and generate answer.
        """
        results = self.retriever.retrieve(question, top_k=top_k)
        context_chunks = [r.content for r in results]
        answer = self.llm.generate_answer(question, context_chunks)

        return {
            'question': question,
            'answer': answer,
            'context': context_chunks,
            'retrieval_results': results
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Provider Test")
    print("=" * 50)

    try:
        llm = create_llm()
        print(f"\nActive provider: {getattr(llm, 'active_provider', LLM_PROVIDER)}")

        # Test simple generation
        print("\nTesting generation...")
        response = llm.generate("What is hypertension in one sentence?")
        print(f"Response: {response[:200]}...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nSetup instructions:")
        print("  For Gemini: Set GEMINI_API_KEY in .env")
        print("  For Ollama: Install from https://ollama.ai and run 'ollama pull mistral'")
