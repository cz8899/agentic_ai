# app/core/fallback_router.py
import logging
from typing import List, Optional, Dict, Any
from app.utils.schema import RetrievedChunk, DocumentChunk
from app.core.policy_store import PolicyStore
from app.core.claude_direct import ClaudeDirectResponder
from app.utils.tracing import trace_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FallbackReason:
    """Enum for fallback types — used in routing decisions."""
    GENERATED = "generated_fallback"
    SESSION_MEMORY = "session_memory"
    PLANNER_CONTEXT = "planner_context"
    RETRIEVAL_EMPTY = "no_retrieval"
    RETRIEVAL_FILTERED = "all_filtered"
    RETRY_EXHAUSTED = "retry_exhausted"
    UNKNOWN = "unknown"


class FallbackRouter:
    """
    Generates fallback responses when primary retrieval fails.
    Uses policy-controlled strategies: session memory, LLM generation, or hybrid.
    """

    def __init__(
        self,
        policy_store: Optional[PolicyStore] = None,
        claude_responder: Optional[ClaudeDirectResponder] = None
    ):
        self.policy_store = policy_store or PolicyStore()
        self.claude = claude_responder or ClaudeDirectResponder(policy_store=self.policy_store)
        self._load_policies()

    def _load_policies(self):
        """Load fallback strategies and thresholds."""
        try:
            self.enable_session_memory = self.policy_store.get_bool("fallback.enable_session_memory", True)
            self.enable_llm_generation = self.policy_store.get_bool("fallback.enable_llm_generation", True)
            self.min_history_length = self.policy_store.get_int("fallback.min_history_length", 1)
            self.fallback_prompt_type = self.policy_store.get_str("fallback.prompt_type", "concise")
            self.enable_metadata_injection = self.policy_store.get_bool("fallback.enable_metadata_injection", True)
        except Exception as e:
            logger.warning(f"Failed to load fallback policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.enable_session_memory = True
        self.enable_llm_generation = True
        self.min_history_length = 1
        self.fallback_prompt_type = "concise"

    @trace_function
    def generate_fallback(
        self,
        query: str,
        conversation_history: Optional[List[RetrievedChunk]] = None,
        context_chunks: Optional[List[RetrievedChunk]] = None
    ) -> List[RetrievedChunk]:
        """
        Generate fallback response using policy-controlled strategy.
        Returns list of RetrievedChunk for downstream ranking and routing.
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query in generate_fallback: %r", query)
            return []

        query = query.strip()
        history = conversation_history or []
        context = context_chunks or []

        logger.info("Generating fallback for query: '%s' | history_len=%d, context_len=%d", query[:50], len(history), len(context))

        # Strategy 1: Use session memory if available
        if self.enable_session_memory and len(history) >= self.min_history_length:
            logger.info("Using session memory as fallback context.")
            return self._fallback_from_history(query, history)

        # Strategy 2: Use LLM to generate a response
        if self.enable_llm_generation:
            logger.info("Using LLM to generate fallback response.")
            return self._generate_with_llm(query, context, history)

        # Strategy 3: Return empty if all else fails
        logger.warning("All fallback strategies disabled or failed.")
        return []

    def _fallback_from_history(
        self,
        query: str,
        history: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Use recent chat history as fallback context."""
        try:
            content = "I'll help based on our conversation history:\n" + "\n".join([
                f"- {self._truncate(c.chunk.content, 100)}"
                for c in history[-3:]
            ])
            return [
                RetrievedChunk(
                    chunk=DocumentChunk(
                        id="fallback-session-memory",
                        content=content,
                        source="fallback_router",
                        title="Session Memory Context"
                    ),
                    score=0.3
                )
            ]
        except Exception as e:
            logger.error("Session memory fallback failed: %s", str(e))
            return []

    def _generate_with_llm(
        self,
        query: str,
        context: List[RetrievedChunk],
        history: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Use Claude to generate a natural language fallback response."""
        try:
            response = self.claude.generate_response(
                query=query,
                context_chunks=context,
                chat_history=[{"role": "user" if i % 2 == 0 else "assistant", "content": h.chunk.content}
                            for i, h in enumerate(history[-4:])] if history else None
            )
            if not response or not isinstance(response, str):
                logger.warning("LLM fallback returned invalid response: %r", response)
                return []

            return [
                RetrievedChunk(
                    chunk=DocumentChunk(
                        id="fallback-generated-response",
                        content=response.strip(),
                        source="fallback_router",
                        title="Generated Fallback"
                    ),
                    score=0.4
                )
            ]
        except Exception as e:
            logger.error("LLM fallback generation failed: %s", str(e), exc_info=True)
            return []

    def _truncate(self, text: str, max_words: int) -> str:
        return " ".join(text.split()[:max_words])
