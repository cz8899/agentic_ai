# app/core/claude_direct.py
import logging
from typing import List, Optional, Dict, Any
from app.utils.schema import RetrievedChunk
from app.core.policy_store import PolicyStore
from app.utils.tracing import trace_function
from app.utils.llm import invoke_claude_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ClaudeDirectResponder:
    """
    Generates final natural language responses using Claude.
    Uses retrieved context and chat history to answer user queries.
    """

    def __init__(self, policy_store: Optional[PolicyStore] = None):
        self.policy_store = policy_store or PolicyStore()
        self._load_policies()

    def _load_policies(self):
        """Load response generation policies at init."""
        try:
            self.model_id = self.policy_store.get_str("claude_direct_model", "anthropic.claude-3-sonnet-20240229-v1:0")
            self.temperature = self.policy_store.get_float("claude_direct.temperature", 0.5)
            self.max_tokens = self.policy_store.get_int("claude_direct.max_tokens", 512)
            self.response_style = self.policy_store.get_str("claude_direct.response_style", "concise")
            self.enable_citations = self.policy_store.get_bool("claude_direct.enable_citations", True)
        except Exception as e:
            logger.warning(f"Failed to load direct response policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.temperature = 0.5
        self.max_tokens = 512
        self.response_style = "concise"
        self.enable_citations = True

    @trace_function
    def generate_response(
        self,
        query: str,
        context_chunks: Optional[List[RetrievedChunk]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a natural language response using Claude.
        Returns empty string if generation fails.
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query in generate_response: %r", query)
            return ""

        query = query.strip()
        context = context_chunks or []
        history = chat_history or []

        try:
            prompt = self._build_response_prompt(query, context, history)
            raw = invoke_claude_model(
                model=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            cleaned = self._clean_response(raw)
            logger.info("Final response generated | query='%s' | response_len=%d", query[:50], len(cleaned))
            return cleaned
        except Exception as e:
            logger.error("Final response generation failed: %s", str(e), exc_info=True)
            return ""

    def _build_response_prompt(
        self,
        query: str,
        context: List[RetrievedChunk],
        history: List[Dict[str, str]]
    ) -> str:
        """Build prompt for final answer generation."""
        context_str = "\n".join([
            f"[{i+1}] {c.chunk.content[:300]}... (Source: {c.chunk.source})"
            for i, c in enumerate(context)
        ]) if context else "None"

        history_str = "\n".join([
            f"{turn['role'].title()}: {turn['content'][:100]}..."
            for turn in history[-4:]
        ]) if history else "None"

        citation_instruction = (
            "At the end, cite sources using [1], [2], etc., based on relevance."
            if self.enable_citations else ""
        )

        style_instruction = {
            "concise": "Be concise. Answer in 2-3 sentences.",
            "detailed": "Be detailed and thorough.",
            "professional": "Use professional, formal language.",
            "friendly": "Use friendly, conversational tone."
        }.get(self.response_style, "Be concise.")

        return f"""You are a helpful AI assistant for financial professionals.
Answer the user's query using only the provided context.
Be accurate, concise, and avoid speculation.

Chat History:
{history_str}

Retrieved Context:
{context_str}

User Query: {query}

Instructions:
1. Answer directly and clearly.
2. Do not repeat the query.
3. {style_instruction}
4. {citation_instruction}

Final Answer:"""

    def _clean_response(self, text: str) -> str:
        """Clean up LLM output."""
        if not text:
            return ""
        return text.strip().strip('"').strip("'").strip()
