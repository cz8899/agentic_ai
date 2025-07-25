# app/core/feedback_loop_controller.py
import logging
import json
from typing import List, Optional, Dict, Any
from app.utils.schema import RetrievedChunk
from app.core.policy_store import PolicyStore
from app.utils.tracing import trace_function
from app.utils.llm import invoke_claude_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FeedbackLoopController:
    """
    Self-correcting feedback loop that:
    - Critiques retrieval quality
    - Decides whether to retry or replan
    - Rewrites queries for better results
    - Logs decisions for observability
    """

    def __init__(self, policy_store: Optional[PolicyStore] = None):
        self.policy_store = policy_store or PolicyStore()
        self._load_policies()

    def _load_policies(self):
        """Load feedback loop policies at init for performance."""
        try:
            self.enable_replan = self.policy_store.get_bool("feedback.enable_replan", True)
            self.enable_critique = self.policy_store.get_bool("feedback.enable_critique", True)
            self.critique_score_threshold = self.policy_store.get_float("feedback.critique_score_threshold", 0.25)
            self.max_rewrite_attempts = self.policy_store.get_int("feedback.max_rewrite_attempts", 2)
            self.critique_prompt_type = self.policy_store.get_str("feedback.critique_prompt_type", "concise")
            self.replan_prompt_type = self.policy_store.get_str("feedback.replan_prompt_type", "detailed")
            self.model_id = self.policy_store.get_str("feedback.model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
        except Exception as e:
            logger.warning(f"Failed to load feedback policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.enable_replan = True
        self.enable_critique = True
        self.critique_score_threshold = 0.25
        self.max_rewrite_attempts = 2
        self.critique_prompt_type = "concise"
        self.replan_prompt_type = "detailed"
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    @trace_function
    def should_retry(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_history: Optional[List[RetrievedChunk]] = None
    ) -> bool:
        """
        Decide whether to retry based on:
        - Low chunk scores
        - Critique from LLM (if enabled)
        """
        if not self.enable_replan:
            return False

        # Rule 1: All chunks below threshold
        if retrieved_chunks:
            avg_score = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
            if avg_score >= self.critique_score_threshold:
                logger.info("Average score %.2f >= threshold %.2f. No retry.", avg_score, self.critique_score_threshold)
                return False
        else:
            logger.info("No chunks retrieved. Retry triggered.")
            return True

        # Rule 2: LLM critique (optional)
        if self.enable_critique:
            try:
                critique = self._critique_retrieval(query, retrieved_chunks, conversation_history)
                if "RETRY" in critique.upper():
                    logger.info("LLM critique: RETRY requested.")
                    return True
            except Exception as e:
                logger.warning(f"LLM critique failed: {e}. Falling back to score-based retry.")

        return False

    @trace_function
    def retry_or_replan(
        self,
        query: str,
        conversation_history: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """
        Generate a rewritten or replanned query.
        Returns original query if rewrite fails.
        """
        if not self.enable_replan:
            logger.info("Replan disabled. Skipping query rewrite.")
            return ""

        try:
            prompt = self._build_rewrite_prompt(query, conversation_history)
            raw = invoke_claude_model(
                model=self.model_id,
                prompt=prompt,
                temperature=0.5,
                max_tokens=200
            )
            rewritten = self._extract_rewritten_query(raw)
            if rewritten and rewritten.strip() and rewritten.strip() != query:
                logger.info("Feedback loop: Rewritten query: '%s'", rewritten.strip())
                return rewritten.strip()
            else:
                logger.warning("Rewritten query is empty or same as original. Skipping retry.")
                return ""
        except Exception as e:
            logger.error("Query rewrite failed: %s", str(e), exc_info=True)
            return ""

    def _build_rewrite_prompt(
        self,
        query: str,
        conversation_history: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """Build prompt for query rewriting."""
        history_context = "\n".join([
            f"User: {h.chunk.content[:100]}..." for h in (conversation_history or [])[-2:]
        ]) if conversation_history else "None"

        return f"""You are an AI query optimization assistant.
Your goal is to rewrite the user's query to improve retrieval quality.
Be concise and focus on intent.

Chat History:
{history_context}

User Query: {query}

Instructions:
1. Identify the core intent.
2. Remove ambiguity.
3. Add context from history if relevant.
4. Return only the rewritten query, no explanations.

Rewritten Query:"""

    def _extract_rewritten_query(self, text: str) -> str:
        """Clean up LLM output."""
        if not text:
            return ""
        return text.strip().strip('"').strip("'").strip()

    def _critique_retrieval(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        conversation_history: Optional[List[RetrievedChunk]] = None
    ) -> str:
        """Ask LLM to critique retrieval quality."""
        context = "\n".join([
            f"[{i+1}] {c.chunk.content[:150]}... (score: {c.score:.2f})"
            for i, c in enumerate(chunks[:3])
        ])

        prompt = f"""Query: {query}

Retrieved Passages:
{context}

On a scale of 1-10, how well do these passages answer the query?
Should the system RETRY with a rewritten query? Answer only 'RETRY' or 'ACCEPT'."""

        return invoke_claude_model(
            model=self.model_id,
            prompt=prompt,
            temperature=0.0,
            max_tokens=20
        )
