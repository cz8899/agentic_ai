# app/core/rank_with_bedrock.py
import logging
import json
from typing import List, Optional, Dict, Any
from app.utils.schema import RetrievedChunk
from app.core.policy_store import PolicyStore
from app.utils.tracing import trace_function
from app.utils.llm import invoke_claude_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BedrockRanker:
    """
    Uses Claude to re-rank retrieved chunks by relevance to the query.
    Returns list of RetrievedChunk with updated scores and explanations.
    """

    def __init__(self, policy_store: Optional[PolicyStore] = None):
        self.policy_store = policy_store or PolicyStore()
        self._load_policies()

    def _load_policies(self):
        """Load ranking policies at init."""
        try:
            self.model_id = self.policy_store.get_str("ranker.model", "anthropic.claude-3-sonnet-20240229-v1:0")
            self.temperature = self.policy_store.get_float("ranker.temperature", 0.0)
            self.max_tokens = self.policy_store.get_int("ranker.max_tokens", 200)
            self.enable_explanation = self.policy_store.get_bool("ranker.enable_explanation", True)
        except Exception as e:
            logger.warning(f"Failed to load ranker policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.temperature = 0.0
        self.max_tokens = 200
        self.enable_explanation = True

    @trace_function
    def rank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rank chunks by relevance using Claude.
        Returns list of RetrievedChunk with updated scores (0.0 to 1.0).
        Returns original list if ranking fails.
        """
        if not query or not chunks:
            logger.warning("Invalid input to rank(): query=%r, chunks=%d", query, len(chunks) if chunks else 0)
            return chunks

        try:
            prompt = self._build_ranking_prompt(query, chunks)
            raw = invoke_claude_model(
                model=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            ranked = self._parse_ranking_response(raw, chunks)
            logger.info("Re-ranking complete | query='%s' | top_chunk_score=%.2f", query[:50], ranked[0].score if ranked else 0.0)
            return ranked
        except Exception as e:
            logger.error("Re-ranking failed: %s", str(e), exc_info=True)
            return chunks  # Fallback to unranked

    def _build_ranking_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Build prompt for re-ranking."""
        context = "\n".join([
            f"[{i+1}] {c.chunk.content[:300]}..." for i, c in enumerate(chunks)
        ])

        return f"""You are an AI relevance ranking assistant.
Your goal is to rank the following passages by how well they answer the user's query.
Return a JSON list with 'index' and 'relevance_score' (0.0 to 1.0).
{ 'Include a brief explanation for each.' if self.enable_explanation else '' }

User Query: {query}

Passages:
{context}

Output format:
[
  {{"index": 0, "relevance_score": 0.95{', "explanation": "Highly relevant"' if self.enable_explanation else ''}}},
  ...
]"""

    def _parse_ranking_response(self, text: str, original_chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Parse LLM response into ranked chunks."""
        if not text:
            raise ValueError("Empty ranking response")

        try:
            parsed = json.loads(text.strip())
            if not isinstance(parsed, list):
                raise ValueError("Response is not a list")

            # Map back to RetrievedChunk
            ranked = []
            used_indices = set()

            for item in parsed:
                idx = item.get("index")
                score = item.get("relevance_score")
                explanation = item.get("explanation", "") if self.enable_explanation else None

                if idx is None or score is None:
                    continue
                if idx < 0 or idx >= len(original_chunks):
                    continue
                if idx in used_indices:
                    continue  # Avoid duplicates
                if not (0.0 <= score <= 1.0):
                    score = 0.5  # Clamp to valid range

                ranked.append(RetrievedChunk(
                    chunk=original_chunks[idx].chunk,
                    score=score,
                    explanation=explanation
                ))
                used_indices.add(idx)

            # Add any missing chunks at the end with default score
            for i, chunk in enumerate(original_chunks):
                if i not in used_indices:
                    ranked.append(RetrievedChunk(
                        chunk=chunk.chunk,
                        score=0.5,
                        explanation="Not ranked by LLM"
                    ))

            return ranked
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in ranking response: {e}")
