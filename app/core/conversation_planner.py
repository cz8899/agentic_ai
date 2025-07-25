# app/core/conversation_planner.py
import logging
import hashlib
from typing import List, Dict, Any
from app.utils.schema import RetrievedChunk, DocumentChunk
from app.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationPlanner:
    def __init__(self, policy_store: PolicyStore):
        self.policy_store = policy_store
        self.intent_keywords: Dict[str, List[str]] = {}
        self.intent_action: Dict[str, str] = {}
        self.intent_confidence: Dict[str, float] = {}
        self.min_confidence = self.policy_store.get_float("planner.min_confidence", 0.7)
        self._load_policies()

    def _load_policies(self):
        """Load all planner policies from PolicyStore."""
        try:
            self.intent_keywords = {
                "retrieve": self.policy_store.get_list("intent_keywords.retrieve", [
                    "show", "find", "lookup", "details", "what", "get", "fetch"
                ]),
                "summarize": self.policy_store.get_list("intent_keywords.summarize", [
                    "summary", "summarize", "tl;dr", "brief", "overview", "recap"
                ]),
                "analyze": self.policy_store.get_list("intent_keywords.analyze", [
                    "trend", "analyze", "chart", "pattern", "insight", "risk", "forecast"
                ]),
                "tool_call": self.policy_store.get_list("intent_keywords.tool_call", [
                    "calculate", "convert", "search", "compute", "translate"
                ]),
                "fallback": self.policy_store.get_list("intent_keywords.fallback", [
                    "unknown", "error", "unclear", "help", "confused", "not sure"
                ])
            }

            self.intent_action = {
                intent: self.policy_store.get_str(f"intent_plan.{intent}.next_action", f"{intent}_router")
                for intent in self.intent_keywords
            }

            self.intent_confidence = {
                intent: self.policy_store.get_float(f"intent_plan.{intent}.confidence", 0.8)
                for intent in self.intent_keywords
            }

            logger.info("✅ ConversationPlanner loaded policies from PolicyStore.")
        except Exception as e:
            logger.warning(f"Failed to load planner policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.intent_keywords = {
            "retrieve": ["show", "find", "lookup", "details"],
            "summarize": ["summary", "summarize", "tl;dr"],
            "analyze": ["trend", "analyze", "insight"],
            "tool_call": ["calculate", "convert"],
            "fallback": ["unknown", "error", "help"]
        }
        self.intent_action = {intent: f"{intent}_router" for intent in self.intent_keywords}
        self.intent_confidence = {intent: 0.8 for intent in self.intent_keywords}

    def _classify_intent(self, query: str) -> str:
        """Classify intent based on keyword matching and confidence scoring."""
        if not query or not query.strip():
            logger.info("[Planner] Empty query → fallback")
            return "fallback"

        query_lower = query.lower()
        scores = {}

        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                # Confidence = matches / total keywords for this intent
                confidence = score / len(keywords)
                if confidence >= self.min_confidence:
                    scores[intent] = confidence

        if not scores:
            logger.info("[Planner] No keywords matched → fallback")
            return "fallback"

        # Pick highest confidence intent
        best_intent = max(scores, key=scores.get)
        logger.info(f"[Planner] Intent: {best_intent}, Confidence: {scores[best_intent]:.2f}")
        return best_intent

    def plan(self, history: List[Dict], user_message: str) -> Dict[str, Any]:
        """
        Legacy method for intent classification with metadata.
        Not used in current RAG flow.
        """
        intent = self._classify_intent(user_message)
        action = self.intent_action.get(intent, "fallback_router")
        confidence = self.intent_confidence.get(intent, 0.5)

        logger.info(f"[Planner] Intent: {intent}, Action: {action}, Confidence: {confidence:.2f}")

        return {
            "intent": intent,
            "next_action": action,
            "confidence": confidence,
            "metadata": {
                "last_turns": [turn.get("message", "") for turn in history[-2:]] if history else []
            }
        }

    def plan_as_context(self, query: str) -> List[RetrievedChunk]:
        """
        Main entry point for HybridRAGRouter.
        Returns planner context as RetrievedChunk if high confidence.
        """
        intent = self._classify_intent(query)
        action = self.intent_action.get(intent, "fallback_router")
        confidence = self.intent_confidence.get(intent, 0.5)

        if confidence < self.min_confidence:
            logger.info(f"[Planner] Low confidence ({confidence:.2f}) → no context")
            return []

        content = f"Intent: {intent} | Next Action: {action} | Confidence: {confidence:.2f}"
        # Ensure unique ID
        chunk_id = f"planner-{intent}-{hashlib.md5(query.encode()).hexdigest()[:8]}"

        chunk = DocumentChunk(
            id=chunk_id,
            content=content,
            source="conversation_planner",
            title=f"Planner Intent: {intent.capitalize()}",
            metadata={"intent": intent, "action": action, "confidence": confidence}
        )

        return [RetrievedChunk(chunk=chunk, score=confidence)]

    def update_state(self, feedback: str):
        """Update planner state based on feedback (e.g., from feedback loop)."""
        logger.info(f"[Planner] Received feedback: {feedback}")
        # Future: Adapt keyword weights or confidence thresholds
