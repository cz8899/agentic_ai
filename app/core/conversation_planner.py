# conversation_planner.py
import hashlib
import logging
from typing import List, Optional, Dict

from app.utils.schema import ChatTurn, RetrievedChunk, DocumentChunk

logger = logging.getLogger(__name__)


class ConversationPlanner:
    def __init__(self, policy_store):
        self.policy_store = policy_store
        self.intent_keywords = {
            intent: self._get_keywords(f"intent_keywords.{intent}")
            for intent in ["retrieve", "summarize", "analyze", "tool_call", "fallback"]
        }
        self.intent_confidence = {
            intent: self._get_confidence(f"intent_plan.{intent}.confidence", default=0.8)
            for intent in self.intent_keywords
        }
        self.intent_action = {
            intent: self.policy_store.get_str(f"intent_plan.{intent}.next_action", "fallback_router")
            for intent in self.intent_keywords
        }
        self.min_confidence = self._get_confidence("planner.min_confidence", default=0.7)

    def _get_keywords(self, key) -> List[str]:
        try:
            keywords = self.policy_store.get_list(key, default=[])
            return [kw.strip().lower() for kw in keywords if kw.strip()]
        except Exception as e:
            logger.warning(f"Failed to load keywords for {key}: {e}")
            return []

    def _get_confidence(self, key, default: float) -> float:
        try:
            return self.policy_store.get_float(key, default=default)
        except Exception as e:
            logger.warning(f"Failed to load confidence for {key}: {e}")
            return default

    def _classify_intent(self, user_message: Optional[str]) -> str:
        if not user_message or not user_message.strip():
            return "fallback"
        user_message = user_message.lower()
        for intent, keywords in self.intent_keywords.items():
            for kw in keywords:
                if kw in user_message:
                    return intent
        return "fallback"

    def plan(self, history: List[ChatTurn], user_message: str) -> Dict:
        intent = self._classify_intent(user_message)
        action = self.intent_action.get(intent, "fallback_router")
        confidence = self.intent_confidence.get(intent, 0.5)

        logger.info(f"[Planner] Intent: {intent}, Action: {action}")

        return {
            "intent": intent,
            "next_action": action,
            "confidence": confidence,
            "metadata": {
                "last_turns": [turn.message for turn in history[-2:]] if history else []
            },
        }

    def plan_as_context(self, query: str) -> List[RetrievedChunk]:
        plan = self.plan(history, user_message)
        if plan["confidence"] < self.min_confidence:
            return []

        intent = plan["intent"]
        action = plan["next_action"]
        confidence = plan["confidence"]

        content = f"Intent: {intent} | Action: {action}"
        chunk_id = f"planner-{intent}"
        hashed = hashlib.md5((user_message or "").encode()).hexdigest()[:6]
        chunk_id = f"{chunk_id}-{hashed}"

        chunk = RetrievedChunk(
            chunk=DocumentChunk(
                id=chunk_id,
                content=content,
                source="planner",
                title=f"Planner Intent: {intent.capitalize()}",
                metadata={},
            ),
            score=confidence,
        )
        return [chunk]

    def update_state(self, feedback: str):
        logger.info(f"Received feedback: {feedback}")
