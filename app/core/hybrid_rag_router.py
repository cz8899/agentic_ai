# app/core/hybrid_rag_router.py
import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Dict, Any, Tuple, Union
import redis

from app.utils.schema import RetrievedChunk, QueryPayload, ChatTurn
from app.core.retrieval_coordinator import RetrievalCoordinator
from app.core.policy_store import PolicyStore
from app.core.rank_with_bedrock import BedrockRanker
from app.core.fallback_router import FallbackRouter
from app.core.conversation_planner import ConversationPlanner
from app.core.feedback_loop_controller import FeedbackLoopController
from app.utils.tracing import trace_function

logger = logging.getLogger(__name__)


class FallbackReason(StrEnum):
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    PLANNER_LOW_SCORE = "planner_low_score"
    RETRIEVAL_FAILED = "retrieval_failed"
    RETRIEVAL_EMPTY = "retrieval_empty"
    GENERATED = "generated"


@dataclass
class RoutingContext:
    query: str
    session_id: str
    turn_number: int = 1
    planner_used: bool = False
    retrieval_used: bool = False
    fallback_used: bool = False
    feedback_retry: bool = False
    retry_depth: int = 0
    max_retry_depth: int = 2
    latency: float = 0.0
    total_context_chunks: int = 0
    retrieval_filtered_count: int = 0
    final_chunk_count: int = 0
    rank_source: Optional[str] = None
    fallback_reason: FallbackReason = FallbackReason.UNKNOWN
    fallback_meta Dict[str, Any] = field(default_factory=dict)


class HybridRAGRouter:
    def __init__(
        self,
        coordinator: Optional[RetrievalCoordinator] = None,
        ranker: Optional[BedrockRanker] = None,
        fallback: Optional[FallbackRouter] = None,
        planner: Optional[ConversationPlanner] = None,
        feedback: Optional[FeedbackLoopController] = None,
        policy_store: Optional[PolicyStore] = None,
        max_retry_depth: int = 2,
        debug_mode: bool = False,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
        enable_pubsub: bool = False,
        redis_client: Optional[redis.Redis] = None
    ):
        self.policy_store = policy_store or PolicyStore()
        self.retrieval_coordinator = coordinator or RetrievalCoordinator(policy_store=self.policy_store)
        self.ranker = ranker or BedrockRanker(policy_store=self.policy_store)
        self.fallback = fallback or FallbackRouter(policy_store=self.policy_store)
        self.planner = planner or ConversationPlanner(policy_store=self.policy_store)
        self.feedback = feedback or FeedbackLoopController(policy_store=self.policy_store)
        self.max_retry_depth = max_retry_depth
        self.debug_mode = debug_mode
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.use_redis = use_redis
        self.enable_pubsub = enable_pubsub
        self.redis_client = redis_client
        self._in_memory_cache: Dict[str, Tuple[List[RetrievedChunk], float]] = {}

        if self.use_redis and self.enable_caching and self.redis_client is None:
            try:
                self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
                self.redis_client.ping()
                logger.info("✅ Redis connected.")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to in-memory.")
                self.redis_client = None

    def _fetch_policies(self) -> Dict[str, Any]:
        try:
            return {
                "enable_fallback": self.policy_store.get_bool("enable_fallback", True),
                "retrieval.top_k": self.policy_store.get_int("retrieval.top_k", 5),
                "retrieval.score_threshold": self.policy_store.get_float("retrieval.score_threshold", 0.0),
                "enable_bedrock_kb": self.policy_store.get_bool("enable_bedrock_kb", True),
                "disable_opensearch": self.policy_store.get_bool("disable_opensearch", False),
                "enable_chroma": self.policy_store.get_bool("enable_chroma", False),
            }
        except Exception as e:
            logger.warning(f"Policy fetch failed: {e}. Using defaults.")
            return {
                "enable_fallback": True,
                "retrieval.top_k": 5,
                "retrieval.score_threshold": 0.0,
                "enable_bedrock_kb": True,
                "disable_opensearch": False,
                "enable_chroma": False,
            }

    def _build_payload(self, query: str, policies: Dict[str, Any]) -> QueryPayload:
        return QueryPayload(
            query=query,
            top_k=policies["retrieval.top_k"],
            use_bedrock_kb=policies["enable_bedrock_kb"],
            use_opensearch=not policies["disable_opensearch"],
            use_chroma=policies["enable_chroma"]
        )

    def _serialize(self, chunks: List[RetrievedChunk]) -> str:
        return json.dumps([
            {
                "chunk": {
                    "id": c.chunk.id,
                    "content": c.chunk.content,
                    "source": c.chunk.source,
                    "title": c.chunk.title,
                    "metadata": c.chunk.metadata
                },
                "score": c.score,
                "explanation": c.explanation
            }
            for c in chunks
        ])

    def _deserialize(self,  str) -> List[RetrievedChunk]:
        raw = json.loads(data)
        return [
            RetrievedChunk(
                chunk=ChatTurn(**c["chunk"]) if "role" in c["chunk"] else DocumentChunk(**c["chunk"]),
                score=c["score"],
                explanation=c["explanation"]
            )
            for c in raw
        ]

    def _get_from_cache(self, key: str) -> Optional[List[RetrievedChunk]]:
        if not self.enable_caching:
            return None

        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data is not None:
                    return self._deserialize(data.decode('utf-8'))
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")

        if key in self._in_memory_cache:
            chunks, timestamp = self._in_memory_cache[key]
            if (time.time() - timestamp) < self.cache_ttl:
                return chunks
            else:
                del self._in_memory_cache[key]

        return None

    def _set_in_cache(self, key: str, chunks: List[RetrievedChunk]):
        if not self.enable_caching:
            return
        try:
            serialized = self._serialize(chunks)
        except Exception:
            return

        if self.redis_client:
            try:
                self.redis_client.setex(key, self.cache_ttl, serialized)
                logger.debug(f"Redis cache set with TTL={self.cache_ttl}: {key}")
                return
            except Exception as e:
                logger.warning(f"Redis SETEX failed: {e}")

        self._in_memory_cache[key] = (chunks, time.time())
        logger.debug(f"In-memory cache set: {key}")

    @trace_function
    def route(
        self,
        query: str,
        conversation_history: Optional[List[ChatTurn]] = None,
        session_id: str = "default",
        user_id: str = "default",
        retry_depth: int = 0,
        _policy_snapshot: Optional[Dict[str, Any]] = None
    ) -> Union[List[RetrievedChunk], Tuple[List[RetrievedChunk], RoutingContext]]:
        start = time.time()
        history = conversation_history or []
        policies = _policy_snapshot or self._fetch_policies()

        if not policies.get("enable_fallback", True):
            logger.info("Fallback disabled. Skipping all processing.")
            ctx = RoutingContext(
                query=query,
                session_id=session_id,
                fallback_reason=FallbackReason.DISABLED,
                latency=time.time() - start
            )
            return self._return([], ctx)

        ctx = RoutingContext(
            query=query,
            session_id=session_id,
            retry_depth=retry_depth,
            max_retry_depth=self.max_retry_depth
        )

        try:
            # Planner First
            try:
                planner_context = self.planner.plan_as_context(history, query)
                if planner_context:
                    ranked = self.ranker.rank(query, planner_context)
                    if ranked and ranked[0].score >= policies["retrieval.score_threshold"]:
                        ctx.planner_used = True
                        ctx.rank_source = "planner"
                        ctx.final_chunk_count = len(ranked)
                        return self._return(ranked, ctx)
                    else:
                        logger.info("Planner returned only low/no-score chunks.")
                else:
                    logger.info("Planner returned no chunks.")
            except Exception as e:
                logger.warning(f"Planner failed: {e}")

            # Hybrid Retrieval
            payload = self._build_payload(query, policies)
            retrieved_chunks = []
            try:
                raw = self.retrieval_coordinator.hybrid_retrieve(payload)
                if raw is None:
                    logger.error("hybrid_retrieve returned None")
                elif not isinstance(raw, list):
                    logger.error("hybrid_retrieve did not return a list")
                else:
                    retrieved_chunks = raw
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                ctx.fallback_reason = FallbackReason.RETRIEVAL_FAILED

            # Ranking
            ranked = self.ranker.rank(query, retrieved_chunks)
            ctx.retrieval_used = True
            ctx.total_context_chunks = len(retrieved_chunks)
            ctx.retrieval_filtered_count = len(retrieved_chunks) - len(ranked)

            # Feedback Loop
            if self.feedback.should_retry(query, ranked, history):
                if retry_depth >= self.max_retry_depth:
                    logger.warning("Max retry depth (%d) reached. Skipping feedback retry.", self.max_retry_depth)
                else:
                    retry_query = self.feedback.retry_or_replan(query, history)
                    if retry_query and retry_query != query:
                        logger.info("Feedback retry with new query: '%s'", retry_query)
                        ctx.feedback_retry = True
                        ctx.latency = time.time() - start
                        return self.route(
                            query=retry_query,
                            conversation_history=history,
                            session_id=session_id,
                            user_id=user_id,
                            retry_depth=retry_depth + 1,
                            _policy_snapshot=policies
                        )

            # Fallback
            if not ranked:
                try:
                    fallback_result = self.fallback.generate_fallback(query)
                    ctx.fallback_reason = FallbackReason.GENERATED
                    ctx.final_chunk_count = len(fallback_result)
                    return self._return(fallback_result or [], ctx)
                except Exception as e:
                    logger.error(f"Fallback failed: {e}")
                    ctx.fallback_reason = FallbackReason.UNKNOWN

            ctx.rank_source = "retrieval"
            ctx.final_chunk_count = len(ranked)
            ctx.latency = time.time() - start

            if self.enable_caching and retry_depth == 0:
                policy_snapshot = _policy_snapshot or self._fetch_policies()
                cache_key = hashlib.md5(f"rag:{query}|{session_id}|{policy_snapshot}".encode()).hexdigest()
                self._set_in_cache(cache_key, ranked)

            logger.info(
                f"RAG route complete| "
                f"sess={ctx.session_id} retry={ctx.retry_depth}/{ctx.max_retry_depth}| "
                f"q='{ctx.query[:40]}{'...' if len(ctx.query) > 40 else ''}'| "
                f"P={ctx.planner_used} R={ctx.retrieval_used} F={ctx.fallback_used} "
                f"reason={ctx.fallback_reason} chunks={ctx.final_chunk_count} took={ctx.latency:.3f}s"
            )

            return self._return(ranked, ctx)

        except Exception as e:
            logger.error("Unexpected error in route(): %s", str(e), exc_info=True)
            ctx.fallback_reason = FallbackReason.UNKNOWN
            return self._return([], ctx)

    def _return(self, result: List[RetrievedChunk], ctx: RoutingContext) -> Union[List[RetrievedChunk], Tuple[List[RetrievedChunk], RoutingContext]]:
        if self.debug_mode:
            return result, ctx
        return result

    def publish_feedback_event(self, event_type: str, payload: Dict[str, Any]):
        if self.redis_client and self.enable_pubsub:
            try:
                channel = f"rag/{event_type}"
                self.redis_client.publish(channel, json.dumps(payload))
                logger.info(f"Published to {channel}: {payload}")
            except Exception as e:
                logger.error(f"Pub/Sub publish failed: {e}")
