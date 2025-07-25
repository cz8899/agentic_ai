# app/core/hybrid_rag_router.py
import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Dict, Any, Tuple, Union, Callable
import redis  # pip install redis

from app.utils.schema import RetrievedChunk, DocumentChunk, QueryPayload, ChatTurn
from app.core.retrieval_coordinator import RetrievalCoordinator
from app.core.policy_store import PolicyStore
from app.core.rank_with_bedrock import BedrockRanker
from app.core.fallback_router import FallbackRouter
from app.core.conversation_planner import ConversationPlanner
from app.core.feedback_loop_controller import FeedbackLoopController
from app.utils.tracing import trace_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FallbackReason(StrEnum):
    UNKNOWN = "unknown"
    DISABLED = "disabled"
    PLANNER_LOW_SCORE = "planner_low_score"
    RETRIEVAL_FAILED = "retrieval_failed"
    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_FILTERED = "retrieval_filtered"
    SESSION_MEMORY = "session_memory"
    PLANNER_CONTEXT = "planner_context"
    PLANNER_CONTEXT_EMPTY = "planner_context_empty"
    GENERATED = "generated"
    CACHE_HIT = "cache_hit"
    RETRY_EXHAUSTED = "retry_exhausted"


class RouterMode(StrEnum):
    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    FALLBACK = "fallback"


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
    fallback_meta: Dict[str, Any] = field(default_factory=dict)
    planner_score: Optional[float] = None
    retrieval_sources: List[str] = field(default_factory=list)
    rerank_performed: bool = False
    cache_hit: bool = False
    mode: RouterMode = RouterMode.RETRIEVAL


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
        redis_client: Optional[redis.Redis] = None,
        time_fn: Callable[[], float] = time.time
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
        self.redis_client: Optional[redis.Redis] = redis_client
        self._in_memory_cache: Dict[str, Tuple[List[RetrievedChunk], float]] = {}
        self.time_fn = time_fn

        if self.use_redis and self.enable_caching and self.redis_client is None:
            try:
                self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
                self.redis_client.ping()
                logger.info("✅ Redis connected.")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to in-memory.")
                self.redis_client = None

        self._load_policies()

    def _load_policies(self):
        try:
            self.enable_planner_first = self.policy_store.get_bool("planner.first", True)
            self.disable_planner = self.policy_store.get_bool("disable_planner", False)
            self.enable_feedback = self.policy_store.get_bool("enable_feedback", True)
            self.enable_rerank = self.policy_store.get_bool("enable_rerank", True)
            self.skip_cache = self.policy_store.get_bool("skip_cache", False)
            self.retrieval_score_threshold = self.policy_store.get_float("retrieval.score_threshold", 0.0)
            self.max_retry_depth = self.policy_store.get_int("max_retry_depth", 2)
        except Exception as e:
            logger.warning(f"Failed to load routing policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        self.enable_planner_first = True
        self.disable_planner = False
        self.enable_feedback = True
        self.enable_rerank = True
        self.skip_cache = False
        self.retrieval_score_threshold = 0.0
        self.max_retry_depth = 2

    def _fetch_policies(self) -> Dict[str, Any]:
        try:
            return {
                "enable_fallback": self.policy_store.get_bool("enable_fallback", True),
                "retrieval.top_k": self.policy_store.get_int("retrieval.top_k", 5),
                "retrieval.score_threshold": self.retrieval_score_threshold,
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

    def _deserialize(self,  str) -> List[RetrievedChunk]:  # ✅ Fixed: added 'data: str'
        """Deserialize JSON to RetrievedChunk list."""
        try:
            raw = json.loads(data)
            return [
                RetrievedChunk(
                    chunk=DocumentChunk(**c["chunk"]),
                    score=c["score"],
                    explanation=c.get("explanation")
                )
                for c in raw
            ]
        except Exception as e:
            logger.error(f"Failed to deserialize cache  {e}")
            return []

    def _sanitize_input(self, s: str) -> str:
        return s.strip().lower()

    def _cache_key(self, query: str, session_id: str, policy_hash: str) -> str:
        sanitized_query = self._sanitize_input(query)
        combined = f"rag:route:{sanitized_query}|{session_id}|{policy_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_from_cache(self, key: str, ctx: RoutingContext) -> Optional[List[RetrievedChunk]]:
        if not self.enable_caching or self.skip_cache:
            return None

        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data is not None:
                    result = self._deserialize(data.decode('utf-8'))
                    if result:
                        ctx.cache_hit = True
                        logger.info("Cache hit (Redis): %s", key)
                        return result
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")

        if key in self._in_memory_cache:
            chunks, timestamp = self._in_memory_cache[key]
            if (self.time_fn() - timestamp) < self.cache_ttl:
                ctx.cache_hit = True
                logger.info("Cache hit (in-memory): %s", key)
                return chunks
            else:
                del self._in_memory_cache[key]
                logger.info("Cache miss (in-memory, expired): %s", key)

        logger.info("Cache miss: %s", key)
        return None

    def _set_in_cache(self, key: str, chunks: List[RetrievedChunk]):
        if not self.enable_caching or self.skip_cache:
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

        self._in_memory_cache[key] = (chunks, self.time_fn())
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
        start = self.time_fn()
        history = conversation_history or []
        policies = _policy_snapshot or self._fetch_policies()

        if not policies.get("enable_fallback", True):
            logger.info("Fallback disabled. Skipping all processing.")
            ctx = RoutingContext(
                query=query,
                session_id=session_id,
                fallback_reason=FallbackReason.DISABLED,
                latency=self.time_fn() - start
            )
            return self._return([], ctx)

        ctx = RoutingContext(
            query=query,
            session_id=session_id,
            retry_depth=retry_depth,
            max_retry_depth=self.max_retry_depth
        )

        if not query or not isinstance(query, str) or query.strip() == "":
            logger.warning("Invalid or empty query received: %r", query)
            ctx.fallback_reason = FallbackReason.UNKNOWN
            ctx.latency = self.time_fn() - start
            return self._return([], ctx)

        query = query.strip()
        logger.info("Processing query: %s", query[:100] + ("..." if len(query) > 100 else ""))

        # === Caching: Check early ===
        if self.enable_caching and retry_depth == 0 and not self.skip_cache:
            policy_snapshot = _policy_snapshot or self._fetch_policies()
            cache_key = self._cache_key(query, session_id, str(policy_snapshot))
            cached = self._get_from_cache(cache_key, ctx)
            if cached:
                ctx.fallback_reason = FallbackReason.CACHE_HIT
                ctx.latency = self.time_fn() - start
                return self._return(cached, ctx)

        # === Planner First ===
        try:
            if not self.disable_planner and self.enable_planner_first:
                planner_context = self.planner.plan_as_context(history, query)
                if planner_context:
                    ranked = self.ranker.rank(query, planner_context)
                    if ranked and ranked[0].score >= self.retrieval_score_threshold:
                        ctx.planner_used = True
                        ctx.mode = RouterMode.PLANNER
                        ctx.rank_source = "planner"
                        ctx.final_chunk_count = len(ranked)
                        ctx.planner_score = ranked[0].score
                        return self._return(ranked, ctx)
                    else:
                        logger.info("Planner returned only low/no-score chunks.")
                        ctx.planner_score = ranked[0].score if ranked else 0.0
                else:
                    logger.info("Planner returned no chunks.")
            else:
                logger.info("Planner skipped (disabled or planner.first=false).")
        except Exception as e:
            logger.warning(f"Planner failed: {e}")

        # === Hybrid Retrieval ===
        payload = self._build_payload(query, policies)
        retrieved_chunks = []
        retrieval_sources = []
        try:
            raw = self.retrieval_coordinator.hybrid_retrieve(payload)
            if raw is None:
                logger.error("hybrid_retrieve returned None")
            elif not isinstance(raw, list):
                logger.error("hybrid_retrieve did not return a list")
            else:
                retrieved_chunks = raw
                retrieval_sources = [c.chunk.metadata.get("retrieval_source", "unknown") for c in raw]
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            ctx.fallback_reason = FallbackReason.RETRIEVAL_FAILED

        # === Ranking ===
        ranked = retrieved_chunks
        if self.enable_rerank and retrieved_chunks:
            ranked = self.ranker.rank(query, retrieved_chunks)
            ctx.rerank_performed = True
        ctx.retrieval_used = True
        ctx.total_context_chunks = len(retrieved_chunks)
        ctx.retrieval_filtered_count = len(retrieved_chunks) - len(ranked)
        ctx.retrieval_sources = list(set(retrieval_sources))

        # === Feedback Loop ===
        if self.enable_feedback and self.feedback.should_retry(query, ranked, history):
            if retry_depth >= self.max_retry_depth:
                logger.warning("Max retry depth (%d) reached. Skipping feedback retry.", self.max_retry_depth)
                ctx.fallback_reason = FallbackReason.RETRY_EXHAUSTED
            else:
                retry_query = self.feedback.retry_or_replan(query, history)
                if retry_query:
                    if retry_query == query:
                        if self.debug_mode:
                            logger.warning("Retry triggered with same query. Possible deadlock.")
                    else:
                        logger.info("Feedback retry with new query: '%s'", retry_query)
                        ctx.feedback_retry = True
                        ctx.latency = self.time_fn() - start
                        return self.route(
                            query=retry_query,
                            conversation_history=history,
                            session_id=session_id,
                            user_id=user_id,
                            retry_depth=retry_depth + 1,
                            _policy_snapshot=policies
                        )

        # === Fallback Path ===
        if not ranked:
            ctx.fallback_used = True  # ✅ Set fallback_used = True
            if history:
                logger.info("Falling back to session history context.")
                ctx.fallback_reason = FallbackReason.SESSION_MEMORY
                ctx.final_chunk_count = len(history)
                return self._return(history, ctx)

            try:
                planner_fallback = self.planner.plan_as_context([], query)
                if planner_fallback:
                    logger.info("Falling back to planner-generated context.")
                    ctx.fallback_reason = FallbackReason.PLANNER_CONTEXT
                    ctx.final_chunk_count = len(planner_fallback)
                    return self._return(planner_fallback, ctx)
                else:
                    ctx.fallback_reason = FallbackReason.PLANNER_CONTEXT_EMPTY
            except Exception as e:
                logger.debug("Planner fallback failed: %s", e)
                ctx.fallback_reason = FallbackReason.PLANNER_CONTEXT_EMPTY

            try:
                fallback_result = self.fallback.generate_fallback(query)
                ctx.fallback_reason = FallbackReason.GENERATED
                ctx.final_chunk_count = len(fallback_result)
                return self._return(fallback_result or [], ctx)
            except Exception as e:
                logger.error(f"Fallback generation failed: {e}")
                ctx.fallback_reason = FallbackReason.UNKNOWN
                return self._return([], ctx)

        ctx.mode = RouterMode.RETRIEVAL
        ctx.rank_source = "retrieval"
        ctx.final_chunk_count = len(ranked)
        ctx.latency = self.time_fn() - start

        # === Cache Result ===
        if self.enable_caching and retry_depth == 0 and not self.skip_cache:
            policy_snapshot = _policy_snapshot or self._fetch_policies()
            cache_key = self._cache_key(query, session_id, str(policy_snapshot))
            self._set_in_cache(cache_key, ranked)

        # === Log Final Result ===
        logger.info(
            f"RAG route complete| "
            f"sess={ctx.session_id} retry={ctx.retry_depth}/{ctx.max_retry_depth}| "
            f"q='{ctx.query[:40]}{'...' if len(ctx.query) > 40 else ''}'| "
            f"P={ctx.planner_used} R={ctx.retrieval_used} F={ctx.fallback_used} "
            f"reason={ctx.fallback_reason} chunks={ctx.final_chunk_count} took={ctx.latency:.3f}s"
        )

        return self._return(ranked, ctx)

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
