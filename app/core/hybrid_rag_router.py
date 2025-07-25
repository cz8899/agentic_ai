# app/core/hybrid_rag_router.py
import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Dict, Any, Tuple, Union
import redis

from app.utils.schema import RetrievedChunk, QueryPayload, DocumentChunk
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
        self.redis_client: Optional[redis.Redis] = redis_client
        self._in_memory_cache: Dict[str, Tuple[List[RetrievedChunk], float]] = {}

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
        """Load routing policies at init."""
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
        """Safe fallback if policy load fails."""
        self.enable_planner_first = True
        self.disable_planner = False
        self.enable_feedback = True
        self.enable_rerank = True
        self.skip_cache = False
        self.retrieval_score_threshold = 0.0
        self.max_retry_depth = 2

    def _fetch_policies(self) -> Dict[str, Any]:
        """Load policies at runtime."""
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

    def _deserialize(self,  str) -> List[RetrievedChunk]:
        try:
            raw = json.loads(data)
            return [
                RetrievedChunk(
                    chunk=DocumentChunk(**c["chunk"]),
                    score=c["score"],
                    explanation=c["explanation"]
                )
                for c in raw
            ]
        except Exception as e:
            logger.error(f"Failed to deserialize cache: {e}")
            return []

    def _cache_key(self, query: str, session_id: str, policy_hash: str) -> str:
        combined = f"rag:route:{query}|{session_id}|{policy_hash}"
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
                        logger.info(f"Cache hit (Redis): {key}")
                        return result
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")

        if key in self._in_memory_cache:
            chunks, timestamp = self._in_memory_cache[key]
            if (time.time() - timestamp) < self.cache_ttl:
                ctx.cache_hit = True
                logger.info(f"Cache hit (in-memory): {key}")
                return chunks
            else:
                del self._in_memory_cache[key]
                logger.info(f"Cache miss (in-memory, expired): {key}")

        logger.info(f"Cache miss: {key}")
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

        self._in_memory_cache[key] = (chunks, time.time())
        logger.debug(f"In-memory cache set: {key}")

    @trace_function
    def route(
        self,
        query: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        retry_depth: int = 0
    ) -> Union[List[RetrievedChunk], Tuple[List[RetrievedChunk], RoutingContext]]:
        ctx = RoutingContext(query=query, metadata=metadata, session_id=session_id)
        chunks: List[RetrievedChunk] = []
        rerank_needed = self.enable_rerank
        planner_first = self._policy("planner.first", default="false", kind="bool")
        disable_planner = self._policy("disable_planner", default="false", kind="bool")
        skip_cache = self.skip_cache or self._policy("skip_cache", default="false", kind="bool")

        if not skip_cache and self.enable_caching:
            cached_chunks = self._get_from_cache(query, ctx)
            if cached_chunks:
                ctx.cache_hit = True
                ctx.mode = RouterMode.CACHE
                chunks = cached_chunks
                if self.debug_mode:
                    return chunks, ctx
                return chunks

        # 1️⃣ Planner path
        if planner_first and not disable_planner and self.planner:
            ctx.planner_used = True
            try:
                planner_chunks = self.planner.plan_as_context(query, metadata)
                if planner_chunks:
                    chunks = planner_chunks
                    ctx.mode = RouterMode.PLANNER
                    rerank_needed = False  # trust planner
            except Exception as e:
                logger.warning(f"Planner failed: {e}")
                ctx.fallback_reason = FallbackReason.PLANNER_FAILED
                ctx.planner_used = False  # mark it failed

        # 2️⃣ Retrieval path if planner path didn’t return chunks
        if not chunks and self.coordinator:
            try:
                ctx.retrieval_used = True
                retrieved = self.coordinator.hybrid_retrieve(query=query, metadata=metadata, session_id=session_id)
                ctx.total_context_chunks = len(retrieved)

                if rerank_needed and self.ranker:
                    reranked = self.ranker.rank(query=query, candidates=retrieved)
                    ctx.final_chunk_count = len(reranked)
                    ctx.retrieval_filtered_count = len(retrieved) - len(reranked)
                    chunks = reranked
                else:
                    chunks = retrieved

                if not chunks:
                    ctx.fallback_reason = FallbackReason.EMPTY_CONTEXT
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                ctx.fallback_reason = FallbackReason.RETRIEVAL_FAILED

        # 3️⃣ Feedback-based retry logic
        low_score = (
            chunks and
            self.enable_rerank and
            self.feedback and
            self.feedback.should_retry(query, chunks) and
            retry_depth < self.max_retry_depth
        )
        if low_score:
            rewritten = self.feedback.retry_or_replan(query, chunks)
            return self.route(rewritten, session_id=session_id, metadata=metadata, retry_depth=retry_depth + 1)

        if self.feedback and retry_depth >= self.max_retry_depth:
            logger.warning(f"Max retry depth ({self.max_retry_depth}) reached. Skipping feedback retry.")
            ctx.fallback_reason = FallbackReason.RETRY_EXHAUSTED

        # 4️⃣ Fallback if needed
        if not chunks and self.enable_fallback and self.fallback:
            fallback_chunks = self.fallback.generate_fallback(query, metadata)
            if fallback_chunks:
                chunks = fallback_chunks
                ctx.fallback_used = True
                ctx.mode = RouterMode.FALLBACK
                ctx.fallback_reason = FallbackReason.GENERATED

        # 5️⃣ Cache results
        if self.enable_caching and chunks and session_id and not ctx.cache_hit:
            self._set_in_cache(query, chunks, session_id, ctx)

        logger.info(f"RAG route complete | query='{query}' | count={len(chunks)} | mode={ctx.mode} | fallback={ctx.fallback_reason}")

        return (chunks, ctx) if self.debug_mode else chunks

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
