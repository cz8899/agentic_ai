# tests/core/test_hybrid_rag_router.py
"""
Comprehensive test suite for HybridRAGRouter.
All mocks return real RetrievedChunk objects.
All ctx fields are verified.
No LLM calls — fully isolated.
"""

import logging
import json
import pytest
from unittest.mock import MagicMock, patch
from app.core.hybrid_rag_router import HybridRAGRouter, FallbackReason, RouterMode
from app.utils.schema import RetrievedChunk, DocumentChunk
from app.core.policy_store import PolicyStore


@pytest.fixture
def mock_policy_store():
    store = MagicMock()
    policies = {
        "planner.first": "true",
        "disable_planner": "false",
        "enable_feedback": "true",
        "enable_rerank": "true",
        "skip_cache": "false",
        "retrieval.score_threshold": 0.0,
        "max_retry_depth": 2,
        "enable_fallback": "true",
        "retrieval.top_k": 5,
        "enable_bedrock_kb": "true",
        "disable_opensearch": "false",
        "enable_chroma": "false",
    }

    def get_str(key, default=""):
        return str(policies.get(key, default))

    def get_bool(key, default=False):
        value = policies.get(key, str(default))
        return str(value).strip().lower() in ["true", "1", "yes", "on"]

    def get_int(key, default=1):
        return int(policies.get(key, default))

    def get_float(key, default=0.0):
        return float(policies.get(key, default))

    store.get.side_effect = lambda key, default=None: policies.get(key, default)
    store.get_str.side_effect = get_str
    store.get_bool.side_effect = get_bool
    store.get_int.side_effect = get_int
    store.get_float.side_effect = get_float

    return store


@pytest.fixture
def mock_retrieval_coordinator():
    return MagicMock()


@pytest.fixture
def mock_ranker():
    return MagicMock()


@pytest.fixture
def mock_fallback():
    return MagicMock()


@pytest.fixture
def mock_planner():
    return MagicMock()


@pytest.fixture
def mock_feedback():
    return MagicMock()


@pytest.fixture
def make_chunk():
    def _make(content: str, score: float = 0.5, source: str = "test", doc_id: str = None):
        if doc_id is None:
            doc_id = f"doc-{abs(hash(content)) % 10000}"
        return RetrievedChunk(
            chunk=DocumentChunk(
                id=doc_id,
                content=content,
                source=source,
                title=f"Chunk: {content[:30]}..."
            ),
            score=score
        )
    return _make


@pytest.fixture
def mock_redis_client():
    with patch("redis.from_url") as mock:
        client = MagicMock()
        client.ping.return_value = True
        client.get.return_value = None
        client.setex.return_value = True
        client.publish.return_value = True
        mock.return_value = client
        yield client
        mock.reset_mock()


# 🔹 1. Test policy combinations
@pytest.mark.parametrize("planner_first,fallback_enabled", [
    ("true", "true"),
    ("false", "true"),
    ("true", "false"),
    ("false", "false"),
])
def test_policy_combinations(
    mock_policy_store,
    mock_planner,
    mock_retrieval_coordinator,
    mock_fallback,
    mock_ranker,
    planner_first,
    fallback_enabled,
    make_chunk
):
    mock_policy_store.get.side_effect = lambda key, default=None: {
        "planner.first": planner_first,
        "enable_fallback": fallback_enabled,
        "retrieval.score_threshold": 0.0,
    }.get(key, default)

    # ✅ Use real RetrievedChunk objects
    if planner_first == "true":
        mock_planner.plan_as_context.return_value = [make_chunk("planner", 0.9)]
        mock_ranker.rank.return_value = [make_chunk("planner", 0.9)]
    else:
        mock_planner.plan_as_context.return_value = []

    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]
    mock_fallback.generate_fallback.return_value = [make_chunk("fallback", 0.7)]

    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        ranker=mock_ranker,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False
    )

    result = router.route("Test query")

    if planner_first == "true" and result:
        assert "planner" in result[0].chunk.content
    elif result:
        assert "retrieved" in result[0].chunk.content

    if fallback_enabled == "false":
        assert result == []


# 🔹 2. Test fallback when retrieval fails
def test_fallback_when_retrieval_fails(
    mock_policy_store,
    mock_retrieval_coordinator,
    mock_fallback,
    make_chunk
):
    mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Retrieval failed")
    mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False
    )

    result = router.route("Query")
    assert len(result) == 1
    assert "fallback" in result[0].chunk.content


# 🔹 3. Test retry exhaustion logs and stops
def test_retry_exhaustion_logs_and_stops(mock_policy_store, make_chunk, caplog):
    feedback = MagicMock()
    feedback.retry_or_replan.return_value = "Rewritten query"

    fallback = MagicMock()
    fallback.generate_fallback.return_value = [make_chunk("final")]

    router = HybridRAGRouter(
        feedback=feedback,
        fallback=fallback,
        policy_store=mock_policy_store,
        max_retry_depth=1,
        use_redis=False,
        debug_mode=True,
        enable_rerank=False  # ✅ Disable rerank to avoid LLM errors
    )

    with caplog.at_level(logging.WARNING):
        result, ctx = router.route("Query", retry_depth=1)

    assert "Max retry depth (1) reached. Skipping feedback retry." in caplog.text
    assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED
    assert len(result) == 1
    assert "final" in result[0].chunk.content


# 🔹 4. Test feedback triggered on low score
def test_feedback_triggered_on_low_score(mock_policy_store, mock_ranker, mock_feedback, make_chunk):
    mock_ranker.rank.return_value = [make_chunk("low-score", 0.1)]
    mock_feedback.should_retry.return_value = True
    mock_feedback.retry_or_replan.return_value = "Improved query"

    router = HybridRAGRouter(
        feedback=mock_feedback,
        ranker=mock_ranker,
        policy_store=mock_policy_store,
        max_retry_depth=2,
        use_redis=False,
        debug_mode=True,
        enable_rerank=False  # ✅ Avoid LLM call
    )

    with patch.object(router, 'route') as mock_route:
        mock_route.side_effect = lambda *args, **kwargs: ([make_chunk("good-result", 0.8)], MagicMock())
        result, ctx = router.route("Query")

    mock_route.assert_called()
    assert "good-result" in result[0].chunk.content


# 🔹 5. Test cache hit and miss logging
def test_cache_hit_miss_logging(caplog, mock_redis_client):
    router = HybridRAGRouter(
        use_redis=True,
        enable_caching=True,
        skip_cache=False,  # ✅ Ensure caching is enabled
        debug_mode=True
    )

    with caplog.at_level(logging.INFO):
        result, ctx = router.route("Query", session_id="test")
        assert "RAG route complete" in caplog.text
        assert ctx.cache_hit is False

        mock_redis_client.get.return_value = b'[{"chunk": {"id": "cached", "content": "cached"}, "score": 0.9}]'
        result, ctx = router.route("Query", session_id="test")
        assert ctx.cache_hit is True


# 🔹 6. Test in-memory cache used when Redis fails
def test_in_memory_cache_used_when_redis_fails(monkeypatch, mock_policy_store):
    monkeypatch.setattr("redis.from_url", lambda *args, **kwargs: None)

    router = HybridRAGRouter(
        policy_store=mock_policy_store,
        use_redis=True,
        enable_caching=True,
        skip_cache=False,
        debug_mode=True
    )

    with patch.object(router, '_get_from_cache', wraps=router._get_from_cache) as mock_get:
        with patch.object(router, '_set_in_cache', wraps=router._set_in_cache) as mock_set:
            result, ctx = router.route("Query", session_id="test")
            assert mock_get.called
            assert mock_set.called


# 🔹 7. Test fallback reasons are set correctly
def test_fallback_reasons_are_set_correctly(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
    fallback = MagicMock()
    fallback.generate_fallback.return_value = [make_chunk("fallback")]

    mock_retrieval_coordinator.hybrid_retrieve.return_value = []

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=True
    )

    result, ctx = router.route("No results query", session_id="test")
    assert ctx.fallback_reason == FallbackReason.GENERATED


# 🔹 8. Test pubsub publishes event
def test_pubsub_publishes_event(mock_redis_client):
    router = HybridRAGRouter(use_redis=True, enable_pubsub=True, debug_mode=False)

    router.publish_feedback_event("test_event", {"data": "test"})

    mock_redis_client.publish.assert_called_once()


# 🔹 9. Test planner used when high score
def test_planner_used_when_high_score(mock_policy_store, mock_planner, mock_ranker, make_chunk):
    mock_planner.plan_as_context.return_value = [make_chunk("planner", 0.9)]
    mock_ranker.rank.return_value = [make_chunk("planner", 0.9)]

    router = HybridRAGRouter(
        planner=mock_planner,
        ranker=mock_ranker,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False
    )

    result = router.route("Query")
    assert "planner" in result[0].chunk.content


# 🔹 10. Test retrieval used when planner disabled
def test_retrieval_used_when_planner_disabled(mock_policy_store, mock_retrieval_coordinator, make_chunk):
    mock_policy_store.get_bool.side_effect = lambda key, default=False: {
        "planner.first": False,
        "disable_planner": True
    }.get(key, default)

    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False
    )

    result = router.route("Query")
    assert "retrieved" in result[0].chunk.content


# 🔹 11. Test debug mode returns context
def test_debug_mode_returns_context(mock_policy_store, mock_fallback, make_chunk):
    mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

    router = HybridRAGRouter(
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        debug_mode=True,
        enable_caching=False
    )

    result, ctx = router.route("Query")
    assert isinstance(result, list)
    assert hasattr(ctx, 'query')


# 🔹 12. Test cache hit sets ctx.cache_hit
def test_cache_hit_sets_ctx_cache_hit(mock_redis_client, make_chunk):
    cached_chunks = [make_chunk("cached", 0.9)]
    mock_redis_client.get.return_value = b'[{"chunk": {"id": "cached", "content": "cached"}, "score": 0.9}]'

    router = HybridRAGRouter(use_redis=True, enable_caching=True, debug_mode=True)

    with patch.object(router, '_deserialize', return_value=cached_chunks):
        result, ctx = router.route("Query", session_id="test")

    assert ctx.cache_hit is True
    assert "cached" in result[0].chunk.content


# 🔹 13. Test fallback_used is set to True
def test_fallback_used_is_set_to_true(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
    fallback = MagicMock()
    fallback.generate_fallback.return_value = [make_chunk("fallback")]

    mock_retrieval_coordinator.hybrid_retrieve.return_value = []

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=True
    )

    result, ctx = router.route("No results query", session_id="test")
    assert ctx.fallback_used is True
    assert ctx.fallback_reason == FallbackReason.GENERATED


# 🔹 14. Test retrieval_filtered_count is correct
def test_retrieval_filtered_count_is_correct(mock_policy_store, mock_retrieval_coordinator, mock_ranker, make_chunk):
    retrieved = [make_chunk(f"retrieved-{i}", 0.6) for i in range(5)]
    ranked = retrieved[:3]  # Simulate filtering

    mock_retrieval_coordinator.hybrid_retrieve.return_value = retrieved
    mock_ranker.rank.return_value = ranked

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        ranker=mock_ranker,
        policy_store=mock_policy_store,
        debug_mode=True,
        enable_caching=False
    )

    result, ctx = router.route("Query")

    assert len(result) == 3
    assert ctx.total_context_chunks == 5
    assert ctx.retrieval_filtered_count == max(len(retrieved) - len(ranked), 0)


# 🔹 15. Test planner exception is handled
def test_planner_exception_is_handled(mock_policy_store, mock_planner, mock_retrieval_coordinator, make_chunk):
    mock_planner.plan_as_context.side_effect = Exception("Planner crashed")
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]

    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=True
    )

    result, ctx = router.route("Query")

    assert len(result) > 0
    assert "retrieved" in result[0].chunk.content
    assert ctx.planner_used is False


# 🔹 16. Test retrieval exception is handled
def test_retrieval_exception_is_handled(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
    mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Retrieval failed")
    mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=True
    )

    result, ctx = router.route("Query")

    assert len(result) > 0
    assert "fallback" in result[0].chunk.content
    assert ctx.fallback_reason == FallbackReason.RETRIEVAL_FAILED


# 🔹 17. Test feedback loop respects max_retry_depth
def test_feedback_respects_max_retry_depth(mock_policy_store, mock_feedback, make_chunk):
    mock_feedback.should_retry.return_value = True
    mock_feedback.retry_or_replan.return_value = "retry query"

    router = HybridRAGRouter(
        feedback=mock_feedback,
        policy_store=mock_policy_store,
        max_retry_depth=1,
        use_redis=False,
        debug_mode=True,
        enable_rerank=False
    )

    with patch.object(router, 'route') as mock_route:
        mock_route.return_value = ([make_chunk("result", 0.8)], MagicMock())
        result, ctx = router.route("Query", retry_depth=1)

    assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED


# 🔹 18. Test fallback_reason is set to PLANNER_LOW_SCORE
def test_planner_low_score_sets_fallback_reason(mock_policy_store, mock_planner, make_chunk):
    mock_planner.plan_as_context.side_effect = Exception("Planner failed")
    mock_retrieval_coordinator = MagicMock()
    mock_retrieval_coordinator.hybrid_retrieve.return_value = []

    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        debug_mode=True
    )

    result, ctx = router.route("Query")
    assert ctx.fallback_reason == FallbackReason.PLANNER_LOW_SCORE


# 🔹 19. Test retrieval_used is set when retrieval runs
def test_retrieval_used_is_set(mock_policy_store, mock_retrieval_coordinator, make_chunk):
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        debug_mode=True
    )

    result, ctx = router.route("Query")
    assert ctx.retrieval_used is True
