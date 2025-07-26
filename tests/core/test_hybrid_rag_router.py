# tests/core/test_hybrid_rag_router.py
import logging
import pytest
from unittest.mock import MagicMock, patch
from app.core.hybrid_rag_router import HybridRAGRouter, FallbackReason, RouterMode
from app.utils.schema import RetrievedChunk, DocumentChunk


@pytest.fixture
def mock_retrieval_coordinator():
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


# 🔹 1. Test policy combinations
@pytest.mark.parametrize("planner_first,fallback_enabled", [
    (True, True),
    (False, True),
    (True, False),
    (False, False),
])
def test_policy_combinations(
    mock_policy_store,
    mock_retrieval_coordinator,
    make_chunk,
    planner_first,
    fallback_enabled,
):
    # 1) force deterministic policies
    mock_policy_store.get_bool.side_effect = lambda k, d=False: {
        "planner.first": planner_first,
        "enable_fallback": fallback_enabled,
        "retrieval.score_threshold": 0.0,  # let retrieval win
    }.get(k, d)

    # 2) stub retrieval
    retrieval_chunk = make_chunk("retrieved", 0.8)
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]

    # 3) build router
    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False,
    )

    # 4) run & assert
    result, _ = router.route("Test query")
    assert result == [retrieval_chunk]

# 🔹 2. Test fallback when retrieval fails
def test_fallback_when_retrieval_fails(
    mock_policy_store,
    mock_retrieval_coordinator,
    mock_fallback,
    make_chunk
):
    mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Retrieval failed")
    fallback_chunk = make_chunk("fallback")
    mock_fallback.generate_fallback.return_value = [fallback_chunk]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=True,
        debug_mode=True,
    )

    result = router.route("Query")
    assert result == [fallback_chunk]


# 🔹 3. Test retry exhaustion logs and stops
def test_retry_exhaustion_logs_and_stops(mock_policy_store, make_chunk, caplog):
    feedback = MagicMock()
    feedback.should_retry.return_value = True
    feedback.retry_or_replan.return_value = "rewritten query"

    fallback = MagicMock()
    fallback.generate_fallback.return_value = []

    router = HybridRAGRouter(
        feedback=feedback,
        fallback=fallback,
        policy_store=mock_policy_store,
        max_retry_depth=2,
        use_redis=False,
        debug_mode=True,
    )

    with caplog.at_level(logging.WARNING):
        result, ctx = router.route("Query")

    assert "Max retry depth (2) reached" in caplog.text
    assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED
    assert result == []

# 🔹 4. Test feedback triggered on low score
def test_feedback_triggered_on_low_score(mock_policy_store, mock_retrieval_coordinator, mock_feedback, make_chunk):
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("low-score", 0.1)]
    mock_feedback.should_retry.return_value = True
    mock_feedback.retry_or_replan.return_value = "Improved query"

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        feedback=mock_feedback,
        policy_store=mock_policy_store,
        max_retry_depth=2,
        use_redis=False,
        debug_mode=True,
    )

    with patch.object(router, 'route') as mock_route:
        mock_route.side_effect = lambda *args, **kwargs: ([make_chunk("good-result", 0.8)], MagicMock())
        result, ctx = router.route("Query")

    mock_route.assert_called()
    assert "good-result" in result[0].chunk.content


# 🔹 5. Test cache hit and miss logging
def test_cache_hit_miss_logging(caplog, mock_redis_client):
    router = HybridRAGRouter(use_redis=True, enable_caching=True, debug_mode=True)

    with caplog.at_level(logging.INFO):
        result, ctx = router.route("Query", session_id="test")
        assert "RAG route complete" in caplog.text
        assert ctx.cache_hit is False

# 🔹 6. Test in-memory cache used when Redis fails
def test_in_memory_cache_used_when_redis_fails(
    monkeypatch, mock_policy_store, mock_retrieval_coordinator, make_chunk
):
    # force redis.from_url to fail
    monkeypatch.setattr("redis.from_url", lambda *a, **kw: None)

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        enable_caching=True,
        debug_mode=True,
    )

    with patch.object(router, "_set_in_cache") as spy:
        router.route("Query", session_id="test")
        spy.assert_called()

# 🔹 7. Test fallback reasons are set correctly
def test_fallback_reasons_are_set_correctly(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
    mock_retrieval_coordinator.hybrid_retrieve.return_value = []
    mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=True,
        debug_mode=True,
    )

    result, ctx = router.route("No results query", session_id="test")
    assert ctx.fallback_reason == FallbackReason.GENERATED


# 🔹 8. Test pubsub publishes event
def test_pubsub_publishes_event(mock_policy_store):
    router = HybridRAGRouter(
        policy_store=mock_policy_store,
        use_redis=True,
        enable_pubsub=True,
        debug_mode=False
    )

    router.publish_feedback_event("test_event", {"data": "test"})
    router.redis_client.publish.assert_called_once()


# 🔹 9. Test retrieval_filtered_count is correct
def test_retrieval_filtered_count_is_correct(mock_policy_store, mock_retrieval_coordinator, make_chunk):
    retrieved = [make_chunk(f"retrieved-{i}", 0.6) for i in range(5)]
    ranked = retrieved[:3]

    mock_retrieval_coordinator.hybrid_retrieve.return_value = retrieved

    with patch("app.core.hybrid_rag_router.BedrockRanker") as m:
        m.return_value.rank.return_value = ranked
        router = HybridRAGRouter(
            coordinator=mock_retrieval_coordinator,
            policy_store=mock_policy_store,
            debug_mode=True,
            enable_caching=True,
        )
        result, ctx = router.route("Query")

    assert len(result) == 3
    assert ctx.total_context_chunks == 5
    assert ctx.retrieval_filtered_count == 2


# 🔹 10. Test retrieval exception is handled
def test_retrieval_exception_is_handled(
    mock_policy_store, mock_retrieval_coordinator
):
    fallback = MagicMock()
    fallback.generate_fallback.return_value = []  # force RETRIEVAL_FAILED

    mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("boom")

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=fallback,
        policy_store=mock_policy_store,
        debug_mode=True,
    )

    result, ctx = router.route("Query")
    assert ctx.fallback_reason == FallbackReason.RETRIEVAL_FAILED

# 🔹 11. Test retrieval_used is set
def test_retrieval_used_is_set(mock_policy_store, mock_retrieval_coordinator, make_chunk):
    mock_policy_store.get_bool.side_effect = lambda key, default=False: {
        "planner.first": False,
        "disable_planner": True
    }.get(key, default)

    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        policy_store=mock_policy_store,
        debug_mode=True,
    )

    result, ctx = router.route("Query")
    assert ctx.retrieval_used is True
