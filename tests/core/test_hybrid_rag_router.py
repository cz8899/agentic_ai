import pytest
from unittest.mock import MagicMock, create_autospec, patch
from app.core.hybrid_rag_router import HybridRAGRouter
from app.schema import RetrievedChunk, DocumentChunk, FallbackReason
from app.core.policy_store import PolicyStore
from app.core.retrieval import HybridRetrievalCoordinator
from app.core.feedback import FeedbackCoordinator


def make_chunk(content: str, score: float = 0.5):
    return RetrievedChunk(
        chunk=DocumentChunk(
            id=f"doc-{hash(content) % 10000}",
            content=content,
            source="test",
            title=f"Chunk: {content}...",
            metadata={},
        ),
        score=score,
        explanation=None,
    )


@pytest.mark.parametrize("planner_first", [True, False])
@pytest.mark.parametrize("fallback_enabled", [True, False])
def test_policy_combinations(
    planner_first,
    fallback_enabled,
):
    mock_policy_store = MagicMock()
    mock_policy_store.get_bool.side_effect = lambda k, d=False: {
        "planner.first": planner_first,
        "enable_fallback": fallback_enabled,
        "retrieval.score_threshold": 0.0,
    }.get(k, d)

    mock_retrieval_coordinator = MagicMock()
    retrieval_chunk = make_chunk("retrieved", 0.8)
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]

    mock_planner = MagicMock()
    mock_planner.plan.return_value = []

    fallback_chunk = make_chunk("fallback", 0.5)
    mock_fallback = MagicMock()
    mock_fallback.generate_fallback.return_value = [fallback_chunk]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        planner=mock_planner,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=True,
    )

    result, _ = router.route("Test query")

    if planner_first or not fallback_enabled:
        assert result == [retrieval_chunk]
    else:
        assert result == [fallback_chunk]


def test_fallback_when_retrieval_fails():
    router = HybridRAGRouter(
        coordinator=MagicMock(hybrid_retrieve=MagicMock(side_effect=Exception("boom"))),
        fallback=MagicMock(generate_fallback=MagicMock(return_value=[make_chunk("fallback")])),
        planner=MagicMock(plan=lambda *_: []),
        debug_mode=True,
    )

    result, _ = router.route("Trigger fallback")
    assert result[0].chunk.content == "fallback"


def test_retry_exhaustion_logs_and_stops(caplog):
    mock_retrieval = MagicMock()
    mock_retrieval.hybrid_retrieve.return_value = []

    feedback = MagicMock(spec=FeedbackCoordinator)
    feedback.retry_or_replan.side_effect = ["retry-1", "retry-2"]

    mock_policy_store = MagicMock()
    mock_policy_store.get_bool.side_effect = lambda k, d=False: {
        "planner.first": False,
        "enable_fallback": False,
    }.get(k, d)

    router = HybridRAGRouter(
        coordinator=mock_retrieval,
        fallback=MagicMock(),
        planner=MagicMock(plan=lambda *_: []),
        feedback=feedback,
        policy_store=mock_policy_store,
        debug_mode=True,
        max_retry_depth=2,
    )

    router.route("Trigger retry")

    assert "Max retry depth (2) reached" in caplog.text


def test_feedback_triggered_on_low_score():
    mock_policy_store = MagicMock()
    mock_policy_store.get_bool.return_value = False
    mock_policy_store.get_float.return_value = 0.9

    mock_feedback = MagicMock()
    mock_feedback.retry_or_replan.return_value = None

    retrieval_chunk = make_chunk("low-score", 0.5)
    mock_retrieval_coordinator = MagicMock()
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        planner=MagicMock(plan=lambda *_: []),
        fallback=MagicMock(),
        feedback=mock_feedback,
        policy_store=mock_policy_store,
        debug_mode=True,
    )

    router.route("Low score")
    mock_feedback.retry_or_replan.assert_called()


def test_cache_hit_miss_logging(caplog):
    mock_cache = MagicMock()
    mock_cache.get_cache.return_value = None
    mock_cache._set_in_cache = MagicMock()

    mock_policy_store = MagicMock()
    mock_policy_store.get_bool.side_effect = lambda k, d=False: {
        "enable_fallback": True,
        "planner.first": False,
    }.get(k, d)

    router = HybridRAGRouter(
        coordinator=MagicMock(hybrid_retrieve=lambda *_: []),
        fallback=MagicMock(generate_fallback=lambda *_: [make_chunk("fallback", 0.6)]),
        planner=MagicMock(plan=lambda *_: []),
        policy_store=mock_policy_store,
        cache=mock_cache,
        debug_mode=True,
    )

    router.route("Check logging", session_id="test")
    assert "Cache miss" in caplog.text
    assert "RAG route complete" in caplog.text


def test_in_memory_cache_used_when_redis_fails():
    with patch("app.utils.redis_utils.StrictRedis.from_url", side_effect=Exception("Redis error")):
        router = HybridRAGRouter(
            coordinator=MagicMock(hybrid_retrieve=lambda *_: []),
            fallback=MagicMock(generate_fallback=lambda *_: [make_chunk("fallback", 0.6)]),
            planner=MagicMock(plan=lambda *_: []),
            enable_caching=True,
            debug_mode=True,
        )

        spy = patch.object(router.cache, "_set_in_cache").start()
        router.route("Trigger in-memory fallback", session_id="test")
        spy.assert_called()


def test_fallback_reasons_are_set_correctly():
    mock_retrieval = MagicMock()
    mock_retrieval.hybrid_retrieve.return_value = []

    router = HybridRAGRouter(
        coordinator=mock_retrieval,
        fallback=MagicMock(generate_fallback=lambda *_: [make_chunk("fallback")]),
        planner=MagicMock(plan=lambda *_: []),
        debug_mode=True,
    )

    _, ctx = router.route("Fallback reason")
    assert ctx.fallback_reason == FallbackReason.LOW_SCORE


def test_pubsub_publishes_event():
    pubsub = MagicMock()
    pubsub.publish = MagicMock()

    router = HybridRAGRouter(
        coordinator=MagicMock(hybrid_retrieve=lambda *_: [make_chunk("ok", 0.8)]),
        fallback=MagicMock(),
        planner=MagicMock(plan=lambda *_: []),
        pubsub=pubsub,
        debug_mode=True,
    )

    router.route("Publish test", session_id="abc")
    pubsub.publish.assert_called()


def test_retrieval_filtered_count_is_correct():
    chunk1 = make_chunk("relevant", 0.9)
    chunk2 = make_chunk("irrelevant", 0.3)

    router = HybridRAGRouter(
        coordinator=MagicMock(hybrid_retrieve=lambda *_: [chunk1, chunk2]),
        fallback=MagicMock(),
        planner=MagicMock(plan=lambda *_: []),
        score_threshold=0.5,
        debug_mode=True,
    )

    result, ctx = router.route("Score filter")
    assert chunk1 in result
    assert chunk2 not in result
    assert ctx.retrieval_filtered == 1


def test_retrieval_exception_is_handled():
    mock_retrieval = MagicMock()
    mock_retrieval.hybrid_retrieve.side_effect = Exception("boom")

    mock_fallback = MagicMock()
    mock_fallback.generate_fallback.side_effect = Exception("fallback failed")

    router = HybridRAGRouter(
        coordinator=mock_retrieval,
        fallback=mock_fallback,
        planner=MagicMock(plan=lambda *_: []),
        debug_mode=True,
    )

    _, ctx = router.route("Fail both")
    assert ctx.fallback_reason == FallbackReason.RETRIEVAL_FAILED


def test_retrieval_used_is_set():
    router = HybridRAGRouter(
        coordinator=MagicMock(hybrid_retrieve=lambda *_: [make_chunk("data")]),
        fallback=MagicMock(),
        planner=MagicMock(plan=lambda *_: []),
        debug_mode=True,
    )

    _, ctx = router.route("Tracking")
    assert ctx.retrieval_used is True
