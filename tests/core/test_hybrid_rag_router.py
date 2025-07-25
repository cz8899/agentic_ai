import pytest
import logging
from unittest.mock import MagicMock, patch
from app.core.hybrid_rag_router import HybridRAGRouter
from app.core.routing_context import RoutingContext
from app.models.data_models import RetrievedChunk, DocumentChunk
from app.utils.enums import FallbackReason

logger = logging.getLogger("app")

@pytest.fixture
def mock_context():
    return RoutingContext(query="Test query", user_id="123")

@pytest.fixture
def router():
    return HybridRAGRouter()

@pytest.mark.parametrize("planner_enabled,retriever_enabled", [
    (False, True),
    (True, False),
    (False, False),
    (True, True)
])
def test_policy_combinations(mock_context, router, planner_enabled, retriever_enabled):
    router.planner = MagicMock()
    router.planner_enabled = planner_enabled
    router.retriever = MagicMock()
    router.retriever_enabled = retriever_enabled
    router.rank = MagicMock()
    router.fallback_router = MagicMock()

    chunk = DocumentChunk(id="doc-123", content="retrieved", source="test", title="title")
    router.rank.return_value = [RetrievedChunk(chunk=chunk, score=0.9)]
    router.fallback_router.route.return_value = [RetrievedChunk(chunk=chunk, score=0.4)]

    result = router.route(mock_context)
    assert isinstance(result, list)
    if planner_enabled or retriever_enabled:
        assert any("retrieved" in r.chunk.content for r in result)


def test_rerank_skipped_if_disabled(mock_context, router):
    router.policy_store.get.side_effect = lambda k: False if "rerank" in k else True
    router.retriever.retrieve = MagicMock(return_value=[DocumentChunk(id="1", content="A")])
    router.rank = MagicMock()

    results = router.route(mock_context)
    router.rank.assert_not_called()
    assert isinstance(results, list)


def test_reranker_runs_with_planner(mock_context, router):
    router.policy_store.get.side_effect = lambda k: True
    router.planner.plan.return_value = [DocumentChunk(id="1", content="planner")]
    router.rank = MagicMock()
    router.rank.return_value = [RetrievedChunk(chunk=DocumentChunk(id="1", content="planner"), score=0.95)]

    results = router.route(mock_context)
    router.rank.assert_called()
    assert results[0].chunk.content == "planner"


def test_feedback_triggered_on_low_score(mock_context, router):
    router.feedback = MagicMock()
    router.rank = MagicMock()
    chunk = DocumentChunk(id="id", content="fallback")
    router.rank.return_value = [RetrievedChunk(chunk=chunk, score=0.3)]

    router.route(mock_context)
    router.feedback.send_feedback.assert_called()


def test_retry_exhaustion_logs_and_stops(mock_context, router, caplog):
    router.planner.plan.side_effect = NameError("name 'history' is not defined")
    router.retry_depth = 1
    with caplog.at_level(logging.WARNING):
        router.route(mock_context)
        assert "Max retry depth (1) reached. Skipping feedback retry." in caplog.text


def test_retrieval_exception_is_handled(mock_context, router):
    router.retriever.retrieve.side_effect = Exception("API failure")
    router.fallback_router.route = MagicMock(return_value=[])

    results = router.route(mock_context)
    assert results == []


def test_planner_exception_is_handled(mock_context, router):
    router.planner.plan.side_effect = Exception("Planner down")
    router.fallback_router.route = MagicMock(return_value=[])

    results = router.route(mock_context)
    assert results == []


def test_cache_hit_sets_ctx_cache_hit(mock_context, router):
    router._get_from_cache = MagicMock(return_value=[RetrievedChunk(chunk=DocumentChunk(id="doc-1", content="cached"), score=0.5)])

    results = router.route(mock_context)
    assert mock_context.cache_hit
    assert results[0].chunk.content == "cached"


def test_cache_miss_sets_ctx_cache_miss(mock_context, router):
    router._get_from_cache = MagicMock(return_value=None)
    router._set_in_cache = MagicMock()
    router.fallback_router.route = MagicMock(return_value=[])

    router.route(mock_context)
    assert not mock_context.cache_hit


def test_cache_hit_miss_logging(mock_context, router, caplog):
    router._get_from_cache = MagicMock(return_value=None)
    router._set_in_cache = MagicMock()
    router.fallback_router.route = MagicMock(return_value=[])

    with caplog.at_level(logging.INFO):
        router.route(mock_context)
        assert "RAG route complete" in caplog.text


def test_in_memory_cache_used_when_redis_fails(mock_context, router):
    router.redis_cache = MagicMock()
    router.redis_cache.get.side_effect = Exception("Redis down")
    router._set_in_cache = MagicMock()
    router.fallback_router.route = MagicMock(return_value=[])

    router.route(mock_context)
    assert router._set_in_cache.called


def test_fallback_triggered_when_empty_result(mock_context, router):
    router.retriever.retrieve = MagicMock(return_value=[])
    router.fallback_router.route = MagicMock(return_value=[RetrievedChunk(chunk=DocumentChunk(id="f1", content="fallback"), score=0.5)])

    results = router.route(mock_context)
    assert "fallback" in results[0].chunk.content


def test_ctx_tracks_fallback_reason(mock_context, router):
    router.retriever.retrieve = MagicMock(return_value=[])
    router.fallback_router.route = MagicMock(return_value=[])

    router.route(mock_context)
    assert FallbackReason.RETRIEVAL_EMPTY in mock_context.fallback_reasons


def test_retrieval_used_when_planner_disabled(mock_context, router):
    router.policy_store.get.side_effect = lambda k: False if "planner" in k else True
    router.retriever.retrieve = MagicMock(return_value=[DocumentChunk(id="id", content="R")])

    results = router.route(mock_context)
    assert "R" in results[0].chunk.content


def test_reranked_list_truncated_to_top_k(mock_context, router):
    chunk = DocumentChunk(id="doc-1", content="A")
    retrieved = [RetrievedChunk(chunk=chunk, score=s) for s in [0.9, 0.8, 0.7, 0.6]]
    router.rank = MagicMock(return_value=retrieved)

    results = router.route(mock_context)
    assert len(results) <= 3
