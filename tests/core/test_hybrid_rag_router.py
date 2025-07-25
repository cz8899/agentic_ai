# tests/core/test_hybrid_rag_router.py
import logging
import pytest
from unittest.mock import MagicMock, patch
from app.core.hybrid_rag_router import HybridRAGRouter, FallbackReason
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
    def _make(content: str, score: float = 0.5, source: str = "test"):
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


# 🔹 Only real, fixable tests — no invented ones
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
    mock_policy_store.get_bool.side_effect = lambda key, default: {
        "planner.first": planner_first == "true",
        "enable_fallback": fallback_enabled == "true",
    }.get(key, default)

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

    if not fallback_enabled == "true":
        assert result == []


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
