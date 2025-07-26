# tests/core/conftest.py
import pytest
from unittest.mock import patch, MagicMock
import os

@pytest.fixture
def mock_policy_store():
    store = MagicMock()
    policies = {
        "planner.first": "false",
        "enable_fallback": "false",
        "retrieval.score_threshold": 1.0,
        "retrieval.top_k": 5,
        "disable_opensearch": "false",
        "enable_chroma": "true",
        "enable_bedrock_kb": "true",
    }

    store.get.side_effect = lambda key, default=None: policies.get(key, default)
    store.get_bool.side_effect = lambda key, default=False: str(policies.get(key, default)).strip().lower() in ["true", "1", "yes", "on"]
    store.get_int.side_effect = lambda key, default=0: int(policies.get(key, default))
    store.get_float.side_effect = lambda key, default=0.0: float(policies.get(key, default))
    return store


@pytest.fixture
def make_chunk():
    def _make(content: str, score: float = 0.5, source: str = "test"):
        from app.utils.schema import RetrievedChunk, DocumentChunk
        return RetrievedChunk(
            chunk=DocumentChunk(
                id=f"doc-{abs(hash(content)) % 10000}",
                content=content,
                source=source,
                title=f"Chunk: {content[:30]}..."
            ),
            score=score
        )
    return _make


@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_USER", "test")
    monkeypatch.setenv("OPENSEARCH_PASS", "test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("BEDROCK_KB_ID", "test-kb-123")


@pytest.fixture(autouse=True)
def patch_redis():
    with patch("redis.from_url") as m:
        client = MagicMock()
        client.ping.return_value = True
        client.get.return_value = None
        client.setex.return_value = True
        client.publish.return_value = True
        m.return_value = client
        yield client


@pytest.fixture(autouse=True)
def stub_ranker():
    with patch("app.core.hybrid_rag_router.BedrockRanker") as m:
        m.return_value.rank.return_value = []
        yield m.return_value

# tests/conftest.py  (append at the bottom)
@pytest.fixture(autouse=True)
def _stub_everything(monkeypatch, mock_policy_store):
    """
    Stub every component that tries to reach AWS, Redis, LLM, etc.
    Keeps the router fast and isolated.
    """
    from app.core.conversation_planner import ConversationPlanner
    from app.core.rank_with_bedrock import BedrockRanker
    from app.core.fallback_router import FallbackRouter
    # inside conftest.py → _stub_everything fixture
    from app.core.retrieval_coordinator import RetrievalCoordinator
    monkeypatch.setattr(
        RetrievalCoordinator, "hybrid_retrieve",
        lambda self, payload: [make_chunk("retrieved", 0.8)]
    )
    # ConversationPlanner → always empty list
    monkeypatch.setattr(
        ConversationPlanner, "plan_as_context", lambda self, query: []
    )

    # BedrockRanker → identity (no filtering)
    monkeypatch.setattr(
        BedrockRanker, "__init__", lambda self, **kw: None
    )
    monkeypatch.setattr(
        BedrockRanker, "rank", lambda self, query, chunks: chunks
    )

    # FallbackRouter → always empty list so we can force failures
    monkeypatch.setattr(
        FallbackRouter, "__init__", lambda self, **kw: None
    )
    monkeypatch.setattr(
        FallbackRouter, "generate_fallback", lambda self, query: []
    )

    # Force redis.from_url to return an in-memory mock
    fake_redis = MagicMock()
    fake_redis.ping.return_value = True
    fake_redis.get.return_value = None          # cache miss
    fake_redis.setex.return_value = True
    fake_redis.publish.return_value = True
    monkeypatch.setattr("redis.from_url", lambda *a, **kw: fake_redis)
