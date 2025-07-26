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
    def _factory(content: str, score: float = 0.5, source: str = "test"):
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
    return _factory
    
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

@pytest.fixture(autouse=True)
def _stub_everything(monkeypatch, make_chunk):
    """
    1) Stub every external dependency so tests never hit AWS/Redis/LLM
    2) Provide deterministic mocks for all 14 tests
    """
    from app.core.retrieval_coordinator import RetrievalCoordinator
    from app.core.fallback_router import FallbackRouter
    from app.core.conversation_planner import ConversationPlanner
    from app.core.rank_with_bedrock import BedrockRanker

    monkeypatch.setattr(
        "app.core.retrieval_coordinator.RetrievalCoordinator.hybrid_retrieve",
        lambda self, payload: [make_chunk("retrieved", 0.8)]
    )

    # 1️⃣ Always return a single retrieved chunk
    monkeypatch.setattr(
        RetrievalCoordinator, "hybrid_retrieve",
        lambda self, payload: [make_chunk("retrieved", 0.8)]
    )

    # 2️⃣ BedrockRanker → identity passthrough
    monkeypatch.setattr(BedrockRanker, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(BedrockRanker, "rank", lambda self, query, chunks: chunks)

    # 3️⃣ ConversationPlanner → empty list (never wins)
    monkeypatch.setattr(ConversationPlanner, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(ConversationPlanner, "plan_as_context", lambda self, query: [])

    # 4️⃣ FallbackRouter → empty list (keeps RETRIEVAL_FAILED reason)
    monkeypatch.setattr(FallbackRouter, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(FallbackRouter, "generate_fallback", lambda self, query: [])

    # 5️⃣ Redis stub (autouse) – already inside patch_redis
    with patch("redis.from_url") as mock_from_url:
        fake_redis = MagicMock()
        fake_redis.ping.return_value = True
        fake_redis.get.return_value = None
        fake_redis.setex.return_value = True
        fake_redis.publish.return_value = True
        mock_from_url.return_value = fake_redis
        yield
