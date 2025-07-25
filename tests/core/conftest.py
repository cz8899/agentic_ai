# tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock
import os

@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://dummy")
    monkeypatch.setenv("OPENSEARCH_HOST", "dummy-host")
    monkeypatch.setenv("OPENSEARCH_USER", "test-user")
    monkeypatch.setenv("OPENSEARCH_PASS", "test-pass")
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

@pytest.fixture
def make_chunk():
    def _make(content: str, score: float = 0.5, source: str = "test", doc_id: str = None):
        if doc_id is None:
            doc_id = f"doc-{abs(hash(content)) % 10000}"
        from app.utils.schema import RetrievedChunk, DocumentChunk
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
def mock_policy_store():
    store = MagicMock()
    policies = {
        "planner.first": "true",
        "enable_fallback": "true",
        "retrieval.score_threshold": 1.0,   # Force fallback
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
