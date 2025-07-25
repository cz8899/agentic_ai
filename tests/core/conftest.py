# tests/core/conftest.py
import logging
import pytest
from unittest.mock import MagicMock, patch
import os
import json

@pytest.fixture(scope="function")
def mock_policy_store():
    store = MagicMock()

    policies = {
        "planner.first": "true",
        "enable_fallback": "true",
        "retrieval.score_threshold": 0.0,
        "retrieval.top_k": 3,
        "disable_opensearch": "false",
        "enable_chroma": "true",
        "enable_bedrock_kb": "true",
        "ranker.model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude_direct_model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock_kb_id": "test-kb-123",
        "feedback.enable_replan": "true",
        "feedback.enable_critique": "true",
        "feedback.critique_score_threshold": 0.25,
    }

    store.get.side_effect = lambda key, default=None: policies.get(key, default)

    def get_bool(key, default=False):
        value = policies.get(key, str(default))
        return str(value).strip().lower() in ["true", "1", "yes", "on", "enabled"]
    store.get_bool.side_effect = get_bool

    def get_int(key, default=3):
        return int(policies.get(key, default))
    store.get_int.side_effect = get_int

    def get_float(key, default=0.0):
        return float(policies.get(key, default))
    store.get_float.side_effect = get_float

    def get_str(key, default=""):
        return str(policies.get(key, default))
    store.get_str.side_effect = get_str

    def get_list(key, default=None):
        value = policies.get(key, "")
        if not value:
            return default or []
        try:
            if value.strip().startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        except:
            return default or []
    store.get_list.side_effect = get_list

    return store


@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://dummy-opensearch")
    monkeypatch.setenv("OPENSEARCH_HOST", "dummy-host")
    monkeypatch.setenv("OPENSEARCH_USER", "test-user")
    monkeypatch.setenv("OPENSEARCH_PASS", "test-pass")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-v2")
    monkeypatch.setenv("CHROMA_HOST", "http://localhost:8000")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("BEDROCK_KB_ID", "test-kb-123")

    test_logs = os.path.join(os.path.dirname(__file__), "test_logs")
    os.makedirs(test_logs, exist_ok=True)
    monkeypatch.setenv("CRITIQUE_LOG_DIR", test_logs)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

def patch_redis():
    """
    Automatically patch redis.from_url for ALL tests.
    Ensures no real Redis connection is ever attempted.
    Applied before HybridRAGRouter is instantiated.
    """
    with patch("redis.from_url") as mock_from_url:
        fake_redis = MagicMock()
        fake_redis.ping.return_value = True
        fake_redis.get.return_value = None  # Default: cache miss
        fake_redis.setex.return_value = True
        fake_redis.publish.return_value = True

        mock_from_url.return_value = fake_redis
        yield fake_redis
        
@pytest.fixture
def mock_redis_client():
    with patch("redis.from_url") as mock:
        client = MagicMock()
        client.ping.return_value = True
        client.get.return_value = None
        client.setex = MagicMock()
        mock.return_value = client
        yield client


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
