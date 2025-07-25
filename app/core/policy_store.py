# tests/conftest.py
import logging
import pytest
from unittest.mock import MagicMock, patch
import os
import json

from app.utils.schema import RetrievedChunk, DocumentChunk


@pytest.fixture(scope="function")
def mock_policy_store():
    """
    Fully isolated mock of PolicyStore.
    Never connects to DynamoDB.
    Returns correct types for all get_* methods.
    """
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

    # Base get
    store.get.side_effect = lambda key, default=None: policies.get(key, default)

    # get_bool: string -> bool
    def get_bool(key, default=False):
        value = policies.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ["true", "1", "yes", "on", "enabled"]

    store.get_bool.side_effect = get_bool

    # get_int: string -> int
    def get_int(key, default=3):
        value = policies.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    store.get_int.side_effect = get_int

    # get_float: string -> float
    def get_float(key, default=0.0):
        value = policies.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    store.get_float.side_effect = get_float

    # get_str: any -> str
    def get_str(key, default=""):
        value = policies.get(key, default)
        return str(value)

    store.get_str.side_effect = get_str

    # get_list: comma-separated or JSON list
    def get_list(key, default=None):
        value = policies.get(key, "")
        if not value:
            return default or []
        try:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                if value.strip().startswith("["):
                    return json.loads(value)
                return [item.strip() for item in value.split(",") if item.strip()]
            return default or []
        except Exception:
            return default or []

    store.get_list.side_effect = get_list

    # get_dict: JSON string or dict
    def get_dict(key, default=None):
        value = policies.get(key, "")
        if not value:
            return default or {}
        try:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return json.loads(value)
            return default or {}
        except Exception:
            return default or {}

    store.get_dict.side_effect = get_dict

    return store


@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    """
    Set all required environment variables.
    Ensures no real AWS, Redis, or OpenSearch connections.
    """
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

    # Fake AWS credentials for boto3 (required even in local mode)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(autouse=True)
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
def make_chunk():
    """
    Factory for creating real RetrievedChunk objects.
    Ensures deterministic, identity-based assertions.
    """

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
