# tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock
import os
from app.utils.schema import RetrievedChunk, DocumentChunk


@pytest.fixture(autouse=True)
def patch_env_vars(monkeypatch):
    """
    Set all required environment variables to prevent KeyError.
    Ensures no real AWS/OpenSearch/Chroma connections.
    """
    # OpenSearch
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "https://dummy")
    monkeypatch.setenv("OPENSEARCH_HOST", "dummy-host")
    monkeypatch.setenv("OPENSEARCH_USER", "test-user")
    monkeypatch.setenv("OPENSEARCH_PASS", "test-pass")

    # AWS / Bedrock
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_MODEL_ID", "anthropic.claude-v2")
    monkeypatch.setenv("BEDROCK_KB_ID", "test-kb-123")

    # Redis
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    # Chroma
    monkeypatch.setenv("CHROMA_HOST", "http://localhost:8000")

    # Critique logs
    test_logs = os.path.join(os.path.dirname(__file__), "test_logs")
    os.makedirs(test_logs, exist_ok=True)
    monkeypatch.setenv("CRITIQUE_LOG_DIR", test_logs)


@pytest.fixture(autouse=True)
def patch_redis():
    """
    Automatically patch redis.from_url before any test runs.
    Ensures no real Redis connection is attempted.
    """
    with patch("redis.from_url") as mock_from_url:
        fake_redis = MagicMock()
        fake_redis.ping.return_value = True
        fake_redis.get.return_value = None  # Default: cache miss
        fake_redis.setex.return_value = True
        fake_redis.publish.return_value = True
        mock_from_url.return_value = fake_redis
        yield fake_redis


@pytest.fixture(autouse=True)
def stub_bedrock_ranker():
    """
    Replace BedrockRanker with a no-op mock.
    Prevents real LLM calls during ranking.
    """
    with patch("app.core.hybrid_rag_router.BedrockRanker") as m:
        ranker = m.return_value
        ranker.rank.return_value = []  # Identity passthrough
        yield ranker


@pytest.fixture(autouse=True)
def stub_retrieval_coordinator():
    """
    Replace RetrievalCoordinator with a no-op mock.
    Prevents real OpenSearch/Chroma/Bedrock KB calls.
    """
    with patch("app.core.hybrid_rag_router.RetrievalCoordinator") as m:
        coordinator = m.return_value
        coordinator.hybrid_retrieve.return_value = []
        yield coordinator


@pytest.fixture(autouse=True)
def stub_policy_store():
    """
    Replace PolicyStore with a fully stubbed version.
    Never connects to DynamoDB.
    Returns correct types for all get_* methods.
    """
    with patch("app.core.hybrid_rag_router.PolicyStore") as m:
        store = m.return_value

        # Default policy values
        policies = {
            "planner.first": "true",
            "disable_planner": "false",
            "enable_feedback": "true",
            "enable_rerank": "true",
            "skip_cache": "false",
            "retrieval.score_threshold": 1.0,   # Force fallback paths
            "max_retry_depth": "2",
            "enable_fallback": "true",
            "retrieval.top_k": "5",
            "enable_bedrock_kb": "true",
            "disable_opensearch": "false",
            "enable_chroma": "false",
            "redis.url": "redis://localhost:6379/0",
            "cache.ttl": "300",
            "pubsub.enable": "true",
        }

        store.get.side_effect = lambda key, default=None: policies.get(key, default)
        store.get_str.side_effect = lambda key, default="": str(policies.get(key, default))
        store.get_bool.side_effect = lambda key, default=False: str(policies.get(key, default)).strip().lower() in ["true", "1", "yes", "on"]
        store.get_int.side_effect = lambda key, default=0: int(policies.get(key, default))
        store.get_float.side_effect = lambda key, default=0.0: float(policies.get(key, default))
        store.get_list.side_effect = lambda key, default=None: (
            default or [] if not policies.get(key) else
            [item.strip() for item in str(policies.get(key)).split(",") if item.strip()]
        )
        store.get_dict.side_effect = lambda key, default=None: (
            default or {} if not policies.get(key) else
            eval(policies.get(key)) if policies.get(key).strip().startswith("{") else {}
        )

        yield store


@pytest.fixture(autouse=True)
def stub_fallback_router():
    """
    Replace FallbackRouter with a no-op mock.
    Prevents real LLM calls during fallback.
    """
    with patch("app.core.hybrid_rag_router.FallbackRouter") as m:
        fallback = m.return_value
        fallback.generate_fallback.return_value = []
        yield fallback


@pytest.fixture(autouse=True)
def stub_conversation_planner():
    """
    Replace ConversationPlanner with a no-op mock.
    Prevents real LLM calls during planning.
    """
    with patch("app.core.hybrid_rag_router.ConversationPlanner") as m:
        planner = m.return_value
        planner.plan_as_context.return_value = []
        yield planner


@pytest.fixture(autouse=True)
def stub_feedback_controller():
    """
    Replace FeedbackLoopController with a no-op mock.
    Prevents real critique/rewrite calls.
    """
    with patch("app.core.hybrid_rag_router.FeedbackLoopController") as m:
        feedback = m.return_value
        feedback.retry_or_replan.return_value = None
        yield feedback


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
