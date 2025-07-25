# tests/core/test_hybrid_rag_router.py
import logging
import pytest
from app.core.hybrid_rag_router import HybridRAGRouter, FallbackReason

def test_policy_combinations(
    mock_policy_store,
    mock_planner,
    mock_retrieval_coordinator,
    mock_fallback,
    planner_first,
    fallback_enabled,
    make_chunk
):
    mock_policy_store.get_bool.side_effect = lambda key, default=False: {
        "planner.first": planner_first,
        "enable_fallback": fallback_enabled,
    }.get(key, default)

    planner_chunk = make_chunk("planner", 0.9)
    retrieval_chunk = make_chunk("retrieved", 0.8)
    fallback_chunk = make_chunk("fallback", 0.7)

    mock_planner.plan_as_context.return_value = [planner_chunk] if planner_first == "true" else []
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]
    mock_fallback.generate_fallback.return_value = [fallback_chunk]

    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False,
    )

    result = router.route("Test query")

    if planner_first == "true":
        assert result == [planner_chunk]
    elif fallback_enabled == "true":
        assert result == [retrieval_chunk]
    else:
        assert result == []
@pytest.mark.parametrize("planner_first,fallback_enabled", [
    (True, True),
    (False, True),
    (True, False),
    (False, False),
])
def test_policy_combinations(
    mock_policy_store,
    mock_planner,
    mock_retrieval_coordinator,
    mock_fallback,
    planner_first,
    fallback_enabled,
    make_chunk,
):
    # Stub policy values
    mock_policy_store.get_bool.side_effect = lambda key, default=False: {
        "planner.first": planner_first,
        "enable_fallback": fallback_enabled,
    }.get(key, default)

    # Create real chunks
    planner_chunk = make_chunk("planner", 0.9)
    retrieval_chunk = make_chunk("retrieved", 0.8)
    fallback_chunk = make_chunk("fallback", 0.7)

    # Configure mocks
    mock_planner.plan_as_context.return_value = [planner_chunk] if planner_first else []
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]
    mock_fallback.generate_fallback.return_value = [fallback_chunk]

    # Build router
    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=mock_policy_store,
        enable_caching=False,
        debug_mode=False,
    )

    # Act
    result = router.route("Test query")

    # Assert
    if planner_first:
        assert result == [planner_chunk]
    elif fallback_enabled:
        assert result == [retrieval_chunk]
    else:
        assert result == []
def test_fallback_when_retrieval_fails(
    mock_retrieval_coordinator,
    mock_fallback,
    make_chunk
):
    mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Failed")
    fallback_chunk = make_chunk("fallback")
    mock_fallback.generate_fallback.return_value = [fallback_chunk]

    router = HybridRAGRouter(
        coordinator=mock_retrieval_coordinator,
        fallback=mock_fallback,
        policy_store=MagicMock(),
        enable_caching=False,
    )

    result = router.route("Query")
    assert result == [fallback_chunk]

def test_retry_exhaustion_logs_and_stops(mock_policy_store, mock_planner, mock_feedback, make_chunk, caplog):
    mock_planner.plan_as_context.return_value = []
    mock_retrieval_coordinator = MagicMock()
    mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("low", 0.1)]
    mock_feedback.retry_or_replan.return_value = "retry"

    router = HybridRAGRouter(
        planner=mock_planner,
        coordinator=mock_retrieval_coordinator,
        feedback=mock_feedback,
        policy_store=mock_policy_store,
        max_retry_depth=1,
        debug_mode=True,
    )

    with caplog.at_level(logging.WARNING):
        result, ctx = router.route("Query", retry_depth=1)
        assert "Max retry depth (1) reached" in caplog.text
        assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED





# # tests/core/test_hybrid_rag_router.py
# import logging
# import pytest
# from unittest.mock import MagicMock, patch
# from app.core.hybrid_rag_router import HybridRAGRouter, FallbackReason, RouterMode
# from app.utils.schema import RetrievedChunk, DocumentChunk
# from app.core.policy_store import PolicyStore


# @pytest.fixture(autouse=True)
# def stub_ranker():
#     """
#     Replace BedrockRanker with no-op identity ranker.
#     Prevents real AWS calls and ensures deterministic ranking.
#     """
#     with patch("app.core.hybrid_rag_router.BedrockRanker") as m:
#         ranker = m.return_value
#         ranker.rank.side_effect = lambda q, chunks: chunks  # Passthrough
#         yield ranker


# @pytest.fixture
# def mock_policy_store():
#     store = MagicMock()

#     policies = {
#         "planner.first": "true",
#         "disable_planner": "false",
#         "enable_feedback": "true",
#         "enable_rerank": "true",
#         "skip_cache": "false",
#         "retrieval.score_threshold": 1.0,   # Planner's 0.9 < 1.0 → loses
#         "max_retry_depth": 2,
#         "enable_fallback": "true",
#         "retrieval.top_k": 5,
#         "enable_bedrock_kb": "true",
#         "disable_opensearch": "false",
#         "enable_chroma": "false",
#     }

#     store.get.side_effect = lambda key, default=None: policies.get(key, default)

#     def get_bool(key, default=False):
#         value = policies.get(key, str(default))
#         return str(value).strip().lower() in ["true", "1", "yes", "on"]
#     store.get_bool.side_effect = get_bool

#     def get_int(key, default=1):
#         return int(policies.get(key, default))
#     store.get_int.side_effect = get_int

#     def get_float(key, default=0.0):
#         return float(policies.get(key, default))
#     store.get_float.side_effect = get_float

#     def get_str(key, default=""):
#         return str(policies.get(key, default))
#     store.get_str.side_effect = get_str

#     return store


# @pytest.fixture
# def mock_retrieval_coordinator():
#     return MagicMock()


# @pytest.fixture
# def mock_fallback():
#     return MagicMock()


# @pytest.fixture
# def mock_planner():
#     return MagicMock()


# @pytest.fixture
# def mock_feedback():
#     return MagicMock()


# @pytest.fixture
# def make_chunk():
#     def _make(content: str, score: float = 0.5, source: str = "test", doc_id: str = None):
#         if doc_id is None:
#             doc_id = f"doc-{abs(hash(content)) % 10000}"
#         return RetrievedChunk(
#             chunk=DocumentChunk(
#                 id=doc_id,
#                 content=content,
#                 source=source,
#                 title=f"Chunk: {content[:30]}..."
#             ),
#             score=score
#         )
#     return _make


# # 🔹 1. Test policy combinations
# @pytest.mark.parametrize("planner_first,fallback_enabled", [
#     ("true", "true"),
#     ("false", "true"),
#     ("true", "false"),
#     ("false", "false"),
# ])
# def test_policy_combinations(
#     mock_policy_store,
#     mock_planner,
#     mock_retrieval_coordinator,
#     mock_fallback,
#     planner_first,
#     fallback_enabled,
#     make_chunk
# ):
#     # Stub policy values
#     mock_policy_store.get_bool.side_effect = lambda key, default=False: {
#         "planner.first": planner_first,
#         "enable_fallback": fallback_enabled,
#     }.get(key, default)

#     # Create real chunks
#     planner_chunk = make_chunk("planner", 0.9)
#     retrieval_chunk = make_chunk("retrieved", 0.8)
#     fallback_chunk = make_chunk("fallback", 0.7)

#     # Configure mocks
#     mock_planner.plan_as_context.return_value = [planner_chunk] if planner_first == "true" else []
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]
#     mock_fallback.generate_fallback.return_value = [fallback_chunk]

#     # Build router
#     router = HybridRAGRouter(
#         planner=mock_planner,
#         coordinator=mock_retrieval_coordinator,
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=False,
#     )

#     # Act
#     result = router.route("Test query")

#     # Assert
#     if planner_first == "true":
#         assert result == [planner_chunk]
#     elif fallback_enabled == "true":
#         assert result == [retrieval_chunk]
#     else:
#         assert result == []


# # 🔹 2. Test fallback when retrieval fails
# def test_fallback_when_retrieval_fails(
#     mock_policy_store,
#     mock_retrieval_coordinator,
#     mock_fallback,
#     make_chunk
# ):
#     mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Retrieval failed")
#     fallback_chunk = make_chunk("fallback")
#     mock_fallback.generate_fallback.return_value = [fallback_chunk]

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=False,
#     )

#     result = router.route("Query")
#     assert result == [fallback_chunk]


# # 🔹 3. Test retry exhaustion logs and stops
# def test_retry_exhaustion_logs_and_stops(mock_policy_store, mock_planner, mock_feedback, make_chunk, caplog):
#     # Planner returns nothing → triggers feedback loop
#     mock_planner.plan_as_context.return_value = []

#     # Retrieval returns low-score chunk → triggers retry
#     mock_retrieval_coordinator = MagicMock()
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("low", 0.1)]

#     # Feedback rewrites query
#     mock_feedback.retry_or_replan.return_value = "retry query"

#     router = HybridRAGRouter(
#         planner=mock_planner,
#         coordinator=mock_retrieval_coordinator,
#         feedback=mock_feedback,
#         policy_store=mock_policy_store,
#         max_retry_depth=1,
#         use_redis=False,
#         debug_mode=True,
#     )

#     with caplog.at_level(logging.WARNING):
#         result, ctx = router.route("Query", retry_depth=1)

#     assert "Max retry depth (1) reached. Skipping feedback retry." in caplog.text
#     assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED
#     assert len(result) > 0


# # 🔹 4. Test feedback triggered on low score
# def test_feedback_triggered_on_low_score(mock_policy_store, mock_retrieval_coordinator, mock_feedback, make_chunk):
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("low-score", 0.1)]
#     mock_feedback.should_retry.return_value = True
#     mock_feedback.retry_or_replan.return_value = "Improved query"

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         feedback=mock_feedback,
#         policy_store=mock_policy_store,
#         max_retry_depth=2,
#         use_redis=False,
#         debug_mode=True,
#     )

#     with patch.object(router, 'route') as mock_route:
#         mock_route.side_effect = lambda *args, **kwargs: ([make_chunk("good-result", 0.8)], MagicMock())
#         result, ctx = router.route("Query")

#     mock_route.assert_called()
#     assert "good-result" in result[0].chunk.content


# # 🔹 5. Test cache hit and miss logging
# def test_cache_hit_miss_logging(mock_policy_store, caplog):
#     router = HybridRAGRouter(
#         policy_store=mock_policy_store,
#         use_redis=True,
#         enable_caching=True,
#         debug_mode=True
#     )

#     with caplog.at_level(logging.INFO):
#         result, ctx = router.route("Query", session_id="test")
#         assert "RAG route complete" in caplog.text
#         assert ctx.cache_hit is False

#         # Simulate cache hit
#         router.redis_client.get.return_value = b'[{"chunk": {"id": "cached", "content": "cached"}, "score": 0.9}]'
#         result, ctx = router.route("Query", session_id="test")
#         assert ctx.cache_hit is True


# # 🔹 6. Test in-memory cache used when Redis fails
# def test_in_memory_cache_used_when_redis_fails(monkeypatch, mock_policy_store):
#     # Force Redis connection to fail
#     monkeypatch.setattr("redis.from_url", MagicMock(side_effect=Exception("Redis failed")))

#     router = HybridRAGRouter(
#         policy_store=mock_policy_store,
#         use_redis=True,
#         enable_caching=True,
#         debug_mode=True,
#     )

#     with patch.object(router, "_set_in_cache") as spy_set:
#         result, ctx = router.route("Query", session_id="test")
#         spy_set.assert_called()


# # 🔹 7. Test fallback reasons are set correctly
# def test_fallback_reasons_are_set_correctly(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = []
#     mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("No results query", session_id="test")
#     assert ctx.fallback_reason == FallbackReason.GENERATED


# # 🔹 8. Test pubsub publishes event
# def test_pubsub_publishes_event(mock_policy_store):
#     router = HybridRAGRouter(
#         policy_store=mock_policy_store,
#         use_redis=True,
#         enable_pubsub=True,
#         debug_mode=False
#     )

#     router.publish_feedback_event("test_event", {"data": "test"})
#     router.redis_client.publish.assert_called_once()


# # 🔹 9. Test planner used when high score
# def test_planner_used_when_high_score(mock_policy_store, mock_planner, make_chunk):
#     # Temporarily lower threshold to let planner win
#     mock_policy_store.get_float.side_effect = lambda key, default=0.0: {
#         "retrieval.score_threshold": 0.5,
#         "max_retry_depth": 2,
#     }.get(key, default)

#     planner_chunk = make_chunk("planner", 0.9)
#     mock_planner.plan_as_context.return_value = [planner_chunk]
#     mock_planner.plan_as_context.return_value = [planner_chunk]

#     router = HybridRAGRouter(
#         planner=mock_planner,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=False,
#     )

#     result = router.route("Query")
#     assert result == [planner_chunk]


# # 🔹 10. Test retrieval used when planner disabled
# def test_retrieval_used_when_planner_disabled(mock_policy_store, mock_retrieval_coordinator, make_chunk):
#     # Disable planner
#     mock_policy_store.get_bool.side_effect = lambda key, default=False: {
#         "planner.first": False,
#         "disable_planner": True
#     }.get(key, default)

#     retrieval_chunk = make_chunk("retrieved", 0.8)
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=False,
#     )

#     result = router.route("Query")
#     assert result == [retrieval_chunk]


# # 🔹 11. Test debug mode returns context
# def test_debug_mode_returns_context(mock_policy_store, mock_fallback, make_chunk):
#     fallback_chunk = make_chunk("fallback")
#     mock_fallback.generate_fallback.return_value = [fallback_chunk]

#     router = HybridRAGRouter(
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         debug_mode=True,
#         enable_caching=False,
#     )

#     result, ctx = router.route("Query")
#     assert isinstance(result, list)
#     assert hasattr(ctx, 'query')
#     assert ctx.fallback_used is True


# # 🔹 12. Test cache hit sets ctx.cache_hit
# def test_cache_hit_sets_ctx_cache_hit(mock_policy_store):
#     router = HybridRAGRouter(
#         policy_store=mock_policy_store,
#         use_redis=True,
#         enable_caching=True,
#         debug_mode=True,
#     )

#     # Simulate cache hit
#     router.redis_client.get.return_value = b'[{"chunk": {"id": "cached", "content": "cached"}, "score": 0.9}]'

#     result, ctx = router.route("Query", session_id="test")
#     assert ctx.cache_hit is True
#     assert "cached" in result[0].chunk.content


# # 🔹 13. Test fallback_used is set to True
# def test_fallback_used_is_set_to_true(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = []
#     mock_fallback.generate_fallback.return_value = [make_chunk("fallback")]

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("No results query", session_id="test")
#     assert ctx.fallback_used is True
#     assert ctx.fallback_reason == FallbackReason.GENERATED


# # 🔹 14. Test retrieval_filtered_count is correct
# def test_retrieval_filtered_count_is_correct(mock_policy_store, mock_retrieval_coordinator, make_chunk):
#     retrieved = [make_chunk(f"retrieved-{i}", 0.6) for i in range(5)]
#     ranked = retrieved[:3]  # Simulate filtering

#     mock_retrieval_coordinator.hybrid_retrieve.return_value = retrieved

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         policy_store=mock_policy_store,
#         debug_mode=True,
#         enable_caching=False,
#     )

#     # Mock ranker to return only 3
#     with patch("app.core.hybrid_rag_router.BedrockRanker") as m:
#         m.return_value.rank.return_value = ranked
#         result, ctx = router.route("Query")

#     assert len(result) == 3
#     assert ctx.total_context_chunks == 5
#     assert ctx.retrieval_filtered_count == 2


# # 🔹 15. Test planner exception is handled
# def test_planner_exception_is_handled(mock_policy_store, mock_planner, mock_retrieval_coordinator, make_chunk):
#     mock_planner.plan_as_context.side_effect = Exception("Planner crashed")
#     retrieval_chunk = make_chunk("retrieved", 0.8)
#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [retrieval_chunk]

#     router = HybridRAGRouter(
#         planner=mock_planner,
#         coordinator=mock_retrieval_coordinator,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("Query")
#     assert "retrieved" in result[0].chunk.content
#     assert ctx.planner_used is False


# # 🔹 16. Test retrieval exception is handled
# def test_retrieval_exception_is_handled(mock_policy_store, mock_retrieval_coordinator, mock_fallback, make_chunk):
#     mock_retrieval_coordinator.hybrid_retrieve.side_effect = Exception("Retrieval failed")
#     mock_fallback.generate_fallback.return_value = []  # Ensure no fallback chunks

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         fallback=mock_fallback,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("Query")
#     assert result == []
#     assert ctx.fallback_reason == FallbackReason.RETRIEVAL_FAILED


# # 🔹 17. Test feedback loop respects max_retry_depth
# def test_feedback_respects_max_retry_depth(mock_policy_store, mock_feedback, make_chunk):
#     mock_feedback.should_retry.return_value = True
#     mock_feedback.retry_or_replan.return_value = "retry query"

#     router = HybridRAGRouter(
#         feedback=mock_feedback,
#         policy_store=mock_policy_store,
#         max_retry_depth=1,
#         use_redis=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("Query", retry_depth=1)
#     assert ctx.fallback_reason == FallbackReason.RETRY_EXHAUSTED


# # 🔹 18. Test fallback_reason is set to PLANNER_LOW_SCORE
# def test_planner_low_score_sets_fallback_reason(mock_policy_store, mock_planner, make_chunk):
#     mock_planner.plan_as_context.return_value = [make_chunk("planner", 0.1)]
#     mock_planner.plan_as_context.return_value = [make_chunk("planner", 0.1)]  # Below threshold

#     router = HybridRAGRouter(
#         planner=mock_planner,
#         policy_store=mock_policy_store,
#         enable_caching=False,
#         debug_mode=True,
#     )

#     result, ctx = router.route("Query")
#     assert ctx.fallback_reason == FallbackReason.PLANNER_LOW_SCORE


# # 🔹 19. Test retrieval_used is set when retrieval runs
# def test_retrieval_used_is_set(mock_policy_store, mock_retrieval_coordinator, make_chunk):
#     mock_policy_store.get_bool.side_effect = lambda key, default=False: {
#         "planner.first": False,
#         "disable_planner": True
#     }.get(key, default)

#     mock_retrieval_coordinator.hybrid_retrieve.return_value = [make_chunk("retrieved", 0.8)]

#     router = HybridRAGRouter(
#         coordinator=mock_retrieval_coordinator,
#         policy_store=mock_policy_store,
#         debug_mode=True,
#     )

#     result, ctx = router.route("Query")
#     assert ctx.retrieval_used is True
