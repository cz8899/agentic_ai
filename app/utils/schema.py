# app/utils/schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from enum import Enum

class FallbackReason(Enum):
    """Reasons why a fallback response was triggered."""
    LOW_SCORE = "low_score"
    RETRIEVAL_FAILED = "retrieval_failed"
    PLANNER_FAILED = "planner_failed"
    CACHE_MISS = "cache_miss"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    explanation: Optional[str] = None


@dataclass
class ChatTurn:
    role: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChatRequest:
    user_id: str
    session_id: str
    message: str
    history: List[ChatTurn]
    use_memory: bool = True
    use_fallback: bool = True
    metadata: Optional[Dict[str, str]] = None


@dataclass
class ChatResponse:
    message: str
    source_chunks: List[RetrievedChunk] = field(default_factory=list)
    planner_actions: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    fallback_triggered: Optional[bool] = None


@dataclass
class MemoryRecord:
    user_id: str
    session_id: str
    turns: List[ChatTurn]
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SessionKey:
    user_id: str
    session_id: str


@dataclass
class FeedbackRecord:
    user_id: str
    session_id: str
    turn_id: str
    rating: str
    comment: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IngestionStatus:
    source: str
    total_docs: int
    total_chunks: int
    success_count: int
    failure_count: int
    last_updated: datetime


@dataclass
class RankingCandidate:
    chunk: DocumentChunk
    raw_score: float
    rerank_score: Optional[float] = None
    explanation: Optional[str] = None


@dataclass
class QueryPayload:
    query: str
    conversation_history: Optional[List[RetrievedChunk]] = None
    top_k: int = 5
    use_bedrock_kb: bool = True
    use_opensearch: bool = True
    use_chroma: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None
    session_id: str = "default"
    user_id: str = "default"


def generate_chunk_id(source: str) -> str:
    return f"{source}-{uuid.uuid4().hex[:8]}"
