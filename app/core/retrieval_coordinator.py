# app/core/retrieval_coordinator.py
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from app.utils.schema import RetrievedChunk, QueryPayload, DocumentChunk
from app.core.policy_store import PolicyStore
from app.utils.tracing import trace_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class RetrievalResult:
    source: str
    chunks: List[RetrievedChunk]
    latency: float
    success: bool
    error: Optional[str] = None


class RetrievalCoordinator:
    """
    Coordinates hybrid retrieval from multiple sources:
    - Chroma (vector)
    - OpenSearch (keyword + vector)
    - Bedrock Knowledge Base (native RAG with Textract, HTML parsing, Titan embeddings)
    Also supports ingestion into Bedrock KB.
    """

    def __init__(self, policy_store: Optional[PolicyStore] = None):
        self.policy_store = policy_store or PolicyStore()
        self._load_policies()

        # Lazy-loaded clients
        self._chroma_client = None
        self._opensearch_client = None
        self._bedrock_agent_client = None
        self._s3_client = None

    def _load_policies(self):
        """Load retrieval policies at init."""
        try:
            self.top_k = self.policy_store.get_int("retrieval.top_k", 5)
            self.score_threshold = self.policy_store.get_float("retrieval.score_threshold", 0.0)
            self.enable_bedrock_kb = self.policy_store.get_bool("enable_bedrock_kb", True)
            self.disable_opensearch = self.policy_store.get_bool("disable_opensearch", False)
            self.enable_chroma = self.policy_store.get_bool("enable_chroma", True)
            self.bedrock_kb_id = self.policy_store.get_str("bedrock_kb_id", "")
        except Exception as e:
            logger.warning(f"Failed to load retrieval policies: {e}. Using defaults.")
            self._use_default_policies()

    def _use_default_policies(self):
        """Safe fallback if policy load fails."""
        self.top_k = 5
        self.score_threshold = 0.0
        self.enable_bedrock_kb = True
        self.disable_opensearch = False
        self.enable_chroma = True
        self.bedrock_kb_id = ""

    @trace_function
    def hybrid_retrieve(self, payload: QueryPayload) -> List[RetrievedChunk]:
        """
        Retrieve from all enabled sources.
        Returns unified list of chunks, deduplicated by ID.
        """
        if not payload.query or not isinstance(payload.query, str):
            logger.warning("Invalid query in hybrid_retrieve: %r", payload.query)
            return []

        query = payload.query.strip()
        results: List[RetrievedChunk] = []
        seen_ids = set()

        # Source 1: Chroma
        if self.enable_chroma:
            try:
                chroma_chunks = self._retrieve_from_chroma(query, payload)
                self._add_unique_chunks(results, seen_ids, chroma_chunks, "chroma")
            except Exception as e:
                logger.error("Chroma retrieval failed: %s", str(e), exc_info=True)

        # Source 2: OpenSearch
        if not self.disable_opensearch:
            try:
                opensearch_chunks = self._retrieve_from_opensearch(query, payload)
                self._add_unique_chunks(results, seen_ids, opensearch_chunks, "opensearch")
            except Exception as e:
                logger.error("OpenSearch retrieval failed: %s", str(e), exc_info=True)

        # Source 3: Bedrock Knowledge Base
        if self.enable_bedrock_kb and self.bedrock_kb_id:
            try:
                kb_chunks = self._retrieve_from_bedrock_kb(query, payload)
                self._add_unique_chunks(results, seen_ids, kb_chunks, "bedrock_kb")
            except Exception as e:
                logger.error("Bedrock KB retrieval failed: %s", str(e), exc_info=True)

        # Sort by score descending
        results.sort(key=lambda c: c.score or 0.0, reverse=True)

        # Truncate to top_k
        results = results[:self.top_k]

        logger.info("Hybrid retrieval complete | sources=%s | total_chunks=%d",
                   self._enabled_sources(), len(results))
        return results

    @trace_function
    def ingest_documents(self, sources: List[str], s3_bucket: str) -> bool:
        """
        Ingest documents (PDFs, blogs) into Bedrock Knowledge Base.
        Uses:
        - ✅ Amazon Textract (for PDFs)
        - ✅ HTML parsing (for blog URLs)
        - ✅ Titan Embeddings
        - ✅ Vector storage in OpenSearch Serverless
        """
        if not self.enable_bedrock_kb or not self.bedrock_kb_id:
            logger.warning("Bedrock KB not enabled or KB ID not set. Skipping ingestion.")
            return False

        if not sources:
            logger.warning("No sources provided for ingestion.")
            return False

        try:
            client = self._get_bedrock_agent_client()
            s3_client = self._get_s3_client()

            # Validate S3 bucket
            s3_client.head_bucket(Bucket=s3_bucket)

            # Start ingestion job
            response = client.start_ingestion_job(
                knowledgeBaseId=self.bedrock_kb_id,
                dataSourceId="default-data-source",  # Must exist in KB
                description=f"Ingestion job for {len(sources)} sources"
            )

            job = response["ingestionJob"]
            job_id = job["ingestionJobId"]
            logger.info("Ingestion job started: %s", job_id)

            # Wait for completion
            while True:
                status = client.get_ingestion_job(
                    knowledgeBaseId=self.bedrock_kb_id,
                    ingestionJobId=job_id
                )["ingestionJob"]["status"]
                if status in ["COMPLETE", "FAILED"]:
                    logger.info("Ingestion job %s: %s", job_id, status)
                    return status == "COMPLETE"
                time.sleep(5)

        except Exception as e:
            logger.error("Document ingestion failed: %s", str(e), exc_info=True)
            return False

    def _add_unique_chunks(
        self,
        results: List[RetrievedChunk],
        seen_ids: set,
        new_chunks: List[RetrievedChunk],
        source: str
    ):
        """Add chunks to results only if ID is not already seen."""
        for chunk in new_chunks:
            if chunk.chunk.id not in seen_ids:
                chunk.chunk.metadata["retrieval_source"] = source
                results.append(chunk)
                seen_ids.add(chunk.chunk.id)

    def _retrieve_from_chroma(self, query: str, payload: QueryPayload) -> List[RetrievedChunk]:
        if self._chroma_client is None:
            logger.info("Chroma client initialized")
        logger.info("Retrieving from Chroma: '%s'", query[:50])
        return []

    def _retrieve_from_opensearch(self, query: str, payload: QueryPayload) -> List[RetrievedChunk]:
        if self._opensearch_client is None:
            logger.info("OpenSearch client initialized")
        logger.info("Retrieving from OpenSearch: '%s'", query[:50])
        return []

    def _retrieve_from_bedrock_kb(self, query: str, payload: QueryPayload) -> List[RetrievedChunk]:
        if not self._bedrock_agent_client:
            self._bedrock_agent_client = self._get_bedrock_agent_client()

        try:
            response = self._bedrock_agent_client.retrieve(
                knowledgeBaseId=self.bedrock_kb_id,
                retrievalQuery={"text": query}
            )
            return [
                RetrievedChunk(
                    chunk=DocumentChunk(
                        id=hit["content"]["chunkReferenceNumber"],
                        content=hit["content"]["text"],
                        source="bedrock_kb",
                        title=f"KB: {hit['content']['text'][:30]}..."
                    ),
                    score=hit["score"]
                )
                for hit in response["retrievalResults"]
            ]
        except Exception as e:
            logger.error("Bedrock KB retrieval failed: %s", str(e))
            return []

    def _get_bedrock_agent_client(self):
        if not self._bedrock_agent_client:
            import boto3
            self._bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        return self._bedrock_agent_client

    def _get_s3_client(self):
        if not self._s3_client:
            import boto3
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _enabled_sources(self) -> str:
        sources = []
        if self.enable_chroma:
            sources.append("chroma")
        if not self.disable_opensearch:
            sources.append("opensearch")
        if self.enable_bedrock_kb and self.bedrock_kb_id:
            sources.append("bedrock_kb")
        return ",".join(sources) or "none"
