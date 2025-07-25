# app/core/policy_store.py
import logging
import json
import os
from typing import Any, Optional, List, Dict, TypeVar, Callable
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
    PartialCredentialsError = Exception
    HAS_BOTO3 = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar('T')

# Default fallback values
DEFAULT_POLICIES = {
    "planner.first": "true",
    "disable_planner": "false",
    "enable_feedback": "true",
    "enable_rerank": "true",
    "skip_cache": "false",
    "retrieval.score_threshold": 0.0,
    "max_retry_depth": "2",
    "enable_fallback": "true",
    "retrieval.top_k": "5",
    "enable_bedrock_kb": "true",
    "disable_opensearch": "false",
    "enable_chroma": "false",
    "planner.min_confidence": "0.7",
    "feedback.critique_score_threshold": "0.25",
    "ranker.model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude_direct_model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "opensearch.host": "localhost",
    "opensearch.port": "443",
    "opensearch.index": "chat-history",
    "opensearch.username": "admin",
    "opensearch.password": "admin-pass",
    "intent_keywords.retrieve": "show, find, lookup, details",
    "intent_keywords.summarize": "summary, summarize, tl;dr",
    "intent_keywords.analyze": "trend, analyze, insight",
    "intent_keywords.fallback": "unknown, error",
    "intent_keywords.tool_call": "calculate, convert",
    "redis.url": "redis://localhost:6379/0",
    "cache.ttl": "300",
    "pubsub.enable": "true",
}


@dataclass
class PolicyConfig:
    table_name: str = ""
    region: str = "us-west-2"
    local_mode: bool = False


class PolicyStore:
    """
    Central policy store for runtime configuration.
    Loads from DynamoDB, environment variables, or defaults.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        if not HAS_BOTO3:
            logger.warning("boto3 not installed. Running in local mode only.")
            self.config = config or PolicyConfig()
            self.config.local_mode = True
            self.table = None
            self._local_cache: Dict[str, str] = {}
            return

        self.config = config or PolicyConfig()
        self.table = None
        self._local_cache: Dict[str, str] = {}

        # Only try to connect if not in local mode
        if not self.config.local_mode:
            self._load_source()
        else:
            logger.info("PolicyStore running in local mode. Using env vars and defaults.")

    def _load_source(self):
        """Initialize DynamoDB client or fall back to local mode."""
        # If table name is empty, force local mode
        if not self.config.table_name:
            logger.warning("DynamoDB table name is empty. Falling back to local mode.")
            self.config.local_mode = True
            return

        try:
            dynamodb = boto3.resource(
                'dynamodb',
                region_name=self.config.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                endpoint_url=os.getenv("AWS_ENDPOINT_URL")  # For local testing
            )
            self.table = dynamodb.Table(self.config.table_name)
            logger.info(f"✅ PolicyStore connected to DynamoDB table: {self.config.table_name}")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.warning(f"Missing AWS credentials: {e}. Falling back to local mode.")
            self.config.local_mode = True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ResourceNotFoundException":
                logger.error(f"DynamoDB table not found: {self.config.table_name}")
            else:
                logger.warning(f"DynamoDB connection failed: {e}")
            self.config.local_mode = True
        except Exception as e:
            logger.warning(f"Unexpected error connecting to DynamoDB: {e}")
            self.config.local_mode = True

    def _get_from_dynamodb(self, key: str) -> Optional[str]:
        """Fetch policy from DynamoDB."""
        if self.config.local_mode or not self.table:
            return None

        try:
            response = self.table.get_item(Key={'policy_key': key})
            item = response.get('Item')
            value = item['value'] if item else None
            if value is not None:
                self._local_cache[key] = value
            return value
        except ClientError as e:
            logger.warning(f"DynamoDB get failed for {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading policy {key}: {e}", exc_info=True)
            return None

    def _get_from_env(self, key: str) -> Optional[str]:
        """Fetch policy from environment variable."""
        env_key = key.upper().replace('.', '_')
        return os.getenv(env_key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get policy value as string, with fallback chain:
        1. Local cache
        2. DynamoDB
        3. Environment variable
        4. Default policies
        """
        if not key:
            return default

        # 1. Cache
        if key in self._local_cache:
            return self._local_cache[key]

        # 2. DynamoDB
        value = self._get_from_dynamodb(key)

        # 3. Env var
        if value is None:
            value = self._get_from_env(key)

        # 4. Default
        if value is None:
            value = DEFAULT_POLICIES.get(key, default)

        if value is not None:
            self._local_cache[key] = value

        return value

    def get_str(self, key: str, default: str = "") -> str:
        return str(self.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key, str(default))
        return str(value).strip().lower() in ["true", "1", "yes", "on"]

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Invalid int for policy {key}: {self.get(key)}")
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Invalid float for policy {key}: {self.get(key)}")
            return default

    def get_list(self, key: str, default: Optional[List[str]] = None) -> List[str]:
        value = self.get(key, "")
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
        except Exception as e:
            logger.warning(f"Failed to parse list for policy {key}: {value} | Error: {e}")
            return default or []

    def get_dict(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        value = self.get(key, "")
        if not value:
            return default or {}
        try:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                return json.loads(value)
            return default or {}
        except Exception as e:
            logger.warning(f"Failed to parse dict for policy {key}: {value} | Error: {e}")
            return default or {}

    def put(self, key: str, value: Any):
        """Write policy back to DynamoDB (optional)."""
        if self.config.local_mode or not self.table:
            logger.warning("Cannot put policy in local mode.")
            return

        try:
            self.table.put_item(Item={'policy_key': key, 'value': str(value)})
            self._local_cache[key] = str(value)
            logger.info(f"Policy updated: {key} = {value}")
        except Exception as e:
            logger.error(f"Failed to update policy {key}: {e}")
