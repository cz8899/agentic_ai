# app/utils/llm.py
"""
Unified LLM interface for Bedrock model invocation.
Handles Claude models with retry, fallback, and observability.
"""

import logging
import json
import time
import os
from typing import Any, Dict, Optional, List
from dataclasses import asdict
from app.core.policy_store import PolicyStore
from app.utils.tracing import trace_function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import AWS SDK
try:
    import boto3
    from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
    HAS_BEDROCK = True
except ImportError:
    boto3 = None
    ClientError = Exception
    HAS_BEDROCK = False


def _default(obj):
    """JSON serializer for objects not serializable by default."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        return str(obj)


@trace_function
def invoke_claude_model(
    model: str,
    prompt: str,
    temperature: float = 0.5,
    max_tokens: int = 200,
    stop_sequences: Optional[List[str]] = None,
    top_p: float = 0.9,
    policy_store: Optional[PolicyStore] = None
) -> str:
    """
    Invoke a Claude model on Amazon Bedrock.

    Args:
        model: Model ID (e.g., anthropic.claude-3-sonnet-20240229-v1:0)
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Max output tokens
        stop_sequences: List of stop sequences
        top_p: Nucleus sampling threshold
        policy_store: Optional policy store for runtime config

    Returns:
        Model response as string, or empty string on failure.
    """
    if not HAS_BEDROCK:
        logger.error("boto3 not installed. Cannot invoke Bedrock model.")
        return ""

    if not prompt or not isinstance(prompt, str):
        logger.warning("Invalid prompt: %r", prompt)
        return ""

    # Load policies
    try:
        policy_store = policy_store or PolicyStore()
        model = policy_store.get_str("llm.model", model)
        temperature = policy_store.get_float("llm.temperature", temperature)
        max_tokens = policy_store.get_int("llm.max_tokens", max_tokens)
        top_p = policy_store.get_float("llm.top_p", top_p)
    except Exception as e:
        logger.warning(f"Failed to load LLM policies: {e}. Using provided values.")

    # Build request body
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    if stop_sequences:
        body["stop_sequences"] = stop_sequences

    # Serialize for logging
    try:
        body_str = json.dumps(body, default=_default, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize request body: {e}")
        body_str = str(body)

    logger.info("Calling Claude model: %s | prompt_len=%d", model, len(prompt))

    # Retry config
    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
            response = client.invoke_model(
                modelId=model,
                body=body_str,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response["body"].read().decode())
            content = response_body.get("content", [])
            text = "".join([c.get("text", "") for c in content if c.get("type") == "text"])
            logger.info("LLM call succeeded | model=%s | response_len=%d | attempt=%d", model, len(text), attempt + 1)
            return text.strip()

        except (ConnectTimeoutError, ReadTimeoutError) as e:
            logger.warning("LLM request timeout (attempt %d/%d): %s", attempt + 1, max_retries, str(e))
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ["ThrottlingException", "ServiceQuotaExceededException"]:
                logger.warning("LLM throttled (attempt %d/%d): %s", attempt + 1, max_retries, error_code)
            elif error_code == "ValidationException":
                logger.error("LLM validation failed: %s", str(e))
                return ""
            else:
                logger.error("LLM client error: %s", str(e), exc_info=True)
                return ""
        except Exception as e:
            logger.error("Unexpected LLM error (attempt %d/%d): %s", attempt + 1, max_retries, str(e), exc_info=True)

        # Exponential backoff
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info("Retrying in %.1f seconds...", delay)
            time.sleep(delay)

    logger.error("LLM call failed after %d attempts: %s", max_retries, model)
    return ""
