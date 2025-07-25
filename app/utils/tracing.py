# app/utils/tracing.py

import os
import logging
import contextvars
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)
ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"

try:
    if ENABLE_TELEMETRY:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
        from opentelemetry.trace import SpanKind, get_current_span

        USE_XRAY = os.getenv("ENABLE_XRAY", "false").lower() == "true"
        if USE_XRAY:
            from opentelemetry.exporter.aws.xray import (
                AWSXRaySpanExporter,
                AWSXRayIdGenerator,
            )
            trace.set_tracer_provider(
                TracerProvider(
                    resource=Resource.create({SERVICE_NAME: "fin-copilot"}),
                    id_generator=AWSXRayIdGenerator(),
                )
            )
            span_processor = BatchSpanProcessor(AWSXRaySpanExporter())
        else:
            trace.set_tracer_provider(
                TracerProvider(resource=Resource.create({SERVICE_NAME: "fin-copilot"}))
            )
            span_processor = BatchSpanProcessor(ConsoleSpanExporter())

        trace.get_tracer_provider().add_span_processor(span_processor)
        tracer = trace.get_tracer(__name__)
        LoggingInstrumentor().instrument(set_logging_format=True)
        TELEMETRY_ENABLED = True
    else:
        raise ImportError("Telemetry disabled")

except ImportError:
    tracer = None
    TELEMETRY_ENABLED = False

# === Local Context ===
_request_id_ctx = contextvars.ContextVar("request_id", default=None)


def set_request_id(req_id: Optional[str]):
    if req_id:
        _request_id_ctx.set(req_id)


def get_request_id() -> Optional[str]:
    return _request_id_ctx.get()

def trace_function(_func: Optional[Callable] = None, *, name: Optional[str] = None):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            logger.info(f"Tracing: {trace_name} called with args={args}, kwargs={kwargs}")
            return func(*args, **kwargs)
        return wrapper

    if _func is None:
        return decorator  # when used with parameters
    else:
        return decorator(_func)  # when used directly

def instrument_fastapi(app):
    if TELEMETRY_ENABLED:
        FastAPIInstrumentor().instrument_app(app)


def instrument_requests():
    if TELEMETRY_ENABLED:
        RequestsInstrumentor().instrument()


def add_span_tag(key: str, value: Any):
    if TELEMETRY_ENABLED:
        span = get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, value)


def log_trace_context():
    if TELEMETRY_ENABLED:
        span = get_current_span()
        if span and span.is_recording():
            trace_id = span.get_span_context().trace_id
            logging.info(f"[Trace] Active Trace ID: {trace_id:032x}")
