"""Monitoring and observability utilities (metrics, tracing)."""

import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional

from fastapi import Request
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from api.logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

REQUEST_IN_PROGRESS = Gauge(
    "api_requests_in_progress",
    "Number of requests currently being processed",
    ["method"],
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions made",
    ["model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of predicted values",
    ["model_version"],
    buckets=[50000, 100000, 250000, 500000, 750000, 1000000, 1500000, 2000000, 5000000],
)

# Model metrics
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is loaded (1) or not (0)",
    ["model_type"],
)

MODEL_LAST_TRAINING = Gauge(
    "model_last_training_timestamp",
    "Unix timestamp of last model training",
)

# Cache metrics
CACHE_HITS = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

# Error metrics
ERROR_COUNT = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["error_type", "endpoint"],
)


# =============================================================================
# METRICS MIDDLEWARE
# =============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request metrics."""
        method = request.method
        REQUEST_IN_PROGRESS.labels(method=method).inc()

        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code

            # Track request count and duration
            endpoint = request.url.path
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            return response

        except Exception as e:
            ERROR_COUNT.labels(
                error_type=type(e).__name__,
                endpoint=request.url.path,
            ).inc()
            raise

        finally:
            REQUEST_IN_PROGRESS.labels(method=method).dec()


class MetricsEndpoint:
    """Prometheus metrics endpoint."""

    @staticmethod
    async def metrics() -> Response:
        """Return Prometheus metrics."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )


# =============================================================================
# TRACING
# =============================================================================

def setup_tracing(service_name: str = "real-estate-api", service_version: str = "1.0.0") -> None:
    """Setup OpenTelemetry tracing.

    Args:
        service_name: Name of the service.
        service_version: Version of the service.
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    resource = Resource.create(
        attributes={
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
        }
    )

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add span processor with OTLP exporter
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)
    except Exception as e:
        logger.warning("failed_to_setup_otlp_exporter", error=str(e))

    # Set the tracer provider
    trace.set_tracer_provider(provider)

    logger.info("tracing_setup", service_name=service_name, endpoint=otlp_endpoint)


def get_tracer(name: str = "real-estate-api") -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)


@contextmanager
def span(name: str, attributes: Optional[dict[str, Any]] = None) -> Iterator[trace.Span]:
    """Context manager for creating spans.

    Usage:
        with span("prediction", {"model_version": "1.0"}) as s:
            result = model.predict(data)
            s.set_attribute("prediction", result)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as current_span:
        if attributes:
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
        yield current_span


def traced(name: str | None = None) -> Callable:
    """Decorator to trace function calls.

    Usage:
        @traced("make_prediction")
        def predict(data):
            return model.predict(data)
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.args_count", len(args))
                span.set_attribute("function.kwargs_count", len(kwargs))

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    span.set_attribute("error_type", type(e).__name__)
                    raise
                finally:
                    duration = time.time() - start_time
                    span.set_attribute("duration_seconds", duration)

        return wrapper
    return decorator


def instrument_fastapi(app: Any) -> None:
    """Instrument FastAPI app with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(app)


# =============================================================================
# PREDICTION TRACKING
# =============================================================================

def track_prediction(
    model_version: str,
    prediction_value: float,
    latency: float,
    cached: bool = False,
    error: Optional[str] = None,
) -> None:
    """Track a prediction in metrics.

    Args:
        model_version: Version of the model used.
        prediction_value: The predicted value.
        latency: Time taken for prediction.
        cached: Whether prediction was from cache.
        error: Error message if prediction failed.
    """
    status = "cached" if cached else ("error" if error else "success")
    PREDICTION_COUNT.labels(model_version=model_version, status=status).inc()
    PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)

    if not error:
        PREDICTION_VALUE.labels(model_version=model_version).observe(prediction_value)

    if cached:
        CACHE_HITS.labels(cache_type="prediction").inc()
    else:
        CACHE_MISSES.labels(cache_type="prediction").inc()


def update_model_metrics(model_type: str, loaded: bool = True) -> None:
    """Update model metrics.

    Args:
        model_type: Type of model.
        loaded: Whether model is loaded.
    """
    MODEL_LOADED.labels(model_type=model_type).set(1 if loaded else 0)


def set_last_training_timestamp() -> None:
    """Set the last training timestamp."""
    MODEL_LAST_TRAINING.set(time.time())


# =============================================================================
# HEALTH CHECKS
# =============================================================================

class HealthChecker:
    """Health check registry."""

    def __init__(self) -> None:
        """Initialize health checker."""
        self.checks: dict[str, Callable[[], tuple[bool, str]]] = {}

    def register(
        self,
        name: str,
        check_func: Callable[[], tuple[bool, str]],
    ) -> None:
        """Register a health check.

        Args:
            name: Name of the check.
            check_func: Function that returns (is_healthy, message).
        """
        self.checks[name] = check_func
        logger.info("health_check_registered", check_name=name)

    def run_checks(self) -> dict[str, Any]:
        """Run all health checks.

        Returns:
            Dictionary with health status of all checks.
        """
        results = {}
        healthy_count = 0

        for name, check_func in self.checks.items():
            try:
                is_healthy, message = check_func()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": message,
                }
                if is_healthy:
                    healthy_count += 1
            except Exception as e:
                logger.error("health_check_failed", check_name=name, error=str(e))
                results[name] = {
                    "status": "error",
                    "message": str(e),
                }

        return {
            "overall_status": "healthy" if healthy_count == len(self.checks) else "degraded",
            "checks": results,
            "healthy_count": healthy_count,
            "total_checks": len(self.checks),
        }


# Global health checker
health_checker = HealthChecker()


def register_health_check(name: str) -> Callable:
    """Decorator to register a health check.

    Usage:
        @register_health_check("database")
        def check_database():
            return True, "Database connection OK"
    """
    def decorator(func: Callable[[], tuple[bool, str]]) -> Callable:
        health_checker.register(name, func)
        return func
    return decorator
