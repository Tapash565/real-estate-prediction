"""FastAPI middleware for logging, monitoring, and request tracking."""

import time
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import structlog

logger = structlog.get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all incoming requests and responses."""

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[list[str]] = None,
    ) -> None:
        """Initialize middleware with excluded paths."""
        super().__init__(app)
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        start_time = time.time()

        # Log request
        await self._log_request(request, request_id)

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            await self._log_response(request, response, process_time, request_id)

            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as exc:
            process_time = time.time() - start_time
            await self._log_error(request, exc, process_time, request_id)
            raise

    async def _log_request(self, request: Request, request_id: str) -> None:
        """Log incoming request details."""
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            client_ip=client_ip,
            user_agent=user_agent,
        )

    async def _log_response(
        self,
        request: Request,
        response: Response,
        process_time: float,
        request_id: str,
    ) -> None:
        """Log response details."""
        logger.info(
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_ms=round(process_time * 1000, 2),
            response_size=len(response.body) if hasattr(response, "body") else 0,
        )

    async def _log_error(
        self,
        request: Request,
        exc: Exception,
        process_time: float,
        request_id: str,
    ) -> None:
        """Log error details."""
        logger.error(
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
            process_time_ms=round(process_time * 1000, 2),
        )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    # Paths that need a relaxed CSP for Swagger/ReDoc to work
    DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}

    # CSP that allows Swagger UI and ReDoc CDN assets
    DOCS_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://unpkg.com https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://unpkg.com https://fonts.googleapis.com https://cdn.jsdelivr.net; "
        "font-src 'self' https://fonts.gstatic.com https://unpkg.com; "
        "img-src 'self' data: https://fastapi.tiangolo.com; "
        "worker-src blob:; "
        "connect-src 'self';"
    )

    # Strict CSP for all other routes
    API_CSP = "default-src 'self'"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Choose CSP based on path
        path = request.url.path
        csp = self.DOCS_CSP if path in self.DOCS_PATHS or path.startswith("/docs") else self.API_CSP

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"  # DENY breaks Swagger iframe
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = csp
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


class CORSMiddlewareConfig:
    """Configuration for CORS middleware."""

    ALLOW_ORIGINS = ["*"]  # Configure for production
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    ALLOW_HEADERS = [
        "*",
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-API-Key",
    ]


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeouts."""

    def __init__(self, app: ASGIApp, timeout_seconds: float = 60.0) -> None:
        """Initialize with timeout."""
        super().__init__(app)
        self.timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce timeout on request processing."""
        import asyncio

        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(
                "request_timeout",
                path=request.url.path,
                method=request.method,
                timeout=self.timeout,
            )
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=504,
                content={"detail": "Request timeout"},
            )


class RateLimitByIPMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting by IP address (use Redis for production)."""

    def __init__(self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 60) -> None:
        """Initialize rate limiter."""
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time
                for req_time in self.requests[client_ip]
                if current_time - req_time < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []

        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            from fastapi.responses import JSONResponse

            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": self.window_seconds,
                },
            )

        # Add request
        self.requests[client_ip].append(current_time)

        return await call_next(request)
