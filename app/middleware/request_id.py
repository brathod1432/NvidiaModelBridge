"""Request ID middleware for tracing and correlation."""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


REQUEST_ID_HEADER = "X-Request-ID"

# Context variable for request ID propagation
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add a unique request ID to each request for tracing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Use existing request ID from header or generate a new one
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Store in context variable for downstream access
        token = request_id_ctx.set(request_id)

        # Store in request state for endpoint access
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
