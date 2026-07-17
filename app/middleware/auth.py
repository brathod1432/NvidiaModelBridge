"""API key authentication middleware for Nvidia Model Bridge."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Any

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader


BRIDGE_API_KEY_HEADER = "X-Bridge-API-Key"
BRIDGE_API_KEY_ENV = "NVIDIA_BRIDGE_API_KEY"

api_key_header = APIKeyHeader(name=BRIDGE_API_KEY_HEADER, auto_error=False)


def get_configured_api_keys() -> list[str]:
    """Get the list of valid API keys from environment."""
    raw = os.getenv(BRIDGE_API_KEY_ENV, "")
    if not raw.strip():
        return []
    return [key.strip() for key in raw.split(",") if key.strip()]


def generate_api_key(prefix: str = "nvbridge") -> str:
    """Generate a new secure API key."""
    token = secrets.token_urlsafe(32)
    return f"{prefix}-{token}"


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Verify the API key from the request header.
    
    Returns the API key if valid, raises HTTPException if invalid.
    If no keys are configured (NVIDIA_BRIDGE_API_KEY not set), 
    authentication is disabled and all requests are allowed.
    """
    configured_keys = get_configured_api_keys()
    
    # If no keys configured, auth is disabled (open access)
    if not configured_keys:
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": f"Missing API key. Provide it in the '{BRIDGE_API_KEY_HEADER}' header.",
                    "type": "authentication_error",
                }
            },
        )

    for valid_key in configured_keys:
        if _constant_time_compare(api_key, valid_key):
            return api_key

    raise HTTPException(
        status_code=403,
        detail={
            "error": {
                "message": "Invalid API key.",
                "type": "authentication_error",
            }
        },
    )


def is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return bool(get_configured_api_keys())
