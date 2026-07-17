"""Server-Sent Events (SSE) streaming support for Nvidia Model Bridge."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

import httpx
from sse_starlette.sse import EventSourceResponse


async def stream_chat_completion(
    base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: int = 90,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream chat completion responses from NVIDIA API."""
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(
            {k: v for k, v in extra_body.items()
             if k not in {"model", "messages", "stream"}}
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                try:
                    error_data = json.loads(error_body)
                except (json.JSONDecodeError, ValueError):
                    error_data = {"error": {"message": error_body.decode(errors="replace")}}
                yield {
                    "event": "error",
                    "data": json.dumps(error_data),
                }
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            yield {"event": "done", "data": "[DONE]"}
                            return
                        try:
                            data = json.loads(data_str)
                            yield {"event": "message", "data": data_str}
                        except json.JSONDecodeError:
                            continue


async def create_sse_response(
    base_url: str,
    api_key: str,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: int = 90,
) -> EventSourceResponse:
    """Create an SSE EventSourceResponse for streaming."""

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        async for event in stream_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
            timeout=timeout,
        ):
            yield {"event": event.get("event", "message"), "data": event["data"]}

    return EventSourceResponse(event_generator())
