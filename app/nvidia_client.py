"""NVIDIA OpenAI-compatible API client helpers."""

from __future__ import annotations

import time
import re
from typing import Any

import httpx
from openai import OpenAI

from app.config import NvidiaSettings


TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _preview(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    text = _redact_sensitive(str(value)).replace("\r", " ").strip()
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _redact_sensitive(text: str) -> str:
    text = re.sub(r"account '([^']+)'", "account '[REDACTED]'", text)
    text = re.sub(r'account "([^"]+)"', 'account "[REDACTED]"', text)
    text = re.sub(
        r"Function '[0-9a-fA-F-]{32,}'",
        "Function '[REDACTED]'",
        text,
    )
    return text


def _classify_error(status_code: int | None, error_type: str, message: str) -> str:
    lower_message = message.lower()
    lower_type = error_type.lower()
    if status_code in {401, 403} or "unauthorized" in lower_message:
        return "unauthorized"
    if status_code == 429 or "rate limit" in lower_message:
        return "rate_limit"
    if "timeout" in lower_type or "timeout" in lower_message:
        return "timeout"
    if status_code == 404 or "not found" in lower_message:
        return "model_unavailable"
    if status_code in {400, 422} or "payload" in lower_message or "invalid" in lower_message:
        return "payload_rejected"
    if (
        "parse" in lower_type
        or "response" in lower_type
        or lower_type in {"keyerror", "indexerror", "attributeerror", "typeerror"}
    ):
        return "response_parse_error"
    if "empty" in lower_type or "empty" in lower_message:
        return "empty_response"
    if "network" in lower_type or "connect" in lower_type:
        return "network_error"
    return "other"


class NvidiaClient:
    """Small wrapper around the NVIDIA OpenAI-compatible API."""

    def __init__(self, settings: NvidiaSettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        self.api_key = settings.api_key or "missing"
        self.timeout = settings.timeout_seconds
        self._sdk_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )

    def list_models(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        result: dict[str, Any] = {
            "success": False,
            "latency_seconds": None,
            "status_code": None,
            "model_ids": [],
            "number_of_models_discovered": 0,
            "response_preview": "",
            "raw_response_shape": "",
            "error_type": "",
            "error_category": "",
            "error_message": "",
        }
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.timeout,
            )
            result["latency_seconds"] = round(time.perf_counter() - started_at, 4)
            result["status_code"] = response.status_code
            result["response_preview"] = _preview(response.text)
            data = self._safe_json(response)
            if response.status_code >= 400:
                message = self._extract_error_message(data, response.text)
                if response.status_code in {401, 403}:
                    message = "NVIDIA_API_KEY was found but rejected by NVIDIA."
                result.update(
                    {
                        "error_type": "HTTPStatusError",
                        "error_category": _classify_error(
                            response.status_code, "HTTPStatusError", message
                        ),
                        "error_message": _preview(message),
                    }
                )
                return result

            model_ids = self._extract_model_ids(data)
            result.update(
                {
                    "success": True,
                    "model_ids": model_ids,
                    "number_of_models_discovered": len(model_ids),
                    "raw_response_shape": self._summarize_model_list_shape(data),
                }
            )
            return result
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            result["latency_seconds"] = round(time.perf_counter() - started_at, 4)
            result["error_type"] = type(exc).__name__
            result["error_message"] = _preview(exc)
            result["error_category"] = _classify_error(None, type(exc).__name__, str(exc))
            return result
        except Exception as exc:  # noqa: BLE001 - audit tool should not crash.
            result["latency_seconds"] = round(time.perf_counter() - started_at, 4)
            result["error_type"] = type(exc).__name__
            result["error_message"] = _preview(exc)
            result["error_category"] = _classify_error(None, type(exc).__name__, str(exc))
            return result

    def chat_completion_sdk(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        try:
            completion = self._sdk_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
                stream=stream,
            )

            if stream:
                content, reasoning = self._extract_sdk_stream_content(completion)
                shape_summary = "streaming chunks"
            else:
                content, reasoning = self._extract_sdk_content(completion)
                shape_summary = self._summarize_sdk_shape(completion)

            latency = round(time.perf_counter() - started_at, 4)
            success = bool((content or "").strip() or (reasoning or "").strip())
            return {
                "success": success,
                "model_id": model_id,
                "method": "sdk",
                "latency_seconds": latency,
                "status_code": None,
                "error_type": "" if success else "EmptyResponse",
                "error_category": "" if success else "empty_response",
                "error_message": "" if success else "No content or reasoning returned.",
                "response_text": content or "",
                "response_preview": _preview(content),
                "reasoning": reasoning or "",
                "reasoning_preview": _preview(reasoning),
                "raw_response_shape": shape_summary,
            }
        except Exception as exc:  # noqa: BLE001 - failures are audit data.
            message = _preview(exc)
            return {
                "success": False,
                "model_id": model_id,
                "method": "sdk",
                "latency_seconds": round(time.perf_counter() - started_at, 4),
                "status_code": self._extract_status_code(exc),
                "error_type": type(exc).__name__,
                "error_category": _classify_error(
                    self._extract_status_code(exc), type(exc).__name__, str(exc)
                ),
                "error_message": message,
                "response_text": "",
                "response_preview": "",
                "reasoning": "",
                "reasoning_preview": "",
                "raw_response_shape": "",
            }

    def chat_completion_http(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra_body:
            payload.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key
                    not in {"model", "messages", "temperature", "top_p", "max_tokens", "stream"}
                }
            )

        last_error: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                latency = round(time.perf_counter() - started_at, 4)
                data = self._safe_json(response)
                if response.status_code >= 400:
                    message = self._extract_error_message(data, response.text)
                    error_result = {
                        "success": False,
                        "model_id": model_id,
                        "method": "http",
                        "latency_seconds": latency,
                        "status_code": response.status_code,
                        "error_type": "HTTPStatusError",
                        "error_category": _classify_error(
                            response.status_code, "HTTPStatusError", message
                        ),
                        "error_message": _preview(message),
                        "response_text": "",
                        "response_preview": _preview(response.text),
                        "reasoning": "",
                        "reasoning_preview": "",
                        "raw_response_shape": self._summarize_http_shape(data),
                    }
                    if (
                        attempt == 0
                        and response.status_code in TRANSIENT_HTTP_STATUS_CODES
                    ):
                        last_error = error_result
                        continue
                    return error_result

                content, reasoning = self._extract_http_content(data)
                success = bool((content or "").strip() or (reasoning or "").strip())
                return {
                    "success": success,
                    "model_id": model_id,
                    "method": "http",
                    "latency_seconds": latency,
                    "status_code": response.status_code,
                    "error_type": "" if success else "EmptyResponse",
                    "error_category": "" if success else "empty_response",
                    "error_message": "" if success else "No content or reasoning returned.",
                    "response_text": content or "",
                    "response_preview": _preview(content),
                    "reasoning": reasoning or "",
                    "reasoning_preview": _preview(reasoning),
                    "raw_response_shape": self._summarize_http_shape(data),
                }
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = {
                    "success": False,
                    "model_id": model_id,
                    "method": "http",
                    "latency_seconds": round(time.perf_counter() - started_at, 4),
                    "status_code": None,
                    "error_type": type(exc).__name__,
                    "error_category": _classify_error(None, type(exc).__name__, str(exc)),
                    "error_message": _preview(exc),
                    "response_text": "",
                    "response_preview": "",
                    "reasoning": "",
                    "reasoning_preview": "",
                    "raw_response_shape": "",
                }
                if attempt == 0:
                    continue
                return last_error
            except Exception as exc:  # noqa: BLE001 - failures are audit data.
                return {
                    "success": False,
                    "model_id": model_id,
                    "method": "http",
                    "latency_seconds": round(time.perf_counter() - started_at, 4),
                    "status_code": None,
                    "error_type": type(exc).__name__,
                    "error_category": _classify_error(None, type(exc).__name__, str(exc)),
                    "error_message": _preview(exc),
                    "response_text": "",
                    "response_preview": "",
                    "reasoning": "",
                    "reasoning_preview": "",
                    "raw_response_shape": "",
                }

        return last_error or {
            "success": False,
            "model_id": model_id,
            "method": "http",
            "latency_seconds": round(time.perf_counter() - started_at, 4),
            "status_code": None,
            "error_type": "UnknownHTTPFailure",
            "error_category": "other",
            "error_message": "HTTP request failed without an exception.",
            "response_text": "",
            "response_preview": "",
            "reasoning": "",
            "reasoning_preview": "",
            "raw_response_shape": "",
        }

    def forward_chat_completion_raw(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }
        if extra_body:
            payload.update(
                {
                    key: value
                    for key, value in extra_body.items()
                    if key not in {"model", "messages", "stream"}
                }
            )
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        return response.status_code, self._safe_json(response)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _extract_sdk_content(completion: Any) -> tuple[str, str]:
        message = completion.choices[0].message
        content = getattr(message, "content", None)
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
        )
        return content or "", reasoning or ""

    @staticmethod
    def _extract_sdk_stream_content(stream: Any) -> tuple[str, str]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
            )
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
        return "".join(content_parts), "".join(reasoning_parts)

    @staticmethod
    def _extract_http_content(data: Any) -> tuple[str, str]:
        message = data["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        return content, reasoning

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return {"raw_text_preview": _preview(response.text)}

    @staticmethod
    def _extract_model_ids(data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []
        items = data.get("data", [])
        ids: list[str] = []
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        return sorted(ids)

    @staticmethod
    def _extract_error_message(data: Any, fallback_text: str) -> str:
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            if data.get("message"):
                return str(data["message"])
        return fallback_text

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status if isinstance(response_status, int) else None

    @staticmethod
    def _summarize_sdk_shape(completion: Any) -> str:
        try:
            data = completion.model_dump(exclude_none=True)
            return NvidiaClient._summarize_http_shape(data)
        except Exception:  # noqa: BLE001 - shape summary is best effort.
            return type(completion).__name__

    @staticmethod
    def _summarize_http_shape(data: Any) -> str:
        if not isinstance(data, dict):
            return type(data).__name__
        top_keys = sorted(data.keys())
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            message_keys = sorted(message.keys()) if isinstance(message, dict) else []
            return f"keys={top_keys}; choices={len(choices)}; message_keys={message_keys}"
        return f"keys={top_keys}; choices={len(choices) if isinstance(choices, list) else 'n/a'}"

    @staticmethod
    def _summarize_model_list_shape(data: Any) -> str:
        if not isinstance(data, dict):
            return type(data).__name__
        items = data.get("data") or []
        first_keys = sorted(items[0].keys()) if items and isinstance(items[0], dict) else []
        return f"keys={sorted(data.keys())}; data_count={len(items)}; first_keys={first_keys}"
