"""Coordinator for model selection and request execution."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.analytics import get_analytics_collector
from app.cache import get_response_cache
from app.circuit_breaker import get_circuit_breaker_registry
from app.config import BridgeSettings
from app.metrics import record_fallback, record_model_request
from app.model_registry import ModelEntry, get_model_by_id
from app.model_router import (
    fallback_candidates_for,
    get_available_model_ids,
    get_model_record,
    is_model_available,
    select_model,
)
from app.nvidia_client import NvidiaClient
from app.prompt_templates import build_messages_with_system_prompt


UNSUPPORTED_STREAMING_ERROR = "Streaming is not supported yet by Nvidia Model Bridge."


@dataclass
class CoordinatorOutcome:
    success: bool
    model: str | None
    task_type: str
    selected_by: str
    selection_reason: str
    fallback_used: bool
    latency_seconds: float | None
    content: str | None
    reasoning: str | None
    error: str | None
    selected_model: str | None = None
    fallback_model: str | None = None
    metadata: dict[str, Any] | None = None
    cache_hit: bool = False

    def to_api_response(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "model": self.model,
            "task_type": self.task_type,
            "selected_by": self.selected_by,
            "selection_reason": self.selection_reason,
            "fallback_used": self.fallback_used,
            "latency_seconds": self.latency_seconds,
            "content": self.content,
            "reasoning": self.reasoning,
            "error": self.error,
            "cache_hit": self.cache_hit,
        }


class Coordinator:
    def __init__(
        self,
        settings: BridgeSettings | None = None,
        client: NvidiaClient | None = None,
    ) -> None:
        self.settings = settings or BridgeSettings.load()
        self.client = client or NvidiaClient(
            settings=_benchmark_settings_for_gateway(self.settings)
        )
        self._cache = get_response_cache(
            maxsize=self.settings.cache_maxsize,
            ttl=self.settings.cache_ttl,
        )
        self._circuit_breakers = get_circuit_breaker_registry(
            failure_threshold=self.settings.circuit_breaker_threshold,
            recovery_timeout=self.settings.circuit_breaker_recovery,
        )
        self._analytics = get_analytics_collector()

    def ask(
        self,
        prompt: str,
        task_type: str | None,
        model: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        stream: bool,
        system_prompt: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> CoordinatorOutcome:
        started_at = time.perf_counter()
        if stream:
            return CoordinatorOutcome(
                success=False,
                model=model,
                task_type=(task_type or self.settings.default_task_type).lower(),
                selected_by="user" if model else "coordinator",
                selection_reason="streaming is currently unsupported",
                fallback_used=False,
                latency_seconds=round(time.perf_counter() - started_at, 4),
                content=None,
                reasoning=None,
                error=UNSUPPORTED_STREAMING_ERROR,
                selected_model=model,
            )

        selection = select_model(task_type or self.settings.default_task_type, model)
        available_ids = get_available_model_ids()
        if selection.user_specified and selection.model_id not in available_ids:
            return CoordinatorOutcome(
                success=False,
                model=selection.model_id,
                task_type=selection.task_type,
                selected_by=selection.selected_by,
                selection_reason=selection.selection_reason,
                fallback_used=False,
                latency_seconds=round(time.perf_counter() - started_at, 4),
                content=None,
                reasoning=None,
                error=_unavailable_model_error(selection.model_id),
                selected_model=selection.model_id,
            )

        # Build messages with system prompt and conversation history
        effective_task = selection.task_type
        if self.settings.enable_system_prompts:
            messages = build_messages_with_system_prompt(
                prompt=prompt,
                task_type=effective_task,
                system_prompt_override=system_prompt,
                conversation_history=conversation_history,
            )
        else:
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

        # Check cache
        if self.settings.enable_cache:
            cached = self._cache.get(
                model_id=selection.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if cached is not None:
                self._analytics.record(
                    model_id=selection.model_id,
                    task_type=effective_task,
                    success=True,
                    latency_seconds=round(time.perf_counter() - started_at, 4),
                    cache_hit=True,
                    endpoint="/ask",
                )
                return CoordinatorOutcome(
                    success=True,
                    model=selection.model_id,
                    task_type=effective_task,
                    selected_by=selection.selected_by,
                    selection_reason=selection.selection_reason,
                    fallback_used=False,
                    latency_seconds=round(time.perf_counter() - started_at, 4),
                    content=cached.get("content", ""),
                    reasoning=cached.get("reasoning"),
                    error=None,
                    selected_model=selection.model_id,
                    cache_hit=True,
                )

        # Check circuit breaker
        if not self._circuit_breakers.is_model_available(selection.model_id):
            # Model circuit is open, go directly to fallback
            if not selection.user_specified:
                fallback_outcome = self._try_fallbacks(
                    selection=selection,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    started_at=started_at,
                    primary_error=f"Circuit breaker open for {selection.model_id}.",
                )
                if fallback_outcome:
                    return fallback_outcome

            return self._failure_outcome(
                selection=selection,
                started_at=started_at,
                model_id=selection.model_id,
                error=f"Model {selection.model_id} circuit breaker is open (too many recent failures).",
            )

        # Try primary model
        selected_record = get_model_record(selection.model_id) or get_model_by_id(
            selection.model_id
        )
        primary_result = self._call_model(
            model_id=selection.model_id,
            messages=messages,
            record=selected_record,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        if primary_result["success"]:
            self._circuit_breakers.record_success(selection.model_id)
            outcome = self._success_outcome(
                selection=selection,
                result=primary_result,
                started_at=started_at,
                selected_model=selection.model_id,
                fallback_used=False,
            )
            # Cache successful response
            if self.settings.enable_cache:
                self._cache.put(
                    model_id=selection.model_id,
                    messages=messages,
                    response={
                        "content": outcome.content,
                        "reasoning": outcome.reasoning,
                    },
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            self._record_analytics(selection, outcome)
            return outcome

        # Primary failed
        self._circuit_breakers.record_failure(selection.model_id)

        if selection.user_specified:
            outcome = self._failure_outcome(
                selection=selection,
                started_at=started_at,
                model_id=selection.model_id,
                error=primary_result.get("error_message") or "Model request failed.",
            )
            self._record_analytics(selection, outcome)
            return outcome

        # Try fallback chain
        fallback_outcome = self._try_fallbacks(
            selection=selection,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            started_at=started_at,
            primary_error=primary_result.get("error_message") or "request failed.",
        )
        if fallback_outcome:
            return fallback_outcome

        outcome = self._failure_outcome(
            selection=selection,
            started_at=started_at,
            model_id=selection.model_id,
            error=primary_result.get("error_message") or "Model request failed.",
        )
        self._record_analytics(selection, outcome)
        return outcome

    def _try_fallbacks(
        self,
        *,
        selection,
        messages: list[dict[str, str]],
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        started_at: float,
        primary_error: str,
    ) -> CoordinatorOutcome | None:
        """Try all available fallback candidates in order."""
        candidates = fallback_candidates_for(selection.model_id)
        for fallback_model_id in candidates:
            if not is_model_available(fallback_model_id):
                continue
            if not self._circuit_breakers.is_model_available(fallback_model_id):
                continue

            fallback_record = get_model_record(fallback_model_id)
            fallback_result = self._call_model(
                model_id=fallback_model_id,
                messages=messages,
                record=fallback_record,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if fallback_result["success"]:
                self._circuit_breakers.record_success(fallback_model_id)
                record_fallback(selection.model_id, fallback_model_id)
                outcome = self._success_outcome(
                    selection=selection,
                    result=fallback_result,
                    started_at=started_at,
                    selected_model=fallback_model_id,
                    fallback_used=True,
                    fallback_model=fallback_model_id,
                )
                # Cache the fallback response too
                if self.settings.enable_cache:
                    self._cache.put(
                        model_id=selection.model_id,
                        messages=messages,
                        response={
                            "content": outcome.content,
                            "reasoning": outcome.reasoning,
                        },
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                self._record_analytics(selection, outcome)
                return outcome
            else:
                self._circuit_breakers.record_failure(fallback_model_id)

        return None

    def _call_model(
        self,
        *,
        model_id: str,
        messages: list[dict[str, str]],
        record: ModelEntry | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        default_temperature = (
            temperature if temperature is not None else record.default_temperature if record else 0.2
        )
        default_top_p = top_p if top_p is not None else record.default_top_p if record else 0.95
        default_max_tokens = (
            max_tokens if max_tokens is not None else record.default_max_tokens if record else 1024
        )
        return self.client.chat_completion_http(
            model_id=model_id,
            messages=messages,
            temperature=default_temperature,
            top_p=default_top_p,
            max_tokens=default_max_tokens,
            extra_body=record.extra_body if record else None,
            stream=False,
        )

    def _record_analytics(self, selection, outcome: CoordinatorOutcome) -> None:
        record_model_request(
            model_id=outcome.model or "",
            task_type=outcome.task_type,
            success=outcome.success,
            latency=outcome.latency_seconds,
        )
        self._analytics.record(
            model_id=outcome.model or "",
            task_type=outcome.task_type,
            success=outcome.success,
            latency_seconds=outcome.latency_seconds or 0,
            fallback_used=outcome.fallback_used,
            cache_hit=outcome.cache_hit,
            endpoint="/ask",
        )

    def _success_outcome(
        self,
        *,
        selection,
        result: dict[str, Any],
        started_at: float,
        selected_model: str,
        fallback_used: bool,
        fallback_model: str | None = None,
    ) -> CoordinatorOutcome:
        return CoordinatorOutcome(
            success=True,
            model=selected_model,
            task_type=selection.task_type,
            selected_by=selection.selected_by,
            selection_reason=selection.selection_reason,
            fallback_used=fallback_used,
            latency_seconds=round(time.perf_counter() - started_at, 4),
            content=result.get("response_text") or "",
            reasoning=result.get("reasoning") or None,
            error=None,
            selected_model=selection.model_id,
            fallback_model=fallback_model,
            metadata={
                "selected_model": selection.model_id,
                "selected_by": selection.selected_by,
                "selection_reason": selection.selection_reason,
                "fallback_used": fallback_used,
                "fallback_model": fallback_model,
                "success": True,
                "latency_seconds": result.get("latency_seconds"),
            },
        )

    def _failure_outcome(
        self,
        *,
        selection,
        started_at: float,
        model_id: str,
        error: str,
        fallback_used: bool = False,
        fallback_model: str | None = None,
    ) -> CoordinatorOutcome:
        return CoordinatorOutcome(
            success=False,
            model=model_id,
            task_type=selection.task_type,
            selected_by=selection.selected_by,
            selection_reason=selection.selection_reason,
            fallback_used=fallback_used,
            latency_seconds=round(time.perf_counter() - started_at, 4),
            content=None,
            reasoning=None,
            error=error,
            selected_model=selection.model_id,
            fallback_model=fallback_model,
            metadata={
                "selected_model": selection.model_id,
                "selected_by": selection.selected_by,
                "selection_reason": selection.selection_reason,
                "fallback_used": fallback_used,
                "fallback_model": fallback_model,
                "success": False,
            },
        )


def _benchmark_settings_for_gateway(settings: BridgeSettings):
    from app.config import NvidiaSettings

    return NvidiaSettings(
        api_key=settings.api_key,
        api_key_source=settings.api_key_source,
        base_url=settings.base_url,
        timeout_seconds=settings.timeout_seconds,
    )


def _unavailable_model_error(model_id: str) -> str:
    return (
        f"Model '{model_id}' is not available in the registry or latest discovered models."
    )
