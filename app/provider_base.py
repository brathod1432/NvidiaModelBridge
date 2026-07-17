"""Provider abstraction layer for multi-provider support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    base_url: str
    api_key: str
    timeout_seconds: int = 90
    max_retries: int = 2
    extra_headers: dict[str, str] | None = None


@dataclass
class CompletionResult:
    """Standardized result from any provider."""
    success: bool
    model_id: str
    provider: str
    content: str = ""
    reasoning: str = ""
    latency_seconds: float = 0.0
    status_code: int | None = None
    error_type: str = ""
    error_message: str = ""
    raw_response: dict[str, Any] | None = None
    usage: dict[str, int] | None = None


class BaseProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Execute a synchronous chat completion request."""
        ...

    @abstractmethod
    async def chat_completion_async(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Execute an async chat completion request."""
        ...

    @abstractmethod
    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a chat completion response."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """List available models from this provider."""
        ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Check provider connectivity and health."""
        ...


class NvidiaProvider(BaseProvider):
    """NVIDIA-specific provider implementation wrapping NvidiaClient."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        # Lazy import to avoid circular dependencies
        self._client = None

    def _get_client(self):
        if self._client is None:
            from app.config import NvidiaSettings
            from app.nvidia_client import NvidiaClient
            settings = NvidiaSettings(
                api_key=self.config.api_key,
                api_key_source="provider_config",
                base_url=self.config.base_url,
                timeout_seconds=self.config.timeout_seconds,
            )
            self._client = NvidiaClient(settings)
        return self._client

    def chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        client = self._get_client()
        result = client.chat_completion_http(
            model_id=model_id,
            messages=messages,
            temperature=temperature or 0.2,
            top_p=top_p or 0.95,
            max_tokens=max_tokens or 1024,
            extra_body=extra_body,
        )
        return CompletionResult(
            success=result.get("success", False),
            model_id=model_id,
            provider=self.name,
            content=result.get("response_text", ""),
            reasoning=result.get("reasoning", ""),
            latency_seconds=result.get("latency_seconds", 0),
            status_code=result.get("status_code"),
            error_type=result.get("error_type", ""),
            error_message=result.get("error_message", ""),
            raw_response=result,
        )

    async def chat_completion_async(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.chat_completion(
                model_id, messages, temperature, top_p, max_tokens, extra_body
            ),
        )

    async def stream_completion(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        from app.streaming import stream_chat_completion
        async for event in stream_chat_completion(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
            timeout=self.config.timeout_seconds,
        ):
            yield event

    def list_models(self) -> list[str]:
        client = self._get_client()
        result = client.list_models()
        return result.get("model_ids", [])

    def health_check(self) -> dict[str, Any]:
        client = self._get_client()
        result = client.list_models()
        return {
            "provider": self.name,
            "base_url": self.config.base_url,
            "reachable": result.get("success", False),
            "status_code": result.get("status_code"),
            "models_found": result.get("number_of_models_discovered", 0),
            "latency_seconds": result.get("latency_seconds"),
            "error": result.get("error_message", ""),
        }


class ProviderRegistry:
    """Registry of available model providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> BaseProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())

    def get_default(self) -> BaseProvider | None:
        """Get the first registered provider as default."""
        if self._providers:
            return next(iter(self._providers.values()))
        return None

    def health_check_all(self) -> dict[str, Any]:
        """Check health of all registered providers."""
        return {
            name: provider.health_check()
            for name, provider in self._providers.items()
        }
