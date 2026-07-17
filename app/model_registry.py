"""Curated NVIDIA Model Bridge registry and routing recommendations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


LATEST_AUDIT_SOURCE = "latest_audit_2026-07-06_01-09-54"
RECOMMENDED_ROUTE_MAP = {
    "default/general": "qwen/qwen3-next-80b-a3b-instruct",
    "reasoning": "qwen/qwen3-next-80b-a3b-instruct",
    "nvidia_native_reasoning": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "coding": "qwen/qwen3-next-80b-a3b-instruct",
    "fast": "nvidia/nemotron-mini-4b-instruct",
    "general_fallback": "mistralai/mistral-nemotron",
    "lightweight_fallback": "openai/gpt-oss-20b",
    "deepseek_fallback": "deepseek-ai/deepseek-v4-pro",
}


class ModelEntry(BaseModel):
    id: str
    display_name: str
    provider: str
    priority: bool = False
    status: str
    recommended_for: list[str] = Field(default_factory=list)
    avg_latency: float | None = None
    benchmark_pass_count: int = 0
    benchmark_total_count: int = 0
    notes: str = ""
    default_temperature: float = 0.2
    default_top_p: float = 0.95
    default_max_tokens: int = 1024
    extra_body: dict[str, Any] = Field(default_factory=dict)
    category: str = "general"
    recommended_use: str = ""
    supports_streaming: bool = False
    uses_reasoning: bool | str = False
    source_reference: str = LATEST_AUDIT_SOURCE
    rank: int | None = None

    @property
    def benchmark_summary(self) -> str:
        return f"{self.benchmark_pass_count}/{self.benchmark_total_count}"


CURATED_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="qwen/qwen3-next-80b-a3b-instruct",
        display_name="Qwen 3 Next 80B A3B Instruct",
        provider="Qwen",
        priority=False,
        status="recommended",
        recommended_for=[
            "recommended_default",
            "recommended_reasoning",
            "recommended_coding",
        ],
        avg_latency=1.13,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Best overall benchmark result and the current default router target.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=4096,
        extra_body={},
        category="general/coding/reasoning",
        recommended_use="default general model, reasoning, coding, and JSON-ish tasks",
        supports_streaming=False,
        uses_reasoning=True,
        rank=1,
    ),
    ModelEntry(
        id="nvidia/nemotron-mini-4b-instruct",
        display_name="Nemotron Mini 4B Instruct",
        provider="NVIDIA",
        priority=False,
        status="recommended",
        recommended_for=["recommended_fast"],
        avg_latency=1.37,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Best fast/light model in the latest audit.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=2048,
        extra_body={},
        category="fast/general",
        recommended_use="fast responses and lightweight assistant calls",
        supports_streaming=False,
        uses_reasoning=False,
        rank=2,
    ),
    ModelEntry(
        id="mistralai/mistral-nemotron",
        display_name="Mistral Nemotron",
        provider="Mistral AI",
        priority=False,
        status="recommended",
        recommended_for=["recommended_general_fallback"],
        avg_latency=1.56,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Reliable general fallback and strong low-friction backup.",
        default_temperature=0.3,
        default_top_p=0.95,
        default_max_tokens=4096,
        extra_body={},
        category="general/fallback",
        recommended_use="general fallback, broad assistant work, safe backup route",
        supports_streaming=False,
        uses_reasoning=True,
        rank=3,
    ),
    ModelEntry(
        id="nvidia/nemotron-nano-12b-v2-vl",
        display_name="Nemotron Nano 12B V2 VL",
        provider="NVIDIA",
        priority=False,
        status="recommended",
        recommended_for=["recommended_multimodal_candidate"],
        avg_latency=1.89,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Strong multimodal-capable candidate from the discovery run.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=2048,
        extra_body={},
        category="multimodal/general",
        recommended_use="multimodal candidate and general text helper",
        supports_streaming=False,
        uses_reasoning=True,
    ),
    ModelEntry(
        id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        display_name="Nemotron 3 Nano Omni 30B A3B Reasoning",
        provider="NVIDIA",
        priority=True,
        status="priority",
        recommended_for=["recommended_nvidia_native_reasoning"],
        avg_latency=2.20,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Official priority model and the best NVIDIA-native reasoning route.",
        default_temperature=0.6,
        default_top_p=0.95,
        default_max_tokens=65536,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 16384,
        },
        category="reasoning",
        recommended_use="NVIDIA-native reasoning, thinking mode, hard reasoning tasks",
        supports_streaming=False,
        uses_reasoning=True,
        rank=1,
    ),
    ModelEntry(
        id="openai/gpt-oss-20b",
        display_name="GPT OSS 20B",
        provider="OpenAI OSS",
        priority=False,
        status="recommended",
        recommended_for=["recommended_lightweight_fallback"],
        avg_latency=2.55,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Lightweight fallback that still passed all benchmark tasks.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=2048,
        extra_body={},
        category="fast/general",
        recommended_use="lightweight fallback and simple assistant responses",
        supports_streaming=False,
        uses_reasoning=True,
    ),
    ModelEntry(
        id="deepseek-ai/deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        provider="DeepSeek AI",
        priority=True,
        status="priority",
        recommended_for=["recommended_deepseek_fallback"],
        avg_latency=4.09,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Official priority model and the DeepSeek fallback route.",
        default_temperature=0.8,
        default_top_p=0.95,
        default_max_tokens=16384,
        extra_body={"chat_template_kwargs": {"thinking": False}},
        category="reasoning/coding",
        recommended_use="reasoning, coding, and DeepSeek-specific fallback use",
        supports_streaming=False,
        uses_reasoning="configurable",
        rank=2,
    ),
    ModelEntry(
        id="qwen/qwen3.5-122b-a10b",
        display_name="Qwen 3.5 122B A10B",
        provider="Qwen",
        priority=True,
        status="priority_but_slow",
        recommended_for=["priority_but_slow"],
        avg_latency=56.26,
        benchmark_pass_count=3,
        benchmark_total_count=3,
        notes="Priority model, but far slower than the newer Qwen default route.",
        default_temperature=0.6,
        default_top_p=0.95,
        default_max_tokens=16384,
        extra_body={},
        category="general/coding/reasoning",
        recommended_use="priority coverage, but not the default router choice",
        supports_streaming=False,
        uses_reasoning="unknown",
        rank=3,
    ),
    ModelEntry(
        id="deepseek-ai/deepseek-v4-flash",
        display_name="DeepSeek V4 Flash",
        provider="DeepSeek AI",
        priority=False,
        status="partial",
        recommended_for=["partial_candidate"],
        avg_latency=32.49,
        benchmark_pass_count=2,
        benchmark_total_count=3,
        notes="Partial candidate; keep for retest but do not default yet.",
        default_temperature=0.5,
        default_top_p=0.95,
        default_max_tokens=4096,
        extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
        category="reasoning/fast",
        recommended_use="partial candidate for later retest",
        supports_streaming=False,
        uses_reasoning=True,
    ),
    ModelEntry(
        id="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        display_name="Llama 3.1 Nemotron Nano VL 8B V1",
        provider="NVIDIA",
        priority=False,
        status="partial",
        recommended_for=["partial_candidate"],
        avg_latency=2.18,
        benchmark_pass_count=2,
        benchmark_total_count=3,
        notes="Partial multimodal candidate; keep as a retest option.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=2048,
        extra_body={},
        category="multimodal/general",
        recommended_use="partial multimodal candidate for later retest",
        supports_streaming=False,
        uses_reasoning=True,
    ),
    ModelEntry(
        id="qwen/qwen3.5-397b-a17b",
        display_name="Qwen 3.5 397B A17B",
        provider="Qwen",
        priority=False,
        status="retest_later",
        recommended_for=["retest_later"],
        avg_latency=240.73,
        benchmark_pass_count=0,
        benchmark_total_count=3,
        notes="Timed out on all tasks; retest only with sequential mode and a longer timeout.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=300,
        extra_body={},
        category="reasoning/coding",
        recommended_use="retest later only, with sequential mode and longer timeout",
        supports_streaming=False,
        uses_reasoning=True,
    ),
    ModelEntry(
        id="deepseek-ai/deepseek-coder-6.7b-instruct",
        display_name="DeepSeek Coder 6.7B Instruct",
        provider="DeepSeek AI",
        priority=False,
        status="avoid",
        recommended_for=["avoid"],
        avg_latency=0.22,
        benchmark_pass_count=0,
        benchmark_total_count=3,
        notes="Unavailable for the current account; returned 404 during the audit.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=1024,
        extra_body={},
        category="coding",
        recommended_use="avoid for this account",
        supports_streaming=False,
        uses_reasoning=False,
    ),
    ModelEntry(
        id="nvidia/nemotron-4-340b-instruct",
        display_name="Nemotron 4 340B Instruct",
        provider="NVIDIA",
        priority=False,
        status="avoid",
        recommended_for=["avoid"],
        avg_latency=0.22,
        benchmark_pass_count=0,
        benchmark_total_count=3,
        notes="Unavailable for the current account; returned 404 during the audit.",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=1024,
        extra_body={},
        category="reasoning",
        recommended_use="avoid for this account",
        supports_streaming=False,
        uses_reasoning=True,
    ),
]


def get_curated_models(limit: int = 0) -> list[ModelEntry]:
    models = list(CURATED_MODELS)
    if limit > 0:
        return models[:limit]
    return models


def get_priority_models() -> list[ModelEntry]:
    return [model for model in CURATED_MODELS if model.priority]


def get_non_priority_models() -> list[ModelEntry]:
    return [model for model in CURATED_MODELS if not model.priority]


def get_model_by_id(model_id: str) -> ModelEntry | None:
    for model in CURATED_MODELS:
        if model.id == model_id:
            return model
    return None


def get_models_by_status(status: str) -> list[ModelEntry]:
    return [model for model in CURATED_MODELS if model.status == status]


def get_avoid_models() -> list[ModelEntry]:
    return get_models_by_status("avoid")


def get_partial_models() -> list[ModelEntry]:
    return get_models_by_status("partial")


def get_retest_models() -> list[ModelEntry]:
    return get_models_by_status("retest_later")


def get_recommended_models() -> list[ModelEntry]:
    return [
        model
        for model in CURATED_MODELS
        if model.status in {"recommended", "priority_but_slow", "priority"}
    ]


def model_to_dict(model: ModelEntry) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
