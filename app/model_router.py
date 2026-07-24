"""Coordinator routing helpers for the NVIDIA Model Bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_project_root
from app.model_registry import (
    CURATED_MODELS,
    RECOMMENDED_ROUTE_MAP,
    ModelEntry,
    get_model_by_id,
    get_priority_models,
)


SUPPORTED_TASK_TYPES = {
    "general",
    "coding",
    "reasoning",
    "nvidia_reasoning",
    "json",
    "fast",
    "fallback",
    "deepseek",
    "lightweight",
}

TASK_TYPE_SELECTION_REASON = {
    "general": "recommended general/default model from latest benchmark",
    "coding": "recommended coding/default model from latest benchmark",
    "reasoning": "recommended reasoning/default model from latest benchmark",
    "nvidia_reasoning": "recommended NVIDIA-native reasoning model from latest benchmark",
    "json": "recommended general/default model for structured outputs",
    "fast": "recommended fast model from latest benchmark",
    "fallback": "recommended general fallback model from latest benchmark",
    "deepseek": "recommended DeepSeek fallback model from latest benchmark",
    "lightweight": "recommended lightweight fallback model from latest benchmark",
}

FALLBACK_CHAINS: dict[str, list[str]] = {
    "general": [
        RECOMMENDED_ROUTE_MAP["general_fallback"],
        RECOMMENDED_ROUTE_MAP["lightweight_fallback"],
        RECOMMENDED_ROUTE_MAP["fast"],
    ],
    "coding": [
        RECOMMENDED_ROUTE_MAP["general_fallback"],
        RECOMMENDED_ROUTE_MAP["lightweight_fallback"],
        RECOMMENDED_ROUTE_MAP["fast"],
    ],
    "reasoning": [
        RECOMMENDED_ROUTE_MAP["general_fallback"],
        RECOMMENDED_ROUTE_MAP["lightweight_fallback"],
    ],
    "fast": [
        RECOMMENDED_ROUTE_MAP["lightweight_fallback"],
        RECOMMENDED_ROUTE_MAP["general_fallback"],
    ],
}

# Default chain for task types not explicitly configured
DEFAULT_FALLBACK_CHAIN = [
    RECOMMENDED_ROUTE_MAP["general_fallback"],
    RECOMMENDED_ROUTE_MAP["lightweight_fallback"],
    RECOMMENDED_ROUTE_MAP["fast"],
]

# Keep backward compatibility:
FALLBACK_CHAIN = DEFAULT_FALLBACK_CHAIN


@dataclass(frozen=True)
class ModelSelection:
    model_id: str
    selected_by: str
    selection_reason: str
    task_type: str
    user_specified: bool
    fallback_model_id: str | None = None
    fallback_reason: str | None = None


def normalize_task_type(task_type: str | None) -> str:
    candidate = (task_type or "general").strip().lower()
    return candidate if candidate in SUPPORTED_TASK_TYPES else "general"


def get_routing_model_id(task_type: str | None) -> str:
    normalized = normalize_task_type(task_type)
    if normalized == "nvidia_reasoning":
        return RECOMMENDED_ROUTE_MAP["nvidia_native_reasoning"]
    if normalized == "fallback":
        return RECOMMENDED_ROUTE_MAP["general_fallback"]
    if normalized == "deepseek":
        return RECOMMENDED_ROUTE_MAP["deepseek_fallback"]
    if normalized == "lightweight":
        return RECOMMENDED_ROUTE_MAP["lightweight_fallback"]
    if normalized == "fast":
        return RECOMMENDED_ROUTE_MAP["fast"]
    return RECOMMENDED_ROUTE_MAP.get(
        f"{normalized}/general",
        RECOMMENDED_ROUTE_MAP["default/general"],
    )


def select_model(task_type: str | None, model: str | None) -> ModelSelection:
    normalized_task_type = normalize_task_type(task_type)
    if model:
        return ModelSelection(
            model_id=model,
            selected_by="user",
            selection_reason="explicit model supplied by caller",
            task_type=normalized_task_type,
            user_specified=True,
        )

    route_model_id = get_routing_model_id(normalized_task_type)
    return ModelSelection(
        model_id=route_model_id,
        selected_by="coordinator",
        selection_reason=TASK_TYPE_SELECTION_REASON[normalized_task_type],
        task_type=normalized_task_type,
        user_specified=False,
    )


def get_fallback_model_id(model_id: str) -> str | None:
    for candidate in FALLBACK_CHAIN:
        if candidate != model_id:
            return candidate
    return None


def get_model_record(model_id: str) -> ModelEntry | None:
    return get_model_by_id(model_id)


@lru_cache(maxsize=1)
def load_latest_discovered_model_ids() -> tuple[str, ...]:
    project_root = get_project_root()
    audit_dir = project_root / "audits"
    if not audit_dir.exists():
        return ()

    audit_files = sorted(audit_dir.glob("nvidia_model_benchmark_*.json"))
    if not audit_files:
        return ()

    latest = audit_files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    discovery = payload.get("discovery", {})
    model_ids = discovery.get("model_ids", [])
    if not isinstance(model_ids, list):
        return ()
    cleaned = [str(model_id) for model_id in model_ids if model_id]
    return tuple(sorted(set(cleaned)))


def get_discovered_model_ids() -> set[str]:
    return set(load_latest_discovered_model_ids())


def get_available_model_ids() -> set[str]:
    avoid_ids = {model.id for model in CURATED_MODELS if model.status == "avoid"}
    registry_ids = {model.id for model in CURATED_MODELS if model.status != "avoid"}
    registry_ids.update(
        model_id for model_id in get_discovered_model_ids() if model_id not in avoid_ids
    )
    return registry_ids


def is_model_available(model_id: str) -> bool:
    return model_id in get_available_model_ids()


def fallback_candidates_for(model_id: str, task_type: str | None = None) -> list[str]:
    """Get fallback candidates for a model, optionally filtered by task type."""
    normalized = normalize_task_type(task_type) if task_type else "general"
    chain = FALLBACK_CHAINS.get(normalized, DEFAULT_FALLBACK_CHAIN)
    candidates = [c for c in chain if c != model_id]
    return candidates


def summarize_known_models() -> dict[str, list[str]]:
    return {
        "priority": [model.id for model in get_priority_models()],
        "discovered": list(load_latest_discovered_model_ids()),
    }
