"""Benchmark runner for NVIDIA models."""

from __future__ import annotations

import platform
import re
import sys
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from statistics import mean
from typing import Any

from rich.console import Console
from rich.table import Table

from app.audit import write_audit_reports, write_supporting_docs
from app.benchmark_tasks import (
    DISCOVERY_BENCHMARK_TASKS,
    FULL_BENCHMARK_TASKS,
    BenchmarkTask,
    build_priority_smoke_task,
)
from app.config import NvidiaSettings
from app.model_registry import ModelEntry, get_curated_models, get_priority_models
from app.nvidia_client import NvidiaClient


console = Console()


def run_nvidia_model_benchmark() -> dict[str, Any]:
    """Run discovery, smoke tests, and benchmarks, then save reports."""

    settings = NvidiaSettings.load()
    timestamp = datetime.now().isoformat(timespec="seconds")
    priority_models = get_priority_models()
    reference_models = get_curated_models()

    if not settings.api_key:
        report = _build_missing_key_report(settings, priority_models, reference_models, timestamp)
        audit_paths = write_audit_reports(report)
        report["audit_files"] = audit_paths
        report["documentation_files"] = write_supporting_docs(report)
        _print_missing_key_summary(audit_paths)
        return report

    client = NvidiaClient(settings)
    discovery = _run_discovery(client)
    if not discovery.get("success"):
        report = _build_discovery_failure_report(
            settings=settings,
            timestamp=timestamp,
            priority_models=priority_models,
            reference_models=reference_models,
            discovery=discovery,
        )
        audit_paths = write_audit_reports(report)
        report["audit_files"] = audit_paths
        report["documentation_files"] = write_supporting_docs(report)
        _print_discovery_failure_summary(report)
        return report

    discovery_inventory = _build_discovery_inventory(
        discovery.get("model_ids", []), priority_models
    )
    priority_ids = [model.id for model in priority_models]
    discovery["timestamp"] = timestamp
    discovery["priority_models_found"] = [
        model_id for model_id in priority_ids if model_id in discovery.get("model_ids", [])
    ]
    discovery["priority_models_missing"] = [
        model_id for model_id in priority_ids if model_id not in discovery.get("model_ids", [])
    ]
    discovery["non_priority_models_discovered"] = [
        model_id
        for model_id in discovery.get("model_ids", [])
        if model_id not in priority_ids
    ]
    selected_candidates = _select_discovery_candidates(
        discovery_inventory, settings.discovery_test_limit
    )
    selected_candidate_models = [
        _discovery_inventory_to_model_entry(item) for item in selected_candidates
    ]

    priority_smoke_results = _run_priority_smoke_tests(
        client, settings, priority_models
    )
    priority_results, priority_workers = _run_parallel_benchmarks(
        settings=settings,
        selected_models=priority_models,
        tasks=FULL_BENCHMARK_TASKS,
        workers_cap=1,
    )
    candidate_results, candidate_workers = _run_parallel_benchmarks(
        settings=settings,
        selected_models=selected_candidate_models,
        tasks=DISCOVERY_BENCHMARK_TASKS,
        workers_cap=settings.max_workers,
    )

    _merge_smoke_results(priority_results, priority_smoke_results)
    all_results = priority_results + candidate_results
    _score_models(all_results)

    priority_ranking = _rank_models(
        priority_results,
        sort_by_priority_rank=True,
    )
    overall_ranking = _rank_models(all_results)
    recommendations = _build_recommendations(all_results)
    discovered_summary = _build_discovery_summary(
        discovery=discovery,
        discovery_inventory=discovery_inventory,
        selected_candidates=selected_candidates,
        candidate_results=candidate_results,
    )
    failure_analysis = _build_failure_analysis(priority_results, candidate_results, discovery)
    report = _build_report(
        settings=settings,
        timestamp=timestamp,
        reference_models=reference_models,
        discovery=discovery,
        priority_smoke_results=priority_smoke_results,
        priority_results=priority_results,
        candidate_results=candidate_results,
        priority_ranking=priority_ranking,
        overall_ranking=overall_ranking,
        recommendations=recommendations,
        discovered_summary=discovered_summary,
        failure_analysis=failure_analysis,
        priority_workers=priority_workers,
        candidate_workers=candidate_workers,
    )
    audit_paths = write_audit_reports(report)
    report["audit_files"] = audit_paths
    report["documentation_files"] = write_supporting_docs(report)
    _print_success_summary(report)
    return report


def _run_discovery(client: NvidiaClient) -> dict[str, Any]:
    return client.list_models()


def _run_priority_smoke_tests(
    client: NvidiaClient, settings: NvidiaSettings, priority_models: list[ModelEntry]
) -> list[dict[str, Any]]:
    smoke_task = build_priority_smoke_task()
    results: list[dict[str, Any]] = []
    for model in priority_models:
        task_result = _run_task(client, model, smoke_task, settings)
        results.append(_smoke_summary(model, task_result))
    return results


def _run_parallel_benchmarks(
    settings: NvidiaSettings,
    selected_models: list[ModelEntry],
    tasks: list[BenchmarkTask],
    workers_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not selected_models:
        return [], _parallel_worker_summary(
            settings=settings,
            selected_models=[],
            model_summaries=[],
            workers_used=0,
            rate_limit_notes="No models selected.",
        )

    max_workers = max(1, min(workers_cap, settings.max_workers, len(selected_models)))
    active_workers = max_workers
    workers_used = 0
    rate_limit_notes = ""
    model_summaries: list[dict[str, Any]] = []
    remaining_models = list(selected_models)

    while remaining_models:
        batch = remaining_models[:active_workers]
        remaining_models = remaining_models[active_workers:]
        workers_used = max(workers_used, min(active_workers, len(batch)))
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            future_to_model = {
                executor.submit(_benchmark_model_worker, settings, model, tasks): model
                for model in batch
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    model_summaries.append(future.result())
                except Exception as exc:  # noqa: BLE001 - one worker must not end the audit.
                    model_summaries.append(_worker_failure_summary(model, exc, tasks))

        if active_workers > 2 and any(
            _model_has_error_category(model, "rate_limit") for model in model_summaries
        ):
            active_workers = 2
            rate_limit_notes = (
                "Rate limit detected during benchmark; remaining work was reduced "
                "to 2 workers."
            )

    order = {model.id: index for index, model in enumerate(selected_models)}
    model_summaries.sort(key=lambda item: order.get(item["model_id"], 9999))
    return model_summaries, _parallel_worker_summary(
        settings=settings,
        selected_models=selected_models,
        model_summaries=model_summaries,
        workers_used=workers_used,
        rate_limit_notes=rate_limit_notes,
    )


def _benchmark_model_worker(
    settings: NvidiaSettings, model: ModelEntry, tasks: list[BenchmarkTask]
) -> dict[str, Any]:
    client = NvidiaClient(settings)
    return _benchmark_model(client, settings, model, tasks)


def _benchmark_model(
    client: NvidiaClient,
    settings: NvidiaSettings,
    model: ModelEntry,
    tasks: list[BenchmarkTask],
) -> dict[str, Any]:
    task_results = [_run_task(client, model, task, settings) for task in tasks]
    latencies = [
        task["latency_seconds"] for task in task_results if task.get("latency_seconds") is not None
    ]
    successful_tasks = sum(1 for task in task_results if task["task_success"])
    api_successful_tasks = sum(1 for task in task_results if task["api_success"])
    failed_tasks = len(task_results) - successful_tasks
    quality_score = mean(task["evaluator_score"] for task in task_results) if task_results else 0
    reliability_score = successful_tasks / len(task_results) if task_results else 0
    timeout_count = sum(1 for task in task_results if task.get("error_category") == "timeout")
    malformed_count = sum(
        1
        for task in task_results
        if task.get("error_category") in {"response_parse_error", "empty_response"}
    )

    task_group_summaries = _task_group_summaries(task_results)
    profile = "priority" if model.priority else "discovery"
    return {
        "model_id": model.id,
        "priority": model.priority,
        "priority_rank": model.rank,
        "display_name": model.display_name,
        "provider": model.provider,
        "category": model.category,
        "recommended_use": model.recommended_use,
        "source_reference": model.source_reference,
        "benchmark_profile": profile,
        "total_tasks": len(task_results),
        "successful_tasks": successful_tasks,
        "api_successful_tasks": api_successful_tasks,
        "failed_tasks": failed_tasks,
        "average_latency_seconds": round(mean(latencies), 4) if latencies else None,
        "fastest_task_latency_seconds": round(min(latencies), 4) if latencies else None,
        "slowest_task_latency_seconds": round(max(latencies), 4) if latencies else None,
        "simple_task_latency_seconds": task_group_summaries["simple"].get(
            "average_latency_seconds"
        ),
        "coding_task_latency_seconds": task_group_summaries["coding"].get(
            "average_latency_seconds"
        ),
        "reasoning_task_latency_seconds": task_group_summaries["reasoning"].get(
            "average_latency_seconds"
        ),
        "task_group_summaries": task_group_summaries,
        "all_tests_passed": successful_tasks == len(task_results),
        "quality_score": round(quality_score, 4),
        "speed_score": 0,
        "reliability_score": round(reliability_score, 4),
        "final_score": 0,
        "timeout_count": timeout_count,
        "malformed_count": malformed_count,
        "tasks": task_results,
        "notes": _model_notes(task_results),
    }


def _worker_failure_summary(
    model: ModelEntry, exc: Exception, tasks: list[BenchmarkTask]
) -> dict[str, Any]:
    task_results = [
        {
            "task_id": task.id,
            "task_category": task.category,
            "run_number": task.run_number,
            "task_name": task.name,
            "task_purpose": task.purpose,
            "task_success": False,
            "api_success": False,
            "method": "failed",
            "latency_seconds": None,
            "status_code": None,
            "error_type": type(exc).__name__,
            "error_category": "other",
            "error_message": str(exc),
            "response_preview": "",
            "reasoning_preview": "",
            "raw_response_shape": "",
            "evaluator_passed": False,
            "evaluator_strong_success": False,
            "evaluator_score": 0.0,
            "evaluator_notes": "Worker failed before task completion.",
            "attempts": [],
        }
        for task in tasks
    ]
    return {
        "model_id": model.id,
        "priority": model.priority,
        "priority_rank": model.rank,
        "display_name": model.display_name,
        "provider": model.provider,
        "category": model.category,
        "recommended_use": model.recommended_use,
        "source_reference": model.source_reference,
        "benchmark_profile": "priority" if model.priority else "discovery",
        "total_tasks": len(task_results),
        "successful_tasks": 0,
        "api_successful_tasks": 0,
        "failed_tasks": len(task_results),
        "average_latency_seconds": None,
        "fastest_task_latency_seconds": None,
        "slowest_task_latency_seconds": None,
        "simple_task_latency_seconds": None,
        "coding_task_latency_seconds": None,
        "reasoning_task_latency_seconds": None,
        "task_group_summaries": _task_group_summaries(task_results),
        "all_tests_passed": False,
        "quality_score": 0,
        "speed_score": 0,
        "reliability_score": 0,
        "final_score": 0,
        "timeout_count": 0,
        "malformed_count": 0,
        "tasks": task_results,
        "notes": f"Worker failed: {type(exc).__name__}",
    }


def _run_task(
    client: NvidiaClient,
    model: ModelEntry,
    task: BenchmarkTask,
    settings: NvidiaSettings,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": task.prompt}]
    stream = settings.test_streaming and model.supports_streaming
    extra_body = _task_extra_body(model, task)
    sdk_result = client.chat_completion_sdk(
        model_id=model.id,
        messages=messages,
        temperature=task.temperature,
        top_p=model.default_top_p,
        max_tokens=task.max_tokens,
        extra_body=extra_body,
        stream=stream,
    )

    attempts = [_attempt_summary(sdk_result)]
    final_result = sdk_result
    if not sdk_result["success"] and _should_retry_sdk(sdk_result):
        time.sleep(2)
        retry_result = client.chat_completion_sdk(
            model_id=model.id,
            messages=messages,
            temperature=task.temperature,
            top_p=model.default_top_p,
            max_tokens=task.max_tokens,
            extra_body=extra_body,
            stream=stream,
        )
        attempts.append(_attempt_summary(retry_result))
        final_result = retry_result

    if not final_result["success"] and _should_try_http_fallback(final_result):
        http_result = client.chat_completion_http(
            model_id=model.id,
            messages=messages,
            temperature=task.temperature,
            top_p=model.default_top_p,
            max_tokens=task.max_tokens,
            extra_body=extra_body,
            stream=False,
        )
        attempts.append(_attempt_summary(http_result))
        final_result = http_result

    response_text = final_result.get("response_text", "")
    evaluation = (
        task.evaluator(response_text)
        if final_result["success"]
        else {
            "passed": False,
            "strong_success": False,
            "score": 0,
            "notes": "API call did not return usable content.",
        }
    )
    task_success = bool(final_result["success"] and evaluation["passed"])

    return {
        "task_id": task.id,
        "task_category": task.category,
        "run_number": task.run_number,
        "task_name": task.name,
        "task_purpose": task.purpose,
        "task_success": task_success,
        "api_success": final_result["success"],
        "method": final_result.get("method", "failed") if final_result["success"] else "failed",
        "latency_seconds": final_result.get("latency_seconds"),
        "status_code": final_result.get("status_code"),
        "error_type": final_result.get("error_type", ""),
        "error_category": final_result.get("error_category", ""),
        "error_message": final_result.get("error_message", ""),
        "response_preview": final_result.get("response_preview", ""),
        "reasoning_preview": final_result.get("reasoning_preview", ""),
        "raw_response_shape": final_result.get("raw_response_shape", ""),
        "evaluator_passed": bool(evaluation["passed"]),
        "evaluator_strong_success": bool(evaluation["strong_success"]),
        "evaluator_score": float(evaluation["score"]),
        "evaluator_notes": str(evaluation["notes"]),
        "attempts": attempts,
    }


def _should_try_http_fallback(result: dict[str, Any]) -> bool:
    return result.get("error_category") in {
        "payload_rejected",
        "response_parse_error",
        "empty_response",
        "network_error",
        "timeout",
        "other",
    }


def _should_retry_sdk(result: dict[str, Any]) -> bool:
    return result.get("error_category") in {
        "rate_limit",
        "network_error",
    } or result.get("status_code") in {408, 409, 425, 429, 500, 502, 503, 504}


def _task_extra_body(model: ModelEntry, task: BenchmarkTask) -> dict[str, Any]:
    extra_body = deepcopy(model.extra_body)
    if task.category not in {"smoke", "simple"}:
        return extra_body

    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        return extra_body

    if "enable_thinking" in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = False
    if "thinking" in chat_template_kwargs:
        chat_template_kwargs["thinking"] = False
    if "reasoning_budget" in extra_body:
        extra_body["reasoning_budget"] = 0
    return extra_body


def _attempt_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": result.get("method", "failed"),
        "success": result.get("success", False),
        "latency_seconds": result.get("latency_seconds"),
        "status_code": result.get("status_code"),
        "error_type": result.get("error_type", ""),
        "error_category": result.get("error_category", ""),
        "error_message": result.get("error_message", ""),
    }


def _score_models(model_summaries: list[dict[str, Any]]) -> None:
    average_latencies = [
        model["average_latency_seconds"]
        for model in model_summaries
        if model.get("average_latency_seconds") and model["average_latency_seconds"] > 0
    ]
    inverse_latencies = [1 / latency for latency in average_latencies]
    min_inverse = min(inverse_latencies) if inverse_latencies else 0
    max_inverse = max(inverse_latencies) if inverse_latencies else 0

    for model in model_summaries:
        average_latency = model.get("average_latency_seconds")
        if average_latency and average_latency > 0:
            inverse = 1 / average_latency
            if max_inverse == min_inverse:
                speed_score = 1.0
            else:
                speed_score = (inverse - min_inverse) / (max_inverse - min_inverse)
        else:
            speed_score = 0.0
        model["speed_score"] = round(speed_score, 4)
        if model.get("successful_tasks", 0) == 0:
            model["speed_score"] = 0.0
        model["final_score"] = round(
            model["reliability_score"] * 0.50
            + model["quality_score"] * 0.30
            + model["speed_score"] * 0.20,
            4,
        )


def _rank_models(
    model_summaries: list[dict[str, Any]], *, sort_by_priority_rank: bool = False
) -> list[dict[str, Any]]:
    if sort_by_priority_rank:
        return sorted(
            model_summaries,
            key=lambda item: (
                item.get("priority_rank") if item.get("priority_rank") is not None else 9999,
                item.get("final_score", 0),
                item.get("reliability_score", 0),
            ),
        )
    return sorted(
        model_summaries,
        key=lambda item: (
            item.get("final_score", 0),
            item.get("reliability_score", 0),
            item.get("quality_score", 0),
            -(item.get("average_latency_seconds") or 999999),
        ),
        reverse=True,
    )


def _build_recommendations(model_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    successful_models = [model for model in model_summaries if model["successful_tasks"] > 0]
    perfect_models = [model for model in model_summaries if model["failed_tasks"] == 0]
    default_pool = perfect_models or successful_models
    default_model = _best_by(default_pool, ["final_score", "reliability_score", "quality_score"])

    simple_successes = [
        model
        for model in model_summaries
        if _group_summary(model, "simple").get("successful_tasks", 0) > 0
        and _group_summary(model, "simple").get("average_latency_seconds") is not None
    ]
    fast_model = (
        sorted(
            simple_successes,
            key=lambda item: _group_summary(item, "simple")["average_latency_seconds"],
        )[0]
        if simple_successes
        else None
    )

    reasoning_successes = [
        model
        for model in model_summaries
        if _group_summary(model, "reasoning").get("successful_tasks", 0) > 0
    ]
    reasoning_model = _best_by(
        reasoning_successes,
        ["reliability_score", "quality_score", "final_score"],
        group_latency_category="reasoning",
    )

    coding_successes = [
        model
        for model in model_summaries
        if _group_summary(model, "coding").get("successful_tasks", 0) > 0
    ]
    coding_model = _best_by(
        coding_successes,
        ["reliability_score", "quality_score", "final_score"],
        group_latency_category="coding",
    )

    retest_models = [
        model
        for model in model_summaries
        if model["successful_tasks"] == 0
        and _model_has_only_transient_failures(model)
    ]
    avoid_models = [
        model
        for model in model_summaries
        if model not in retest_models
        and (
            model["successful_tasks"] == 0
            or model["reliability_score"] < 0.5
            or model["malformed_count"] >= max(3, model["total_tasks"] // 2)
        )
    ]
    additional_good_models = [
        model
        for model in _rank_models(successful_models)
        if not model.get("priority")
    ][:5]

    return {
        "recommended_default_model": _recommendation_summary(default_model),
        "recommended_fast_model": _recommendation_summary(fast_model),
        "recommended_reasoning_model": _recommendation_summary(reasoning_model),
        "recommended_coding_model": _recommendation_summary(coding_model),
        "additional_good_models_found": [
            _recommendation_summary(model) for model in additional_good_models
        ],
        "retest_models": [_recommendation_summary(model) for model in retest_models],
        "avoid_models": [_recommendation_summary(model) for model in avoid_models],
    }


def _build_failure_analysis(
    priority_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    discovery: dict[str, Any],
) -> dict[str, list[str]]:
    analysis: dict[str, list[str]] = {
        "missing_api_key": [],
        "unauthorized": [],
        "rate_limit": [],
        "timeout": [],
        "model_unavailable": [],
        "payload_rejected": [],
        "evaluation_failed": [],
        "response_parse_error": [],
        "empty_response": [],
        "other": [],
    }
    if discovery and not discovery.get("success") and discovery.get("error_category"):
        category = discovery.get("error_category", "other")
        if category not in analysis:
            category = "other"
        analysis[category].append(f"/v1/models: {discovery.get('error_message', '')}")

    for model in priority_results + candidate_results:
        for task in model.get("tasks", []):
            if task.get("task_success"):
                continue
            if task.get("api_success"):
                category = "evaluation_failed"
            else:
                category = task.get("error_category") or "empty_response"
            if category not in analysis:
                category = "other"
            detail = (
                f"{model['model_id']} / {task['task_id']}: "
                f"{task.get('error_message') or task.get('evaluator_notes')}"
            )
            analysis[category].append(detail)
    return analysis


def _best_by(
    models: list[dict[str, Any]],
    score_keys: list[str],
    latency_key: str | None = None,
    group_latency_category: str | None = None,
) -> dict[str, Any] | None:
    if not models:
        return None

    def sort_key(model: dict[str, Any]) -> tuple[Any, ...]:
        score_values = tuple(model.get(key, 0) for key in score_keys)
        if group_latency_category:
            latency = _group_summary(model, group_latency_category).get(
                "average_latency_seconds"
            )
        else:
            latency = model.get(latency_key) if latency_key else model.get("average_latency_seconds")
        latency_sort = -(latency or 999999)
        return (*score_values, latency_sort)

    return sorted(models, key=sort_key, reverse=True)[0]


def _recommendation_summary(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if not model:
        return None
    return {
        "model_id": model["model_id"],
        "display_name": model["display_name"],
        "average_latency_seconds": model.get("average_latency_seconds"),
        "final_score": model.get("final_score"),
        "successful_tasks": model.get("successful_tasks"),
        "failed_tasks": model.get("failed_tasks"),
        "all_tests_passed": model.get("all_tests_passed"),
        "task_group_summaries": model.get("task_group_summaries", {}),
    }


def _build_discovery_inventory(
    discovered_model_ids: list[str], priority_models: list[ModelEntry]
) -> list[dict[str, Any]]:
    priority_ids = {model.id for model in priority_models}
    inventory: list[dict[str, Any]] = []
    for model_id in discovered_model_ids:
        if model_id in priority_ids:
            continue
        categories = _categorize_discovered_model(model_id)
        skip_reason = _skip_reason_for_model(model_id)
        inventory.append(
            {
                "model_id": model_id,
                "family_key": _family_key(model_id),
                "categories": categories,
                "selection_score": _candidate_score(model_id, categories, skip_reason),
                "selected_for_testing": False,
                "tested": False,
                "skip_reason": skip_reason,
                "selection_reason": "",
            }
        )
    return inventory


def _select_discovery_candidates(
    discovery_inventory: list[dict[str, Any]], discovery_test_limit: int
) -> list[dict[str, Any]]:
    limit = max(0, discovery_test_limit)
    if limit == 0:
        return []

    ordered = sorted(
        discovery_inventory,
        key=lambda item: (
            item.get("selection_score", 0),
            item.get("family_key", ""),
            item.get("model_id", ""),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in ordered:
        if item.get("skip_reason"):
            continue
        family_key = item.get("family_key", item["model_id"])
        if family_key in seen_families:
            continue
        selected_item = dict(item)
        selected_item["selected_for_testing"] = True
        selected_item["selection_reason"] = (
            "high-scoring discovery candidate with chat/instruct-style naming"
        )
        selected.append(selected_item)
        seen_families.add(family_key)
        if len(selected) >= limit:
            break
    return selected


def _build_discovery_summary(
    discovery: dict[str, Any],
    discovery_inventory: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any]:
    tested_candidate_ids = {result["model_id"] for result in candidate_results}
    selected_candidates = [
        {**item, "tested": item["model_id"] in tested_candidate_ids}
        for item in selected_candidates
    ]
    priority_models_found = discovery.get("priority_models_found", [])
    non_priority_models = discovery.get("non_priority_models_discovered", [])
    category_counts = {
        "reasoning_candidates": sum(
            1
            for item in discovery_inventory
            if "reasoning candidates" in item.get("categories", [])
        ),
        "coding_general_candidates": sum(
            1
            for item in discovery_inventory
            if "coding/general candidates" in item.get("categories", [])
        ),
        "fast_light_candidates": sum(
            1
            for item in discovery_inventory
            if "fast/light candidates" in item.get("categories", [])
        ),
        "multimodal_candidates": sum(
            1
            for item in discovery_inventory
            if "multimodal candidates" in item.get("categories", [])
        ),
    }
    return {
        "source": "/v1/models",
        "timestamp": discovery.get("timestamp", ""),
        "total_discovered": discovery.get("number_of_models_discovered", 0),
        "priority_models_found": priority_models_found,
        "priority_models_missing": discovery.get("priority_models_missing", []),
        "non_priority_models_discovered": non_priority_models,
        "category_counts": category_counts,
        "discovery_inventory": discovery_inventory,
        "selected_candidates": selected_candidates,
        "selected_candidate_ids": [item["model_id"] for item in selected_candidates],
        "candidate_results": candidate_results,
    }


def _build_report(
    settings: NvidiaSettings,
    timestamp: str,
    reference_models: list[ModelEntry],
    discovery: dict[str, Any],
    priority_smoke_results: list[dict[str, Any]],
    priority_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    priority_ranking: list[dict[str, Any]],
    overall_ranking: list[dict[str, Any]],
    recommendations: dict[str, Any],
    discovered_summary: dict[str, Any],
    failure_analysis: dict[str, list[str]],
    priority_workers: dict[str, Any],
    candidate_workers: dict[str, Any],
) -> dict[str, Any]:
    benchmark_models = priority_results + candidate_results
    total_tasks = sum(model["total_tasks"] for model in benchmark_models)
    total_successes = sum(model["successful_tasks"] for model in benchmark_models)
    total_failures = total_tasks - total_successes
    smoke_successes = sum(1 for item in priority_smoke_results if item["smoke_test_status"] == "PASS")
    smoke_failures = len(priority_smoke_results) - smoke_successes
    api_key_works = bool(discovery.get("success"))

    report = {
        "timestamp": timestamp,
        "environment": _environment_section(settings, api_key_found=True),
        "connectivity": _connectivity_section(settings, discovery),
        "summary": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "base_url": settings.base_url,
            "checked_variable": settings.checked_variable,
            "api_key_found": True,
            "api_key_source": settings.api_key_source,
            "masked_api_key": settings.masked_api_key,
            "api_key_works": api_key_works,
            "discovery_enabled": settings.enable_discovery,
            "priority_models_tested": len(priority_results),
            "priority_smoke_tests": len(priority_smoke_results),
            "discovery_candidates_tested": len(candidate_results),
            "benchmark_tasks_per_priority_model": len(FULL_BENCHMARK_TASKS),
            "benchmark_tasks_per_discovery_model": len(DISCOVERY_BENCHMARK_TASKS),
            "total_tasks": total_tasks,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "smoke_successes": smoke_successes,
            "smoke_failures": smoke_failures,
            "test_streaming": settings.test_streaming,
            "timeout_seconds": settings.timeout_seconds,
            "max_workers": settings.max_workers,
            "discovery_test_limit": settings.discovery_test_limit,
        },
        "discovery": {
            "success": bool(discovery.get("success")),
            "status_code": discovery.get("status_code"),
            "latency_seconds": discovery.get("latency_seconds"),
            "response_preview": discovery.get("response_preview", ""),
            "raw_response_shape": discovery.get("raw_response_shape", ""),
            "number_of_models_discovered": discovery.get("number_of_models_discovered", 0),
            "model_ids": discovery.get("model_ids", []),
            "priority_models_found": discovered_summary["priority_models_found"],
            "priority_models_missing": discovered_summary["priority_models_missing"],
            "non_priority_models_discovered": discovered_summary[
                "non_priority_models_discovered"
            ],
            "selected_candidates": discovered_summary["selected_candidates"],
            "selected_candidate_ids": discovered_summary["selected_candidate_ids"],
            "candidate_results": discovered_summary["candidate_results"],
            "category_counts": discovered_summary["category_counts"],
            "source": discovered_summary["source"],
            "timestamp": discovered_summary["timestamp"],
        },
        "priority_smoke_results": priority_smoke_results,
        "priority_model_registry": [_model_to_dict(model) for model in reference_models],
        "priority_model_results": priority_results,
        "priority_model_ranking": priority_ranking,
        "overall_model_ranking": overall_ranking,
        "candidate_results": candidate_results,
        "recommendations": recommendations,
        "failure_analysis": failure_analysis,
        "priority_workers": priority_workers,
        "candidate_workers": candidate_workers,
        "benchmark_tasks": {
            "priority": [_task_to_dict(task) for task in FULL_BENCHMARK_TASKS],
            "discovery": [_task_to_dict(task) for task in DISCOVERY_BENCHMARK_TASKS],
            "smoke": [_task_to_dict(build_priority_smoke_task())],
        },
        "model_selection": {
            "priority_model_ids": [model.id for model in get_priority_models()],
            "priority_models": [_model_to_dict(model) for model in get_priority_models()],
            "reference_models": [_model_to_dict(model) for model in reference_models],
            "selected_discovery_candidates": discovered_summary["selected_candidates"],
        },
        "final_decision_notes": _final_decision_notes(recommendations),
    }
    return report


def _build_missing_key_report(
    settings: NvidiaSettings,
    priority_models: list[ModelEntry],
    reference_models: list[ModelEntry],
    timestamp: str,
) -> dict[str, Any]:
    recommendations = {
        "recommended_default_model": None,
        "recommended_fast_model": None,
        "recommended_reasoning_model": None,
        "recommended_coding_model": None,
        "additional_good_models_found": [],
        "retest_models": [],
        "avoid_models": [],
    }
    discovery = {
        "success": False,
        "skipped": True,
        "model_ids": [],
        "priority_models_found": [],
        "priority_models_missing": [model.id for model in priority_models],
        "non_priority_models_discovered": [],
        "selected_candidates": [],
        "selected_candidate_ids": [],
        "candidate_results": [],
        "category_counts": {},
        "source": "/v1/models",
        "timestamp": timestamp,
        "error_category": "missing_api_key",
        "error_message": "NVIDIA_API_KEY was not found in this Python process.",
    }
    return {
        "timestamp": timestamp,
        "environment": _environment_section(settings, api_key_found=False),
        "connectivity": {
            "base_url": settings.base_url,
            "models_success": False,
            "status_code": None,
            "latency_seconds": None,
            "number_of_models_discovered": 0,
            "error_message": "NVIDIA_API_KEY was not found in this Python process.",
            "response_preview": "",
        },
        "summary": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "base_url": settings.base_url,
            "checked_variable": settings.checked_variable,
            "api_key_found": False,
            "api_key_source": "missing",
            "masked_api_key": "",
            "api_key_works": False,
            "discovery_enabled": settings.enable_discovery,
            "priority_models_tested": 0,
            "priority_smoke_tests": 0,
            "discovery_candidates_tested": 0,
            "benchmark_tasks_per_priority_model": len(FULL_BENCHMARK_TASKS),
            "benchmark_tasks_per_discovery_model": len(DISCOVERY_BENCHMARK_TASKS),
            "total_tasks": 0,
            "total_successes": 0,
            "total_failures": 0,
            "smoke_successes": 0,
            "smoke_failures": 0,
            "test_streaming": settings.test_streaming,
            "timeout_seconds": settings.timeout_seconds,
            "max_workers": settings.max_workers,
            "discovery_test_limit": settings.discovery_test_limit,
        },
        "discovery": discovery,
        "priority_smoke_results": [],
        "priority_model_registry": [_model_to_dict(model) for model in reference_models],
        "priority_model_results": [],
        "priority_model_ranking": [],
        "overall_model_ranking": [],
        "candidate_results": [],
        "recommendations": recommendations,
        "failure_analysis": {
            "missing_api_key": ["NVIDIA_API_KEY was not found in this Python process."],
            "unauthorized": [],
            "rate_limit": [],
            "timeout": [],
            "model_unavailable": [],
            "payload_rejected": [],
            "evaluation_failed": [],
            "response_parse_error": [],
            "empty_response": [],
            "other": [],
        },
        "priority_workers": _parallel_worker_summary(
            settings=settings,
            selected_models=priority_models,
            model_summaries=[],
            workers_used=0,
            rate_limit_notes="No model tests were run because NVIDIA_API_KEY is missing.",
        ),
        "candidate_workers": _parallel_worker_summary(
            settings=settings,
            selected_models=[],
            model_summaries=[],
            workers_used=0,
            rate_limit_notes="No model tests were run because NVIDIA_API_KEY is missing.",
        ),
        "benchmark_tasks": {
            "priority": [_task_to_dict(task) for task in FULL_BENCHMARK_TASKS],
            "discovery": [_task_to_dict(task) for task in DISCOVERY_BENCHMARK_TASKS],
            "smoke": [_task_to_dict(build_priority_smoke_task())],
        },
        "model_selection": {
            "priority_model_ids": [model.id for model in priority_models],
            "priority_models": [_model_to_dict(model) for model in priority_models],
            "reference_models": [_model_to_dict(model) for model in reference_models],
            "selected_discovery_candidates": [],
        },
        "final_decision_notes": _final_decision_notes(recommendations),
    }


def _build_discovery_failure_report(
    settings: NvidiaSettings,
    timestamp: str,
    priority_models: list[ModelEntry],
    reference_models: list[ModelEntry],
    discovery: dict[str, Any],
) -> dict[str, Any]:
    report = _build_missing_key_report(settings, priority_models, reference_models, timestamp)
    report["environment"]["api_key_found"] = True
    report["environment"]["masked_api_key"] = settings.masked_api_key
    report["connectivity"] = _connectivity_section(settings, discovery)
    report["summary"].update(
        {
            "api_key_found": True,
            "api_key_source": settings.api_key_source,
            "masked_api_key": settings.masked_api_key,
        }
    )
    report["discovery"] = {
        "success": bool(discovery.get("success")),
        "status_code": discovery.get("status_code"),
        "latency_seconds": discovery.get("latency_seconds"),
        "response_preview": discovery.get("response_preview", ""),
        "raw_response_shape": discovery.get("raw_response_shape", ""),
        "number_of_models_discovered": discovery.get("number_of_models_discovered", 0),
        "model_ids": discovery.get("model_ids", []),
        "priority_models_found": [],
        "priority_models_missing": [model.id for model in priority_models],
        "non_priority_models_discovered": [],
        "selected_candidates": [],
        "selected_candidate_ids": [],
        "candidate_results": [],
        "category_counts": {},
        "source": "/v1/models",
        "timestamp": timestamp,
        "error_category": discovery.get("error_category", "other"),
        "error_message": discovery.get("error_message", ""),
    }
    report["failure_analysis"] = {
        "missing_api_key": [],
        "unauthorized": [f"/v1/models: {discovery.get('error_message', '')}"],
        "rate_limit": [],
        "timeout": [],
        "model_unavailable": [],
        "payload_rejected": [],
        "response_parse_error": [],
        "empty_response": [],
        "other": [],
    }
    if discovery.get("error_category") != "unauthorized":
        report["failure_analysis"]["other"].append(
            f"/v1/models: {discovery.get('error_message', '')}"
        )
    else:
        report["failure_analysis"]["unauthorized"] = [
            f"/v1/models: {discovery.get('error_message', '')}"
        ]
    return report


def _environment_section(settings: NvidiaSettings, api_key_found: bool) -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "working_directory": settings.working_directory,
        "project_root": settings.project_root,
        "loaded_env_files": list(settings.loaded_env_files),
        "checked_variable": settings.checked_variable,
        "api_key_found": api_key_found,
        "api_key_source": settings.api_key_source,
        "masked_api_key": settings.masked_api_key,
        "key_shape_warning": settings.key_shape_warning,
        "injected_for_run": settings.api_key_injected_for_run,
        "injection_note": settings.api_key_injection_note,
        "sources": list(settings.environment_sources),
    }


def _connectivity_section(
    settings: NvidiaSettings, discovery: dict[str, Any]
) -> dict[str, Any]:
    return {
        "base_url": settings.base_url,
        "models_success": bool(discovery.get("success")),
        "status_code": discovery.get("status_code"),
        "latency_seconds": discovery.get("latency_seconds"),
        "number_of_models_discovered": len(discovery.get("model_ids", [])),
        "error_message": discovery.get("error_message", ""),
        "response_preview": discovery.get("response_preview", ""),
    }


def _parallel_worker_summary(
    settings: NvidiaSettings,
    selected_models: list[ModelEntry],
    model_summaries: list[dict[str, Any]],
    workers_used: int,
    rate_limit_notes: str,
) -> dict[str, Any]:
    completed = len(model_summaries)
    failed = sum(1 for model in model_summaries if model.get("successful_tasks", 0) == 0)
    return {
        "max_workers_configured": settings.max_workers,
        "workers_used": workers_used,
        "models_submitted": len(selected_models) if workers_used else 0,
        "models_completed": completed,
        "models_failed": failed,
        "rate_limit_notes": rate_limit_notes,
    }


def _model_has_error_category(model: dict[str, Any], category: str) -> bool:
    return any(task.get("error_category") == category for task in model.get("tasks", []))


def _model_has_only_transient_failures(model: dict[str, Any]) -> bool:
    failed_tasks = [task for task in model.get("tasks", []) if not task.get("task_success")]
    if not failed_tasks:
        return False
    transient_categories = {"rate_limit", "timeout", "network_error"}
    return all(
        task.get("error_category") in transient_categories for task in failed_tasks
    )


def _final_decision_notes(recommendations: dict[str, Any]) -> dict[str, str]:
    default_model = recommendations.get("recommended_default_model")
    fast_model = recommendations.get("recommended_fast_model")
    reasoning_model = recommendations.get("recommended_reasoning_model")
    coding_model = recommendations.get("recommended_coding_model")
    avoid_models = recommendations.get("avoid_models", [])
    return {
        "default_use": _note_for(default_model, "No default model selected."),
        "fast_fallback": _note_for(fast_model, "No fast model selected."),
        "reasoning": _note_for(reasoning_model, "No reasoning model selected."),
        "coding": _note_for(coding_model, "No coding model selected."),
        "avoid": (
            ", ".join(model["model_id"] for model in avoid_models)
            if avoid_models
            else "No avoid list entries from this run."
        ),
    }


def _note_for(model: dict[str, Any] | None, fallback: str) -> str:
    if not model:
        return fallback
    return (
        f"{model['model_id']} scored {model.get('final_score')} with "
        f"{model.get('successful_tasks')} successful tasks and "
        f"{model.get('failed_tasks')} failures."
    )


def _task_by_id(model: dict[str, Any], task_id: str) -> dict[str, Any]:
    for task in model.get("tasks", []):
        if task.get("task_id") == task_id:
            return task
    return {}


def _task_group_summaries(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for category in ["simple", "coding", "reasoning"]:
        category_tasks = [
            task for task in tasks if task.get("task_category") == category
        ]
        latencies = [
            task.get("latency_seconds")
            for task in category_tasks
            if task.get("latency_seconds") is not None
        ]
        successful_tasks = sum(1 for task in category_tasks if task.get("task_success"))
        total_tasks = len(category_tasks)
        summaries[category] = {
            "category": category,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "all_passed": bool(total_tasks and successful_tasks == total_tasks),
            "latencies_seconds": [round(float(value), 4) for value in latencies],
            "average_latency_seconds": round(mean(latencies), 4) if latencies else None,
        }
    return summaries


def _group_summary(model: dict[str, Any], category: str) -> dict[str, Any]:
    return model.get("task_group_summaries", {}).get(category, {})


def _model_notes(tasks: list[dict[str, Any]]) -> str:
    failed = [task for task in tasks if not task["task_success"]]
    if not failed:
        return "All benchmark tasks passed."
    categories = sorted({task.get("error_category") or "evaluation_failed" for task in failed})
    return "Failures: " + ", ".join(categories)


def _model_to_dict(model: ModelEntry) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _task_to_dict(task: BenchmarkTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "category": task.category,
        "run_number": task.run_number,
        "name": task.name,
        "purpose": task.purpose,
        "max_tokens": task.max_tokens,
        "temperature": task.temperature,
    }


def _print_missing_key_summary(audit_paths: dict[str, str]) -> None:
    console.print("[bold]NVIDIA Model Bridge Benchmark[/bold]\n")
    console.print("Checked variable:")
    console.print("NVIDIA_API_KEY\n")
    console.print("NVIDIA_API_KEY found:")
    console.print("false\n")
    console.print("API key:")
    console.print("[bold red]MISSING[/bold red]\n")
    console.print("No model tests were run.\n")
    console.print("This project only supports one key name:")
    console.print("NVIDIA_API_KEY\n")
    console.print("Fix options:")
    console.print("1. Add NVIDIA_API_KEY to the PyCharm Run Configuration environment variables.")
    console.print("2. Or create a local .env file in the project root with:")
    console.print("   NVIDIA_API_KEY=your_key_here")
    console.print("3. Or set it in the current PowerShell session:")
    console.print('   $env:NVIDIA_API_KEY="your_key_here"')
    console.print("4. Or set it permanently:")
    console.print('   setx NVIDIA_API_KEY "your_key_here"\n')
    console.print("After using setx, fully close and reopen PyCharm.\n")
    console.print("Audit files:")
    console.print(f"- {audit_paths['json']}")
    console.print(f"- {audit_paths['markdown']}")


def _print_discovery_failure_summary(report: dict[str, Any]) -> None:
    connectivity = report["connectivity"]
    console.print("[bold]NVIDIA Model Bridge Benchmark[/bold]\n")
    console.print("Checked variable:")
    console.print("NVIDIA_API_KEY\n")
    console.print("NVIDIA_API_KEY found:")
    console.print("true\n")
    console.print("API key:")
    console.print(f"FOUND: {report['summary']['masked_api_key']}\n")
    console.print("Connectivity:")
    console.print(f"GET /v1/models: FAILED")
    console.print(f"Status code: {connectivity.get('status_code')}")
    console.print(f"Latency: {connectivity.get('latency_seconds')}")
    console.print(f"Error: {connectivity.get('error_message')}\n")
    console.print("No model tests were run because discovery did not succeed.\n")
    console.print("Audit files:")
    console.print(f"- {report['audit_files']['json']}")
    console.print(f"- {report['audit_files']['markdown']}")


def _print_success_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    discovery = report["discovery"]
    recs = report["recommendations"]
    priority_workers = report.get("priority_workers", {})
    candidate_workers = report.get("candidate_workers", {})
    console.print("[bold]NVIDIA Model Bridge Benchmark[/bold]\n")
    console.print("Checked variable:")
    console.print("NVIDIA_API_KEY\n")
    console.print("NVIDIA_API_KEY found:")
    console.print("true\n")
    console.print("API key:")
    console.print(f"FOUND: {summary['masked_api_key']}\n")
    console.print("Connectivity:")
    console.print(f"GET /v1/models: {'SUCCESS' if discovery.get('success') else 'FAILED'}")
    console.print(f"Discovered models: {len(discovery.get('model_ids', []))}\n")
    console.print("Workers:")
    console.print(f"Priority workers used: {priority_workers.get('workers_used', 0)}")
    console.print(f"Discovery workers used: {candidate_workers.get('workers_used', 0)}\n")
    console.print("Benchmark:")
    console.print(f"Total tests: {summary['total_tasks']}")
    console.print(f"Successful tests: {summary['total_successes']}")
    console.print(f"Failed tests: {summary['total_failures']}\n")
    console.print("Recommended:")
    console.print(f"Default: {_rec_id(recs.get('recommended_default_model'))}")
    console.print(f"Fast: {_rec_id(recs.get('recommended_fast_model'))}")
    console.print(f"Reasoning: {_rec_id(recs.get('recommended_reasoning_model'))}")
    console.print(f"Coding: {_rec_id(recs.get('recommended_coding_model'))}\n")

    table = Table(title="Model Summary")
    table.add_column("Model")
    table.add_column("Priority")
    table.add_column("All")
    table.add_column("Simple", justify="right")
    table.add_column("Coding", justify="right")
    table.add_column("Reasoning", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Score", justify="right")
    for model in report["overall_model_ranking"]:
        groups = model.get("task_group_summaries", {})
        table.add_row(
            model["model_id"],
            "yes" if model.get("priority") else "no",
            "ALL" if model.get("all_tests_passed") else "",
            _terminal_group_pass(groups.get("simple", {})),
            _terminal_group_pass(groups.get("coding", {})),
            _terminal_group_pass(groups.get("reasoning", {})),
            f"{model['successful_tasks']}/{model['total_tasks']}",
            _format_latency(model.get("average_latency_seconds")),
            f"{model['final_score']:.3f}",
        )
    console.print(table)
    console.print("\nAudit files:")
    console.print(f"- {report['audit_files']['json']}")
    console.print(f"- {report['audit_files']['markdown']}")


def _rec_id(model: dict[str, Any] | None) -> str:
    return model["model_id"] if model else "None"


def _format_latency(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}s"


def _terminal_group_pass(group: dict[str, Any]) -> str:
    return f"{group.get('successful_tasks', 0)}/{group.get('total_tasks', 0)}"


def _categorize_discovered_model(model_id: str) -> list[str]:
    slug = model_id.lower()
    categories: list[str] = []
    if any(token in slug for token in ["reasoning", "nemotron", "deepseek", "qwen", "glm", "kimi", "gpt-oss"]):
        categories.append("reasoning candidates")
    if any(token in slug for token in ["qwen", "deepseek", "llama", "mistral", "mixtral", "gemma", "gpt-oss", "codestral", "starcoder", "granite", "codellama", "dbrx"]):
        categories.append("coding/general candidates")
    if any(token in slug for token in ["nano", "small", "mini", "flash", "20b", "8b", "7b"]):
        categories.append("fast/light candidates")
    if any(token in slug for token in ["vision", "vl", "multimodal", "omni", "phi"]):
        categories.append("multimodal candidates")
    if not categories:
        categories.append("uncategorized")
    return categories


def _skip_reason_for_model(model_id: str) -> str:
    slug = model_id.lower()
    skip_tokens = [
        "embed",
        "rerank",
        "retriever",
        "guard",
        "safety",
        "audio",
        "image",
        "video",
        "detector",
        "parse",
        "translate",
        "clip",
        "deplot",
        "topic-control",
        "content-safety",
        "pii",
        "reward",
    ]
    if any(token in slug for token in skip_tokens):
        return "utility or non-chat model"
    return ""


def _candidate_score(model_id: str, categories: list[str], skip_reason: str) -> float:
    if skip_reason:
        return -999.0
    slug = model_id.lower()
    score = 0.0
    if "reasoning candidates" in categories:
        score += 3.0
    if "coding/general candidates" in categories:
        score += 2.0
    if "fast/light candidates" in categories:
        score += 0.5
    if "multimodal candidates" in categories:
        score += 0.25
    if any(token in slug for token in ["instruct", "it", "chat", "reasoning", "pro"]):
        score += 1.0
    if any(token in slug for token in ["openai/gpt-oss", "mistral", "llama", "qwen", "deepseek", "nemotron"]):
        score += 0.5
    if any(token in slug for token in ["mini", "small", "nano", "8b", "7b", "20b"]):
        score += 0.25
    return score


def _family_key(model_id: str) -> str:
    slug = model_id.split("/", 1)[-1].lower()
    slug = re.sub(r"-(?:\d+(?:\.\d+)?(?:b|m|k)?)(?:-[a-z0-9.]+)*$", "", slug)
    slug = re.sub(r"-(?:instruct|it|reasoning|flash|pro|chat|base|v\d+(?:\.\d+)?)$", "", slug)
    slug = re.sub(r"-(?:\d{3,})$", "", slug)
    return slug


def _merge_smoke_results(
    priority_results: list[dict[str, Any]], smoke_results: list[dict[str, Any]]
) -> None:
    smoke_by_model = {result["model_id"]: result for result in smoke_results}
    for model in priority_results:
        model_id = model["model_id"]
        smoke_result = smoke_by_model.get(model_id)
        model["smoke_test"] = smoke_result
        if smoke_result:
            model["smoke_test_status"] = smoke_result["smoke_test_status"]
            model["smoke_test_latency_seconds"] = smoke_result["latency_seconds"]
            model["smoke_test_method"] = smoke_result["method"]
            model["smoke_test_status_code"] = smoke_result["status_code"]
            model["smoke_test_response_preview"] = smoke_result["response_preview"]
            model["smoke_test_error_message"] = smoke_result["error_message"]
        else:
            model["smoke_test_status"] = "MISSING"
            model["smoke_test_latency_seconds"] = None
            model["smoke_test_method"] = ""
            model["smoke_test_status_code"] = None
            model["smoke_test_response_preview"] = ""
            model["smoke_test_error_message"] = "Smoke result missing."


def _smoke_summary(model: ModelEntry, task_result: dict[str, Any]) -> dict[str, Any]:
    status = "PASS" if task_result.get("task_success") else "FAIL"
    return {
        "model_id": model.id,
        "priority": model.priority,
        "priority_rank": model.rank,
        "smoke_test_status": status,
        "latency_seconds": task_result.get("latency_seconds"),
        "status_code": task_result.get("status_code"),
        "method": task_result.get("method", "failed"),
        "response_preview": task_result.get("response_preview", ""),
        "reasoning_preview": task_result.get("reasoning_preview", ""),
        "error_message": task_result.get("error_message", ""),
        "error_category": task_result.get("error_category", ""),
        "task_result": task_result,
    }


def _discovery_inventory_to_model_entry(item: dict[str, Any]) -> ModelEntry:
    model_id = item["model_id"]
    provider = model_id.split("/", 1)[0] if "/" in model_id else "NVIDIA"
    slug = model_id.split("/", 1)[-1]
    display_name = " ".join(part for part in re.split(r"[-_]", slug) if part)
    categories = item.get("categories", [])
    if "reasoning candidates" in categories and "coding/general candidates" in categories:
        category = "reasoning/coding"
    elif "reasoning candidates" in categories:
        category = "reasoning"
    elif "coding/general candidates" in categories:
        category = "general/coding"
    elif "fast/light candidates" in categories:
        category = "fast/general"
    elif "multimodal candidates" in categories:
        category = "multimodal/general"
    else:
        category = "general"
    return ModelEntry(
        id=model_id,
        display_name=display_name[:1].upper() + display_name[1:] if display_name else model_id,
        provider=provider,
        status="discovered",
        recommended_for=["discovered_candidate"],
        avg_latency=None,
        benchmark_pass_count=0,
        benchmark_total_count=0,
        notes="Discovered from /v1/models during benchmark run.",
        category=category,
        recommended_use=item.get("selection_reason")
        or "discovered candidate selected from /v1/models",
        default_temperature=0.2,
        default_top_p=0.95,
        default_max_tokens=500,
        supports_streaming=False,
        uses_reasoning="unknown",
        extra_body={},
        source_reference="Discovered from /v1/models",
        priority=False,
        rank=None,
    )
