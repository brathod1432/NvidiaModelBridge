"""JSON and Markdown audit report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUDIT_DIR = Path("audits")
DOCS_DIR = Path("docs")
PREVIEW_LIMIT = 500


def write_audit_reports(report: dict[str, Any]) -> dict[str, str]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = report["timestamp"].replace(":", "-").replace("T", "_")[:19]
    json_path = AUDIT_DIR / f"nvidia_model_benchmark_{timestamp}.json"
    markdown_path = AUDIT_DIR / f"nvidia_model_benchmark_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def write_supporting_docs(report: dict[str, Any]) -> dict[str, str]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    priority_path = DOCS_DIR / "nvidia_priority_models.md"
    discovered_path = DOCS_DIR / "discovered_nvidia_models.md"
    selection_notes_path = DOCS_DIR / "model_selection_notes.md"

    priority_path.write_text(render_priority_models_doc(report), encoding="utf-8")
    discovered_path.write_text(render_discovered_models_doc(report), encoding="utf-8")
    selection_notes_path.write_text(render_model_selection_notes_doc(report), encoding="utf-8")
    return {
        "priority_models": str(priority_path),
        "discovered_models": str(discovered_path),
        "selection_notes": str(selection_notes_path),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = ["# NVIDIA Model Bridge Benchmark Audit", ""]
    lines.extend(_environment_section(report))
    lines.extend(_connectivity_section(report))
    lines.extend(_summary_section(report))
    lines.extend(_priority_smoke_section(report))
    lines.extend(_priority_ranking_section(report))
    lines.extend(_overall_ranking_section(report))
    lines.extend(_discovery_section(report))
    lines.extend(_recommendations_section(report))
    lines.extend(_worker_section(report))
    lines.extend(_failure_analysis_section(report))
    return "\n".join(lines).rstrip() + "\n"


def render_priority_models_doc(report: dict[str, Any]) -> str:
    priority_ids = [model.get("model_id") for model in report.get("priority_model_ranking", [])]
    priority_results = {
        model.get("model_id"): model for model in report.get("priority_model_results", [])
    }
    smoke_results = {
        item.get("model_id"): item for item in report.get("priority_smoke_results", [])
    }
    lines = [
        "# NVIDIA Priority Models",
        "",
        "Current priority models:",
        "1. nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "2. deepseek-ai/deepseek-v4-pro",
        "3. qwen/qwen3.5-122b-a10b",
        "",
        "| Rank | Model ID | Provider | Purpose | Known NVIDIA UI settings | Benchmark status | Latest latency | Notes |",
        "| ---: | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for rank, model_id in enumerate(priority_ids or [
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "deepseek-ai/deepseek-v4-pro",
        "qwen/qwen3.5-122b-a10b",
    ], start=1):
        model = _find_model_entry(report, model_id)
        result = priority_results.get(model_id, {})
        smoke = smoke_results.get(model_id, {})
        status = _priority_benchmark_status(result, smoke)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    _cell(model_id),
                    _cell(model.get("provider", "")),
                    _cell(model.get("recommended_use", "")),
                    _cell(_ui_settings_summary(model)),
                    _cell(status),
                    _latency(result.get("average_latency_seconds")),
                    _cell(result.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_discovered_models_doc(report: dict[str, Any]) -> str:
    discovery = report.get("discovery", {})
    selected = {
        item.get("model_id"): item for item in discovery.get("selected_candidates", [])
    }
    tested = {
        item.get("model_id"): item for item in report.get("candidate_results", [])
    }
    lines = [
        "# Discovered NVIDIA Models",
        "",
        "Source:",
        "GET /v1/models",
        "",
        "Timestamp:",
        str(discovery.get("timestamp", report.get("timestamp", ""))),
        "",
        "Total discovered:",
        str(discovery.get("number_of_models_discovered", 0)),
        "",
        "Priority models found:",
    ]
    priority_found = discovery.get("priority_models_found", [])
    if priority_found:
        lines.extend(f"- {model_id}" for model_id in priority_found)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "Non-priority models discovered:",
        ]
    )
    non_priority = discovery.get("non_priority_models_discovered", [])
    if non_priority:
        lines.extend(f"- {model_id}" for model_id in non_priority)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "Selected candidate models:",
            "| Model ID | Categories | Selected | Tested | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in discovery.get("selected_candidates", []):
        model_id = item.get("model_id", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(model_id),
                    _cell(", ".join(item.get("categories", []))),
                    str(bool(item.get("selected_for_testing"))).lower(),
                    str(bool(item.get("tested"))).lower(),
                    _cell(item.get("selection_reason", "") or "Selected for benchmark testing."),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Tested candidate models:",
            "| Model ID | Status | Average latency | Notes |",
            "| --- | --- | ---: | --- |",
        ]
    )
    if tested:
        for model_id, result in tested.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(model_id),
                        "PASS" if result.get("all_tests_passed") else "FAIL",
                        _latency(result.get("average_latency_seconds")),
                        _cell(result.get("notes", "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| None | - | - | No additional candidates were tested. |")
    lines.append("")
    return "\n".join(lines)


def render_model_selection_notes_doc(report: dict[str, Any]) -> str:
    recs = report.get("recommendations", {})
    priority_ranking = report.get("priority_model_ranking", [])
    overall_ranking = report.get("overall_model_ranking", [])
    lines = [
        "# Model Selection Notes",
        "",
        "## Priority Models",
        "The three priority models are tested first because they represent the current desired routing order and the most relevant NVIDIA-native or high-value reasoning/coding candidates.",
        "",
        "Priority ranking is kept separate from overall discovered-model ranking. A discovered model can score higher in a short benchmark without invalidating the priority list.",
        "",
        "Latest priority ranking:",
    ]
    if priority_ranking:
        lines.extend(
            f"- {index}. {_model_ref(model)}: {model.get('successful_tasks', 0)}/{model.get('total_tasks', 0)} tasks passed, average latency {_latency(model.get('average_latency_seconds'))}, score {_score(model.get('final_score'))}"
            for index, model in enumerate(priority_ranking, start=1)
        )
    else:
        lines.append("- No priority benchmark results available.")

    lines.extend(
        [
            "",
            "Latest overall discovered ranking:",
        ]
    )
    if overall_ranking:
        lines.extend(
            f"- {index}. {_model_ref(model)}: priority={_yes_no(bool(model.get('priority')))}, {model.get('successful_tasks', 0)}/{model.get('total_tasks', 0)} tasks passed, average latency {_latency(model.get('average_latency_seconds'))}, score {_score(model.get('final_score'))}"
            for index, model in enumerate(overall_ranking[:10], start=1)
        )
    else:
        lines.append("- No overall benchmark results available.")

    lines.extend(
        [
        "",
        "## Benchmark Criteria",
        "- reliability",
        "- latency",
        "- simple task success",
        "- coding task success",
        "- reasoning task success",
        "- response parse stability",
        "- payload compatibility",
        "",
        "## Recommended Usage",
        f"- default model: {_model_ref(recs.get('recommended_default_model'))}",
        f"- reasoning model: {_model_ref(recs.get('recommended_reasoning_model'))}",
        f"- coding model: {_model_ref(recs.get('recommended_coding_model'))}",
        f"- fast fallback model: {_model_ref(recs.get('recommended_fast_model'))}",
        f"- retest models: {_avoid_list(recs.get('retest_models', []))}",
        f"- avoid list: {_avoid_list(recs.get('avoid_models', []))}",
        "",
        "Additional good models found:",
        ]
    )
    additional = recs.get("additional_good_models_found", [])
    if additional:
        lines.extend(f"- {_model_ref(model)}" for model in additional)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _summary_section(report: dict[str, Any]) -> list[str]:
    summary = report.get("summary", {})
    return [
        "## Summary",
        f"- Timestamp: {report.get('timestamp', '')}",
        f"- Python version: {summary.get('python_version', '')}",
        f"- Base URL: {summary.get('base_url', '')}",
        f"- Checked variable: {summary.get('checked_variable', 'NVIDIA_API_KEY')}",
        f"- API key source: {summary.get('api_key_source', '')}",
        f"- Masked API key: {summary.get('masked_api_key', '')}",
        f"- Discovery enabled: {summary.get('discovery_enabled', False)}",
        f"- Priority smoke tests: {summary.get('priority_smoke_tests', 0)}",
        f"- Priority models tested: {summary.get('priority_models_tested', 0)}",
        f"- Discovery candidates tested: {summary.get('discovery_candidates_tested', 0)}",
        f"- Total tasks: {summary.get('total_tasks', 0)}",
        f"- Total successes: {summary.get('total_successes', 0)}",
        f"- Total failures: {summary.get('total_failures', 0)}",
        f"- Smoke successes: {summary.get('smoke_successes', 0)}",
        f"- Smoke failures: {summary.get('smoke_failures', 0)}",
        "",
        (
            "Scoring: reliability is successful tasks divided by total tasks; "
            "quality is the average local evaluator score; speed is normalized "
            "from inverse average latency. Final score = reliability * 0.50 + "
            "quality * 0.30 + speed * 0.20."
        ),
        "",
    ]


def _environment_section(report: dict[str, Any]) -> list[str]:
    environment = report.get("environment", {})
    loaded_env_files = environment.get("loaded_env_files", [])
    lines = [
        "## Environment",
        f"- Python executable: {environment.get('python_executable', '')}",
        f"- Working directory: {environment.get('working_directory', '')}",
        f"- Project root: {environment.get('project_root', '')}",
        "- Loaded .env files: "
        + (", ".join(loaded_env_files) if loaded_env_files else "None"),
        f"- Checked variable: {environment.get('checked_variable', 'NVIDIA_API_KEY')}",
        f"- API key found: {environment.get('api_key_found', False)}",
        f"- API key source: {environment.get('api_key_source', '')}",
        f"- Masked key: {environment.get('masked_api_key', '')}",
        f"- Key shape warning: {environment.get('key_shape_warning', '') or 'None'}",
        f"- Injected for this run: {environment.get('injected_for_run', False)}",
        f"- Injection note: {environment.get('injection_note', '') or 'None'}",
        "",
        "| Source | Found | Masked | Length | Starts nvapi- | Whitespace | Newline |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for source in environment.get("sources", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(source.get("source", "")),
                    str(source.get("found", False)),
                    _cell(source.get("masked", "")),
                    str(source.get("length", 0)),
                    str(source.get("starts_with_nvapi", False)),
                    str(source.get("had_leading_or_trailing_whitespace", False)),
                    str(source.get("has_newline", False)),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _connectivity_section(report: dict[str, Any]) -> list[str]:
    connectivity = report.get("connectivity", {})
    return [
        "## Connectivity",
        f"- Base URL: {connectivity.get('base_url', '')}",
        f"- /models success: {connectivity.get('models_success', False)}",
        f"- Status code: {connectivity.get('status_code', '')}",
        f"- Latency: {_latency(connectivity.get('latency_seconds'))}",
        (
            "- Number of discovered models: "
            f"{connectivity.get('number_of_models_discovered', 0)}"
        ),
        f"- Error message: {connectivity.get('error_message', '') or 'None'}",
        f"- Response preview: {_clip(connectivity.get('response_preview', ''))}",
        "",
    ]


def _priority_smoke_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Priority Smoke Tests",
        "| Model | Status | Latency | Method | Response preview | Error message |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item in report.get("priority_smoke_results", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(item.get("model_id", "")),
                    _cell(item.get("smoke_test_status", "")),
                    _latency(item.get("latency_seconds")),
                    _cell(item.get("method", "")),
                    _clip(item.get("response_preview", "")),
                    _clip(item.get("error_message", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _priority_ranking_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Priority Model Ranking",
        "| Rank | Model | Smoke | Overall pass | Avg latency | Final score | Notes |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for index, model in enumerate(report.get("priority_model_ranking", []), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _cell(model.get("model_id", "")),
                    _cell(model.get("smoke_test_status", "")),
                    f"{model.get('successful_tasks', 0)}/{model.get('total_tasks', 0)}",
                    _latency(model.get("average_latency_seconds")),
                    _score(model.get("final_score")),
                    _cell(model.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _overall_ranking_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Overall Discovered Model Ranking",
        "| Rank | Model | Priority | Pass | Avg latency | Final score | Notes |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for index, model in enumerate(report.get("overall_model_ranking", []), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _cell(model.get("model_id", "")),
                    "yes" if model.get("priority") else "no",
                    f"{model.get('successful_tasks', 0)}/{model.get('total_tasks', 0)}",
                    _latency(model.get("average_latency_seconds")),
                    _score(model.get("final_score")),
                    _cell(model.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _discovery_section(report: dict[str, Any]) -> list[str]:
    discovery = report.get("discovery", {})
    lines = [
        "## Discovery",
        f"- Source: {discovery.get('source', '/v1/models')}",
        f"- Timestamp: {discovery.get('timestamp', '')}",
        f"- Success: {discovery.get('success', False)}",
        f"- Status code: {discovery.get('status_code', '')}",
        f"- Latency: {_latency(discovery.get('latency_seconds'))}",
        f"- Total discovered models: {discovery.get('number_of_models_discovered', 0)}",
        f"- Priority models found: {_list_or_none(discovery.get('priority_models_found', []))}",
        f"- Priority models missing: {_list_or_none(discovery.get('priority_models_missing', []))}",
        f"- Non-priority models discovered: {len(discovery.get('non_priority_models_discovered', []))}",
        f"- Selected candidates: {_list_or_none(discovery.get('selected_candidate_ids', []))}",
        "",
        "### Candidate Categories",
        f"- reasoning candidates: {discovery.get('category_counts', {}).get('reasoning_candidates', 0)}",
        f"- coding/general candidates: {discovery.get('category_counts', {}).get('coding_general_candidates', 0)}",
        f"- fast/light candidates: {discovery.get('category_counts', {}).get('fast_light_candidates', 0)}",
        f"- multimodal candidates: {discovery.get('category_counts', {}).get('multimodal_candidates', 0)}",
        "",
    ]
    return lines


def _recommendations_section(report: dict[str, Any]) -> list[str]:
    recs = report.get("recommendations", {})
    return [
        "## Recommendations",
        f"- Recommended default model: {_model_ref(recs.get('recommended_default_model'))}",
        f"- Recommended reasoning model: {_model_ref(recs.get('recommended_reasoning_model'))}",
        f"- Recommended coding model: {_model_ref(recs.get('recommended_coding_model'))}",
        f"- Recommended fast fallback model: {_model_ref(recs.get('recommended_fast_model'))}",
        f"- Additional good models found: {_avoid_list(recs.get('additional_good_models_found', []))}",
        f"- Models to retest: {_avoid_list(recs.get('retest_models', []))}",
        f"- Models to avoid: {_avoid_list(recs.get('avoid_models', []))}",
        "",
    ]


def _worker_section(report: dict[str, Any]) -> list[str]:
    priority = report.get("priority_workers", {})
    discovery = report.get("candidate_workers", {})
    return [
        "## Worker Section",
        "- Priority smoke tests were sequential: yes",
        f"- Priority benchmark workers used: {priority.get('workers_used', 0)}",
        f"- Discovery candidate workers used: {discovery.get('workers_used', 0)}",
        f"- Priority rate limit notes: {priority.get('rate_limit_notes', '') or 'None'}",
        f"- Discovery rate limit notes: {discovery.get('rate_limit_notes', '') or 'None'}",
        "",
    ]


def _failure_analysis_section(report: dict[str, Any]) -> list[str]:
    analysis = report.get("failure_analysis", {})
    lines = ["## Failure Analysis"]
    categories = [
        "missing_api_key",
        "unauthorized",
        "rate_limit",
        "timeout",
        "model_unavailable",
        "payload_rejected",
        "evaluation_failed",
        "response_parse_error",
        "empty_response",
        "other",
    ]
    for category in categories:
        failures = analysis.get(category, [])
        lines.append(f"- {category}: {len(failures)}")
        for item in failures:
            lines.append(f"  - {_cell(item)}")
    lines.append("")
    return lines


def _find_model_entry(report: dict[str, Any], model_id: str) -> dict[str, Any]:
    for model in report.get("priority_model_registry", []):
        if model.get("id") == model_id:
            return model
    for model in report.get("model_selection", {}).get("priority_models", []):
        if model.get("id") == model_id:
            return model
    return {}


def _ui_settings_summary(model: dict[str, Any]) -> str:
    if not model:
        return ""
    return (
        f"temp={model.get('default_temperature')} top_p={model.get('default_top_p')} "
        f"max_tokens={model.get('default_max_tokens')} extra_body={_compact_json(model.get('extra_body', {}))}"
    )


def _priority_benchmark_status(result: dict[str, Any], smoke: dict[str, Any]) -> str:
    smoke_status = smoke.get("smoke_test_status", "UNKNOWN")
    benchmark_status = "PASS" if result.get("all_tests_passed") else "FAIL"
    return f"smoke={smoke_status}, benchmark={benchmark_status}"


def _compact_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return text if len(text) <= PREVIEW_LIMIT else text[: PREVIEW_LIMIT - 3] + "..."


def _model_ref(model: Any) -> str:
    if not model:
        return "None"
    if isinstance(model, dict):
        return str(model.get("model_id") or model.get("id") or "None")
    return str(model)


def _avoid_list(models: list[Any]) -> str:
    if not models:
        return "None"
    return ", ".join(_model_ref(model) for model in models)


def _list_or_none(values: list[Any]) -> str:
    if not values:
        return "None"
    return ", ".join(_model_ref(value) for value in values)


def _latency(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return ""


def _score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _clip(value: Any) -> str:
    text = _cell(value)
    if len(text) <= PREVIEW_LIMIT:
        return text or "None"
    return text[: PREVIEW_LIMIT - 3] + "..."
