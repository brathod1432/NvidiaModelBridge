"""Benchmark tasks and local response evaluators."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Callable


Evaluator = Callable[[str], dict[str, object]]


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    run_number: int
    name: str
    purpose: str
    prompt: str
    max_tokens: int
    temperature: float
    evaluator: Evaluator


SIMPLE_EXPECTED = "NVIDIA Model Bridge simple test successful."
PRIORITY_SMOKE_EXPECTED = "NVIDIA priority smoke test successful."


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _evaluate_exact_sentence(expected: str, response_text: str) -> dict[str, object]:
    normalized = _normalize_text(response_text)
    contains_expected = expected in response_text
    exact_match = normalized == expected
    score = 1.0 if exact_match else 0.8 if contains_expected else 0.0
    if exact_match:
        notes = "exact match"
    elif contains_expected:
        notes = "contains expected sentence but includes extra text"
    else:
        notes = "expected sentence not found"
    return {
        "passed": exact_match,
        "strong_success": exact_match,
        "score": score,
        "notes": notes,
    }


def evaluate_simple_task(response_text: str) -> dict[str, object]:
    return _evaluate_exact_sentence(SIMPLE_EXPECTED, response_text)


def evaluate_priority_smoke_task(response_text: str) -> dict[str, object]:
    return _evaluate_exact_sentence(PRIORITY_SMOKE_EXPECTED, response_text)


def evaluate_coding_task(response_text: str) -> dict[str, object]:
    lower_text = response_text.lower()
    has_function = "def is_palindrome" in lower_text
    has_bool = "true" in lower_text or "false" in lower_text
    test_markers = len(re.findall(r"\bassert\b|\btest\b|is_palindrome\(", lower_text))
    appears_three_tests = test_markers >= 4
    checks = [has_function, has_bool, appears_three_tests]
    score = sum(1 for item in checks if item) / len(checks)
    return {
        "passed": all(checks),
        "strong_success": all(checks),
        "score": score,
        "notes": (
            f"function={has_function}, bool={has_bool}, "
            f"three_tests={appears_three_tests}"
        ),
    }


def evaluate_reasoning_task(response_text: str) -> dict[str, object]:
    normalized = _normalize_text(response_text)
    required_format = "Final position: 7" in normalized
    contains_seven = re.search(r"(?<!\d)7(?!\d)", normalized) is not None
    score = 1.0 if required_format else 0.5 if contains_seven else 0.0
    return {
        "passed": required_format,
        "strong_success": required_format,
        "score": score,
        "notes": (
            "required format present"
            if required_format
            else "contains 7 without required final format"
            if contains_seven
            else "final position not detected"
        ),
    }


def build_benchmark_tasks(
    run_count: int = 5,
    *,
    simple_max_tokens: int = 80,
    coding_max_tokens: int = 600,
    reasoning_max_tokens: int = 500,
    simple_temperature: float = 0.0,
    coding_temperature: float = 0.2,
    reasoning_temperature: float = 0.2,
    prompt_total: int | None = None,
) -> list[BenchmarkTask]:
    """Build simple, coding, and reasoning tasks for a benchmark profile."""

    total = prompt_total or run_count
    total = max(1, total)
    tasks: list[BenchmarkTask] = []

    for run_number in range(1, run_count + 1):
        tasks.append(
            BenchmarkTask(
                id=f"simple_{run_number}",
                category="simple",
                run_number=run_number,
                name=f"Simple task {run_number}",
                purpose="speed and basic response test",
                prompt=(
                    f"Simple benchmark run {run_number}/{total}.\n"
                    "Reply with exactly this sentence and nothing else:\n"
                    f"{SIMPLE_EXPECTED}"
                ),
                max_tokens=simple_max_tokens,
                temperature=simple_temperature,
                evaluator=evaluate_simple_task,
            )
        )

    for run_number in range(1, run_count + 1):
        tasks.append(
            BenchmarkTask(
                id=f"coding_{run_number}",
                category="coding",
                run_number=run_number,
                name=f"Coding task {run_number}",
                purpose="test coding/usefulness",
                prompt=(
                    f"Coding benchmark run {run_number}/{total}.\n"
                    "Write a Python function named is_palindrome(text: str) -> bool.\n"
                    "Requirements:\n"
                    "- Ignore case.\n"
                    "- Ignore spaces.\n"
                    "- Ignore punctuation.\n"
                    "- Return True or False.\n"
                    "Then provide exactly 3 short test cases.\n"
                    "Keep the answer concise."
                ),
                max_tokens=coding_max_tokens,
                temperature=coding_temperature,
                evaluator=evaluate_coding_task,
            )
        )

    for run_number in range(1, run_count + 1):
        tasks.append(
            BenchmarkTask(
                id=f"reasoning_{run_number}",
                category="reasoning",
                run_number=run_number,
                name=f"Reasoning task {run_number}",
                purpose="test reasoning and instruction following",
                prompt=(
                    f"Reasoning benchmark run {run_number}/{total}.\n"
                    "A robot starts at position 0 on a number line.\n"
                    "It performs these moves:\n"
                    "+3, -1, +4, -2, -2, +5.\n"
                    "First calculate the final position.\n"
                    "Then explain the calculation in 3 short steps.\n"
                    "Final answer must be in this format:\n"
                    "Final position: X"
                ),
                max_tokens=reasoning_max_tokens,
                temperature=reasoning_temperature,
                evaluator=evaluate_reasoning_task,
            )
        )

    return tasks


def build_priority_smoke_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="priority_smoke",
        category="smoke",
        run_number=1,
        name="Priority smoke test",
        purpose="verify basic instruction following before parallel work",
        prompt=(
            "Reply with exactly this sentence and nothing else:\n"
            f"{PRIORITY_SMOKE_EXPECTED}"
        ),
        max_tokens=80,
        temperature=0.0,
        evaluator=evaluate_priority_smoke_task,
    )


FULL_BENCHMARK_TASKS = build_benchmark_tasks(1, prompt_total=1)
DISCOVERY_BENCHMARK_TASKS = build_benchmark_tasks(
    1,
    simple_max_tokens=500,
    coding_max_tokens=500,
    reasoning_max_tokens=500,
    prompt_total=1,
)


def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))
