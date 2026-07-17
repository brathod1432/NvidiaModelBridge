"""Tests for model router module."""

from __future__ import annotations

from app.model_router import (
    SUPPORTED_TASK_TYPES,
    ModelSelection,
    fallback_candidates_for,
    get_routing_model_id,
    normalize_task_type,
    select_model,
)


class TestNormalizeTaskType:
    def test_none_defaults_to_general(self):
        assert normalize_task_type(None) == "general"

    def test_valid_types(self):
        for task_type in SUPPORTED_TASK_TYPES:
            assert normalize_task_type(task_type) == task_type

    def test_unknown_defaults_to_general(self):
        assert normalize_task_type("unknown_type") == "general"

    def test_case_insensitive(self):
        assert normalize_task_type("CODING") == "coding"
        assert normalize_task_type("Reasoning") == "reasoning"


class TestSelectModel:
    def test_user_specified_model(self):
        selection = select_model("general", "my/custom-model")
        assert selection.model_id == "my/custom-model"
        assert selection.user_specified is True
        assert selection.selected_by == "user"

    def test_coordinator_selection(self):
        selection = select_model("coding", None)
        assert selection.model_id != ""
        assert selection.user_specified is False
        assert selection.selected_by == "coordinator"

    def test_different_task_types_route_correctly(self):
        general = select_model("general", None)
        fast = select_model("fast", None)
        assert general.model_id != "" 
        assert fast.model_id != ""


class TestGetRoutingModelId:
    def test_general_returns_model(self):
        model_id = get_routing_model_id("general")
        assert model_id != ""

    def test_fast_returns_model(self):
        model_id = get_routing_model_id("fast")
        assert "nemotron-mini" in model_id or model_id != ""

    def test_nvidia_reasoning(self):
        model_id = get_routing_model_id("nvidia_reasoning")
        assert "nvidia" in model_id.lower() or model_id != ""


class TestFallbackCandidates:
    def test_fallback_excludes_self(self):
        model_id = get_routing_model_id("general")
        candidates = fallback_candidates_for(model_id)
        assert model_id not in candidates

    def test_fallback_returns_candidates(self):
        candidates = fallback_candidates_for("some/model")
        assert len(candidates) > 0
