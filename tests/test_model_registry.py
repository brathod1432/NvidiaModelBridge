"""Tests for model registry module."""

from __future__ import annotations

from app.model_registry import (
    CURATED_MODELS,
    RECOMMENDED_ROUTE_MAP,
    ModelEntry,
    get_avoid_models,
    get_curated_models,
    get_model_by_id,
    get_models_by_status,
    get_partial_models,
    get_priority_models,
    get_recommended_models,
    get_retest_models,
    model_to_dict,
)


class TestModelEntry:
    def test_benchmark_summary(self):
        entry = ModelEntry(
            id="test/model",
            display_name="Test",
            provider="Test",
            status="recommended",
            benchmark_pass_count=3,
            benchmark_total_count=5,
        )
        assert entry.benchmark_summary == "3/5"


class TestCuratedModels:
    def test_curated_models_not_empty(self):
        assert len(CURATED_MODELS) > 0

    def test_all_models_have_required_fields(self):
        for model in CURATED_MODELS:
            assert model.id
            assert model.display_name
            assert model.provider
            assert model.status

    def test_priority_models_exist(self):
        priority = get_priority_models()
        assert len(priority) >= 1

    def test_recommended_route_map_keys(self):
        required_keys = {"default/general", "fast", "general_fallback"}
        assert required_keys.issubset(set(RECOMMENDED_ROUTE_MAP.keys()))


class TestModelLookup:
    def test_get_model_by_id_found(self):
        model = get_model_by_id("qwen/qwen3-next-80b-a3b-instruct")
        assert model is not None
        assert model.provider == "Qwen"

    def test_get_model_by_id_not_found(self):
        assert get_model_by_id("nonexistent/model") is None

    def test_get_avoid_models(self):
        avoid = get_avoid_models()
        for model in avoid:
            assert model.status == "avoid"

    def test_get_partial_models(self):
        partial = get_partial_models()
        for model in partial:
            assert model.status == "partial"

    def test_model_to_dict(self):
        model = CURATED_MODELS[0]
        d = model_to_dict(model)
        assert isinstance(d, dict)
        assert "id" in d
        assert "display_name" in d
