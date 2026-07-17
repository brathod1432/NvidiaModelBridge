"""Tests for analytics module."""

from __future__ import annotations

from app.analytics import AnalyticsCollector


class TestAnalyticsCollector:
    def test_record_and_dashboard(self):
        collector = AnalyticsCollector(max_records=100)
        collector.record(
            model_id="test/model",
            task_type="general",
            success=True,
            latency_seconds=1.5,
            endpoint="/ask",
        )
        data = collector.get_dashboard_data(hours=1)
        assert data["total_requests"] == 1
        assert data["successful_requests"] == 1

    def test_empty_dashboard(self):
        collector = AnalyticsCollector()
        data = collector.get_dashboard_data(hours=1)
        assert data["total_requests"] == 0

    def test_max_records(self):
        collector = AnalyticsCollector(max_records=5)
        for i in range(10):
            collector.record("m", "g", True, 1.0)
        assert collector.get_record_count() == 5

    def test_clear(self):
        collector = AnalyticsCollector()
        collector.record("m", "g", True, 1.0)
        collector.clear()
        assert collector.get_record_count() == 0
