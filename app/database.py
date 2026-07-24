"""SQLite database for audit trails and usage analytics."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from app.config import get_project_root

logger = logging.getLogger(__name__)


DB_PATH = get_project_root() / "data" / "nvidia_bridge.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL,
    description TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL DEFAULT 'POST',
    client_ip TEXT DEFAULT '',
    model_id TEXT DEFAULT '',
    task_type TEXT DEFAULT '',
    prompt_length INTEGER DEFAULT 0,
    response_length INTEGER DEFAULT 0,
    latency_seconds REAL DEFAULT 0,
    status_code INTEGER DEFAULT 0,
    success INTEGER DEFAULT 0,
    fallback_used INTEGER DEFAULT 0,
    fallback_model TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    cache_hit INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    task_type TEXT DEFAULT '',
    success INTEGER DEFAULT 0,
    latency_seconds REAL DEFAULT 0,
    tokens_used INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audit_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    timestamp REAL NOT NULL,
    total_models_tested INTEGER DEFAULT 0,
    total_tasks INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    total_failures INTEGER DEFAULT 0,
    report_json TEXT DEFAULT '',
    duration_seconds REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    previous_state TEXT DEFAULT '',
    new_state TEXT DEFAULT '',
    failure_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_request_log_timestamp ON request_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_request_log_model ON request_log(model_id);
CREATE INDEX IF NOT EXISTS idx_model_usage_model ON model_usage(model_id);
CREATE INDEX IF NOT EXISTS idx_model_usage_timestamp ON model_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_runs_timestamp ON audit_runs(timestamp);
"""

_MIGRATIONS = {
    1: ("Initial schema", _SCHEMA),
    2: ("Add job timeout tracking", """
        ALTER TABLE request_log ADD COLUMN job_id TEXT DEFAULT '';
    """),
}


def _get_current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()
        return row[0] if row[0] is not None else 0
    except sqlite3.OperationalError:
        return 0


def _apply_migration(conn: sqlite3.Connection, version: int, description: str, sql: str) -> None:
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO schema_versions (version, applied_at, description) VALUES (?, ?, ?)",
        (version, time.time(), description),
    )
    conn.commit()


def ensure_db_dir() -> None:
    """Ensure the database directory exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with WAL mode for concurrent reads."""
    ensure_db_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database schema and apply pending migrations."""
    with get_db_connection() as conn:
        current = _get_current_version(conn)
        if current == 0:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT INTO schema_versions (version, applied_at, description) VALUES (?, ?, ?)",
                (1, time.time(), "Initial schema"),
            )
            conn.commit()
            current = 1
        for ver in sorted(_MIGRATIONS.keys()):
            if ver > current:
                desc, sql = _MIGRATIONS[ver]
                try:
                    _apply_migration(conn, ver, desc, sql)
                    logger.info("migration_applied", extra={"version": ver, "description": desc})
                except Exception as exc:
                    logger.warning("migration_failed", extra={"version": ver, "error": str(exc)})


def log_request(
    request_id: str,
    endpoint: str,
    method: str = "POST",
    client_ip: str = "",
    model_id: str = "",
    task_type: str = "",
    prompt_length: int = 0,
    response_length: int = 0,
    latency_seconds: float = 0.0,
    status_code: int = 0,
    success: bool = False,
    fallback_used: bool = False,
    fallback_model: str = "",
    error_message: str = "",
    cache_hit: bool = False,
) -> None:
    """Log an API request to the database."""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO request_log
            (request_id, timestamp, endpoint, method, client_ip, model_id,
             task_type, prompt_length, response_length, latency_seconds,
             status_code, success, fallback_used, fallback_model,
             error_message, cache_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id, time.time(), endpoint, method, client_ip,
                model_id, task_type, prompt_length, response_length,
                latency_seconds, status_code, int(success),
                int(fallback_used), fallback_model, error_message,
                int(cache_hit),
            ),
        )
        conn.commit()


def log_model_usage(
    model_id: str,
    task_type: str = "",
    success: bool = False,
    latency_seconds: float = 0.0,
    tokens_used: int = 0,
) -> None:
    """Log model usage statistics."""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO model_usage
            (model_id, timestamp, task_type, success, latency_seconds, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (model_id, time.time(), task_type, int(success), latency_seconds, tokens_used),
        )
        conn.commit()


def log_audit_run(
    run_id: str,
    total_models_tested: int = 0,
    total_tasks: int = 0,
    total_successes: int = 0,
    total_failures: int = 0,
    report_json: str = "",
    duration_seconds: float = 0.0,
) -> None:
    """Log a benchmark audit run."""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO audit_runs
            (run_id, timestamp, total_models_tested, total_tasks,
             total_successes, total_failures, report_json, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, time.time(), total_models_tested, total_tasks,
                total_successes, total_failures, report_json, duration_seconds,
            ),
        )
        conn.commit()


def log_circuit_breaker_event(
    model_id: str,
    event_type: str,
    previous_state: str = "",
    new_state: str = "",
    failure_count: int = 0,
) -> None:
    """Log a circuit breaker state change."""
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO circuit_breaker_events
            (model_id, timestamp, event_type, previous_state, new_state, failure_count)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (model_id, time.time(), event_type, previous_state, new_state, failure_count),
        )
        conn.commit()


def get_usage_stats(hours: int = 24) -> dict[str, Any]:
    """Get usage statistics for the last N hours."""
    cutoff = time.time() - (hours * 3600)
    with get_db_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]

        successful = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE timestamp > ? AND success = 1",
            (cutoff,),
        ).fetchone()[0]

        avg_latency = conn.execute(
            "SELECT AVG(latency_seconds) FROM request_log WHERE timestamp > ? AND success = 1",
            (cutoff,),
        ).fetchone()[0]

        model_breakdown = conn.execute(
            """SELECT model_id, COUNT(*) as count, AVG(latency_seconds) as avg_latency,
            SUM(success) as successes
            FROM request_log WHERE timestamp > ? AND model_id != ''
            GROUP BY model_id ORDER BY count DESC LIMIT 20""",
            (cutoff,),
        ).fetchall()

        cache_hits = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE timestamp > ? AND cache_hit = 1",
            (cutoff,),
        ).fetchone()[0]

        fallbacks = conn.execute(
            "SELECT COUNT(*) FROM request_log WHERE timestamp > ? AND fallback_used = 1",
            (cutoff,),
        ).fetchone()[0]

    return {
        "period_hours": hours,
        "total_requests": total,
        "successful_requests": successful,
        "failed_requests": total - successful,
        "success_rate": round(successful / total, 4) if total > 0 else 0.0,
        "average_latency_seconds": round(avg_latency or 0, 4),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total, 4) if total > 0 else 0.0,
        "fallback_activations": fallbacks,
        "model_breakdown": [
            {
                "model_id": row["model_id"],
                "request_count": row["count"],
                "avg_latency": round(row["avg_latency"] or 0, 4),
                "successes": row["successes"],
            }
            for row in model_breakdown
        ],
    }


def get_recent_errors(limit: int = 50) -> list[dict[str, Any]]:
    """Get recent error entries."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """SELECT request_id, timestamp, endpoint, model_id, task_type,
            error_message, status_code, latency_seconds
            FROM request_log WHERE success = 0
            ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
