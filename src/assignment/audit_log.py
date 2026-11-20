"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], deque[tuple[str, str, float]]] = (
            defaultdict(deque)
        )

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record the input and start time for a pending request."""
        self._open[user_id, request_id].append((text, utc_now_iso(), time.monotonic()))

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Append the completed interaction and its guardrail decision."""
        key = (user_id, request_id)
        pending = self._open.get(key)
        input_text = started_at = latency_ms = None
        if pending:
            input_text, started_at, started_clock = pending.popleft()
            latency_ms = round((time.monotonic() - started_clock) * 1000, 2)
            if not pending:
                del self._open[key]

        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": input_text,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
