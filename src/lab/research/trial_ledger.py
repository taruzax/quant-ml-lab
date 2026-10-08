"""Transactional local campaign and trial accounting."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrialLedger:
    def __init__(self, path: str | Path = "artifacts/trials.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS trials (
                logical_trial_id TEXT PRIMARY KEY,
                campaign TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                snapshot_hash TEXT NOT NULL,
                evaluation_hash TEXT NOT NULL,
                settings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                logical_trial_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                error TEXT,
                result_json TEXT,
                FOREIGN KEY(logical_trial_id) REFERENCES trials(logical_trial_id)
            )"""
        )

    def register_attempt(
        self,
        *,
        campaign: str,
        config_hash: str,
        snapshot_hash: str,
        evaluation_hash: str,
        settings: dict[str, Any],
        logical_trial_id: str | None = None,
    ) -> tuple[str, str, bool]:
        logical = logical_trial_id or self._logical_id(campaign, config_hash, snapshot_hash, evaluation_hash, settings)
        attempt = f"attempt-{uuid.uuid4().hex}"
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO trials VALUES (?, ?, ?, ?, ?, ?, ?)",
                (logical, campaign, config_hash, snapshot_hash, evaluation_hash, json.dumps(settings, default=str, sort_keys=True), _now()),
            )
            self.connection.execute(
                "INSERT INTO attempts VALUES (?, ?, 'running', ?, NULL, NULL, NULL)",
                (attempt, logical, _now()),
            )
        return logical, attempt, self._attempt_count(logical) > 1

    def update_attempt(self, attempt_id: str, *, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        if status not in {"running", "completed", "failed", "cancelled"}:
            raise ValueError(f"Invalid attempt status: {status}")
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET status=?, finished_at=?, error=?, result_json=? WHERE attempt_id=?",
                (status, None if status == "running" else _now(), error, json.dumps(result, default=str, sort_keys=True) if result else None, attempt_id),
            )

    def trials(self, campaign: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT logical_trial_id, campaign, config_hash, snapshot_hash, evaluation_hash, settings_json, created_at FROM trials"
        params: tuple[Any, ...] = ()
        if campaign is not None:
            query += " WHERE campaign=?"
            params = (campaign,)
        rows = self.connection.execute(query + " ORDER BY created_at", params).fetchall()
        return [
            {
                "logical_trial_id": row[0],
                "campaign": row[1],
                "config_hash": row[2],
                "snapshot_hash": row[3],
                "evaluation_hash": row[4],
                "settings": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]

    def attempts(self, logical_trial_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT attempt_id, logical_trial_id, status, started_at, finished_at, error, result_json FROM attempts"
        params: tuple[Any, ...] = ()
        if logical_trial_id is not None:
            query += " WHERE logical_trial_id=?"
            params = (logical_trial_id,)
        rows = self.connection.execute(query + " ORDER BY started_at", params).fetchall()
        return [
            {
                "attempt_id": row[0],
                "logical_trial_id": row[1],
                "status": row[2],
                "started_at": row[3],
                "finished_at": row[4],
                "error": row[5],
                "result": json.loads(row[6]) if row[6] else None,
            }
            for row in rows
        ]

    def comparable_evidence(
        self,
        *,
        campaign: str | None = None,
        snapshot_hash: str,
        evaluation_hash: str,
        frequency: str,
        cost_treatment: str | None = None,
        task: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return comparable logical trials and retain attempts for exclusion reporting."""
        query = "SELECT logical_trial_id, campaign, config_hash, snapshot_hash, evaluation_hash, settings_json, created_at FROM trials WHERE snapshot_hash=? AND evaluation_hash=?"
        params: list[Any] = [snapshot_hash, evaluation_hash]
        if campaign is not None:
            query += " AND campaign=?"
            params.append(campaign)
        rows = self.connection.execute(query + " ORDER BY created_at", tuple(params)).fetchall()
        evidence = []
        for row in rows:
            settings = json.loads(row[5])
            if settings.get("frequency", frequency) != frequency:
                continue
            if cost_treatment is not None and settings.get("cost_treatment") != cost_treatment:
                continue
            if task is not None and settings.get("task") != task:
                continue
            attempts = self.attempts(row[0])
            evidence.append({
                "logical_trial_id": row[0],
                "campaign": row[1],
                "config_hash": row[2],
                "snapshot_hash": row[3],
                "evaluation_hash": row[4],
                "settings": settings,
                "created_at": row[6],
                "attempts": attempts,
            })
        return evidence

    def comparable_return_series(
        self,
        *,
        campaign: str | None = None,
        snapshot_hash: str,
        evaluation_hash: str,
        frequency: str,
        cost_treatment: str | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Select one verified successful attempt per logical trial."""
        evidence = self.comparable_evidence(
            campaign=campaign,
            snapshot_hash=snapshot_hash,
            evaluation_hash=evaluation_hash,
            frequency=frequency,
            cost_treatment=cost_treatment,
            task=task,
        )
        complete: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for trial in evidence:
            successful = [
                attempt for attempt in trial["attempts"]
                if attempt["status"] == "completed" and (attempt.get("result") or {}).get("development_net_returns")
            ]
            selected = successful[-1] if successful else None
            if selected is not None:
                complete.append({"logical_trial_id": trial["logical_trial_id"], "attempt_id": selected["attempt_id"], "returns": selected["result"]["development_net_returns"]})
            for attempt in trial["attempts"]:
                if selected is not None and attempt["attempt_id"] == selected["attempt_id"]:
                    continue
                exclusions.append({
                    "logical_trial_id": trial["logical_trial_id"],
                    "attempt_id": attempt["attempt_id"],
                    "status": attempt["status"],
                    "reason": "duplicate successful attempt" if attempt["status"] == "completed" else attempt.get("error") or "missing returns",
                })
        return {"complete": complete, "exclusions": exclusions, "population": len(evidence)}

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _logical_id(campaign: str, config_hash: str, snapshot_hash: str, evaluation_hash: str, settings: dict[str, Any]) -> str:
        payload = json.dumps([campaign, config_hash, snapshot_hash, evaluation_hash, settings], default=str, sort_keys=True)
        return f"trial-{uuid.uuid5(uuid.NAMESPACE_URL, payload).hex}"

    def _attempt_count(self, logical_trial_id: str) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM attempts WHERE logical_trial_id=?", (logical_trial_id,)).fetchone()[0])
