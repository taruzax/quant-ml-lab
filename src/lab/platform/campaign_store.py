"""Durable campaign plans and variation status records."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.core.contracts import CampaignPlan, CampaignResultRef, CampaignRunEntry
from lab.platform.artifacts import canonical_json_bytes


class CampaignStore:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root) / "campaigns"
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.root / "campaigns.sqlite", timeout=30)
        self.connection.execute("CREATE TABLE IF NOT EXISTS campaign_runs (campaign_id TEXT NOT NULL, variation_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS campaign_statuses (campaign_id TEXT NOT NULL, variation_id TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY(campaign_id, variation_id))")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS campaign_results (campaign_id TEXT NOT NULL, state_hash TEXT NOT NULL, generation INTEGER NOT NULL, path TEXT NOT NULL, PRIMARY KEY(campaign_id, state_hash), UNIQUE(campaign_id, generation))"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS campaign_claims (campaign_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at TEXT NOT NULL, updated_at TEXT NOT NULL, fencing_generation INTEGER NOT NULL DEFAULT 0)"
        )
        claim_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(campaign_claims)")}
        if "fencing_generation" not in claim_columns:
            self.connection.execute("ALTER TABLE campaign_claims ADD COLUMN fencing_generation INTEGER NOT NULL DEFAULT 0")
        self.connection.commit()

    def write_plan(self, plan: CampaignPlan) -> Path:
        path = self.root / f"{plan.campaign_id}.json"
        data = plan.canonical_json().encode("utf-8")
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Campaign plan collision: {plan.campaign_id}")
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise ValueError(f"Campaign plan collision: {plan.campaign_id}")
            finally:
                temporary.unlink(missing_ok=True)
        return path

    def upsert(
        self, entry: CampaignRunEntry, *, owner_id: str | None = None,
        fencing_generation: int | None = None,
    ) -> None:
        self._validate_mutation_credentials(owner_id, fencing_generation)
        campaign_id = entry.planned_identity
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._assert_owner(campaign_id, owner_id, fencing_generation)
            row = self.connection.execute(
                "SELECT payload_json FROM campaign_statuses WHERE campaign_id=? AND variation_id=?",
                (campaign_id, entry.variation_id),
            ).fetchone()
            existing = CampaignRunEntry.model_validate_json(row[0]) if row else None
            if existing is not None:
                if existing.status == "completed":
                    if entry.status != "completed" or entry.run_id != existing.run_id:
                        raise ValueError(f"Completed campaign variation is immutable: {entry.variation_id}")
                    self.connection.commit()
                    return
                if entry.previous_status != existing.status:
                    raise ValueError(f"Campaign status transition must reference current state {existing.status!r}")
                if entry.created_at != existing.created_at:
                    raise ValueError("Campaign variation created_at is immutable across status transitions")
            elif entry.status != "planned" or entry.previous_status is not None:
                raise ValueError("Campaign variations must be initialized in planned status")
            with self.connection:
                self.connection.execute(
                    "INSERT INTO campaign_statuses(campaign_id, variation_id, payload_json) VALUES (?, ?, ?) ON CONFLICT(campaign_id, variation_id) DO UPDATE SET payload_json=excluded.payload_json",
                    (campaign_id, entry.variation_id, entry.model_dump_json()),
                )
        except Exception:
            self.connection.rollback()
            raise

    def entries(self, campaign_id: str) -> list[CampaignRunEntry]:
        rows = self.connection.execute("SELECT payload_json FROM campaign_statuses WHERE campaign_id=? ORDER BY variation_id", (campaign_id,)).fetchall()
        return [CampaignRunEntry.model_validate_json(row[0]) for row in rows]

    def acquire_claim(
        self,
        campaign_id: str,
        owner_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("Campaign claim lease_seconds must be positive")
        if not campaign_id.strip() or not owner_id.strip():
            raise ValueError("Campaign claim requires nonempty campaign_id and owner_id")
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("Campaign claim time must be timezone-aware")
        expires = datetime.fromtimestamp(instant.timestamp() + lease_seconds, timezone.utc)
        now_text = instant.astimezone(timezone.utc).isoformat()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT owner_id, expires_at, fencing_generation FROM campaign_claims WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
            if row and datetime.fromisoformat(row[1]) > instant:
                self.connection.rollback()
                return False
            generation = (int(row[2]) + 1) if row else 1
            self.connection.execute(
                "INSERT INTO campaign_claims(campaign_id, owner_id, expires_at, updated_at, fencing_generation) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(campaign_id) DO UPDATE SET owner_id=excluded.owner_id, expires_at=excluded.expires_at, updated_at=excluded.updated_at, fencing_generation=excluded.fencing_generation",
                (campaign_id, owner_id, expires.isoformat(), now_text, generation),
            )
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def renew_claim(
        self, campaign_id: str, owner_id: str, *, lease_seconds: float,
        fencing_generation: int | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("Campaign claim lease_seconds must be positive")
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + lease_seconds, timezone.utc)
        cursor = self.connection.execute(
            "UPDATE campaign_claims SET expires_at=?, updated_at=? WHERE campaign_id=? AND owner_id=? AND fencing_generation=? AND expires_at>?",
            (expires.isoformat(), now.isoformat(), campaign_id, owner_id, fencing_generation or 0, now.isoformat()),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self, campaign_id: str, owner_id: str, *, fencing_generation: int | None = None,
    ) -> bool:
        cursor = self.connection.execute(
            "DELETE FROM campaign_claims WHERE campaign_id=? AND owner_id=? AND fencing_generation=?",
            (campaign_id, owner_id, fencing_generation or 0),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def claim(self, campaign_id: str) -> dict[str, str | int] | None:
        row = self.connection.execute(
            "SELECT owner_id, expires_at, updated_at, fencing_generation FROM campaign_claims WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        return {"owner_id": row[0], "expires_at": row[1], "updated_at": row[2], "fencing_generation": row[3]} if row else None

    def assert_claim(self, campaign_id: str, owner_id: str, fencing_generation: int) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        row = self.connection.execute(
            "SELECT 1 FROM campaign_claims WHERE campaign_id=? AND owner_id=? AND fencing_generation=? AND expires_at>?",
            (campaign_id, owner_id, fencing_generation, now),
        ).fetchone()
        return row is not None

    @staticmethod
    def _validate_mutation_credentials(owner_id: str | None, fencing_generation: int | None) -> None:
        if (owner_id is None) != (fencing_generation is None):
            raise ValueError("Campaign mutations require both owner_id and fencing_generation")

    def _assert_owner(
        self, campaign_id: str, owner_id: str | None, fencing_generation: int | None,
    ) -> None:
        if owner_id is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        row = self.connection.execute(
            "SELECT 1 FROM campaign_claims WHERE campaign_id=? AND owner_id=? AND fencing_generation=? AND expires_at>?",
            (campaign_id, owner_id, fencing_generation, now),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Campaign claim ownership lost: {campaign_id}")

    def finalize(
        self, plan: CampaignPlan, *, owner_id: str | None = None,
        fencing_generation: int | None = None,
    ) -> CampaignResultRef | None:
        self._validate_mutation_credentials(owner_id, fencing_generation)
        plan_path = self.write_plan(plan)
        plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._assert_owner(plan.campaign_id, owner_id, fencing_generation)
            self._recover_result_index(plan.campaign_id)
            entries = self.entries(plan.campaign_id)
            terminal = {"completed", "failed", "cancelled"}
            if len(entries) != len(plan.variations) or any(entry.status not in terminal for entry in entries):
                self.connection.rollback()
                return None
            results = [entry.model_dump(mode="json") for entry in entries]
            state = {"campaign_id": plan.campaign_id, "plan_hash": plan_hash, "entries": results}
            state_hash = hashlib.sha256(canonical_json_bytes(state)).hexdigest()
            indexed = self.connection.execute(
                "SELECT path FROM campaign_results WHERE campaign_id=? AND state_hash=?",
                (plan.campaign_id, state_hash),
            ).fetchone()
            if indexed:
                path = self.root / indexed[0]
                ref = self._read_result_ref(path, plan.campaign_id, state_hash)
                self.connection.commit()
                return ref
            generation = int(self.connection.execute(
                "SELECT COALESCE(MAX(generation), 0) + 1 FROM campaign_results WHERE campaign_id=?",
                (plan.campaign_id,),
            ).fetchone()[0])
            result_id = f"campaign-result-{state_hash[:24]}"
            path = self.root / f"{plan.campaign_id}-result-{state_hash}.json"
            existing = self._read_result_ref(path, plan.campaign_id, state_hash) if path.exists() else None
            if existing is None:
                payload = {
                    "schema_version": "campaign-result.v2",
                    **state,
                    "state_hash": state_hash,
                    "generation": generation,
                    "result_id": result_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                checksum = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
                self._write_create_only(path, canonical_json_bytes({**payload, "checksum": checksum}))
                ref = CampaignResultRef(
                    campaign_id=plan.campaign_id, result_id=result_id, plan_hash=plan_hash,
                    bundle_path=str(path.resolve()), checksum=checksum,
                    created_at=datetime.fromisoformat(payload["created_at"]),
                )
            else:
                ref = existing
                generation = int(json.loads(path.read_text()).get("generation", generation))
            self.connection.execute(
                "INSERT INTO campaign_results(campaign_id, state_hash, generation, path) VALUES (?, ?, ?, ?)",
                (plan.campaign_id, state_hash, generation, path.name),
            )
            self.connection.commit()
            return ref
        except Exception:
            self.connection.rollback()
            raise

    def latest_result(self, campaign_id: str) -> CampaignResultRef | None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._recover_result_index(campaign_id)
            row = self.connection.execute(
                "SELECT state_hash, path FROM campaign_results WHERE campaign_id=? ORDER BY generation DESC, state_hash DESC LIMIT 1",
                (campaign_id,),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            result = self._read_result_ref(self.root / row[1], campaign_id, row[0])
            self.connection.commit()
            return result
        except Exception:
            self.connection.rollback()
            raise

    def _recover_result_index(self, campaign_id: str) -> None:
        paths = sorted(self.root.glob(f"{campaign_id}-result-*.json"))
        legacy_path = self.root / f"{campaign_id}-result.json"
        if legacy_path.exists():
            paths.insert(0, legacy_path)
        for path in paths:
            payload = self._verified_result_payload(path, campaign_id)
            state = {"campaign_id": campaign_id, "plan_hash": payload["plan_hash"], "entries": payload["entries"]}
            state_hash = hashlib.sha256(canonical_json_bytes(state)).hexdigest()
            if payload.get("state_hash", state_hash) != state_hash:
                raise ValueError(f"Campaign result identity collision: {path.name}")
            generation = int(payload.get("generation", 0))
            row = self.connection.execute(
                "SELECT generation, path FROM campaign_results WHERE campaign_id=? AND state_hash=?",
                (campaign_id, state_hash),
            ).fetchone()
            if row and (int(row[0]) != generation or row[1] != path.name):
                raise ValueError(f"Campaign result identity collision: {path.name}")
            if not row:
                conflicting = self.connection.execute(
                    "SELECT state_hash, path FROM campaign_results WHERE campaign_id=? AND generation=?",
                    (campaign_id, generation),
                ).fetchone()
                if conflicting and conflicting[0] != state_hash:
                    raise ValueError(f"Campaign result generation collision: {campaign_id}:{generation}")
                self.connection.execute(
                    "INSERT INTO campaign_results(campaign_id, state_hash, generation, path) VALUES (?, ?, ?, ?)",
                    (campaign_id, state_hash, generation, path.name),
                )

    def _read_result_ref(self, path: Path, campaign_id: str, state_hash: str) -> CampaignResultRef:
        payload = self._verified_result_payload(path, campaign_id)
        actual_state = {"campaign_id": campaign_id, "plan_hash": payload["plan_hash"], "entries": payload["entries"]}
        actual_hash = hashlib.sha256(canonical_json_bytes(actual_state)).hexdigest()
        if actual_hash != state_hash or payload.get("state_hash", actual_hash) != actual_hash:
            raise ValueError(f"Campaign result identity collision: {path.name}")
        if payload.get("result_id") != f"campaign-result-{actual_hash[:24]}":
            raise ValueError(f"Campaign result identity collision: {path.name}")
        checksum = payload["checksum"]
        return CampaignResultRef(
            campaign_id=campaign_id, result_id=payload["result_id"], plan_hash=payload["plan_hash"],
            bundle_path=str(path.resolve()), checksum=checksum,
            created_at=datetime.fromisoformat(payload["created_at"]),
        )

    @staticmethod
    def _verified_result_payload(path: Path, campaign_id: str) -> dict[str, Any]:
        payload = json.loads(path.read_text())
        checksum = payload.pop("checksum", None)
        if checksum != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
            raise ValueError(f"Campaign result checksum mismatch: {campaign_id}")
        if payload.get("campaign_id") != campaign_id:
            raise ValueError(f"Campaign result identity collision: {campaign_id}")
        payload["checksum"] = checksum
        return payload

    def _write_create_only(self, path: Path, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Campaign result identity collision: {path.name}")
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        self.connection.close()
