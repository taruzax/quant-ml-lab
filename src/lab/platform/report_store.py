"""Immutable locators for generated reports."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.platform.artifacts import canonical_json_bytes, sha256_file


class ReportStore:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root) / "reports"
        self.root.mkdir(parents=True, exist_ok=True)

    def write_locator(self, *, report_path: str | Path, run_id: str, manifest_path: str | Path, evidence: dict[str, Any] | None = None, report_type: str = "development") -> dict[str, Any]:
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"Report does not exist: {path}")
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(f"Run manifest does not exist: {manifest}")
        if evidence:
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_file() or sha256_file(evidence_path) != evidence["sha256"]:
                raise ValueError("Evidence sidecar checksum does not match its referenced file")
        report_checksum = sha256_file(path)
        manifest_checksum = sha256_file(manifest)
        locator_identity = {
            "report_checksum": report_checksum,
            "report_path": str(path.resolve()),
            "source_run_id": run_id,
            "source_manifest_path": str(manifest.resolve()),
            "source_manifest_checksum": manifest_checksum,
            "evidence": evidence,
            "report_type": report_type,
        }
        payload = {
            "schema_version": "report-locator.v1",
            "report_id": f"report-{hashlib.sha256(canonical_json_bytes(locator_identity)).hexdigest()[:24]}",
            "report_checksum": report_checksum,
            "source_run_id": run_id,
            "source_manifest_path": str(manifest.resolve()),
            "source_manifest_checksum": manifest_checksum,
            "evidence": evidence,
            "report_type": report_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_path": str(path.resolve()),
        }
        locator_path = self.root / f"{payload['report_id']}.json"
        if locator_path.exists():
            existing = json.loads(locator_path.read_text())
            immutable_fields = (
                "report_id", "report_checksum", "source_run_id", "source_manifest_path",
                "source_manifest_checksum", "evidence", "report_type", "report_path",
            )
            if any(existing.get(field) != payload[field] for field in immutable_fields):
                raise ValueError(f"Report locator identity collision: {payload['report_id']}")
            payload = existing
        else:
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(canonical_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, locator_path)
            except FileExistsError:
                existing = json.loads(locator_path.read_text())
                immutable_fields = (
                    "report_id", "report_checksum", "source_run_id", "source_manifest_path",
                    "source_manifest_checksum", "evidence", "report_type", "report_path",
                )
                if any(existing.get(field) != payload[field] for field in immutable_fields):
                    raise ValueError(f"Report locator identity collision: {payload['report_id']}")
                payload = existing
            finally:
                temporary.unlink(missing_ok=True)
        payload["locator_path"] = str(locator_path)
        return payload

    def load_locator(self, report_id: str) -> dict[str, Any]:
        if not report_id or Path(report_id).name != report_id or report_id in {".", ".."}:
            raise ValueError("report_id must be a single nonempty path component")
        locator_path = self.root / f"{report_id}.json"
        if not locator_path.exists():
            raise FileNotFoundError(f"Report locator not found: {report_id}")
        payload = json.loads(locator_path.read_text())
        if payload.get("schema_version") != "report-locator.v1" or payload.get("report_id") != report_id:
            raise ValueError(f"Invalid report locator: {report_id}")
        report_path = Path(payload["report_path"])
        if not report_path.is_file() or sha256_file(report_path) != payload["report_checksum"]:
            raise ValueError(f"Report checksum mismatch: {report_id}")
        manifest_path = Path(payload["source_manifest_path"])
        if not manifest_path.is_file() or sha256_file(manifest_path) != payload["source_manifest_checksum"]:
            raise ValueError(f"Source manifest checksum mismatch: {report_id}")
        evidence = payload.get("evidence")
        if evidence:
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_file() or sha256_file(evidence_path) != evidence["sha256"]:
                raise ValueError(f"Evidence sidecar checksum mismatch: {report_id}")
        return {**payload, "locator_path": str(locator_path)}
