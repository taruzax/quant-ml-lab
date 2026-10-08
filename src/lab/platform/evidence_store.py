"""Write-once campaign evidence sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lab.platform.artifacts import canonical_json_bytes


class EvidenceStore:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root) / "evidence"
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, report: dict[str, Any]) -> dict[str, Any]:
        payload = canonical_json_bytes(report)
        evidence_id = f"evidence-{hashlib.sha256(payload).hexdigest()[:24]}"
        path = self.root / f"{evidence_id}.json"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Evidence identity collision: {evidence_id}")
        if not path.exists():
            path.write_bytes(payload)
        return {"evidence_id": evidence_id, "path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}

    def load(self, evidence_id: str) -> dict[str, Any]:
        path = self.root / f"{evidence_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Evidence sidecar not found: {evidence_id}")
        payload = path.read_bytes()
        expected = evidence_id.removeprefix("evidence-")
        if not evidence_id.startswith("evidence-") or not hashlib.sha256(payload).hexdigest().startswith(expected):
            raise ValueError(f"Evidence sidecar checksum mismatch: {evidence_id}")
        return json.loads(payload)
