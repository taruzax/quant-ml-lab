"""Collision-resistant, local, immutable research artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import polars as pl


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class ArtifactStore:
    """Write-once local artifacts underneath one run directory."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ValueError("run_id must be a single nonempty path component")
        return self.root / "runs" / run_id

    def create_run(self, run_id: str | None = None) -> str:
        resolved = run_id or make_run_id()
        path = self.run_dir(resolved)
        if path.exists():
            raise FileExistsError(f"Run already exists: {resolved}")
        path.mkdir(parents=True)
        return resolved

    def _path(self, run_id: str, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("Artifact path escapes the run directory")
        return self.run_dir(run_id) / relative

    def write_bytes(self, run_id: str, relative_path: str | Path, data: bytes) -> dict[str, Any]:
        path = self._path(run_id, relative_path)
        if path.exists():
            raise FileExistsError(f"Artifact already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": str(path.relative_to(self.run_dir(run_id))), "bytes": len(data), "sha256": sha256_bytes(data)}

    def write_json(self, run_id: str, relative_path: str | Path, value: Any) -> dict[str, Any]:
        return self.write_bytes(run_id, relative_path, canonical_json_bytes(value))

    def write_parquet(self, run_id: str, relative_path: str | Path, frame: pl.DataFrame) -> dict[str, Any]:
        if pl.Object in frame.schema.values():
            raise TypeError(f"Cannot persist object-typed artifact {relative_path}: {frame.schema}")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            frame.write_parquet(temporary)
            return self.write_bytes(run_id, relative_path, temporary.read_bytes())
        finally:
            temporary.unlink(missing_ok=True)

    def finalize(self, run_id: str, manifest: dict[str, Any], *, required_artifacts: list[str] | None = None) -> dict[str, Any]:
        required = required_artifacts or []
        missing = [item for item in required if not self._path(run_id, item).exists()]
        if missing:
            raise FileNotFoundError(f"Cannot finalize incomplete run; missing artifacts: {missing}")
        completed = dict(manifest)
        completed.update({"schema_version": "run-manifest.v1", "run_id": run_id, "status": "completed"})
        self.write_json(run_id, "manifest.json", completed)
        return completed

    def mark_failed(self, run_id: str, error: str, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        failed = dict(manifest or {})
        failed.update({"schema_version": "run-manifest.v1", "run_id": run_id, "status": "failed", "error": error})
        path = self._path(run_id, "manifest.json")
        if path.exists():
            raise FileExistsError(f"Run already has a terminal manifest: {run_id}")
        self.write_json(run_id, "manifest.json", failed)
        return failed

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id, "manifest.json")
        if not path.exists():
            raise FileNotFoundError(f"Run manifest not found: {run_id}")
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "completed":
            raise ValueError(f"Run is not complete: {run_id}")
        return manifest

    def verify_artifact(self, run_id: str, record: dict[str, Any]) -> Path:
        path = self._path(run_id, record["path"])
        if not path.exists() or path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Artifact is missing or changed: {record['path']}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {record['path']}")
        return path
