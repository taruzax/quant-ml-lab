"""Create-only storage for canonical provider responses."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from lab.core.contracts import IngestionAttemptSummary, MarketSnapshot, SnapshotCatalogEntry
from lab.platform.artifacts import canonical_json_bytes, sha256_file

_IDENTITY_COLUMNS = ("ticker", "timestamp")


def _safe_component(value: str, field_name: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or any(char in value for char in "/\\"):
        raise ValueError(f"Invalid {field_name} path component: {value!r}")
    return value


def _canonicalize(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        raise ValueError("Provider response is empty")
    required = set(_IDENTITY_COLUMNS) | {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Provider response is missing columns: {missing}")
    frame = frame.sort(["ticker", "timestamp"])
    duplicate_keys = frame.group_by(list(_IDENTITY_COLUMNS)).len().filter(pl.col("len") > 1)
    if not duplicate_keys.is_empty():
        duplicated = frame.join(duplicate_keys.select(list(_IDENTITY_COLUMNS)), on=list(_IDENTITY_COLUMNS), how="inner")
        if duplicated.unique().height != duplicate_keys.height:
            raise ValueError("Provider response contains conflicting duplicate bars")
        frame = frame.unique(subset=list(_IDENTITY_COLUMNS), keep="first")
    return frame.sort(["ticker", "timestamp"])


class ProviderBatchStore:
    """Persist provider responses once and verify later reads by checksum."""

    def __init__(self, root: str | Path = "data/market_batches") -> None:
        self.root = Path(root)
        self.batch_root = self.root / "batches"
        self.batch_root.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        frame: pl.DataFrame,
        *,
        provider: str,
        stream_id: str,
        request: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider = _safe_component(provider, "provider")
        stream_id = _safe_component(stream_id, "stream_id")
        canonical = _canonicalize(frame)
        payload = canonical.to_dicts()
        identity = {
            "provider": provider,
            "stream_id": stream_id,
            "request": request,
            "provenance": provenance or {},
            "content": payload,
        }
        batch_id = f"batch-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]}"
        batch_dir = self.batch_root / batch_id
        bars_path = batch_dir / "bars.parquet"
        manifest_path = batch_dir / "manifest.json"
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            canonical.write_parquet(temporary)
            bars_bytes = temporary.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": "provider-batch.v1",
            "batch_id": batch_id,
            "provider": provider,
            "stream_id": stream_id,
            "request": request,
            "provenance": provenance or {},
            "rows": canonical.height,
            "bars_sha256": hashlib.sha256(bars_bytes).hexdigest(),
        }
        manifest["manifest_checksum"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        if batch_dir.exists():
            if not bars_path.exists() or not manifest_path.exists():
                raise ValueError(f"Incomplete existing provider batch: {batch_id}")
            existing = json.loads(manifest_path.read_text())
            if existing != manifest or sha256_file(bars_path) != manifest["bars_sha256"]:
                raise ValueError(f"Provider batch identity collision with different content: {batch_id}")
            return {**existing, "path": str(batch_dir)}
        staging = Path(tempfile.mkdtemp(prefix=f".{batch_id}-", dir=self.batch_root))
        try:
            self._create_file(staging / "bars.parquet", bars_bytes)
            self._create_file(staging / "manifest.json", canonical_json_bytes(manifest))
            try:
                os.rename(staging, batch_dir)
            except OSError:
                if not batch_dir.exists():
                    raise
                existing = json.loads((batch_dir / "manifest.json").read_text())
                if existing != manifest or sha256_file(batch_dir / "bars.parquet") != manifest["bars_sha256"]:
                    raise ValueError(f"Provider batch identity collision with different content: {batch_id}")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {**manifest, "path": str(batch_dir)}

    def load(self, batch_id: str) -> pl.DataFrame:
        batch_id = _safe_component(batch_id, "batch_id")
        batch_dir = self.batch_root / batch_id
        manifest_path = batch_dir / "manifest.json"
        bars_path = batch_dir / "bars.parquet"
        if not manifest_path.exists() or not bars_path.exists():
            raise FileNotFoundError(f"Provider batch is incomplete or missing: {batch_id}")
        manifest = json.loads(manifest_path.read_text())
        unsigned = {key: value for key, value in manifest.items() if key != "manifest_checksum"}
        if (
            manifest.get("batch_id") != batch_id
            or hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != manifest.get("manifest_checksum")
            or sha256_file(bars_path) != manifest.get("bars_sha256")
        ):
            raise ValueError(f"Provider batch checksum mismatch: {batch_id}")
        return pl.read_parquet(bars_path)

    @staticmethod
    def _create_file(path: Path, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise


class SnapshotCatalog:
    """Index and verify immutable snapshot bundles."""

    def __init__(self, root: str | Path = "data/snapshots", *, read_only: bool = False) -> None:
        self.root = Path(root)
        self.database = self.root / "catalog.sqlite"
        if read_only:
            if not self.database.is_file():
                raise FileNotFoundError(f"Snapshot catalog does not exist: {self.database}")
            self.connection = sqlite3.connect(f"{self.database.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.database, timeout=30)
            self.connection.execute(
                """CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                stream_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                calendar TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ticker_order_json TEXT NOT NULL,
                first_completed_bar TEXT NOT NULL,
                last_completed_bar TEXT NOT NULL,
                published_at TEXT NOT NULL,
                coverage_status TEXT NOT NULL,
                batch_ids_json TEXT NOT NULL,
                bundle_path TEXT NOT NULL,
                manifest_checksum TEXT NOT NULL
                )"""
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS ingestion_attempts (attempt_id TEXT PRIMARY KEY, stream_id TEXT NOT NULL, status TEXT NOT NULL, finished_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            self.connection.commit()
            self._recover_catalog()

    def _recover_catalog(self) -> None:
        indexed = {row[0] for row in self.connection.execute("SELECT snapshot_id FROM snapshots").fetchall()}
        for bundle in sorted(path for path in self.root.iterdir() if path.is_dir() and not path.name.startswith(".")):
            if bundle.name in indexed or not (bundle / "manifest.json").exists():
                continue
            manifest = json.loads((bundle / "manifest.json").read_text())
            entry = SnapshotCatalogEntry.model_validate({key: manifest[key] for key in SnapshotCatalogEntry.model_fields})
            self.verify(entry)
            self._index(entry)

    def publish(
        self,
        snapshot: MarketSnapshot,
        *,
        stream_id: str | None = None,
        selected_batch_ids: tuple[str, ...] = (),
        coverage_status: str = "accepted",
    ) -> SnapshotCatalogEntry:
        if coverage_status not in {"accepted", "rejected", "incomplete"}:
            raise ValueError(f"Unknown snapshot coverage status: {coverage_status}")
        snapshot_dir = self.root / _safe_component(snapshot.snapshot_id, "snapshot_id")
        bars_path = snapshot_dir / "bars.parquet"
        manifest_path = snapshot_dir / "manifest.json"
        first_bar = snapshot.bars["timestamp"].min()
        last_bar = snapshot.bars["timestamp"].max()
        if first_bar is None or last_bar is None:
            raise ValueError("Cannot publish a snapshot without bars")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            snapshot.bars.write_parquet(temporary)
            bars_bytes = temporary.read_bytes()
        finally:
            temporary.unlink(missing_ok=True)
        if snapshot_dir.exists():
            if not bars_path.exists() or not manifest_path.exists():
                raise ValueError(f"Incomplete existing snapshot bundle: {snapshot.snapshot_id}")
            existing = json.loads(manifest_path.read_text())
            if self._manifest_checksum(existing) != existing.get("manifest_checksum"):
                raise ValueError(f"Snapshot manifest checksum mismatch: {snapshot.snapshot_id}")
            if hashlib.sha256(bars_path.read_bytes()).hexdigest() != existing.get("bars_sha256"):
                raise ValueError(f"Snapshot bars checksum mismatch: {snapshot.snapshot_id}")
            if (
                existing.get("snapshot_hash") != snapshot.snapshot_hash
                or existing.get("bars_sha256") != hashlib.sha256(bars_bytes).hexdigest()
            ):
                raise ValueError(f"Snapshot identity collision with different content: {snapshot.snapshot_id}")
            entry = SnapshotCatalogEntry.model_validate({key: existing[key] for key in SnapshotCatalogEntry.model_fields})
            self._index(entry)
            return entry
        now = datetime.now(timezone.utc)
        entry = SnapshotCatalogEntry(
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.snapshot_hash,
            stream_id=stream_id or str(snapshot.provenance.get("stream_id", "research")),
            provider=str(snapshot.provenance.get("provider", snapshot.provenance.get("source", "unknown"))),
            calendar=snapshot.calendar,
            timeframe=snapshot.timeframe,
            ticker_order=snapshot.ticker_order,
            first_completed_bar=first_bar,
            last_completed_bar=last_bar,
            published_at=now,
            coverage_status=coverage_status,
            batch_ids=selected_batch_ids,
            bundle_path=str(snapshot_dir),
            manifest_checksum="pending",
        )
        manifest = entry.model_dump(mode="json")
        manifest["snapshot_provenance"] = snapshot.provenance
        manifest["validation_diagnostics"] = snapshot.validation_diagnostics
        manifest["source_manifest"] = snapshot.source_manifest
        manifest["bars_sha256"] = hashlib.sha256(bars_bytes).hexdigest()
        manifest_checksum = self._manifest_checksum(manifest)
        entry = entry.model_copy(update={"manifest_checksum": manifest_checksum})
        manifest = entry.model_dump(mode="json")
        manifest["snapshot_provenance"] = snapshot.provenance
        manifest["validation_diagnostics"] = snapshot.validation_diagnostics
        manifest["source_manifest"] = snapshot.source_manifest
        manifest["bars_sha256"] = hashlib.sha256(bars_bytes).hexdigest()
        manifest_bytes = canonical_json_bytes(manifest)
        staging = Path(tempfile.mkdtemp(prefix=f".{snapshot.snapshot_id}-", dir=self.root))
        try:
            self._create_file(staging / "bars.parquet", bars_bytes)
            self._create_file(staging / "manifest.json", manifest_bytes)
            try:
                os.rename(staging, snapshot_dir)
            except OSError:
                if not snapshot_dir.exists():
                    raise
                existing = json.loads(manifest_path.read_text())
                if self._manifest_checksum(existing) != existing.get("manifest_checksum"):
                    raise ValueError(f"Snapshot manifest checksum mismatch: {snapshot.snapshot_id}")
                if hashlib.sha256(bars_path.read_bytes()).hexdigest() != existing.get("bars_sha256"):
                    raise ValueError(f"Snapshot bars checksum mismatch: {snapshot.snapshot_id}")
                if (
                    existing.get("snapshot_hash") != snapshot.snapshot_hash
                    or existing.get("bars_sha256") != hashlib.sha256(bars_bytes).hexdigest()
                ):
                    raise ValueError(f"Snapshot identity collision with different content: {snapshot.snapshot_id}")
                entry = SnapshotCatalogEntry.model_validate({key: existing[key] for key in SnapshotCatalogEntry.model_fields})
            self._index(entry)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return entry

    def _index(self, entry: SnapshotCatalogEntry) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(snapshot_id) DO NOTHING",
                (
                    entry.snapshot_id,
                    entry.snapshot_hash,
                    entry.stream_id,
                    entry.provider,
                    entry.calendar,
                    entry.timeframe.value,
                    json.dumps(entry.ticker_order),
                    entry.first_completed_bar.isoformat(),
                    entry.last_completed_bar.isoformat(),
                    entry.published_at.isoformat(),
                    entry.coverage_status,
                    json.dumps(entry.batch_ids),
                    entry.bundle_path,
                    entry.manifest_checksum,
                ),
            )

    def resolve(self, snapshot_id: str) -> SnapshotCatalogEntry:
        row = self.connection.execute(
            "SELECT snapshot_id, snapshot_hash, stream_id, provider, calendar, timeframe, ticker_order_json, first_completed_bar, last_completed_bar, published_at, coverage_status, batch_ids_json, bundle_path, manifest_checksum FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown snapshot: {snapshot_id}")
        entry = SnapshotCatalogEntry(
            snapshot_id=row[0],
            snapshot_hash=row[1],
            stream_id=row[2],
            provider=row[3],
            calendar=row[4],
            timeframe=row[5],
            ticker_order=tuple(json.loads(row[6])),
            first_completed_bar=datetime.fromisoformat(row[7]),
            last_completed_bar=datetime.fromisoformat(row[8]),
            published_at=datetime.fromisoformat(row[9]),
            coverage_status=row[10],
            batch_ids=tuple(json.loads(row[11])),
            bundle_path=row[12],
            manifest_checksum=row[13],
        )
        self.verify(entry)
        return entry

    def load(self, snapshot_id: str) -> MarketSnapshot:
        entry = self.resolve(snapshot_id)
        manifest = json.loads((Path(entry.bundle_path) / "manifest.json").read_text())
        bars = pl.read_parquet(Path(entry.bundle_path) / "bars.parquet")
        return MarketSnapshot(
            snapshot_id=entry.snapshot_id,
            snapshot_hash=entry.snapshot_hash,
            bars=bars,
            ticker_order=entry.ticker_order,
            calendar=entry.calendar,
            timeframe=entry.timeframe,
            provenance=manifest.get("snapshot_provenance", {"provider": entry.provider, "stream_id": entry.stream_id}),
            validation_diagnostics=manifest.get("validation_diagnostics", {"coverage_status": entry.coverage_status}),
            source_manifest=manifest.get("source_manifest", [manifest]),
        )

    def list(self, *, coverage_status: str | None = None) -> list[SnapshotCatalogEntry]:
        query = "SELECT snapshot_id FROM snapshots"
        params: tuple[Any, ...] = ()
        if coverage_status is not None:
            query += " WHERE coverage_status=?"
            params = (coverage_status,)
        ids = self.connection.execute(query + " ORDER BY published_at, snapshot_id", params).fetchall()
        return [self.resolve(row[0]) for row in ids]

    def latest_accepted(self, stream_id: str | None = None) -> SnapshotCatalogEntry | None:
        accepted = [entry for entry in self.list(coverage_status="accepted") if stream_id is None or entry.stream_id == stream_id]
        return (
            max(accepted, key=lambda entry: (entry.last_completed_bar, entry.published_at, entry.snapshot_id))
            if accepted
            else None
        )

    def record_attempt(self, summary: IngestionAttemptSummary) -> None:
        payload = summary.model_dump_json()
        with self.connection:
            self.connection.execute(
                "INSERT INTO ingestion_attempts(attempt_id, stream_id, status, finished_at, payload_json) VALUES (?, ?, ?, ?, ?)",
                (
                    summary.attempt_id,
                    summary.stream_id,
                    summary.status,
                    (summary.finished_at or datetime.now(timezone.utc)).isoformat(),
                    payload,
                ),
            )

    def attempts(self, stream_id: str | None = None) -> list[IngestionAttemptSummary]:
        if stream_id is None:
            rows = self.connection.execute(
                "SELECT payload_json FROM ingestion_attempts ORDER BY finished_at, attempt_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT payload_json FROM ingestion_attempts WHERE stream_id=? ORDER BY finished_at, attempt_id", (stream_id,)
            ).fetchall()
        return [IngestionAttemptSummary.model_validate_json(row[0]) for row in rows]

    def verify(self, entry: SnapshotCatalogEntry) -> None:
        bundle = Path(entry.bundle_path)
        bars_path = bundle / "bars.parquet"
        manifest_path = bundle / "manifest.json"
        if not bars_path.exists() or not manifest_path.exists():
            raise ValueError(f"Snapshot bundle is missing: {entry.snapshot_id}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("snapshot_id") != entry.snapshot_id:
            raise ValueError(f"Snapshot manifest identity mismatch: {entry.snapshot_id}")
        if self._manifest_checksum(manifest) != entry.manifest_checksum:
            raise ValueError(f"Snapshot manifest checksum mismatch: {entry.snapshot_id}")
        if hashlib.sha256(bars_path.read_bytes()).hexdigest() != manifest.get("bars_sha256"):
            raise ValueError(f"Snapshot bars checksum mismatch: {entry.snapshot_id}")
        snapshot_hash = hashlib.sha256(
            canonical_json_bytes(
                {
                    "bars": pl.read_parquet(bars_path).to_dicts(),
                    "provenance": manifest.get("snapshot_provenance", {}),
                }
            )
        ).hexdigest()
        if snapshot_hash != entry.snapshot_hash:
            raise ValueError(f"Snapshot content identity mismatch: {entry.snapshot_id}")

    @staticmethod
    def _manifest_checksum(manifest: dict[str, Any]) -> str:
        unsigned = {key: value for key, value in manifest.items() if key not in {"manifest_checksum", "bars_sha256"}}
        return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()

    @staticmethod
    def _create_file(path: Path, data: bytes) -> None:
        ProviderBatchStore._create_file(path, data)
