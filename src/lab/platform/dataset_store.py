"""Persistent checksummed prepared-dataset bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from lab.core.config import PipelineConfig, get_platform_config
from lab.core.contracts import PreparedDataset, PreparedDatasetRef, SplitPlan
from lab.platform.artifacts import canonical_json_bytes, sha256_file
from lab.platform.market_store import SnapshotCatalog


class DatasetStore:
    def __init__(self, root: str | Path = "artifacts", *, snapshot_root: str | Path | None = None) -> None:
        self.root = Path(root) / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root = Path(snapshot_root) if snapshot_root is not None else get_platform_config().paths.snapshot_dir

    def write(
        self,
        dataset: PreparedDataset,
        *,
        config_hash: str | None = None,
        feature_spec_hash: str | None = None,
        split_hash: str,
    ) -> PreparedDatasetRef:
        effective_config_hash = config_hash or (dataset.config.resolved_config_hash if isinstance(dataset.config, PipelineConfig) else "unresolved")
        effective_feature_hash = feature_spec_hash or hashlib.sha256(canonical_json_bytes(dataset.feature_specification)).hexdigest()
        identity = {
            "snapshot_hash": dataset.snapshot.snapshot_hash,
            "config_hash": effective_config_hash,
            "feature_spec_hash": effective_feature_hash,
            "split_hash": split_hash,
        }
        dataset_id = f"dataset-{hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:24]}"
        bundle = self.root / dataset_id
        if bundle.exists():
            reference = self._load_ref(bundle)
            self.verify(reference)
            return reference
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=self.root))
        try:
            dataset.features.write_parquet(staging / "features.parquet")
            dataset.labels.write_parquet(staging / "labels.parquet")
            if isinstance(dataset.exclusions, pl.DataFrame):
                dataset.exclusions.write_parquet(staging / "exclusions.parquet")
                exclusions_format = "parquet"
            else:
                (staging / "exclusions.json").write_bytes(canonical_json_bytes(dataset.exclusions))
                exclusions_format = "json"
            checksums = {
                name: sha256_file(staging / name)
                for name in ("features.parquet", "labels.parquet", "exclusions.parquet" if exclusions_format == "parquet" else "exclusions.json")
            }
            manifest = {
                "schema_version": "prepared-dataset-bundle.v1",
                "dataset_id": dataset_id,
                "snapshot_id": dataset.snapshot.snapshot_id,
                "snapshot_hash": dataset.snapshot.snapshot_hash,
                "config_hash": effective_config_hash,
                "feature_spec_hash": effective_feature_hash,
                "split_hash": split_hash,
                "feature_specification": dataset.feature_specification,
                "split_plan": dataset.split_plan.model_dump(mode="json"),
                "config": dataset.config.model_dump(mode="json") if isinstance(dataset.config, PipelineConfig) else None,
                "exclusions_format": exclusions_format,
                "file_checksums": checksums,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest["checksum"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
            (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            try:
                os.rename(staging, bundle)
            except OSError:
                if not bundle.exists():
                    raise
                reference = self._load_ref(bundle)
                self.verify(reference)
                return reference
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return self._load_ref(bundle)

    def load(self, reference: PreparedDatasetRef) -> PreparedDataset:
        self.verify(reference)
        bundle = Path(reference.bundle_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        snapshot = SnapshotCatalog(self.snapshot_root).load(reference.snapshot_id)
        exclusions_path = bundle / ("exclusions.parquet" if manifest["exclusions_format"] == "parquet" else "exclusions.json")
        exclusions = pl.read_parquet(exclusions_path) if manifest["exclusions_format"] == "parquet" else json.loads(exclusions_path.read_text())
        return PreparedDataset(
            snapshot=snapshot,
            features=pl.read_parquet(bundle / "features.parquet"),
            feature_specification=tuple(manifest["feature_specification"]),
            labels=pl.read_parquet(bundle / "labels.parquet"),
            split_plan=SplitPlan.model_validate(manifest["split_plan"]),
            exclusions=exclusions,
            config=PipelineConfig.model_validate(manifest["config"]) if manifest.get("config") else None,
        )

    def resolve(self, dataset_id: str) -> PreparedDatasetRef:
        if not dataset_id or Path(dataset_id).name != dataset_id:
            raise ValueError("dataset_id must be one safe path component")
        return self._load_ref(self.root / dataset_id)

    def verify(self, reference: PreparedDatasetRef) -> None:
        bundle = Path(reference.bundle_path)
        if bundle.resolve().parent != self.root.resolve():
            raise ValueError(f"Prepared dataset bundle is outside the configured store: {reference.dataset_id}")
        manifest_path = bundle / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prepared dataset manifest missing: {reference.dataset_id}")
        manifest = json.loads(manifest_path.read_text())
        checksum = manifest.pop("checksum", None)
        if checksum != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() or checksum != reference.checksum:
            raise ValueError(f"Prepared dataset manifest checksum mismatch: {reference.dataset_id}")
        for name, expected in manifest["file_checksums"].items():
            path = bundle / name
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError(f"Prepared dataset file checksum mismatch: {name}")

    def _load_ref(self, bundle: Path) -> PreparedDatasetRef:
        manifest_path = bundle / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Incomplete prepared dataset bundle: {bundle.name}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset_id") != bundle.name:
            raise ValueError(f"Prepared dataset bundle identity mismatch: {bundle.name}")
        return PreparedDatasetRef(
            dataset_id=manifest["dataset_id"],
            snapshot_id=manifest["snapshot_id"],
            snapshot_hash=manifest["snapshot_hash"],
            config_hash=manifest["config_hash"],
            feature_spec_hash=manifest["feature_spec_hash"],
            split_hash=manifest["split_hash"],
            bundle_path=str(bundle),
            checksum=manifest["checksum"],
            created_at=datetime.fromisoformat(manifest["created_at"]),
        )
