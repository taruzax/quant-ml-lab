"""Thin CLI delegating to the shared research preparation API."""

from __future__ import annotations

import argparse
from pathlib import Path

from lab.core.config import PipelineConfig
from lab.research.experiment import prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare one local quant research snapshot")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"), help="research YAML configuration")
    args = parser.parse_args(argv)
    config = PipelineConfig.from_yaml(args.config)
    dataset = prepare_dataset(config)
    print(f"snapshot_id={dataset.snapshot.snapshot_id}")
    print(f"snapshot_hash={dataset.snapshot.snapshot_hash}")
    print(f"rows={dataset.features.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
