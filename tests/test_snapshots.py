import polars as pl
import pytest

from lab.core.config import Timeframe
from lab.platform.data_access import build_market_snapshot, consolidate_partitions, load_market_snapshot
from lab.core.config import PipelineConfig
from lab.quant.timing import bar_times_for_session


def _bars(sessions: tuple[str, ...], price: float = 100.0) -> pl.DataFrame:
    rows = []
    for index, session in enumerate(sessions):
        opened, closed, session_id = bar_times_for_session("XNYS", session, Timeframe.D1)[0]
        rows.append(
            {
                "ticker": "AAA",
                "bar_open_time": opened,
                "bar_close_time": closed,
                "timestamp": closed,
                "open": price + index,
                "high": price + index + 1,
                "low": price + index - 1,
                "close": price + index + 0.5,
                "volume": 1000.0,
                "session_id": session_id,
            }
        )
    return pl.DataFrame(rows)


def test_identical_partitions_have_one_snapshot_identity():
    first = _bars(("2024-01-02", "2024-01-03", "2024-01-04"))
    second = _bars(("2024-01-05", "2024-01-08", "2024-01-09"))
    one = build_market_snapshot(
        [pl.concat([first, second])],
        calendar="XNYS",
        timeframe=Timeframe.D1,
        provenance={"source": "test"},
        metadata={"timestamp_role": "bar_close"},
    )
    partitioned = build_market_snapshot(
        [first, second],
        calendar="XNYS",
        timeframe=Timeframe.D1,
        provenance={"source": "test"},
        metadata={"timestamp_role": "bar_close"},
    )
    assert one.snapshot_hash == partitioned.snapshot_hash


def test_price_or_provenance_changes_snapshot_identity():
    bars = _bars(("2024-01-02", "2024-01-03", "2024-01-04"))
    original = build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})
    changed_price = bars.with_columns(pl.when(pl.arange(0, pl.len()) == 0).then(101.0).otherwise(pl.col("open")).alias("open"))
    changed = build_market_snapshot([changed_price], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "test"})
    changed_provenance = build_market_snapshot([bars], calendar="XNYS", timeframe=Timeframe.D1, provenance={"source": "other"})
    assert original.snapshot_hash != changed.snapshot_hash
    assert original.snapshot_hash != changed_provenance.snapshot_hash


def test_conflicting_duplicate_bars_are_rejected():
    left = _bars(("2024-01-02",))
    right = left.with_columns(pl.lit(999.0).alias("close"))
    with pytest.raises(Exception, match="Conflicting duplicate"):
        consolidate_partitions([left, right], calendar="XNYS", timeframe=Timeframe.D1)


def test_local_source_does_not_fallback_when_input_is_missing(tmp_path):
    config = PipelineConfig(
        data={"source": "local", "input_path": str(tmp_path / "missing.csv"), "calendar": "XNYS", "timeframe": "1d"}
    )
    with pytest.raises(FileNotFoundError):
        load_market_snapshot(config)
