import datetime
import hashlib
import json
import pathlib
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import polars as pl
import yaml
import yfinance as yf

from lab.core.config import PipelineConfig, Timeframe, get_platform_config
from lab.core.contracts import MarketSnapshot
from lab.platform.market_store import SnapshotCatalog
from lab.quant.timing import as_utc, normalize_explicit_bar_times, normalize_provider_timestamp
from lab.quant.validators import DataValidationError, validate_canonical_bars


def load_tickers(config_path):
    """Migrated from: ffd_adf/src/pipelines/ingestion.py"""
    path = pathlib.Path(config_path)
    with open(path) as f:
        data = yaml.safe_load(f)
    tickers = [item["ticker"] for item in data]
    return tickers


def fetch_stock_data(tickers: list[str], interval: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetches OHLCV data for given tickers.

    Migrated from: ffd_adf/src/data/loader.py"""
    stocks = yf.download(
        tickers=tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by="column"
    )
    return stocks


def get_sector_industry_yf(symbol: str):
    """
    Returns (sector, industry) for equities when Yahoo has it.
    For ETFs/funds, these fields are often missing.

    Migrated from: ffd_adf/src/data/loader.py
    """
    try:
        info = yf.Ticker(symbol).get_info()
        return info.get("sector"), info.get("industry")
    except Exception:
        return None, None


def fetch_sector_data(tickers: list[str]) -> pd.DataFrame:
    """Fetches sector and industry data for given tickers.
    Migrated from: ffd_adf/src/data/loader.py"""
    rows = []
    for i, sym in enumerate(tickers, 1):
        sector, industry = get_sector_industry_yf(sym)
        rows.append({"ticker": sym, "sector": sector, "industry": industry})
        time.sleep(0.2)

    sector_df = pd.DataFrame(rows)
    unique_syms = pd.Index(tickers, name="ticker").unique()
    sector_df = sector_df[sector_df["ticker"].isin(unique_syms)]

    return sector_df


def restructure_and_merge_data(stocks_df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """Restructures stock data to long format and merges with sector info."""
    stocks_df.index.name = "timestamp"

    try:
        long_df = stocks_df.stack(level=1, future_stack=True)
    except TypeError:
        long_df = stocks_df.stack(level=1)

    long_df = long_df.rename(columns=str.lower)
    long_df = long_df.swaplevel().sort_index()
    long_df.index.names = ["ticker", "timestamp"]

    # Drop any bars where OHLCV data is missing/NaN from Yahoo Finance
    price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in long_df.columns]
    long_df = long_df.dropna(subset=price_cols)

    merged_df = long_df.join(sector_df.set_index("ticker"))
    merged_df = merged_df.fillna({"sector": "Unknown", "industry": "Unknown"})

    return merged_df


def load_market_data(tickers: list[str], interval: str, start: str, end: str | None = None) -> pl.DataFrame:
    """
    Main ingestion orchestrator
    Migrated from: ffd_adf/src/data/loader.py
    """

    stocks_df = fetch_stock_data(tickers, interval, start, end)
    sector_df = fetch_sector_data(tickers)
    merged_pd = restructure_and_merge_data(stocks_df, sector_df)

    print("Data successfully loaded.")
    df_out = pl.DataFrame(merged_pd.reset_index())

    casts = []
    if "timestamp" in df_out.columns and df_out["timestamp"].dtype != pl.Datetime:
        casts.append(pl.col("timestamp").cast(pl.Datetime))
    if "volume" in df_out.columns and df_out["volume"].dtype != pl.Float64:
        casts.append(pl.col("volume").cast(pl.Float64))

    if casts:
        df_out = df_out.with_columns(casts)

    # Ensure no residual NaNs or nulls in OHLCV
    df_out = df_out.filter(
        pl.col("close").is_not_nan()
        & pl.col("close").is_not_null()
        & pl.col("open").is_not_nan()
        & pl.col("open").is_not_null()
        & pl.col("high").is_not_nan()
        & pl.col("high").is_not_null()
        & pl.col("low").is_not_nan()
        & pl.col("low").is_not_null()
        & pl.col("volume").is_not_nan()
        & pl.col("volume").is_not_null()
    )

    return df_out.sort(["ticker", "timestamp"])


def save_model_data(df: pl.DataFrame, directory: str = "data/raw", filename: str = "model_data"):
    """Migrated from: ffd_adf/src/pipelines/ingestion.py"""
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    full_path = path / f"{filename}_{timestamp}.parquet"

    df.write_parquet(full_path)
    print(f"Data saved successfully to {full_path}")
    return full_path


def _read_input_frame(path: pathlib.Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Configured market-data input does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported market-data input format: {path.suffix}")


def _read_metadata(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Configured market-data metadata does not exist: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in market-data metadata: {path}")
    return raw


def canonicalize_bars(
    df: pl.DataFrame,
    *,
    calendar: str,
    timeframe: Timeframe,
    metadata: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Normalize a raw OHLCV frame into the canonical UTC bar contract."""
    metadata = metadata or {}
    required = {"ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise DataValidationError(f"Raw market data is missing columns: {missing}")
    has_open = "bar_open_time" in df.columns
    has_close = "bar_close_time" in df.columns
    if has_open != has_close:
        raise DataValidationError("bar_open_time and bar_close_time must be supplied together")
    timestamp_role = metadata.get("timestamp_role")
    timezone_name = metadata.get("timezone")
    if not has_open and timestamp_role not in {"session_date", "bar_open", "bar_close"}:
        raise DataValidationError(
            "Ambiguous timestamp convention: provide bar_open_time/bar_close_time or metadata.timestamp_role "
            "and, for naive timestamps, metadata.timezone"
        )
    if not has_open and "timestamp" not in df.columns:
        raise DataValidationError("Raw market data requires timestamp when explicit bar times are absent")

    records: list[dict[str, Any]] = []
    canonical_names = {
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "bar_open_time",
        "bar_close_time",
        "raw_bar_index",
        "session_id",
    }
    for row in df.to_dicts():
        try:
            if has_open:
                opened, closed, session_id = normalize_explicit_bar_times(
                    row["bar_open_time"], row["bar_close_time"], calendar, timeframe
                )
                if "timestamp" in row and row["timestamp"] is not None and as_utc(row["timestamp"]) != closed:
                    raise ValueError("timestamp must equal bar_close_time")
            else:
                opened, closed, session_id = normalize_provider_timestamp(
                    row["timestamp"],
                    calendar,
                    timeframe,
                    role=timestamp_role,
                    timezone_name=timezone_name,
                )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Could not normalize bar for ticker={row.get('ticker')}: {exc}") from exc
        normalized = {
            "ticker": str(row["ticker"]),
            "bar_open_time": opened,
            "bar_close_time": closed,
            "timestamp": closed,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "session_id": session_id,
        }
        for name, value in row.items():
            if name not in canonical_names:
                normalized[name] = value
        records.append(normalized)

    result = pl.DataFrame(records).sort(["ticker", "bar_open_time"])
    result = result.with_columns(pl.int_range(0, pl.len()).over("ticker").cast(pl.Int64).alias("raw_bar_index"))
    result = result.select(
        [
            "ticker",
            "raw_bar_index",
            "bar_open_time",
            "bar_close_time",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_id",
        ]
        + [
            column
            for column in result.columns
            if column
            not in {
                "ticker",
                "raw_bar_index",
                "bar_open_time",
                "bar_close_time",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "session_id",
            }
        ]
    )
    validate_canonical_bars(result, calendar=calendar, timeframe=timeframe)
    return result


def consolidate_partitions(
    partitions: Iterable[pl.DataFrame | pathlib.Path | str],
    *,
    calendar: str,
    timeframe: Timeframe,
    metadata: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Normalize and consolidate validated partitions before research preparation."""
    frames: list[pl.DataFrame] = []
    for partition in partitions:
        if isinstance(partition, (str, pathlib.Path)):
            frames.append(
                canonicalize_bars(
                    _read_input_frame(pathlib.Path(partition)), calendar=calendar, timeframe=timeframe, metadata=metadata
                )
            )
        else:
            frames.append(canonicalize_bars(partition, calendar=calendar, timeframe=timeframe, metadata=metadata))
    if not frames:
        raise DataValidationError("No market-data partitions were supplied")
    combined = pl.concat(frames, how="diagonal_relaxed").sort(["ticker", "bar_open_time"])
    keys = ("ticker", "bar_open_time")
    unique: dict[tuple[Any, Any], dict[str, Any]] = {}
    duplicate_count = 0
    compare_columns = [column for column in combined.columns if column != "raw_bar_index"]
    for row in combined.to_dicts():
        key = (row["ticker"], row["bar_open_time"])
        comparable = {column: row.get(column) for column in compare_columns}
        if key in unique:
            if unique[key] != comparable:
                raise DataValidationError(f"Conflicting duplicate market bars for ticker={key[0]}, bar_open_time={key[1]}")
            duplicate_count += 1
        else:
            unique[key] = comparable
    result = pl.DataFrame(list(unique.values())).sort(["ticker", "bar_open_time"])
    result = result.with_columns(pl.int_range(0, pl.len()).over("ticker").cast(pl.Int64).alias("raw_bar_index")).select(
        [
            "ticker",
            "raw_bar_index",
            "bar_open_time",
            "bar_close_time",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_id",
        ]
        + [
            column
            for column in combined.columns
            if column
            not in {
                "ticker",
                "raw_bar_index",
                "bar_open_time",
                "bar_close_time",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "session_id",
            }
        ]
    )
    validate_canonical_bars(result, calendar=calendar, timeframe=timeframe)
    return result, {"input_rows": combined.height, "duplicate_rows_collapsed": duplicate_count, "output_rows": result.height}


def _source_manifest(paths: Iterable[pathlib.Path]) -> list[dict[str, Any]]:
    manifest = []
    for path in paths:
        data = path.read_bytes()
        manifest.append({"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return manifest


def _snapshot_hash(bars: pl.DataFrame, provenance: dict[str, Any]) -> str:
    payload = {"bars": bars.to_dicts(), "provenance": provenance}
    encoded = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_market_snapshot(
    partitions: Iterable[pl.DataFrame | pathlib.Path | str],
    *,
    calendar: str,
    timeframe: Timeframe,
    provenance: dict[str, Any],
    source_manifest: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MarketSnapshot:
    bars, diagnostics = consolidate_partitions(partitions, calendar=calendar, timeframe=timeframe, metadata=metadata)
    snapshot_hash = _snapshot_hash(bars, provenance)
    return MarketSnapshot(
        snapshot_id=f"snapshot-{snapshot_hash[:16]}",
        snapshot_hash=snapshot_hash,
        bars=bars,
        ticker_order=tuple(sorted(bars["ticker"].unique().to_list())),
        calendar=calendar,
        timeframe=timeframe,
        provenance=provenance,
        validation_diagnostics=diagnostics,
        source_manifest=source_manifest or [],
    )


def load_market_snapshot(config: PipelineConfig) -> MarketSnapshot:
    """Load one explicit source and return an immutable logical snapshot."""
    source = config.data.source
    metadata_path = config.data.metadata_path
    if source == "bundled_demo":
        input_path = config.data.input_path or pathlib.Path("data/demo/ohlcv.csv")
        metadata_path = metadata_path or pathlib.Path("data/demo/metadata.yaml")
    elif source == "local":
        if config.data.input_path is None:
            raise ValueError("data.input_path is required when data.source='local'")
        input_path = config.data.input_path
    elif source == "snapshot":
        if config.data.snapshot_id is None:
            raise ValueError("data.snapshot_id is required when data.source='snapshot'")
        snapshot = SnapshotCatalog(get_platform_config().paths.snapshot_dir).load(config.data.snapshot_id)
        if snapshot.calendar != config.data.calendar or snapshot.timeframe != config.data.timeframe:
            raise ValueError(
                "Saved snapshot calendar/timeframe does not match the research configuration: "
                f"snapshot=({snapshot.calendar}, {snapshot.timeframe.value}), "
                f"config=({config.data.calendar}, {config.data.timeframe.value})"
            )
        return snapshot
    else:
        tickers = load_tickers(config.data.ticker_config_path)
        frame = load_market_data(tickers, config.ingestion_interval, config.data.ingestion_start)
        timestamp_role = "session_date" if config.data.timeframe == Timeframe.D1 else "bar_open"
        provenance = {
            "source": "provider",
            "provider": "yfinance",
            "price_adjustment": config.data.price_adjustment,
            "tickers": tickers,
        }
        return build_market_snapshot(
            [frame],
            calendar=config.data.calendar,
            timeframe=config.data.timeframe,
            provenance=provenance,
            metadata={
                "timestamp_role": timestamp_role,
                "timezone": "America/New_York",
            },
        )

    metadata = _read_metadata(metadata_path)
    frame = _read_input_frame(pathlib.Path(input_path))
    provenance = {
        "source": source,
        "price_adjustment": metadata.get("price_adjustment", config.data.price_adjustment),
        "metadata": metadata,
    }
    return build_market_snapshot(
        [frame],
        calendar=config.data.calendar,
        timeframe=config.data.timeframe,
        provenance=provenance,
        source_manifest=_source_manifest([pathlib.Path(input_path)] + ([pathlib.Path(metadata_path)] if metadata_path else [])),
        metadata=metadata,
    )


def persist_snapshot(snapshot: MarketSnapshot, directory: pathlib.Path | str) -> pathlib.Path:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"{snapshot.snapshot_id}.parquet"
    manifest_path = directory / f"{snapshot.snapshot_id}.yaml"
    snapshot.bars.write_parquet(data_path)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_hash": snapshot.snapshot_hash,
                "calendar": snapshot.calendar,
                "timeframe": snapshot.timeframe.value,
                "provenance": snapshot.provenance,
                "validation_diagnostics": snapshot.validation_diagnostics,
                "source_manifest": snapshot.source_manifest,
                "data_path": str(data_path),
            },
            sort_keys=False,
        )
    )
    return data_path
