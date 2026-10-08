"""Incremental provider ingestion and immutable snapshot publication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from lab.core.config import IngestionStreamConfig, Timeframe
from lab.core.contracts import IngestionAttemptSummary, MarketSnapshot
from lab.platform.data_access import build_market_snapshot, canonicalize_bars, load_tickers
from lab.platform.market_store import ProviderBatchStore, SnapshotCatalog
from lab.platform.providers import MarketDataProvider, YFinanceProvider
from lab.quant.timing import expected_bar_keys
from lab.quant.validators import compare_bar_coverage


@dataclass(frozen=True)
class IngestionPlan:
    stream: IngestionStreamConfig
    start: datetime
    end: datetime
    completed_boundary: datetime | None
    tickers: tuple[str, ...]
    no_op: bool = False


class IngestionPlanner:
    def __init__(self, catalog: SnapshotCatalog) -> None:
        self.catalog = catalog

    def plan(self, stream: IngestionStreamConfig, *, now: datetime | None = None) -> IngestionPlan:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        start_boundary = datetime.fromisoformat(stream.start_boundary.replace("Z", "+00:00"))
        if start_boundary.tzinfo is None:
            start_boundary = start_boundary.replace(tzinfo=timezone.utc)
        tickers = tuple(stream.tickers)
        if stream.ticker_config_path is not None:
            tickers = tuple(load_tickers(stream.ticker_config_path))
        expected = expected_bar_keys(
            stream.calendar,
            stream.timeframe,
            start_boundary,
            current,
            completion_delay=timedelta(minutes=stream.completion_delay_minutes),
            now=current,
        )
        if expected.is_empty():
            return IngestionPlan(stream, start_boundary, max(current, start_boundary + timedelta(microseconds=1)), None, tickers, no_op=True)
        completed_boundary = expected["bar_close_time"].max()
        stream_entries = [entry for entry in self.catalog.list() if entry.stream_id == stream.name]
        if not stream_entries:
            fetch_start = start_boundary
        else:
            latest = max(stream_entries, key=lambda entry: (entry.last_completed_bar, entry.published_at))
            new_keys = expected.filter(pl.col("bar_close_time") > latest.last_completed_bar)
            if new_keys.is_empty():
                if latest.coverage_status != "incomplete":
                    return IngestionPlan(stream, latest.last_completed_bar, current, latest.last_completed_bar, tickers, no_op=True)
                diagnostics = self.catalog.load(latest.snapshot_id).validation_diagnostics
                missing = diagnostics.get("coverage", {}).get("missing_keys", [])
                missing_opens = sorted(
                    datetime.fromisoformat(row["bar_open_time"].replace("Z", "+00:00"))
                    if isinstance(row.get("bar_open_time"), str) else row["bar_open_time"]
                    for row in missing
                )
                repair_bar = missing_opens[0] if missing_opens else latest.last_completed_bar
                repair_keys = expected.filter(pl.col("bar_open_time") <= repair_bar).sort("bar_open_time")
                if repair_keys.is_empty():
                    return IngestionPlan(stream, latest.last_completed_bar, current, latest.last_completed_bar, tickers, no_op=True)
                overlap_index = max(0, repair_keys.height - stream.refresh_overlap_bars - 1)
                fetch_start = repair_keys["bar_open_time"][overlap_index]
                return IngestionPlan(stream, fetch_start, completed_boundary, completed_boundary, tickers)
            prior = expected.filter(pl.col("bar_close_time") <= new_keys["bar_close_time"].min())
            overlap_index = max(0, prior.height - stream.refresh_overlap_bars - 1)
            fetch_start = prior["bar_open_time"][overlap_index] if prior.height else start_boundary
        return IngestionPlan(stream, fetch_start, completed_boundary, completed_boundary, tickers)


class IngestionService:
    def __init__(
        self,
        *,
        batch_store: ProviderBatchStore,
        catalog: SnapshotCatalog,
        providers: dict[str, MarketDataProvider] | None = None,
    ) -> None:
        self.batch_store = batch_store
        self.catalog = catalog
        self.providers = providers or {"yfinance": YFinanceProvider()}
        self.planner = IngestionPlanner(catalog)

    def execute(self, stream: IngestionStreamConfig, *, now: datetime | None = None) -> IngestionAttemptSummary:
        attempt_id = f"ingestion-{uuid.uuid4().hex}"
        started = datetime.now(timezone.utc)
        plan = self.planner.plan(stream, now=now)
        if plan.no_op:
            return self._record(IngestionAttemptSummary(
                attempt_id=attempt_id, stream_id=stream.name, status="noop", planned_start=plan.start,
                planned_end=plan.end, completed_boundary=plan.completed_boundary, started_at=started,
                finished_at=datetime.now(timezone.utc),
            ))
        provider = self.providers.get(stream.provider)
        if provider is None:
            summary = IngestionAttemptSummary(
                attempt_id=attempt_id, stream_id=stream.name, status="failed", planned_start=plan.start,
                planned_end=plan.end, completed_boundary=plan.completed_boundary, started_at=started,
                finished_at=datetime.now(timezone.utc), error=f"No provider adapter configured for {stream.provider!r}",
            )
            return self._record(summary)
        batch_ref = None
        try:
            raw = provider.fetch(list(plan.tickers), timeframe=stream.timeframe, start=plan.start, end=plan.end)
            duplicate_rows = raw.height - raw.unique(subset=["ticker", "timestamp"]).height
            metadata = provider.timestamp_metadata(stream.timeframe) if hasattr(provider, "timestamp_metadata") else {
                "timestamp_role": stream.timestamp_role or ("session_date" if stream.timeframe == Timeframe.D1 else "bar_open"),
                "timezone": stream.provider_timezone,
            }
            incoming = canonicalize_bars(raw, calendar=stream.calendar, timeframe=stream.timeframe, metadata=metadata)
            if incoming.filter(~pl.col("ticker").is_in(plan.tickers)).height:
                raise ValueError("Provider returned tickers outside the configured stream")
            missing_tickers = sorted(set(plan.tickers) - set(incoming["ticker"].unique().to_list()))
            if missing_tickers and stream.gap_policy == "reject":
                raise ValueError(f"Provider returned a partial ticker set: {missing_tickers}")
            batch = self.batch_store.store(
                incoming,
                provider=stream.provider,
                stream_id=stream.name,
                request={"tickers": list(plan.tickers), "start": plan.start.isoformat(), "end": plan.end.isoformat(), "timeframe": stream.timeframe.value},
            )
            batch_ref = batch
            prior_entries = [entry for entry in self.catalog.list() if entry.stream_id == stream.name]
            prior_entries.sort(key=lambda entry: (entry.last_completed_bar, entry.published_at))
            partitions = [self.catalog.load(prior_entries[-1].snapshot_id).bars] if prior_entries else []
            prior_bars = partitions[-1] if partitions else pl.DataFrame()
            previous_keys = set(zip(prior_bars.get_column("ticker").to_list(), prior_bars.get_column("bar_open_time").to_list())) if partitions else set()
            revisions = 0
            if partitions:
                prior_by_key = {(row["ticker"], row["bar_open_time"]): row for row in prior_bars.to_dicts()}
                revisions = sum(
                    1 for row in incoming.to_dicts()
                    if (row["ticker"], row["bar_open_time"]) in prior_by_key
                    and any(prior_by_key[(row["ticker"], row["bar_open_time"])].get(col) != row.get(col) for col in ("open", "high", "low", "close", "volume"))
                )
            reconciled = self._reconcile(partitions + [incoming])
            expected = expected_bar_keys(
                stream.calendar, stream.timeframe, plan.start, plan.end,
                completion_delay=timedelta(minutes=stream.completion_delay_minutes), now=now or datetime.now(timezone.utc),
            )
            observed_window = reconciled.filter(
                (pl.col("bar_open_time") >= plan.start) & (pl.col("bar_close_time") <= plan.end)
            )
            coverage = compare_bar_coverage(expected, observed_window, tickers=plan.tickers)
            provider_coverage = compare_bar_coverage(expected, incoming, tickers=plan.tickers)
            missing_count = coverage["missing_count"]
            unexpected_count = max(coverage["unexpected_count"], provider_coverage["unexpected_count"])
            if stream.gap_policy == "reject" and (missing_count or unexpected_count or duplicate_rows):
                raise ValueError(
                    f"Calendar coverage rejected stream={stream.name}: missing={missing_count}, "
                    f"unexpected={unexpected_count}, duplicate={duplicate_rows}"
                )
            snapshot = build_market_snapshot(
                [reconciled],
                calendar=stream.calendar,
                timeframe=stream.timeframe,
                provenance={"source": "provider", "provider": stream.provider, "stream_id": stream.name, "batch_ids": [*(prior_entries[-1].batch_ids if prior_entries else ()), batch["batch_id"]]},
                metadata=metadata,
            )
            diagnostics = {
                **snapshot.validation_diagnostics,
                "coverage": {
                    "missing_count": missing_count,
                    "unexpected_count": unexpected_count,
                    "duplicate_count": duplicate_rows,
                    "valid_count": coverage["valid_count"],
                    "missing_keys": coverage["missing"].to_dicts(),
                },
                "revised_rows": revisions,
                "gap_policy": stream.gap_policy,
            }
            snapshot = snapshot.model_copy(update={"validation_diagnostics": diagnostics})
            entry = self.catalog.publish(
                snapshot,
                stream_id=stream.name,
                selected_batch_ids=tuple(dict.fromkeys([*(prior_entries[-1].batch_ids if prior_entries else ()), batch["batch_id"]])),
                coverage_status="incomplete" if missing_count and stream.gap_policy == "record" else "accepted",
            )
            return self._record(IngestionAttemptSummary(
                attempt_id=attempt_id, stream_id=stream.name, status="succeeded", planned_start=plan.start,
                planned_end=plan.end, completed_boundary=plan.completed_boundary, batch_id=batch["batch_id"],
                snapshot_id=entry.snapshot_id, observed_rows=raw.height, missing_rows=missing_count,
                unexpected_rows=unexpected_count, duplicate_rows=duplicate_rows, revised_rows=revisions,
                batch_ids=entry.batch_ids, started_at=started,
                finished_at=datetime.now(timezone.utc),
            ))
        except Exception as exc:
            return self._record(IngestionAttemptSummary(
                attempt_id=attempt_id, stream_id=stream.name, status="failed", planned_start=plan.start,
                planned_end=plan.end, completed_boundary=plan.completed_boundary,
                batch_id=batch_ref["batch_id"] if batch_ref else None,
                batch_ids=(batch_ref["batch_id"],) if batch_ref else (), started_at=started,
                finished_at=datetime.now(timezone.utc), error=str(exc)[:1000],
            ))

    def _record(self, summary: IngestionAttemptSummary) -> IngestionAttemptSummary:
        self.catalog.record_attempt(summary)
        return summary

    @staticmethod
    def _reconcile(partitions: list[pl.DataFrame]) -> pl.DataFrame:
        if not partitions:
            raise ValueError("At least one partition is required for reconciliation")
        tagged = [frame.with_columns(pl.lit(index).alias("_revision_order")) for index, frame in enumerate(partitions)]
        combined = pl.concat(tagged, how="diagonal_relaxed").sort(["ticker", "bar_open_time", "_revision_order"])
        reconciled = combined.unique(subset=["ticker", "bar_open_time"], keep="last").drop("_revision_order").sort(["ticker", "bar_open_time"])
        return reconciled.with_columns(pl.int_range(0, pl.len()).over("ticker").cast(pl.Int64).alias("raw_bar_index"))
