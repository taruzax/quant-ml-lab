"""Calendar-aware schedule helpers for ingestion polling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dagster import RunRequest, schedule

from lab.core.config import IngestionConfig, IngestionStreamConfig, Timeframe
from lab.quant.timing import expected_bar_keys


def enabled_streams(config: IngestionConfig, timeframe: Timeframe) -> tuple[IngestionStreamConfig, ...]:
    return tuple(stream for stream in config.streams if stream.enabled and stream.timeframe == timeframe)


def stable_run_key(stream: IngestionStreamConfig, completed_boundary: datetime) -> str:
    boundary = completed_boundary.astimezone(timezone.utc).isoformat()
    return f"{stream.name}:{stream.timeframe.value}:{boundary}"


def _schedule_requests(context, timeframe: Timeframe) -> list[RunRequest]:
    config_path = "config/ingestion.yaml"
    config = IngestionConfig.from_yaml(config_path)
    now = context.scheduled_execution_time
    requests = []
    for stream in enabled_streams(config, timeframe):
        start = datetime.fromisoformat(stream.start_boundary.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        expected = expected_bar_keys(
            stream.calendar,
            stream.timeframe,
            start,
            now,
            completion_delay=timedelta(minutes=stream.completion_delay_minutes),
            now=now,
        )
        boundary = expected["bar_close_time"].max() if not expected.is_empty() else now
        requests.append(
            RunRequest(
                run_key=stable_run_key(stream, boundary),
                run_config={
                    "resources": {
                        "config_py": {
                            "config": {
                                "ingestion_config_path": config_path,
                                "ingestion_timeframe": timeframe.value,
                                "ingestion_stream_name": stream.name,
                            }
                        }
                    }
                },
            )
        )
    return requests


def build_ingestion_schedules(job):
    @schedule(job=job, cron_schedule="15 * * * *", execution_timezone="UTC", name="hourly_market_data_ingestion")
    def hourly_ingestion_schedule(context):
        return _schedule_requests(context, Timeframe.H1)

    @schedule(job=job, cron_schedule="30 22 * * 1-5", execution_timezone="UTC", name="daily_market_data_ingestion")
    def daily_ingestion_schedule(context):
        return _schedule_requests(context, Timeframe.D1)

    return daily_ingestion_schedule, hourly_ingestion_schedule
