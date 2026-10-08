from datetime import datetime, timedelta, timezone
from typing import Literal

import exchange_calendars as xcals
import pandas as pd
import polars as pl

from lab.core.config import Timeframe


def get_exchange_calendar(name: str):
    try:
        return xcals.get_calendar(name)
    except Exception as exc:
        raise ValueError(f"Unknown exchange calendar '{name}'") from exc


def as_utc(value: datetime | str | pd.Timestamp, *, timezone_name: str | None = None) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        if timezone_name is None:
            raise ValueError("Ambiguous timestamp: supply a timezone or explicit UTC offset")
        timestamp = timestamp.tz_localize(timezone_name)
    return timestamp.tz_convert("UTC").to_pydatetime()


def session_schedule(calendar_name: str, session: str | datetime | pd.Timestamp) -> tuple[datetime, datetime, str]:
    calendar = get_exchange_calendar(calendar_name)
    label = calendar.date_to_session(pd.Timestamp(session).normalize(), direction="none")
    row = calendar.schedule.loc[label]
    open_column = "market_open" if "market_open" in row.index else "open"
    close_column = "market_close" if "market_close" in row.index else "close"
    opened = pd.Timestamp(row[open_column]).tz_convert("UTC").to_pydatetime()
    closed = pd.Timestamp(row[close_column]).tz_convert("UTC").to_pydatetime()
    return opened, closed, label.strftime("%Y-%m-%d")


def bar_times_for_session(
    calendar_name: str,
    session: str | datetime | pd.Timestamp,
    timeframe: Timeframe,
) -> list[tuple[datetime, datetime, str]]:
    opened, closed, session_id = session_schedule(calendar_name, session)
    if timeframe == Timeframe.D1:
        return [(opened, closed, session_id)]
    step = timedelta(hours=1)
    bars: list[tuple[datetime, datetime, str]] = []
    current = opened
    while current < closed:
        end = min(current + step, closed)
        bars.append((current, end, session_id))
        current = end
    return bars


def normalize_provider_timestamp(
    value: datetime | str | pd.Timestamp,
    calendar_name: str,
    timeframe: Timeframe,
    *,
    role: Literal["session_date", "bar_open", "bar_close"],
    timezone_name: str | None = None,
) -> tuple[datetime, datetime, str]:
    if role == "session_date":
        timestamp = pd.Timestamp(value)
        _, _, session_id = session_schedule(calendar_name, timestamp)
        return bar_times_for_session(calendar_name, session_id, timeframe)[0]

    timestamp = as_utc(value, timezone_name=timezone_name)
    calendar = get_exchange_calendar(calendar_name)
    session = calendar.minute_to_session(pd.Timestamp(timestamp), direction="none")
    candidates = bar_times_for_session(calendar_name, session, timeframe)
    for opened, closed, session_id in candidates:
        if role == "bar_open" and timestamp == opened:
            return opened, closed, session_id
        if role == "bar_close" and timestamp == closed:
            return opened, closed, session_id
    raise ValueError(f"Timestamp {timestamp.isoformat()} does not match a declared {timeframe.value} {role} bar")


def normalize_explicit_bar_times(
    opened: datetime | str | pd.Timestamp,
    closed: datetime | str | pd.Timestamp,
    calendar_name: str,
    timeframe: Timeframe,
) -> tuple[datetime, datetime, str]:
    open_time = as_utc(opened)
    close_time = as_utc(closed)
    if close_time <= open_time:
        raise ValueError("bar_close_time must be after bar_open_time")
    calendar = get_exchange_calendar(calendar_name)
    session = calendar.minute_to_session(pd.Timestamp(open_time), direction="none")
    expected = bar_times_for_session(calendar_name, session, timeframe)
    if (open_time, close_time) not in [(item[0], item[1]) for item in expected]:
        raise ValueError(
            f"Bar interval {open_time.isoformat()}–{close_time.isoformat()} is not valid for "
            f"calendar={calendar_name}, timeframe={timeframe.value}"
        )
    return open_time, close_time, session.strftime("%Y-%m-%d")


def expected_bar_keys(
    calendar_name: str,
    timeframe: Timeframe,
    start: datetime | str | pd.Timestamp,
    end: datetime | str | pd.Timestamp,
    *,
    completion_delay: timedelta = timedelta(0),
    now: datetime | None = None,
) -> pl.DataFrame:
    """Generate completed exchange bars for a closed interval."""
    start_time = as_utc(start, timezone_name="UTC")
    end_time = as_utc(end, timezone_name="UTC")
    if end_time < start_time:
        raise ValueError("Expected-bar interval must be ordered")
    cutoff = as_utc(now or datetime.now(timezone.utc), timezone_name="UTC") - completion_delay
    calendar = get_exchange_calendar(calendar_name)
    sessions = calendar.sessions_in_range(pd.Timestamp(start_time.date()), pd.Timestamp(end_time.date()))
    rows: list[dict[str, object]] = []
    for session in sessions:
        for opened, closed, session_id in bar_times_for_session(calendar_name, session, timeframe):
            if opened < start_time or closed > end_time or closed > cutoff:
                continue
            rows.append({"session_id": session_id, "bar_open_time": opened, "bar_close_time": closed})
    return pl.DataFrame(
        rows,
        schema={
            "session_id": pl.Utf8,
            "bar_open_time": pl.Datetime("us", "UTC"),
            "bar_close_time": pl.Datetime("us", "UTC"),
        },
    )
