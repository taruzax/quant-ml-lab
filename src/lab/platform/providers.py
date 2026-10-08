"""Provider adapters used by scheduled ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl

from lab.core.config import Timeframe
from lab.platform.data_access import load_market_data


class MarketDataProvider(Protocol):
    def fetch(
        self,
        tickers: list[str],
        *,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Fetch provider bars for one explicit bounded window."""


class YFinanceProvider:
    def timestamp_metadata(self, timeframe: Timeframe) -> dict[str, str]:
        return {
            "timestamp_role": "session_date" if timeframe == Timeframe.D1 else "bar_open",
            "timezone": "America/New_York",
        }

    def fetch(
        self,
        tickers: list[str],
        *,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        interval = "1d" if timeframe == Timeframe.D1 else "60m"
        return load_market_data(tickers, interval, start.isoformat(), end.isoformat())
