from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

from lab.core.config import Timeframe
from lab.quant.timing import bar_times_for_session, get_exchange_calendar


DEMO_TICKERS = ("ALFA", "BRAV", "CHAR", "DELT", "ECHO")
DEMO_SEED = 42
DEMO_GENERATOR_VERSION = "synthetic-demo-v1"


def generate_demo_bars(
    *,
    n_sessions: int = 360,
    tickers: tuple[str, ...] = DEMO_TICKERS,
    seed: int = DEMO_SEED,
    calendar: str = "XNYS",
    timeframe: Timeframe = Timeframe.D1,
) -> pl.DataFrame:
    """Generate deterministic correlated demonstration bars with explicit UTC times."""
    if n_sessions < 300:
        raise ValueError("The bundled demonstration requires at least 300 sessions")
    exchange_calendar = get_exchange_calendar(calendar)
    end_session = pd.Timestamp("2024-12-31")
    sessions = exchange_calendar.sessions[exchange_calendar.sessions <= end_session][-n_sessions:]
    rng = np.random.default_rng(seed)
    common_factor = rng.normal(0.0002, 0.008, size=n_sessions)
    rows: list[dict[str, Any]] = []
    for asset_index, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0.0, 0.006, size=n_sessions)
        prices = 80.0 + asset_index * 15.0
        for row_index, session in enumerate(sessions):
            opened, closed, session_id = bar_times_for_session(calendar, session, timeframe)[0]
            overnight = rng.normal(0.0, 0.002)
            opened_price = prices * (1.0 + overnight)
            close_return = common_factor[row_index] * (0.5 + asset_index * 0.1) + idiosyncratic[row_index]
            closed_price = opened_price * (1.0 + close_return)
            spread = abs(rng.normal(0.002, 0.001))
            high = max(opened_price, closed_price) * (1.0 + spread)
            low = min(opened_price, closed_price) * (1.0 - spread)
            rows.append(
                {
                    "ticker": ticker,
                    "bar_open_time": opened,
                    "bar_close_time": closed,
                    "timestamp": closed,
                    "open": round(float(opened_price), 6),
                    "high": round(float(high), 6),
                    "low": round(float(low), 6),
                    "close": round(float(closed_price), 6),
                    "volume": round(float(rng.lognormal(12.0, 0.15)), 3),
                    "session_id": session_id,
                }
            )
            prices = closed_price
    return pl.DataFrame(rows).sort(["ticker", "bar_open_time"])


def demo_metadata(
    *,
    seed: int = DEMO_SEED,
    calendar: str = "XNYS",
    timeframe: Timeframe = Timeframe.D1,
) -> dict[str, Any]:
    return {
        "dataset": "synthetic demonstration",
        "generator_version": DEMO_GENERATOR_VERSION,
        "seed": seed,
        "calendar": calendar,
        "timeframe": timeframe.value,
        "timestamp_role": "bar_close",
        "timezone": "UTC",
        "price_adjustment": "unadjusted",
        "provenance": "Generated common factor plus independent asset noise; not market evidence.",
    }


def write_demo_dataset(
    data_path: Path | str,
    metadata_path: Path | str,
    *,
    n_sessions: int = 360,
    seed: int = DEMO_SEED,
) -> tuple[Path, Path]:
    """Write the committed offline fixture from deterministic generator inputs."""
    data_path = Path(data_path)
    metadata_path = Path(metadata_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_bars(n_sessions=n_sessions, seed=seed).write_csv(data_path)
    metadata_path.write_text(yaml.safe_dump(demo_metadata(seed=seed), sort_keys=False))
    return data_path, metadata_path
