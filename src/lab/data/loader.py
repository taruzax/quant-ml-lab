import datetime
import pathlib
import time

import pandas as pd
import polars as pl
import yaml
import yfinance as yf


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
    """Restructures stock data to long format and merges with sector info.
    Migrated from: ffd_adf/src/data/loader.py"""
    stocks_df.index.name = "timestamp"

    long_df = stocks_df.stack(level=1)
    long_df = long_df.rename(columns=str.lower)
    long_df = long_df.swaplevel().sort_index()
    long_df.index.names = ["ticker", "timestamp"]

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
