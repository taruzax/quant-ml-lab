import time
import pandas as pd
import polars as pl
import yfinance as yf

def fetch_stock_data(tickers: list[str], interval: str, start: str, end: str = None) -> pd.DataFrame:
    """Fetches OHLCV data for given tickers."""
    stocks = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )
    return stocks

def get_sector_industry_yf(symbol: str):
    """
    Returns (sector, industry) for equities when Yahoo has it.
    For ETFs/funds, these fields are often missing.
    """
    try:
        info = yf.Ticker(symbol).get_info()
        return info.get("sector"), info.get("industry")
    except Exception:
        return None, None

def fetch_sector_data(tickers: list[str]) -> pd.DataFrame:
    """Fetches sector and industry data for given tickers."""
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
    stocks_df.index.name = 'date'

    long_df = stocks_df.stack(level=1)
    long_df = long_df.rename(columns=str.lower)
    long_df = long_df.swaplevel().sort_index()
    long_df.index.names = ['ticker', 'date']

    merged_df = long_df.join(sector_df.set_index('ticker'))
    merged_df = merged_df.dropna(subset=['sector', 'industry'])
    
    return merged_df

def load_market_data(tickers: list[str], interval:str, start: str, end: str = None) -> pl.DataFrame:
    """
    Main ingestion orchestrator
    """

    stocks_df = fetch_stock_data(tickers, interval, start, end)
    sector_df = fetch_sector_data(tickers)
    merged_pd = restructure_and_merge_data(stocks_df, sector_df)

    print("Data successfully loaded.")
    return pl.DataFrame(merged_pd.reset_index())
