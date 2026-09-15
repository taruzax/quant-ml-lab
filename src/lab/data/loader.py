"""Compatibility exports for the relocated data-access module."""

from lab.platform.data_access import (
    fetch_sector_data,
    fetch_stock_data,
    get_sector_industry_yf,
    load_market_data,
    load_tickers,
    restructure_and_merge_data,
    save_model_data,
)

__all__ = [
    "fetch_sector_data",
    "fetch_stock_data",
    "get_sector_industry_yf",
    "load_market_data",
    "load_tickers",
    "restructure_and_merge_data",
    "save_model_data",
]
