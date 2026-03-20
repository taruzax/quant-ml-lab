# FFD — Fractional Feature Data Module

This module is the **data factory** for `quant-ml-lab`. It downloads
raw OHLCV market data, engineers financial features using polars, and applies
**Fractional Differencing (FFD)** to produce stationary, memory-preserving inputs ready for downstream ML models.
---

## Project Structure

```
ffd/
├── src/
│   ├── data/
│   │   ├── loader.py       # Market data ingestion (Yahoo Finance → Polars)
│   │   └── transform.py    # Feature engineering + FFD + ADF optimizer
│   └── pipelines/
│       └── ingestion.py    # End-to-end pipeline entry point
tests/
└── test_frac_diff.py       # Unit + integration tests for FFD and ADF
config/
└── tickers.yaml            # Ticker universe with sector/tier metadata
data/
└── raw/                    # Output parquet files live here
```

---

## How It Works

### 1. Data Loading — `loader.py`

```
load_market_data(tickers, interval, start)
```

- Downloads OHLCV data from Yahoo Finance via `yfinance`
- Fetches `sector` and `industry` metadata per ticker
- Reshapes from wide (multi-index columns) to **long Polars DataFrame**
  — one row per `(ticker, date)`
- Drops tickers where sector/industry metadata is unavailable (e.g. some ETFs)

### 2. Feature Engineering — `transform.py`

`apply_features()` runs these steps in sequence:

| Step | Function | Output |
|---|---|---|
| Dollar volume | `calculate_dollar_volume` | `dollar_vol`, `dollar_vol_1m`, `dollar_vol_rank` |
| Technical indicators | `calculate_technical_indicators` | `ema5`, `macd`, `macdsignal`, `cdl2crows`, `wclprice` |
| Multi-horizon returns | `calculate_returns` | `return_1d` … `return_63d` + lagged versions + forward targets |
| Log price | inline | `log_close = log(close)` |
| FFD (per ticker) | `apply_dynamic_frac_diff` | `log_close_frac_diff` |
| Final features | `create_final_features` | time dummies, one-hot sector |

### 3. Fractional Differencing — the core

**`get_weights_ffd(d, threshold)`**

Generates weights using the iterative formula derived from Newton's Generalized
Binomial Theorem:

```
w_0 = 1.0
w_k = -w_{k-1} * (d - k + 1) / k
```

Stops when `|w_k| < threshold` — ensuring the window is finite but captures
meaningful historical memory.

**`frac_diff_polars(df, col_name, d, threshold)`**

Applies the weighted sum as a single Polars expression:

```
FFD_t = w_0 * close_t + w_1 * close_{t-1} + w_2 * close_{t-2} + ...
```

Uses `.over("ticker")` so lag operations are strictly within each ticker's
history — no cross-ticker contamination.

**`find_min_ffd(df_ticker, col_name, min_d=0.1)`**

Sweeps `d` from `min_d` to `1.0` in 10 steps and returns the smallest `d`
that makes the series stationary (ADF p-value < 0.05).

> `min_d=0.1` is intentional. At `d=0` the series is unchanged, and a
> random walk can accidentally pass ADF on a finite window. Setting a floor
> ensures we always apply at least minimal differencing.

**`apply_dynamic_frac_diff(df)`**

Runs `find_min_ffd` independently per ticker, then applies the optimal `d`.
Each asset gets the minimum differencing it actually needs.

---

## Running the Pipeline

```bash
# Full pipeline: fetch → transform → save parquet
python -m ffd.src.pipelines.ingestion
```

Output is saved to `data/raw/model_data_<timestamp>.parquet`.

---

## Configuration

**`config/tickers.yaml`** — defines the ticker universe:

Tickers missing sector/industry from Yahoo Finance are automatically dropped by the loader.

---

## Dependencies

Managed with `uv`. See `pyproject.toml` for pinned versions.

```bash
uv sync          # install all dependencies
uv add <pkg>     # add a new dependency
```

Key packages: `polars`, `polars-talib`, `statsmodels`, `yfinance`, `numpy`.
