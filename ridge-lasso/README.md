# Stock Price Prediction Model

A machine learning project that predicts stock price movements using technical indicators and historical data.

## What This Project Does

This project downloads stock market data, calculates technical indicators, and uses machine learning models to predict future price movements. It focuses on stocks from energy, technology, ETFs, and quantum computing sectors, providing a comprehensive framework for analyzing over 150 different securities using hourly data.

<img width="953" height="790" alt="image" src="https://github.com/user-attachments/assets/3d18b9ab-490e-4d79-b1df-c634352aa38a" />

## Main Features

### 1\. Data Collection

The system downloads hourly stock data from Yahoo Finance starting January 2024, covering 150+ stocks across multiple sectors including energy companies like XOM, CVX, and SHEL, technology companies like NVDA, MSFT, and AAPL, popular ETFs like VOO, SPY, and QQQ, as well as emerging quantum computing stocks like IONQ and RGTI. For each stock, the system retrieves not only price and volume data but also sector and industry classification information to enable sector-based analysis.

### 2\. Feature Engineering

The feature engineering process creates a rich set of indicators to help predict price movements. Technical indicators form the foundation of the analysis, including RSI (Relative Strength Index) which measures if a stock is overbought or oversold, Bollinger Bands which show price volatility, ATR (Average True Range) which measures price movement size, and MACD (Moving Average Convergence Divergence) which shows trend changes.

Volume metrics play a crucial role in understanding stock liquidity and trading interest. The system calculates dollar volume by multiplying price and volume, computes 21-day average dollar volume to smooth out daily fluctuations, and ranks stocks by volume to focus on the most liquid securities.

Return calculations form the core of both features and prediction targets. The system calculates historical returns over different periods ranging from 1 day to 3 months, creates lagged returns to capture past performance at different time points, and generates future returns which serve as the prediction targets. Time-based features including year and month indicators, along with sector categories, help the model understand seasonal patterns and sector-specific behavior.

### 3\. Machine Learning Models

The project implements three regression models, each serving a specific purpose. Linear Regression provides a baseline to understand which features are most important and establishes minimum expected performance. Ridge Regression prevents overfitting by penalizing large coefficients, tests multiple penalty strengths to find the optimal balance, and includes a custom implementation built from scratch to demonstrate the underlying mathematics. Lasso Regression automatically selects important features by setting weak ones to exactly zero, also tests multiple penalty strengths, and similarly includes a custom implementation for educational purposes.

### 4\. Validation Strategy

The validation strategy uses time-series cross-validation specifically designed for financial data. This approach splits data into multiple training and testing periods, typically training on 63 days of historical data and testing on the subsequent 10 days. The process repeats moving backward through time, ensuring the model never sees future data during training. This prevents data leakage, which is a critical concern in financial prediction where even small amounts of forward-looking information can create misleading results.

### 5\. Performance Metrics

Two primary metrics evaluate model performance. The Information Coefficient (IC) measures how well predictions rank stocks relative to each other, with higher IC indicating better predictions. It's calculated daily and averaged to provide an overall performance measure. Root Mean Squared Error (RMSE) measures prediction accuracy in absolute terms, with lower RMSE indicating more accurate price predictions. While IC focuses on relative ranking ability, RMSE captures absolute prediction error.

### 6\. Analysis and Visualization

The analysis phase creates multiple visualizations to understand model behavior. Feature correlation heatmaps reveal which inputs move together and which provide independent information. Top feature importance rankings show which variables the model relies on most. Coefficient stability analysis across time periods reveals whether feature importance remains consistent or changes over time. Daily IC distribution histograms show the consistency of predictions, while rolling IC performance charts track how model quality evolves over time. RMSE trend analysis helps identify when model accuracy may be degrading.

### 7\. Data Storage

All processed data is saved to Google Drive in HDF5 format, providing compressed storage for efficiency while maintaining fast read access. The format preserves all calculated features and makes it easy to reload data for future analysis without reprocessing raw data from scratch.

---

## Custom Components Explained

### MultipleTimeSeriesCV (Custom Cross-Validation)

Regular cross-validation randomly splits data, which doesn't work for time-series because it would let the model "peek" into the future. This custom splitter respects time order and ensures that training data always comes before testing data, just like in real trading scenarios.

The splitter works by dividing the timeline into overlapping windows that move backward through time. Imagine you have 2 years of data: the first split might train on months 1-3 and test on month 4, the second split trains on months 4-6 and tests on month 7, and so on. This creates multiple independent evaluations of the model's ability to predict future prices based on past data.

The implementation accepts several parameters that control how the data is split. The `n_splits` parameter determines how many different train-test pairs to create, defaulting to 3 but typically set much higher for thorough evaluation. The `train_period_length` specifies how many days of history to use for training, with 126 days (about 6 months) being a common choice. The `test_period_length` sets how many days to evaluate on, typically 21 days (about one month). An optional `lookahead` parameter creates a gap between training and testing periods to prevent any overlap that could leak information from one period to the next.

The key benefit of this approach is that it ensures the model is tested on truly unseen future data. If the model performs well across multiple time periods, you can be more confident that it will work on genuinely new data rather than simply memorizing patterns from the training set.

---

## Main Libraries Used

The project depends on several Python libraries:

- `pandas` for data manipulation and time series operations
- `numpy` for numerical calculations and linear algebra
- `yfinance` for downloading stock data from Yahoo Finance
- `talib` for calculating technical indicators
- `sklearn` for machine learning models, preprocessing, and pipelines
- `matplotlib` and `seaborn` for creating visualizations
- `scipy` for statistical functions like Spearman correlation
- `statsmodels` for advanced time series analysis

---

## Usage Notes

This project is designed to run in Google Colab, the code saves processed data to Google Drive in HDF5 format, ensuring you don't need to reprocess raw data every time you run the analysis.

The analysis focuses on liquid stocks by filtering to the top 100 by dollar volume ranking. This ensures the model trains on actively traded securities where predictions are more reliable and actually tradeable. Using hourly data instead of daily data provides more frequent predictions, which can be valuable for short-term trading strategies.

The custom model implementations serve educational purposes, showing exactly how ridge and lasso regression work under the hood. For production use, you might prefer sklearn's optimized implementations, but these custom versions help you understand what's really happening during training and why these techniques work.

---
