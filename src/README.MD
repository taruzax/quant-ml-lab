This repository is a Quantitative Machine Learning Infrastructure Pipeline. It is designed to ingest raw financial time-series data (OHLCV), validate it, engineer predictive features (including advanced stationarity techniques), convert it into sliding-window tensors for deep learning models (PyTorch), and finally pass predictions through a sophisticated Risk Engine for portfolio optimization.

It relies heavily on Polars for ultra-fast, memory-efficient data manipulation and PyTorch for deep learning integration.


### 📝 Article Behind Zero-Copy Sliding Windows:

**[Zero-Copy Sliding Windows: How to Reduce Memory Cost in Quantitative Deep Learning](https://zaurtarunov.substack.com/p/zero-copy-sliding-windows)**

Deep dive into the architecture behind this data pipeline, including:
- Why naive sliding windows are redundant and waste memory
- How PyTorch Storage/TensorImpl split enables zero-copy views
- The Copy-on-Write trap with multiprocessing workers
- Benchmark results: 38% less RAM, 3.1x faster DataLoader