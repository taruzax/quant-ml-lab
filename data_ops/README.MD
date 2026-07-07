This subproject (data_ops) is structured as a standalone workspace member within the quant-ml-lab monorepo. It maintains its own localized settings, data schemas, and test suites to remain fully decoupled and portable, while sharing a unified root virtual environment managed via uv workspace configuration. For development, executing the pipelines, or running the testing suite, navigate directly into this directory so that local environment variables (.env) and configurations are correctly resolved relative to the subproject root.

### 📝 Article Behind Zero-Copy Sliding Windows:

**[Zero-Copy Sliding Windows: How to Reduce Memory Cost in Quantitative Deep Learning](https://zaurtarunov.substack.com/p/zero-copy-sliding-windows)**

Deep dive into the architecture behind this data pipeline, including:
- Why naive sliding windows are redundant and waste memory
- How PyTorch Storage/TensorImpl split enables zero-copy views
- The Copy-on-Write trap with multiprocessing workers
- Benchmark results: 38% less RAM, 3.1x faster DataLoader