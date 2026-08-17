FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install system dependencies (including those for TA-Lib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr --build=$(uname -m)-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies (CPU-only torch to save space)
RUN uv sync --frozen --no-dev --no-install-project --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the code
COPY README.md ./
COPY src/ ./src/
COPY config/ ./config/

# Ensure the app code is installed
RUN uv sync --frozen --no-dev

# Set dagster home
ENV DAGSTER_HOME=/app/data/dagster_home
RUN mkdir -p $DAGSTER_HOME

# Expose ports
EXPOSE 3000
