# ===== Stage 1: Build =====
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build tools for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY Makefile Makefile
COPY pyproject.toml pyproject.toml
RUN pip install --upgrade pip
RUN make install

# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY src/ src/

# Copy pre-embedded FAISS index and support docs
COPY data/faiss_index/ data/faiss_index/

# Copy models
COPY models/ models/

# Expose API
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

