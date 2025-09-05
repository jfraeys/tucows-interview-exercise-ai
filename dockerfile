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
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and install as package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --upgrade pip
RUN pip install -e . --no-deps
# Install packages with specific llama-cpp-python configuration for ARM64
ENV CMAKE_ARGS="-DLLAMA_METAL=off -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
RUN pip install fastapi faiss-cpu uvicorn pydantic torch sentence-transformers tf-keras pypdf
# Install llama-cpp-python with pre-built wheel or fallback
RUN pip install llama-cpp-python --prefer-binary --no-cache-dir || \
    pip install llama-cpp-python --no-binary=llama-cpp-python --no-cache-dir

# ===== Stage 2: Runtime =====
FROM python:3.12-slim

WORKDIR /app

# Minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed package from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --from=builder /app/src /app/src

# Copy pre-embedded FAISS index and support docs
COPY data/faiss_index/ data/faiss_index/

# Copy models (if available)
COPY models/ models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose API
EXPOSE 8000

# Start using uvicorn directly to avoid model download issues
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

