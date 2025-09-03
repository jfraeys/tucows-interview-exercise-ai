PYTHON := python3
VENV := .venv
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
RAW_DATA_DIR := $(DATA_DIR)/raw
FAISS_INDEX_DIR := $(DATA_DIR)/faiss_index
CONFIG := config.yaml

# Add LLAMA_BACKEND variable (CPU by default)
LLAMA_BACKEND ?= CPU

# Map LLAMA_BACKEND to CMAKE_ARGS
ifeq ($(LLAMA_BACKEND),CUDA)
    CMAKE_ARGS := -DGGML_CUDA=on
else ifeq ($(LLAMA_BACKEND),METAL)
    CMAKE_ARGS := -DGGML_METAL=on
else ifeq ($(LLAMA_BACKEND),ACCELERATE)
    CMAKE_ARGS := -DGGML_BLAS_VENDOR=Accelerate
else
    CMAKE_ARGS := -DGGML_BLAS=ON   # default fallback
endif


.PHONY: help install dev test lint format embed run clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies with hardware acceleration
	@echo "Installing with LLAMA_BACKEND=$(CMAKE_ARGS)"
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	CMAKE_ARGS="$(CMAKE_ARGS)" $(PYTHON) -m pip install -e .

dev: ## Install dev dependencies with hardware acceleration
	@echo "Installing dev dependencies with LLAMA_BACKEND=$(LLAMA_BACKEND)"
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	CMAKE_ARGS="$(CMAKE_ARGS)" $(PYTHON) -m pip install -e ".[dev]" --upgrade --force-reinstall --no-cache-dir

test: ## Run unit tests
	$(PYTHON) -m pytest -v $(TEST_DIR) -s

lint: ## Lint with ruff check
	ruff check $(SRC_DIR) $(TEST_DIR)

format: ## Format code with ruff format
	ruff format $(SRC_DIR) $(TEST_DIR)

embed: ## Build FAISS index from raw docs
	$(PYTHON) -m src.data_handling.embed_docs

run: ## Run API (FastAPI/uvicorn)
	uvicorn  src.api.main:app --reload

run_local:
	$(PYTHON) -m src.core.rag_pipeline --model-repo=microsoft/Phi-3-mini-4k-instruct-gguf --model-filename=Phi-3-mini-4k-instruct-fp16.gguf --n-gpu-layers=-1

clean: ## Remove build artifacts, caches, and FAISS index
	find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.pyc" -type f -delete
	rm -rf .ruff_cache .pytest_cache $(FAISS_INDEX_DIR)

