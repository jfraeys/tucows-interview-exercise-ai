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


.PHONY: help install dev test lint format embed run clean docs-pdf

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'


venv: ## Create a virtual environment
	$(PYTHON) -m venv $(VENV)


install: venv ## Install runtime dependencies with hardware acceleration
	@echo "Installing with LLAMA_BACKEND=$(CMAKE_ARGS)"
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	CMAKE_ARGS="$(CMAKE_ARGS)" $(PYTHON) -m pip install -e .

dev: venv ## Install dev dependencies with hardware acceleration
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
	$(PYTHON) -m cli embed --documents-path=$(RAW_DATA_DIR)

# Check if FAISS index exists, build if needed
check_index:
	@if [ ! -f "$(FAISS_INDEX_DIR)/faiss.index" ]; then \
		echo "🚀 FAISS index not found. Building from raw documents..."; \
		$(PYTHON) -m cli embed --documents-path=$(RAW_DATA_DIR); \
		echo "FAISS index built successfully!"; \
	fi

run: check-index ## Run API with reload and custom model support (auto-builds index if needed)
	MODEL_PATH=$(MODEL_PATH) MODEL_REPO=$(MODEL_REPO) MODEL_FILE=$(MODEL_FILE) \
	uvicorn src.api.main:app --reload

setup: ## First-time setup: embed documents and run server
	@echo "🚀 First-time setup: Building FAISS index..."
	$(PYTHON) -m cli embed --documents-path=$(RAW_DATA_DIR)
	@echo "✅ FAISS index built successfully!"
	@echo "🌐 Starting development server..."
	MODEL_PATH=$(MODEL_PATH) MODEL_REPO=$(MODEL_REPO) MODEL_FILE=$(MODEL_FILE) \
	uvicorn src.api.main:app --reload

run_prod: ## Run production server with custom model support
	ENV=prod GGML_LOG_LEVEL=ERROR \
	MODEL_PATH=$(MODEL_PATH) MODEL_REPO=$(MODEL_REPO) MODEL_FILE=$(MODEL_FILE) \
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --log-level warning

run_local: ## Run local CLI with custom model (use MODEL_REPO and MODEL_FILE variables)
	$(PYTHON) -m cli query \
		--model-repo=$(or $(MODEL_REPO),microsoft/Phi-3-mini-4k-instruct-gguf) \
		--model-filename=$(or $(MODEL_FILE),Phi-3-mini-4k-instruct-fp16.gguf) \
		--n-gpu-layers=$(or $(GPU_LAYERS),-1)

run_custom: ## Run with custom model parameters
	@echo "Usage: make run_custom MODEL_REPO=<repo> MODEL_FILE=<file> [GPU_LAYERS=<n>]"
	@echo "Example: make run_custom MODEL_REPO=microsoft/Phi-3-mini-4k-instruct-gguf MODEL_FILE=Phi-3-mini-4k-instruct-q4_k_m.gguf"
	$(PYTHON) -m cli query \
		--model-repo=$(MODEL_REPO) \
		--model-filename=$(MODEL_FILE) \
		--n-gpu-layers=$(or $(GPU_LAYERS),-1)

clean: ## Remove build artifacts, caches, and FAISS index
	@echo "🧹 Cleaning up build artifacts..."
	find . \( -name "__pycache__" -type d -exec rm -rf {} + \) \
		-o \( -name ".mypy_cache" -type d -exec rm -rf {} + \) \
		-o \( -name "*.egg-info" -type d -exec rm -rf {} + \) \
		-o \( -name "*.pyc" -type f -delete \) \
		-o \( -name ".coverage*" -type f -delete \)
	rm -rf .ruff_cache .pytest_cache $(FAISS_INDEX_DIR) models
	@echo "✅ Cleanup complete!"

docs-pdf: ## Convert docs/ markdown files to PDF with mermaid support
	@echo "Converting documentation to PDF..."
	@mkdir -p docs/output
	@for md_file in docs/*.md; do \
		if [ -f "$$md_file" ]; then \
			base_name=$$(basename "$$md_file" .md); \
			echo "Converting $$md_file to docs/output/$$base_name.pdf"; \
			pandoc "$$md_file" \
				--from markdown \
				--to pdf \
				--pdf-engine=xelatex \
				--filter mermaid-filter \
				--output "docs/output/$$base_name.pdf" \
				--variable geometry:margin=1in \
				--variable fontsize=11pt \
				--variable colorlinks=true \
				--toc \
				2>/dev/null || echo "Warning: Failed to convert $$md_file (missing pandoc/mermaid-filter?)"; \
		fi; \
	done
	@echo "PDF generation complete. Files saved to docs/output/"

