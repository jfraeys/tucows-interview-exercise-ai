PYTHON := python3
VENV := .venv
SRC_DIR := src
TEST_DIR := tests
DATA_DIR := data
RAW_DATA_DIR := $(DATA_DIR)/raw
FAISS_INDEX_DIR := $(DATA_DIR)/faiss_index
CONFIG := config.yaml

.PHONY: help install dev test lint format embed run clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install .

dev: ## Install dev dependencies
	$(PYTHON) -m pip install .[dev]

test: ## Run unit tests
	$(PYTHON) -m pytest $(TEST_DIR)

lint: ## Lint with ruff check
	ruff check $(SRC_DIR) $(TEST_DIR)

format: ## Format code with ruff format
	ruff format $(SRC_DIR) $(TEST_DIR)

embed: ## Build FAISS index from raw docs
	$(PYTHON) src/data_handling/embed_docs.py --config $(CONFIG)

run: ## Run API (FastAPI/uvicorn)
	uvicorn src.api.main:app --reload

clean: ## Remove build artifacts, caches, and FAISS index
	find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.pyc" -type f -delete
	rm -rf .ruff_cache .pytest_cache $(FAISS_INDEX_DIR)

