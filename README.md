# Knowledge Assistant Prototype

A Retrieval-Augmented Generation (RAG) system for customer support, designed to help support teams respond to tickets efficiently using relevant documentation. Built as part of an AI Engineer coding challenge, this prototype emphasizes MCP-compliant responses, simplicity, and extensibility.

-----

## Features

  - **RAG Pipeline**: Retrieve relevant support documentation using FAISS vector search.
  - **LLM Integration**: Generate structured answers with a local LLaMA-based model.
  - **MCP Compliance**: Ensures output follows the **Model Context Protocol** JSON schema:
    ```json
    {
      "answer": "...",
      "references": ["..."],
      "action_required": "..."
    }
    ```
  - **API Endpoint**: Single POST endpoint `/resolve-ticket` for submitting tickets.
  - **Fallback Mechanism**: Uses built-in prompts and embeddings if updated files are missing.
  - **Dockerized Deployment**: Easily reproducible across environments.

-----

## Tech Stack

  - **Language & API**: Python 3.11+, FastAPI
  - **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
  - **Vector Store**: FAISS
  - **LLM**: `cpp-llama` (local inference)
  - **Containerization**: Docker & Docker Compose
  - **Testing**: `pytest`

-----

## Installation

1.  Clone your forked repository
    ```bash
    git clone <your-fork-url>
    cd knowledge-assistant
    ```
2.  Install runtime dependencies
    ```bash
    make install
    ```
3.  Install dev dependencies
    ```bash
    make dev
    ```

## ⚠️ IMPORTANT: First-Time Setup

**Before running the server for the first time**, you MUST build the FAISS index from the raw documents:

```bash
make embed
```

This creates the vector embeddings required for document retrieval. Without this step, the server will fail to start with a `FileNotFoundError: FAISS index not found`.

### Quick Start (First Time)

**For first-time setup, use `make run_local` to automatically download and install the LLM model:**

```bash
# First time - downloads model and runs CLI
make run_local

# After model is installed, you can use other commands:
make embed    # Build FAISS index (required!)
make run      # Start development server
```

**Alternative complete setup:**
```bash
make setup    # Build FAISS index and start server
```
4.  Run API locally
    ```bash
    make run
    ```
5.  Run API in production mode
    ```bash
    make run_server
    ```

### Optional: Documentation PDF Generation

To generate PDF documentation from markdown files with mermaid diagram support:

1.  Install pandoc and dependencies:

    **macOS (Homebrew):**
    ```bash
    brew install pandoc
    npm install -g mermaid-filter
    brew install --cask mactex  # For XeLaTeX PDF engine
    ```

    **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install pandoc texlive-xetex
    npm install -g mermaid-filter
    ```

    **Windows:**
    ```bash
    # Install pandoc from https://pandoc.org/installing.html
    # Install MiKTeX from https://miktex.org/download
    npm install -g mermaid-filter
    ```

    **Alternative (Docker):**
    ```bash
    # Use pandoc Docker image with all dependencies included
    docker pull pandoc/extra
    ```

2.  Generate PDFs:
    ```bash
    make docs-pdf
    ```
    
    This converts all `docs/*.md` files to PDF format in `docs/output/`.

-----

## Usage

### Resolve a support ticket

**Request:**

```bash
POST /resolve-ticket
Content-Type: application/json

{
  "ticket_text": "My domain was suspended and I didn’t get any notice. How can I reactivate it?"
}
```

**Response (MCP-compliant):**

```json
{
  "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "escalate_to_abuse_team"
}
```

-----

## Architecture Overview

The Knowledge Assistant uses a minimal RAG pipeline with a local LLM to produce structured responses.

### High-Level Workflow

#### RAG API FLOW

1.  User submits ticket text.
2.  API receives the request and triggers the retrieval pipeline.
3.  Prompt Builder combines ticket and retrieved context into an MCP-compliant prompt.
4.  Retrieval Module fetches top-k relevant document chunks from FAISS.
5.  LLM generates the answer based on context.
6.  Structured JSON response is returned.

### Data Flow Diagram

#### FAISS Doc Pipeline

  - **Raw Docs**: Stored in `data/raw/`, pre-processed and chunked.
  - **FAISS Vector Store**: Stores embeddings with provenance metadata.
  - **Prompt Builder**: Combines user query and retrieved chunks.
  - **LLM** (`cpp-llama`): Generates structured output.
  - **Response**: Includes `answer`, `references`, `action_required`.

-----

## Module Breakdown

| Module | Responsibility |
| :--- | :--- |
| `data_handling` | Parse, chunk, embed documents; build FAISS index; manage fallback embeddings. |
| `core` | Retrieval module and LLM integration; constructs MCP prompt and generates responses. |
| `api` | FastAPI endpoint; receives tickets and returns structured JSON. |
| `utils` | Configuration management, logging, and helper functions. |

-----

## Makefile Commands

| Command | Description |
| :--- | :--- |
| `make install` | Install runtime dependencies |
| `make dev` | Install dev dependencies |
| `make embed` | Build FAISS index from raw docs |
| `make run` | Run API locally with reload |
| `make run_server` | Run API in prod mode (no reload, warning log level) |
| `make test` | Run unit and integration tests |
| `make lint` | Lint source code with `ruff` |
| `make format` | Format source code with `ruff` |
| `make clean` | Remove caches and build artifacts |
| `make docs-pdf` | Convert docs/ markdown files to PDF with mermaid support |

-----

## Future Enhancements

  - Hot-reload prompts and embeddings.
  - Human-in-the-loop evaluation and MLflow tracking for prompt optimization.
  - Add config files and prompt files for easier experimentation with models and responses.
  - Reranking models for improved retrieval precision.
  - Hybrid retrieval (keyword + vector search).
  - Swap FAISS for cloud vector DBs for scaling.
  - Kubernetes deployment for high availability and scaling.
  - Observability with logging and metrics.

-----

## Testing

Run all tests:

```bash
make test
```

Lint and format code:

```bash
make lint
make format
```

-----

## Disclaimer

  - An AI assistant contributed to planning, decision-making, research, and generating code for testing.
  - All final choices, implementations, and code were reviewed and validated by the project author.
  - AI was used as a supportive tool, not the sole source of technical decisions or production code.

-----

## Submission

  - Push your forked repo to GitHub.
  - Submit the repository link through the portal in the original email.

