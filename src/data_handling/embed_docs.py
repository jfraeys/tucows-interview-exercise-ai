"""
Embedding Builder Module

Handles the complete pipeline for building a FAISS-based vector search index:
- Loading and parsing documents from a directory
- Converting text content to embeddings using SentenceTransformer
- Building a FAISS index for efficient similarity search
- Saving the index and metadata for later retrieval
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import models
from src.data_handling.parser import parse_file
from src.utils import logging_helper

# Initialize logger for this module
logger = logging_helper.get_logger(__name__)


def load_documents(docs_dir: Path) -> models.ParserResponse:
    """
    Load and parse all documents from the specified directory.
    Skips unsupported files and continues processing others.

    Args:
        docs_dir: Directory containing documents to process

    Returns:
        ParserResponse containing all successfully parsed chunks

    Raises:
        FileNotFoundError: If the documents directory doesn't exist
        ValueError: If no documents were successfully processed
    """
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    response: models.ParserResponse = models.ParserResponse()
    processed_files = 0

    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            try:
                logger.info(f"Processing file: {file_path.name}")
                parsed = parse_file(str(file_path))
                response.chunks.extend(parsed.chunks)
                processed_files += 1

            except ValueError:
                # Skip unsupported file types - parser will handle this
                logger.warning(f"Skipping unsupported file: {file_path.name}")
                continue
            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")
                continue

    if processed_files == 0:
        raise ValueError("No documents were successfully processed")

    logger.info(f"Loaded {len(response.chunks)} chunks from {processed_files} files")
    return response


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """
    Convert texts into embeddings using SentenceTransformer.

    Args:
        texts: List of text strings to embed
        model_name: SentenceTransformer model name

    Returns:
        Float32 numpy array of embeddings suitable for FAISS

    Raises:
        ValueError: If texts list is empty or invalid
        RuntimeError: If model loading or embedding generation fails
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for empty text list")

    # Filter out empty texts
    valid_texts = [
        text for text in texts if text and isinstance(text, str) and text.strip()
    ]
    if not valid_texts:
        raise ValueError("No valid text content found")

    try:
        logger.info(f"Loading model: {model_name}")
        embedder = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")

    try:
        logger.info(f"Generating embeddings for {len(valid_texts)} chunks...")
        embeddings = embedder.encode(
            valid_texts, convert_to_numpy=True, show_progress_bar=True
        )
        return np.array(embeddings).astype("float32")
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {e}")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS L2 index for similarity search.
    Uses IndexFlatL2 for exact search - good for smaller datasets.

    Args:
        embeddings: Array of embeddings with shape (n_vectors, dim)

    Returns:
        Trained FAISS index ready for search

    Raises:
        ValueError: If embeddings array is invalid
        RuntimeError: If index creation fails
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build index from empty embeddings array")

    if len(embeddings.shape) != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    logger.info(f"Building FAISS index: {n_vectors} vectors, {dim} dimensions")

    try:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to build FAISS index: {e}")


def save_index_and_metadata(
    index: faiss.Index,
    metadata: models.ParserResponse,
    index_file: Path,
    metadata_file: Path,
) -> None:
    """
    Save FAISS index and metadata to disk.
    Creates output directory if needed.

    Args:
        index: Trained FAISS index to save
        metadata: Metadata containing original text chunks
        index_file: Path where FAISS index should be saved
        metadata_file: Path where metadata JSON should be saved
        force: If True, overwrite existing files

    Raises:
        RuntimeError: If saving fails
    """
    try:
        # Create output directory
        index_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_file))
        logger.info(f"FAISS index saved to {index_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save index: {e}")

    try:
        # Save metadata as JSON
        metadata_json = metadata.to_json()
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to save metadata: {e}")


def main(model_name: str) -> None:
    """
    Main pipeline: load docs → embed → index → save.
    """
    try:
        logger.info(f"Starting embedding pipeline with model: {model_name}")

        # Load and parse all documents
        parser_response = load_documents(DOCS_DIR)

        # Extract text content for embedding
        texts = [chunk.text for chunk in parser_response.chunks]

        # Generate embeddings
        embeddings = embed_texts(texts, model_name)

        # Build search index
        index = build_faiss_index(embeddings)

        # Save everything to disk
        save_index_and_metadata(index, parser_response, INDEX_FILE, METADATA_FILE)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed support documents and create FAISS index for similarity search"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)",
    )

    args = parser.parse_args()
    main(args.model)
