"""
Vector store implementation for semantic document retrieval using FAISS and SentenceTransformers.

Core functionality for RAG systems with clean device management and async operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import llm_helper, logging_helper

logger = logging_helper.get_logger(__name__)

MODEL_NAME_DEFAULT = "all-MiniLM-L6-v2"

DeviceType = Literal["cpu", "cuda", "mps", "auto"]


class VectorStore:
    """
    High-performance vector store for semantic document retrieval.

    Combines FAISS indexing with SentenceTransformers embeddings for
    fast similarity search over document collections.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        metadata_path: Union[str, Path],
        model_name: str = MODEL_NAME_DEFAULT,
        device: DeviceType = "auto",
    ) -> None:
        """
        Initialize vector store with FAISS index and embedding model.

        Args:
            index_path: Path to pre-built FAISS index file
            metadata_path: Path to JSON file containing document metadata
            model_name: HuggingFace model identifier for embeddings
            device: Target device ('cpu', 'cuda', 'mps', 'auto')
        """
        self.device = self._resolve_device(device)
        logger.info(f"Initializing VectorStore on device: {self.device}")

        # Load embedding model
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Loaded embedding model '{model_name}'")
        logger.info(f"Model device: {self.model.device}")

        # Load FAISS index and metadata
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._load_index_and_metadata()

    def _resolve_device(self, device: DeviceType) -> str:
        """Resolve device specification to actual device string."""
        if device == "auto":
            detected = llm_helper.detect_device()
            logger.info(f"Auto-detected device: {detected}")
            return detected

        if device in ["cpu", "cuda", "mps"]:
            return device

        raise ValueError(f"Invalid device '{device}'. Must be: cpu, cuda, mps, auto")

    def _load_index_and_metadata(self) -> None:
        """Load FAISS index and document metadata from disk."""

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load document metadata
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"Loaded metadata for {len(self.data)} documents")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse metadata JSON: {e}") from e

    def _to_similarity(self, distance: float) -> float:
        """Convert FAISS distance to similarity score (0 to 1)."""
        # L2 distance to cosine similarity approximation
        if isinstance(self.index, faiss.IndexFlatL2):
            return 1 / (1 + distance)
        else:
            return distance

    def reload_index(self) -> None:
        """Reload FAISS index and metadata from disk.
        Useful if the index or metadata files have been updated.
        """
        try:
            self._load_index_and_metadata()
            logger.info("Successfully reloaded FAISS index and metadata")
        except Exception as e:
            logger.error(f"Failed to reload index or metadata: {e}")
            raise

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search over the document collection.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of search results with 'chunk', 'score', and 'rank' keys
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = await asyncio.to_thread(
            self.model.encode,
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Ensure embedding is float32
        query_embedding = np.array(query_embedding, dtype="float32")

        # FAISS similarity search
        distances, indices = self.index.search(query_embedding, top_k)

        if len(indices) == 0 or len(distances) == 0:
            return []

        # Format results
        results = []
        for rank, (distance, doc_index) in enumerate(zip(distances[0], indices[0])):
            if doc_index < 0 or doc_index >= len(self.data):
                continue

            results.append(
                {
                    "chunk": self.data[doc_index],
                    "score": self._to_similarity(float(distance)),
                    "rank": rank + 1,
                }
            )

        return results
