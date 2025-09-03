"""
Vector store implementation for semantic document retrieval using FAISS and SentenceTransformers.

Core functionality for RAG systems with clean device management and async operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import faiss
from sentence_transformers import SentenceTransformer

from src.utils import llm_helper, logging_helper

logger = logging_helper.get_logger(__name__)

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
        model_name: str = "all-MiniLM-L6-v2",
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

        # Load FAISS index and metadata
        self._load_index_and_metadata(index_path, metadata_path)

    def _resolve_device(self, device: DeviceType) -> str:
        """Resolve device specification to actual device string."""
        if device == "auto":
            detected = llm_helper.detect_device()
            logger.info(f"Auto-detected device: {detected}")
            return detected

        if device in ["cpu", "cuda", "mps"]:
            return device

        raise ValueError(f"Invalid device '{device}'. Must be: cpu, cuda, mps, auto")

    def _load_index_and_metadata(
        self, index_path: Union[str, Path], metadata_path: Union[str, Path]
    ) -> None:
        """Load FAISS index and document metadata from disk."""
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load document metadata
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info(f"Loaded metadata for {len(self.data)} documents")

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

        # Generate query embedding with device-appropriate async handling
        if self.device == "cpu":
            query_embedding = await asyncio.to_thread(
                self.model.encode,
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            query_embedding = self.model.encode(
                [query], convert_to_numpy=True, show_progress_bar=False
            )

        # FAISS similarity search
        distances, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for rank, (distance, doc_index) in enumerate(zip(distances[0], indices[0])):
            if doc_index < 0 or doc_index >= len(self.data):
                continue

            results.append(
                {
                    "chunk": self.data[doc_index],
                    "score": float(distance),
                    "rank": rank + 1,
                }
            )

        return results
