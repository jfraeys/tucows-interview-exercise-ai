"""
Retrieval-Augmented Generation (RAG) system with Model Context Protocol (MCP) support.

This module implements a RAG pipeline that combines vector-based document retrieval
with large language model generation to provide contextually relevant answers.
"""

import asyncio
import json
from pathlib import Path

from src.core import llm_wrapper, prompt_builder, retrieval
from src.utils import json_schemas, logging_helper, validate_helper

logger = logging_helper.get_logger(__name__)

INDEX_PATH = "data/faiss_index/faiss.index"
METADATA_PATH = "data/faiss_index/metadata.json"

TOP_K_DEFAULT = 3
TIMEOUT_DEFAULT = 30.0  # seconds - reduced from 60
RETRIEVAL_TIMEOUT = 10.0  # seconds - faster retrieval timeout


class _RAGSingleton:
    """Singleton to cache vectorstore and LLM instances."""
    def __init__(self):
        self._vectorstore = None
        self._llm = None
        self._vectorstore_config = None
        self._llm_config = None
    
    def get_vectorstore(self, index_path=None, metadata_path=None):
        config = (index_path or INDEX_PATH, metadata_path or METADATA_PATH)
        if self._vectorstore is None or self._vectorstore_config != config:
            self._vectorstore = retrieval.VectorStore(
                index_path=config[0],
                metadata_path=config[1]
            )
            self._vectorstore_config = config
        return self._vectorstore
    
    def get_llm(self, model_path=None, model_repo=None, model_filename=None, n_gpu_layers=-1):
        config = (model_path, model_repo, model_filename, n_gpu_layers)
        if self._llm is None or self._llm_config != config:
            validate_helper.validate_model_arguments(
                model_path=model_path,
                model_repo=model_repo,
                model_filename=model_filename,
            )
            if model_path:
                self._llm = llm_wrapper.LLMWrapper(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                )
            else:
                self._llm = llm_wrapper.LLMWrapper(
                    repo_id=model_repo,
                    filename=model_filename,
                    n_gpu_layers=n_gpu_layers,
                )
            self._llm_config = config
        return self._llm


_rag_singleton = _RAGSingleton()



class RetrievalQA:
    """
    Retrieval-Augmented Generation (RAG) question-answering system.

    This class orchestrates the RAG pipeline by:
    1. Retrieving semantically relevant document chunks from a vector store
    2. Constructing prompts with retrieved context using MCP format
    3. Generating answers using a large language model

    Attributes:
        vectorstore: Vector database for semantic document retrieval
        llm: Language model wrapper for text generation
        top_k: Number of most relevant chunks to retrieve for context
    """

    def __init__(
        self,
        vectorstore: retrieval.VectorStore,
        llm: llm_wrapper.LLMWrapper,
        top_k: int = 2,  # Reduced from 3 for faster retrieval
        stop_sequences: list[str] = ["}\n"],
    ) -> None:
        """
        Initialize the RAG system with required components.

        Args:
            vectorstore: Pre-built vector store containing document embeddings
            llm: Configured language model wrapper for generation
            top_k: Maximum number of relevant chunks to retrieve (default: 3)
        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.stop_sequences = stop_sequences

    @staticmethod
    def _parse_llm_json(output: str) -> dict[str, str]:
        """Parse JSON output from an LLM that may be truncated before the final '}'.
        Ensures 'answer' and 'action_required' exist.

        Args:
            output: Raw string output from the LLM.

        Returns:
            Parsed string containing a valid JSON object.
        Raises:
            RuntimeError: If no JSON object is found or parsing fails.
        """

        output = output.strip()

        # Find the first '{' and take everything after it
        start = output.find("{")

        if start == -1:
            raise RuntimeError(f"No JSON object found in output:\n{output}")

        output = output[start:]

        # Ensure it ends with a closing brace
        if not output.endswith("}"):
            output += "}"

        # Remove trailing commas before the closing brace
        output = output.replace(",}", "}")

        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM output as JSON: {output}")
            raise RuntimeError("Invalid JSON output from LLM") from e

    async def warmup(self) -> None:
        """
        Warm up the LLM to reduce initial latency.
        Sends a dummy prompt to trigger model loading into memory.
        """
        try:
            dummy_prompt = "Warmup prompt: Say 'ready'."
            await self.llm.generate(dummy_prompt, stop_sequences=self.stop_sequences)
            logger.info("LLM warmup completed successfully.")
        except Exception as e:
            logger.warning(f"LLM warmup failed (ignored): {e}")

    async def answer(self, query: str) -> dict[str, str]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        try:
            # Add timeout for retrieval operations
            retrieved_chunks = await asyncio.wait_for(
                self.vectorstore.search(query, top_k=self.top_k), timeout=RETRIEVAL_TIMEOUT
            )

            if not retrieved_chunks:
                logger.warning("No relevant chunks retrieved for query")
                # Fallback strategy needed here

        except asyncio.TimeoutError:
            raise TimeoutError("Document retrieval timed out")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Failed to retrieve context: {e}") from e

        try:
            prompt = prompt_builder.build_mcp_prompt(query, retrieved_chunks)
        except Exception as e:
            logger.error(f"Prompt building failed: {e}")
            raise RuntimeError(f"Failed to build prompt: {e}") from e

        try:
            raw_output = await asyncio.wait_for(
                self.llm.generate(
                    prompt, 
                    max_tokens=300,  # Reduced from default for faster generation
                    temperature=0.1,  # Lower temperature for faster, more deterministic responses
                    stop_sequences=self.stop_sequences
                ),
                timeout=TIMEOUT_DEFAULT,
            )

            # Safe access to nested response structure
            if not raw_output or "choices" not in raw_output:
                raise ValueError("Invalid LLM response format")

            choices = raw_output["choices"]
            if not choices or not isinstance(choices, list):
                raise ValueError("No choices in LLM response")

            if "text" not in choices[0]:
                raise ValueError("No text in LLM response")

            # Parse the JSON from the LLM output
            res = self._parse_llm_json(choices[0]["text"])

            return json_schemas.enforce_ticket_schema(res)

        except asyncio.TimeoutError:
            raise TimeoutError("LLM generation timed out")
        except KeyError as e:
            raise ValueError(f"Unexpected LLM response format: missing {e}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e


def get_vectorstore(
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> retrieval.VectorStore:
    """Get cached vector store instance (creates once, reuses thereafter)."""
    return _rag_singleton.get_vectorstore(index_path, metadata_path)


def get_llm(
    model_path: Path | None = None,
    model_repo: str | None = None,
    model_filename: str | None = None,
    n_gpu_layers: int = -1,
) -> llm_wrapper.LLMWrapper:
    """
    Get cached LLM wrapper instance (creates once, reuses thereafter).

    Args:
        model_path: Local filesystem path to the LLM model file (GGUF format)
        model_repo: Hugging Face repository ID for the model (if not using local file)
        model_filename: Filename pattern for the GGUF file in the HF repo
        n_gpu_layers: Number of model layers to offload to GPU (-1 for auto, 0 for CPU only)

    Returns:
        LLMWrapper instance
    """
    return _rag_singleton.get_llm(model_path, model_repo, model_filename, n_gpu_layers)


def initialize_rag(
    n_gpu_layers: int,
    model_path: Path | None = None,
    model_repo: str | None = None,
    model_filename: str | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[retrieval.VectorStore, llm_wrapper.LLMWrapper]:
    """
    Initialize the vector store and LLM wrapper based on configuration. Validates input arguments.

    Args:
        model_path: Local filesystem path to the LLM model file (GGUF format)
        model_repo: Hugging Face repository ID for the model (if not using local file)
        model_filename: Filename pattern for the GGUF file in the HF repo
        index_path: Path to the FAISS index file for the vector store
        metadata_path: Path to the metadata JSON file for the vector store
        n_gpu_layers: Number of model layers to offload to GPU (-1 for auto, 0 for CPU only)

    Returns:
        Tuple of (VectorStore, LLMWrapper) instances
    """
    vectorstore = get_vectorstore(index_path=index_path, metadata_path=metadata_path)
    llm = get_llm(
        model_path=model_path,
        model_repo=model_repo,
        model_filename=model_filename,
        n_gpu_layers=n_gpu_layers,
    )

    return vectorstore, llm
