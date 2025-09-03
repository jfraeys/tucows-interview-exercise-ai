"""
Retrieval-Augmented Generation (RAG) system with Model Context Protocol (MCP) support.

This module implements a RAG pipeline that combines vector-based document retrieval
with large language model generation to provide contextually relevant answers.
"""

import argparse
import asyncio
import json

from src.core import llm_wrapper, prompt_builder, retrieval
from src.utils import json_schemas, logging_helper, validate_helper

logger = logging_helper.get_logger(__name__)

INDEX_PATH = "data/faiss_index/faiss.index"
METADATA_PATH = "data/faiss_index/metadata.json"


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
        top_k: int = 3,
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
    def _parse_llm_json(output: str) -> str:
        """
        Parse JSON output from an LLM that may be truncated before the final '}'.
        Ensures 'answer' and 'action_required' exist.

        Args:
            output: Raw string output from the LLM.

        Returns:
            Parsed string containing a valid JSON object.
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

    async def answer(self, query: str) -> dict[str, str]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        try:
            # Add timeout for retrieval operations
            retrieved_chunks = await asyncio.wait_for(
                self.vectorstore.search(query, top_k=self.top_k), timeout=30.0
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
                self.llm.generate(prompt, stop_sequences=self.stop_sequences),
                timeout=120.0,
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


def _create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.

    Supports two model loading modes:
    - Local file: --model-path
    - Hugging Face Hub: --model-repo + --model-filename

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Run RAG-based question answering with configurable LLM backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
              # Using local model file
              python script.py --model-path ./models/llama-7b-q8_0.gguf --n-gpu-layers 32

              # Using Hugging Face model
              python script.py --model-repo microsoft/DialoGPT-medium --model-filename "*q8_0.gguf"
        """,
    )

    # Model source configuration (mutually exclusive groups would be better here)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to local LLM model file (GGUF format)",
        metavar="PATH",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        help="Hugging Face repository ID for pre-trained model",
        metavar="REPO_ID",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        help="Filename pattern for GGUF file in HF repo (e.g., '*q8_0.gguf')",
        metavar="PATTERN",
    )

    parser.add_argument(
        "--index-path",
        type=str,
        default=INDEX_PATH,
        help="Path to FAISS index file",
        metavar="PATH",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=METADATA_PATH,
        help="Path to metadata JSON file for vector store",
        metavar="PATH",
    )

    # Performance and runtime configuration
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of model layers to offload to GPU (0=CPU only, -1=auto)",
        metavar="N",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and debug output",
    )

    # user query input (for future extension)
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="User query to process (if not provided, a default example is used)",
        metavar="TEXT",
    )
    return parser


def _initialize_components(
    model_path: str,
    model_repo: str,
    model_filename: str,
    index_path: str,
    metadata_path: str,
    n_gpu_layers: int,
) -> tuple[retrieval.VectorStore, llm_wrapper.LLMWrapper]:
    """
    Initialize the vector store and LLM wrapper based on configuration.

    Args:
        args: Validated command-line arguments

    Returns:
        Tuple of (VectorStore, LLMWrapper) instances
    """
    # Initialize vector store with pre-built FAISS index
    vectorstore = retrieval.VectorStore(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    # Initialize LLM based on model source (local file vs Hugging Face)
    if model_path:
        # Load model from local filesystem
        llm = llm_wrapper.LLMWrapper(model_path=model_path, n_gpu_layers=n_gpu_layers)
    else:
        # Download and load model from Hugging Face Hub
        llm = llm_wrapper.LLMWrapper(
            repo_id=model_repo,
            filename=model_filename,
            n_gpu_layers=n_gpu_layers,
        )

    return vectorstore, llm


async def main() -> None:
    """
    Main application entry point.

    Orchestrates the complete RAG pipeline:
    1. Parse and validate command-line arguments
    2. Initialize vector store and language model
    3. Create RAG system and process example query
    4. Display results
    """
    # Parse and validate command-line configuration
    parser = _create_argument_parser()
    args = parser.parse_args()
    validate_helper.validate_model_arguments(
        args.model_path, args.model_repo, args.model_filename
    )

    try:
        # Initialize core RAG components
        vectorstore, llm = _initialize_components(
            args.model_path,
            args.model_repo,
            args.model_filename,
            args.index_path,
            args.metadata_path,
            args.n_gpu_layers,
        )

        # Create the RAG question-answering system
        qa_system = RetrievalQA(
            vectorstore=vectorstore,
            llm=llm,
            top_k=10,  # Retrieve top 3 most relevant chunks
        )

        # Process example query (TODO: Make this configurable)
        example_query = "How do I reactivate a suspended domain?"

        user_query = args.query if args.query else example_query

        if args.verbose:
            logger.info(f"Processing query: {user_query}")
            logger.info("Retrieving relevant context and generating answer...")

        # Generate answer using the RAG pipeline
        answer = await qa_system.answer(user_query)

        logger.info("LLM Answer:\n\n%s", json.dumps(answer, indent=2))
    except Exception as e:
        logger.exception(f"Error during RAG processing: {e}")
        raise


if __name__ == "__main__":
    # Run the async main function with proper event loop handling
    asyncio.run(main())
