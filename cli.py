#!/usr/bin/env python3
"""
Command-line interface for the Retrieval-Augmented Generation (RAG) system.

This CLI provides two main operations:
- query: Run question-answering queries
- embed: Build vector embeddings from documents
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from src.utils import logging_helper

logger = logging_helper.get_logger(__name__)

# Default paths
INDEX_PATH = "data/faiss_index/faiss.index"
METADATA_PATH = "data/faiss_index/metadata.json"


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="Retrieval-Augmented Generation (RAG) system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query with local model
  %(prog)s query --model-path ./models/llama-7b.gguf -q "How do I reset my password?"

  # Query with HuggingFace model
  %(prog)s query --model-repo microsoft/DialoGPT-medium --model-filename "*q8_0.gguf" -q "Help with domains"

  # Build embeddings from documents
  %(prog)s embed --documents-path ./docs/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run question-answering queries")
    _add_query_args(query_parser)

    # Embed subcommand
    embed_parser = subparsers.add_parser(
        "embed", help="Build vector embeddings from documents"
    )
    _add_embed_args(embed_parser)

    return parser


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the query subcommand."""
    # Model configuration
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

    # Index paths
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

    # Performance configuration
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=0,
        help="Number of model layers to offload to GPU (0=CPU only, -1=auto)",
        metavar="N",
    )

    # Query configuration
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="User query to process (if not provided, uses default example)",
        metavar="TEXT",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of most relevant chunks to retrieve (default: 10)",
        metavar="N",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and debug output",
    )


def _add_embed_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the embed subcommand."""
    parser.add_argument(
        "--documents-path",
        type=str,
        required=True,
        help="Path to documents directory to embed",
        metavar="PATH",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=INDEX_PATH,
        help="Output path for FAISS index file",
        metavar="PATH",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=METADATA_PATH,
        help="Output path for metadata JSON file",
        metavar="PATH",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embeddings",
        metavar="MODEL",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging and debug output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing index and metadata files",
    )


async def handle_query_command(args) -> None:
    """Handle the query subcommand - runs the RAG pipeline."""
    try:
        if args.verbose:
            logger.info("Initializing RAG components...")

        # Import required components from your rag_pipeline module
        from src.core import RetrievalQA, initialize_rag

        # initialize_rag already handles validation, so we just call it directly
        vectorstore, llm = initialize_rag(
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            n_gpu_layers=args.n_gpu_layers,
            model_path=args.model_path,
            model_repo=args.model_repo,
            model_filename=args.model_filename,
        )

        # Create the RAG question-answering system
        qa_system = RetrievalQA(
            vectorstore=vectorstore,
            llm=llm,
            top_k=args.top_k,
        )

        # Process query
        example_query = "How do I reactivate a suspended domain?"
        user_query = args.query if args.query else example_query

        if args.verbose:
            logger.info(f"Processing query: {user_query}")
            logger.info("Retrieving relevant context and generating answer...")

        # Generate answer using the RAG pipeline
        answer = await qa_system.answer(user_query)

        # answer() already returns a validated dict, just display it
        print("RAG Answer:")
        print(json.dumps(answer, indent=2))

    except Exception as e:
        logger.error(f"Query command failed: {e}")
        sys.exit(1)


async def handle_embed_command(args) -> None:
    """Handle the embed subcommand - builds vector embeddings."""
    try:
        # Check if documents path exists
        docs_path = Path(args.documents_path)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents path does not exist: {docs_path}")

        # Check if index already exists
        index_path = Path(args.index_path)
        if index_path.exists() and not args.force:
            response = input(
                f"Index already exists at {index_path}. Overwrite? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                logger.info("Embedding cancelled.")
                return

        if args.verbose:
            logger.info(f"Building embeddings from documents in: {docs_path}")
            logger.info(f"Embedding model: {args.embedding_model}")

        # Create output directory if it doesn't exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("Starting document embedding process...")

        # Import embedding functionality
        from src.data_handling import build_embeddings

        # Build embeddings using your existing embed_docs module
        build_embeddings(
            documents_path=str(docs_path),
            index_path=args.index_path,
            metadata_path=args.metadata_path,
            embedding_model=args.embedding_model,
        )

        logger.info("Embeddings built successfully!")
        logger.info(f"  Index saved to: {args.index_path}")
        logger.info(f"  Metadata saved to: {args.metadata_path}")

    except Exception as e:
        logger.error(f"Embedding command failed: {e}")
        sys.exit(1)


async def main() -> None:
    """Main CLI entry point."""
    parser = create_main_parser()
    args = parser.parse_args()

    # Set up logging level
    if hasattr(args, "verbose") and args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Handle different commands
    if args.command == "query":
        await handle_query_command(args)
    elif args.command == "embed":
        await handle_embed_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
