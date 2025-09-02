import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_handling.parser import parse_file
from src.models import ParserResponse
from src.utils import logger as logger_module

logger = logger_module.get_logger(__name__)

DOCS_DIR = Path("data/raw")
INDEX_DIR = Path("data/faiss_index")
METADATA_FILE = INDEX_DIR / "metadata.json"
INDEX_FILE = INDEX_DIR / "faiss.index"


def load_documents(docs_dir: Path) -> ParserResponse:
    """
    Load all documents from the directory using the parser module.
    Returns a list of chunks with 'text' and metadata.
    """
    response: ParserResponse = ParserResponse()
    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            parsed = parse_file(str(file_path))
            response.chunks.extend(parsed.chunks)
    return response


def embed_texts(texts, model_name: str):
    """
    Convert a list of texts into embeddings using a SentenceTransformer model.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return np.array(embeddings).astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index from embeddings and return it.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index_and_metadata(
    index, metadata: list, index_file: Path, metadata_file: Path
):
    """
    Save the FAISS index and the metadata to disk.
    """
    index_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"FAISS index saved to {index_file}")
    logger.info(f"Metadata saved to {metadata_file}")


def main(model_name: str):
    logger.info(f"Using transformer model: {model_name}")
    logger.info(f"Loading documents from {DOCS_DIR}...")
    chunks = load_documents(DOCS_DIR)
    logger.info(f"Total chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name)
    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, chunks, INDEX_FILE, METADATA_FILE)
    logger.info("Embedding and FAISS index creation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed support docs and create FAISS index"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model to use",
    )
    args = parser.parse_args()
    main(args.model)
