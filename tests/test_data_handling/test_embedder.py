import json

import faiss
import numpy as np
import pytest

from src.data_handling.embedder import (build_faiss_index, embed_texts,
                                        load_documents,
                                        save_index_and_metadata)


@pytest.fixture
def tmp_docs_dir(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Q1: What is test?\nA: This is a test answer.")

    md_file = tmp_path / "sample.md"
    md_file.write_text("## 1. Section Title\n### 1.1 Subsection\nContent here.")

    return tmp_path


def test_load_documents(tmp_docs_dir):
    chunks = load_documents(tmp_docs_dir)
    assert len(chunks.chunks) >= 2

    # Check attributes
    for c in chunks.chunks:
        assert isinstance(c.text, str)
        assert isinstance(c.metadata.source, str)


def test_unsupported_file_type(tmp_path):
    unsupported_file = tmp_path / "sample.unsupported"
    unsupported_file.write_text("This file has an unsupported extension.")
    with pytest.raises(ValueError, match="Unsupported file type: .unsupported"):
        load_documents(tmp_path)


def test_embed_texts():
    texts = ["This is a test.", "Another sentence."]
    embeddings = embed_texts(texts, "all-MiniLM-L6-v2")
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2


def test_build_faiss_index():
    embeddings = np.random.rand(5, 384).astype("float32")
    index = build_faiss_index(embeddings)
    assert isinstance(index, faiss.IndexFlatL2)
    assert index.ntotal == 5


def test_save_index_and_metadata(tmp_path):
    embeddings = np.random.rand(3, 384).astype("float32")
    index = build_faiss_index(embeddings)
    metadata = [
        {"text": f"Text {i}", "source": f"file{i}.txt", "section": f"Chunk {i}"}
        for i in range(3)
    ]
    index_file = tmp_path / "faiss.index"
    metadata_file = tmp_path / "metadata.json"

    save_index_and_metadata(index, metadata, index_file, metadata_file)

    assert index_file.exists()
    assert metadata_file.exists()

    loaded_metadata = json.loads(metadata_file.read_text())
    assert loaded_metadata == metadata
