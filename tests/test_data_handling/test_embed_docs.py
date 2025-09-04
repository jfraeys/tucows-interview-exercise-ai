import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest

from src import models
from src.data_handling import embed_docs


@pytest.fixture
def tmp_docs_dir(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Q1: What is test?\nA: This is a test answer.")

    md_file = tmp_path / "sample.md"
    md_file.write_text("## 1. Section Title\n### 1.1 Subsection\nContent here.")

    return tmp_path


def test_load_documents(tmp_docs_dir):
    chunks = embed_docs.load_documents(tmp_docs_dir)
    assert len(chunks.chunks) >= 2

    # Check attributes
    for c in chunks.chunks:
        assert isinstance(c.text, str)
        assert isinstance(c.metadata.source, str)


def test_unsupported_file_type(tmp_path):
    unsupported_file = tmp_path / "sample.unsupported"
    unsupported_file.write_text("This file has an unsupported extension.")
    with pytest.raises(ValueError, match="No documents were successfully processed"):
        embed_docs.load_documents(tmp_path)


def test_embed_texts():
    texts = ["This is a test.", "Another sentence."]
    embeddings = embed_docs.embed_texts(texts, "all-MiniLM-L6-v2")
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2


def test_build_faiss_index():
    embeddings = np.random.rand(5, 384).astype("float32")
    index = embed_docs.build_faiss_index(embeddings)
    assert isinstance(index, faiss.IndexFlatL2)
    assert index.ntotal == 5


def test_save_index_and_metadata(tmp_path):
    embeddings = np.random.rand(3, 384).astype("float32")
    index = embed_docs.build_faiss_index(embeddings)
    metadata = models.ParserResponse(
        chunks=[
            models.Chunks(
                id=f"id_{i}",
                text=f"Text {i}",
                metadata=models.ChunksMetadata(
                    source=f"file{i}.txt",
                    section=f"Chunk {i}",
                ),
            )
            for i in range(3)
        ]
    )
    index_file = tmp_path / "faiss.index"
    metadata_file = tmp_path / "metadata.json"

    embed_docs.save_index_and_metadata(index, metadata, index_file, metadata_file)

    assert index_file.exists()
    assert metadata_file.exists()

    loaded_metadata = json.loads(metadata_file.read_text())
    expected_metadata = [chunk.model_dump() for chunk in metadata.chunks]

    def strip_none(d):
        if isinstance(d, dict):
            return {k: strip_none(v) for k, v in d.items() if v is not None}
        if isinstance(d, list):
            return [strip_none(i) for i in d]
        return d

    assert strip_none(loaded_metadata) == strip_none(expected_metadata)


class TestLoadDocuments:
    """Test the core document loading functionality"""

    def test_directory_not_found(self):
        """Should raise FileNotFoundError for non-existent directory"""
        with pytest.raises(FileNotFoundError):
            embed_docs.load_documents(Path("/nonexistent"))

    def test_no_processable_files(self, tmp_path):
        """Should raise ValueError when no files can be processed"""
        # Create directory with no supported files
        with pytest.raises(
            ValueError, match="No documents were successfully processed"
        ):
            embed_docs.load_documents(tmp_path)

    @patch("src.data_handling.embed_docs.parse_file")
    def test_mixed_success_failure(self, mock_parse, tmp_path):
        """Should continue processing when some files fail"""
        # Create test files
        (tmp_path / "good.txt").write_text("content")
        (tmp_path / "bad.txt").write_text("content")

        def parse_side_effect(path):
            if "good" in path:
                return models.ParserResponse(
                    chunks=[
                        models.Chunks(
                            id="1",
                            text="Good content",
                            metadata=models.ChunksMetadata(
                                source=path, section="section1"
                            ),
                        )
                    ]
                )
            raise ValueError("Parse error")

        mock_parse.side_effect = parse_side_effect
        result = embed_docs.load_documents(tmp_path)

        assert len(result.chunks) == 1
        assert result.chunks[0].text == "Good content"


class TestEmbedTexts:
    """Test embedding generation with practical scenarios"""

    def test_empty_input(self):
        """Should reject empty text list"""
        with pytest.raises(ValueError, match="empty text list"):
            embed_docs.embed_texts([], "model")

    def test_no_valid_text(self):
        """Should reject when no valid text after filtering"""
        with pytest.raises(ValueError, match="No valid text content"):
            embed_docs.embed_texts(["", "   "], "model")

    @patch("src.data_handling.embed_docs.SentenceTransformer")
    def test_model_loading_error(self, mock_st):
        """Should handle model loading failures gracefully"""
        mock_st.side_effect = Exception("Model not found")

        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            embed_docs.embed_texts(["test"], "bad-model")

    @patch("src.data_handling.embed_docs.SentenceTransformer")
    def test_encoding_error(self, mock_st):
        """Should handle encoding failures"""
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = Exception("Encoding failed")
        mock_st.return_value = mock_embedder

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            embed_docs.embed_texts(["test"], "model")

    @patch("src.data_handling.embed_docs.SentenceTransformer")
    def test_filters_invalid_text(self, mock_st):
        """Should filter out invalid text but process valid ones"""
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[1, 2], [3, 4]])
        mock_st.return_value = mock_embedder

        result = embed_docs.embed_texts(["good text", "", "another good"], "model")

        # Should have called encode with only valid texts
        call_args = mock_embedder.encode.call_args[0][0]
        assert call_args == ["good text", "another good"]
        assert result.shape == (2, 2)


class TestBuildFaissIndex:
    """Test FAISS index creation with key error cases"""

    def test_empty_embeddings(self):
        """Should reject empty embeddings"""
        with pytest.raises(ValueError, match="empty embeddings"):
            embed_docs.build_faiss_index(np.array([]))

    def test_wrong_dimensions(self):
        """Should reject non-2D arrays"""
        with pytest.raises(ValueError, match="2D array"):
            embed_docs.build_faiss_index(np.array([1, 2, 3]))

    @patch("faiss.IndexFlatL2")
    def test_faiss_creation_error(self, mock_index):
        """Should handle FAISS index creation errors"""
        mock_index.side_effect = Exception("FAISS error")

        with pytest.raises(RuntimeError, match="Failed to build FAISS index"):
            embed_docs.build_faiss_index(np.random.rand(3, 5).astype("float32"))


class TestSaveIndexAndMetadata:
    """Test saving functionality with common failure scenarios"""

    @patch("faiss.write_index")
    def test_index_save_error(self, mock_write, tmp_path):
        """Should handle index writing errors"""
        mock_write.side_effect = Exception("Write failed")

        with pytest.raises(RuntimeError, match="Failed to save index"):
            embed_docs.save_index_and_metadata(
                MagicMock(),
                models.ParserResponse(chunks=[]),
                tmp_path / "index",
                tmp_path / "meta.json",
            )

    @patch("faiss.write_index")
    def test_metadata_save_error(self, mock_write, tmp_path):
        """Should handle metadata writing errors"""
        metadata = MagicMock()
        metadata.to_json.side_effect = Exception("JSON error")

        with pytest.raises(RuntimeError, match="Failed to save metadata"):
            embed_docs.save_index_and_metadata(
                MagicMock(), metadata, tmp_path / "index", tmp_path / "meta.json"
            )


class TestBuildEmbeddings:
    """Test the main pipeline function"""

    @patch("src.data_handling.embed_docs.load_documents")
    @patch("src.data_handling.embed_docs.embed_texts")
    @patch("src.data_handling.embed_docs.build_faiss_index")
    @patch("src.data_handling.embed_docs.save_index_and_metadata")
    def test_full_pipeline_success(self, mock_save, mock_index, mock_embed, mock_load):
        """Should execute full pipeline successfully"""
        # Setup mocks
        mock_load.return_value = models.ParserResponse(
            chunks=[
                models.Chunks(
                    id="1",
                    text="test",
                    metadata=models.ChunksMetadata(
                        source="test.txt", section="section1"
                    ),
                )
            ]
        )
        mock_embed.return_value = np.array([[1, 2, 3]])
        mock_index_obj = MagicMock()
        mock_index_obj.ntotal = 1
        mock_index.return_value = mock_index_obj

        # Execute
        embed_docs.build_embeddings("/docs", "/index", "meta.json")

        # Verify pipeline was called
        mock_load.assert_called_once()
        mock_embed.assert_called_once_with(["test"], "all-MiniLM-L6-v2")
        mock_index.assert_called_once()
        mock_save.assert_called_once()

    @patch("src.data_handling.embed_docs.load_documents")
    def test_pipeline_error_propagation(self, mock_load):
        """Should propagate errors from pipeline steps"""
        mock_load.side_effect = FileNotFoundError("No docs")

        with pytest.raises(FileNotFoundError):
            embed_docs.build_embeddings("/bad", "/index", "meta.json")

    @patch("src.data_handling.embed_docs.load_documents")
    @patch("src.data_handling.embed_docs.embed_texts")
    @patch("src.data_handling.embed_docs.build_faiss_index")
    @patch("src.data_handling.embed_docs.save_index_and_metadata")
    def test_custom_embedding_model(self, mock_save, mock_index, mock_embed, mock_load):
        """Should use custom embedding model when specified"""
        mock_load.return_value = models.ParserResponse(
            chunks=[
                models.Chunks(
                    id="1",
                    text="test",
                    metadata=models.ChunksMetadata(
                        source="test.txt", section="section1"
                    ),
                )
            ]
        )
        mock_embed.return_value = np.array([[1, 2]])
        mock_index.return_value = MagicMock()

        embed_docs.build_embeddings("/docs", "/index", "meta.json", "custom-model")

        mock_embed.assert_called_once_with(["test"], "custom-model")
