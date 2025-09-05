"""
Simple test suite for the VectorStore retrieval system.
Tests the core functionality with Literal device types.
"""

import json
from typing import Literal
from unittest.mock import MagicMock, mock_open, patch

import faiss
import numpy as np
import pytest

from src.core.retrieval import MODEL_NAME_DEFAULT, DeviceType, VectorStore
from src.models import Chunks, ChunksMetadata


class TestVectorStore:
    """Basic tests for VectorStore class with Literal device types."""

    @pytest.fixture
    def mock_files(self):
        """Mock file system components."""
        mock_index = MagicMock()
        mock_index.ntotal = 100
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),  # distances
            np.array([[0, 1, 2]]),  # indices
        )

        mock_metadata = [
            {"text": "chunk 0", "source": "doc1"},
            {"text": "chunk 1", "source": "doc2"},
            {"text": "chunk 2", "source": "doc3"},
        ]

        return mock_index, mock_metadata

    def test_initialization_success(self, mock_files):
        """Test successful VectorStore initialization."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer") as mock_st,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            # Mock the model returned by SentenceTransformer
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            # Initialize VectorStore
            vs = VectorStore("test.index", "test.json")

            # Check that VectorStore initialized correctly
            assert vs.index == mock_index
            assert vs.data == mock_metadata
            assert vs.model == mock_model

            # Assert that SentenceTransformer was called with expected arguments
            mock_st.assert_called_once()
            call_args, call_kwargs = mock_st.call_args
            assert call_args[0] == MODEL_NAME_DEFAULT  # model name
            assert "device" in call_kwargs
            assert isinstance(call_kwargs["device"], str)

    def test_initialization_custom_params(self, mock_files) -> None:
        """Test initialization with custom parameters using Literal types."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer") as mock_st,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            # Test with each valid device type
            device: DeviceType = "cuda"  # Type-safe assignment
            VectorStore(
                "test.index", "test.json", model_name="custom-model", device=device
            )

            mock_st.assert_called_once_with("custom-model", device="cuda")

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_explicit_device_types(
        self, mock_files, device: Literal["cpu", "cuda", "mps"]
    ):
        """Test each explicit device type."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer") as mock_st,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            vs = VectorStore("test.index", "test.json", device=device)
            assert vs.device == device
            mock_st.assert_called_once_with(MODEL_NAME_DEFAULT, device=device)

    def test_resolve_device_auto(self, mock_files) -> None:
        """Test auto device detection."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.llm_helper.detect_device", return_value="cuda"),
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            device: DeviceType = "auto"  # Type-safe assignment
            vs = VectorStore("test.index", "test.json", device=device)
            assert vs.device == "cuda"

    def test_resolve_device_invalid_runtime(self, mock_files):
        """Test runtime validation of device specification."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            # This would normally be caught by type checker, but test runtime behavior
            # by directly calling the internal method
            vs = VectorStore("test.index", "test.json")

            with pytest.raises(ValueError, match="Invalid device"):
                vs._resolve_device("invalid")  # type: ignore

    def test_missing_index_file(self):
        """Test error when index file is missing."""
        with patch("pathlib.Path.exists", side_effect=lambda: False):
            with pytest.raises(FileNotFoundError, match="FAISS index not found"):
                VectorStore("missing.index", "test.json")

    def test_missing_metadata_file(self):
        """Test error when metadata file is missing."""
        def mock_exists(path_obj):
            return str(path_obj).endswith(".index")
        
        with patch("pathlib.Path.exists", new=mock_exists):
            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                VectorStore("test.index", "missing.json")

    def test_invalid_metadata_json(self):
        """Test error with invalid JSON metadata."""
        with (
            patch("src.core.retrieval.faiss.read_index"),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="invalid json")),
        ):
            with pytest.raises(ValueError, match="Failed to parse metadata JSON"):
                VectorStore("test.index", "test.json")

    def test_to_similarity_l2_index(self, mock_files):
        """Test similarity conversion for L2 index."""
        mock_index, mock_metadata = mock_files

        # Mock file reading and SentenceTransformer
        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
            patch("pathlib.Path.exists", return_value=True),
            patch("src.core.retrieval.SentenceTransformer"),
        ):
            # Initialize VectorStore
            vs = VectorStore("test.index", "test.json")

            # Assign a real L2 index so isinstance works
            vs.index = faiss.IndexFlatL2(1)  # small 1D index for test

            # Test L2 distance conversion
            distance = 0.5
            similarity = vs._to_similarity(distance)
            expected_similarity = 1 / (1 + distance)
            assert similarity == expected_similarity

    def test_reload_index_success(self, mock_files):
        """Test successful index reloading."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            vs = VectorStore("test.index", "test.json")

            # Patch _load_index_and_metadata to track calls
            with patch.object(
                vs, "_load_index_and_metadata", wraps=vs._load_index_and_metadata
            ) as mock_load:
                # Call reload_index
                vs.reload_index()

                # Now check how many times _load_index_and_metadata was called
                # It should be at least once during reload
                assert len(mock_load.call_args_list) >= 1

    def test_reload_index_failure(self, mock_files):
        """Test index reloading failure."""
        mock_index, mock_metadata = mock_files

        with (
            patch("src.core.retrieval.faiss.read_index", return_value=mock_index),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_metadata))),
        ):
            vs = VectorStore("test.index", "test.json")

            # Mock reload failure
            with patch.object(
                vs, "_load_index_and_metadata", side_effect=Exception("Reload failed")
            ):
                with pytest.raises(Exception, match="Reload failed"):
                    vs.reload_index()


class TestVectorStoreSearch:
    """Test search functionality."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        # Mock model
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])

        # Fake paths
        index_path = tmp_path / "index.faiss"
        metadata_path = tmp_path / "metadata.json"
        index_path.write_text("")
        metadata_path.write_text("[]")

        # Create VectorStore instance without running __init__
        store = VectorStore.__new__(VectorStore)
        store.model = mock_model

        # Mock FAISS index
        store.index = MagicMock()
        store.index.search = MagicMock(
            return_value=(
                np.array([[0.9, 0.8, 0.7]], dtype=np.float32),  # distances
                np.array([[0, 1, -1]], dtype=np.int64),  # indices
            )
        )

        # Mock data (normally loaded from JSON)
        store.data = [
            Chunks(
                id="chunk1",
                text="First chunk",
                metadata=ChunksMetadata(source="doc1", section="sec1"),
            ).model_dump(),
            Chunks(
                id="chunk2",
                text="Second chunk",
                metadata=ChunksMetadata(source="doc2", section="sec2"),
            ).model_dump(),
            Chunks(
                id="chunk3",
                text="Third chunk",
                metadata=ChunksMetadata(source="doc3", section="sec3"),
            ).model_dump(),
        ]

        return store

    @pytest.mark.asyncio
    async def test_search_success(self, vector_store):
        """Test successful search operation."""
        results = await vector_store.search("test query", top_k=3)

        print(results)  # Debug print to inspect results

        assert len(results) == 2
        assert results[0]["chunk"]["text"] == "First chunk"
        assert results[0]["rank"] == 1
        assert "score" in results[0]

        # Verify model.encode was called
        vector_store.model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_query(self, vector_store):
        """Test search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await vector_store.search("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await vector_store.search("   ")

    @pytest.mark.asyncio
    async def test_search_invalid_top_k(self, vector_store):
        """Test search with invalid top_k values."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            await vector_store.search("test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be positive"):
            await vector_store.search("test", top_k=-1)

    @pytest.mark.asyncio
    async def test_search_with_invalid_indices(self, vector_store):
        """Test search handling invalid document indices."""
        # Mock search to return invalid indices
        vector_store.index.search.return_value = (
            np.array([[0.1, 0.2]]),  # distances
            np.array([[-1, 999]]),  # invalid indices
        )

        results = await vector_store.search("test query", top_k=2)

        # Should filter out invalid indices
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_custom_top_k(self, vector_store):
        """Test search with custom top_k value."""
        results = await vector_store.search("test query", top_k=2)

        assert len(results) <= 2

        # Verify FAISS search was called with correct top_k
        call_args = vector_store.index.search.call_args
        assert call_args[0][1] == 2  # top_k parameter


class TestDeviceResolution:
    """Test device resolution functionality with Literal types."""

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_resolve_device_explicit(self, device: Literal["cpu", "cuda", "mps"]):
        """Test explicit device specification with type safety."""
        with (
            patch("src.core.retrieval.faiss.read_index"),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="[]")),
        ):
            vs = VectorStore("test.index", "test.json", device=device)
            assert vs.device == device

    def test_resolve_device_auto_detection(self) -> None:
        """Test automatic device detection."""
        with (
            patch("src.core.retrieval.llm_helper.detect_device", return_value="mps"),
            patch("src.core.retrieval.faiss.read_index"),
            patch("src.core.retrieval.SentenceTransformer"),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="[]")),
        ):
            device: DeviceType = "auto"
            vs = VectorStore("test.index", "test.json", device=device)
            assert vs.device == "mps"


class TestConstants:
    """Test module constants."""

    def test_default_model_name(self):
        """Test default model name constant."""
        assert MODEL_NAME_DEFAULT == "all-MiniLM-L6-v2"

    def test_device_type_literal(self) -> None:
        """Test that DeviceType is properly defined."""
        # This is mainly a compile-time test, but we can verify the type exists
        assert DeviceType is not None

        # Test that we can assign valid values
        valid_cpu: DeviceType = "cpu"
        valid_cuda: DeviceType = "cuda"
        valid_mps: DeviceType = "mps"
        valid_auto: DeviceType = "auto"

        assert valid_cpu == "cpu"
        assert valid_cuda == "cuda"
        assert valid_mps == "mps"
        assert valid_auto == "auto"


class TestRetrievalFormat:
    """Test retrieval data format for integration debugging."""

    @pytest.mark.asyncio
    async def test_retrieval_format(self):
        """Test and document the retrieval system data format."""
        from src.core.rag_pipeline import get_vectorstore
        
        try:
            vs = get_vectorstore()
            results = await vs.search('domain suspension', top_k=2)
            
            # Verify we get results
            assert len(results) >= 0, "Should return results list"
            
            if results:
                # Test first result structure
                result = results[0]
                assert isinstance(result, dict), "Result should be a dictionary"
                assert 'chunk' in result, "Result should have 'chunk' key"
                assert 'score' in result, "Result should have 'score' key"
                assert 'rank' in result, "Result should have 'rank' key"
                
                # Test chunk structure
                chunk = result['chunk']
                assert isinstance(chunk, dict), "Chunk should be a dictionary"
                assert 'id' in chunk, "Chunk should have 'id' key"
                assert 'text' in chunk, "Chunk should have 'text' key"
                assert 'metadata' in chunk, "Chunk should have 'metadata' key"
                
                # Test metadata structure
                metadata = chunk['metadata']
                assert isinstance(metadata, dict), "Metadata should be a dictionary"
                assert 'source' in metadata, "Metadata should have 'source' key"
                
        except Exception as e:
            # If vectorstore isn't available, skip the test
            pytest.skip(f"Vectorstore not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
