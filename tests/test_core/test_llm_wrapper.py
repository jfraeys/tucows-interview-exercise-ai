"""
Tests for LLM wrapper - focused on core functionality for week-long project.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core import llm_wrapper


class TestLLMWrapper:
    """Core tests for llm_wrapper.LLMWrapper."""

    @pytest.fixture
    def mock_llama(self):
        """Mock Llama model."""
        mock_model = Mock()
        mock_model.create_completion.return_value = {
            "choices": [{"text": " Generated response"}],
            "usage": {"total_tokens": 10},
        }
        return mock_model

    @pytest.fixture
    def temp_model_file(self):
        """Temporary model file."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"fake model")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_init_local_model(self, mock_llama, temp_model_file):
        """Test initialization with local model."""
        with (
            patch("src.core.llm_wrapper.Llama", return_value=mock_llama),
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
        ):
            wrapper = llm_wrapper.LLMWrapper(model_path=temp_model_file)

            assert wrapper.model == mock_llama
            assert wrapper.device_mode == "cpu"  # default with 0 GPU layers

    def test_init_huggingface_model(self, mock_llama):
        """Test initialization with HuggingFace model."""
        with (
            patch("src.core.llm_wrapper.Llama") as mock_llama_class,
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
        ):
            mock_llama_class.from_pretrained.return_value = mock_llama

            wrapper = llm_wrapper.LLMWrapper(
                repo_id="test/model", filename="model.gguf"
            )

            assert wrapper.model == mock_llama
            mock_llama_class.from_pretrained.assert_called_once()

    def test_device_mode_resolution(self):
        """Test device mode is set to CPU when no GPU layers."""
        with (
            patch("src.core.llm_wrapper.Llama"),
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            wrapper = llm_wrapper.LLMWrapper(
                model_path="fake.gguf", device_mode="gpu", n_gpu_layers=0
            )

            assert wrapper.device_mode == "cpu"

    def test_model_file_not_found(self):
        """Test error when model file doesn't exist."""
        with patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"):
            with pytest.raises(FileNotFoundError):
                llm_wrapper.LLMWrapper(model_path="/nonexistent/model.gguf")

    def test_missing_huggingface_params(self):
        """Test error when HF params are incomplete."""
        with patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"):
            with pytest.raises(
                ValueError, match="Both repo_id and filename are required"
            ):
                llm_wrapper.LLMWrapper(repo_id="test/repo")  # missing filename

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_llama):
        """Test successful text generation."""
        with (
            patch("src.core.llm_wrapper.Llama", return_value=mock_llama),
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            wrapper = llm_wrapper.LLMWrapper(model_path="fake.gguf")

            result = await wrapper.generate("Hello")

            mock_llama.create_completion.assert_called_once()
            assert result["choices"][0]["text"] == " Generated response"

    @pytest.mark.asyncio
    async def test_generate_with_params(self, mock_llama):
        """Test generation with custom parameters."""
        with (
            patch("src.core.llm_wrapper.Llama", return_value=mock_llama),
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            wrapper = llm_wrapper.LLMWrapper(model_path="fake.gguf")

            await wrapper.generate(
                "Hello", max_tokens=100, temperature=0.5, stop_sequences=["<end>"]
            )

            mock_llama.create_completion.assert_called_once_with(
                prompt="Hello",
                max_tokens=100,
                temperature=0.5,
                top_p=1.0,
                stop=["<end>"],
                echo=False,
                stream=False,
            )

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, mock_llama):
        """Test error with empty prompt."""
        with (
            patch("src.core.llm_wrapper.Llama", return_value=mock_llama),
            patch("src.core.llm_wrapper.validate_helper.validate_model_arguments"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            wrapper = llm_wrapper.LLMWrapper(model_path="fake.gguf")

            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                await wrapper.generate("")

    def test_ensure_local_directory_directly(self):
        """Test the _ensure_local_directory method directly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "models")
            new_dir_path = Path(new_dir)

            with patch("os.makedirs") as mock_makedirs:
                # Create wrapper without calling __init__
                wrapper = llm_wrapper.LLMWrapper.__new__(llm_wrapper.LLMWrapper)
                wrapper.local_dir = new_dir_path

                # Test the directory creation method
                wrapper._ensure_local_directory()

                # Verify makedirs was called correctly
                mock_makedirs.assert_called_once_with(new_dir_path, exist_ok=True)

    @patch("llama_cpp.Llama")
    @patch("llama_cpp.Llama.from_pretrained")
    @patch("src.utils.validate_helper.validate_model_arguments")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)  # No cached model
    def test_local_dir_creation_integration(
        self,
        mock_exists,
        mock_makedirs,
        mock_validate,
        mock_from_pretrained,
        mock_llama,
    ):
        """Test that directory creation works during normal initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, "models")
            new_dir_path = Path(new_dir)

            # Setup mocks
            mock_from_pretrained.return_value = MagicMock()

            # Create wrapper - should trigger directory creation
            wrapper = llm_wrapper.LLMWrapper(
                repo_id="test/models",
                filename="model.gguf",
                local_dir=new_dir,
            )

            # Verify directory creation and final state
            mock_makedirs.assert_called_with(new_dir_path, exist_ok=True)
            assert wrapper.local_dir == new_dir_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
