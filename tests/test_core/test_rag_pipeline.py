"""
Simple test suite for the RAG pipeline system.
Tests the core functionality without over-engineering.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.rag_pipeline import RetrievalQA, get_llm, get_vectorstore, initialize_rag


class TestRetrievalQA:
    """Basic tests for RetrievalQA class."""

    @pytest.fixture
    def mock_components(self):
        """Create mock vectorstore and LLM."""
        vectorstore = MagicMock()
        vectorstore.search = AsyncMock()
        llm = MagicMock()
        llm.generate = AsyncMock()
        return vectorstore, llm

    def test_initialization(self, mock_components):
        """Test basic initialization."""
        vectorstore, llm = mock_components
        rag = RetrievalQA(vectorstore, llm, top_k=5)

        assert rag.vectorstore == vectorstore
        assert rag.llm == llm
        assert rag.top_k == 5

    def test_parse_llm_json_success(self):
        """Test successful JSON parsing."""
        json_str = '{"answer": "test answer", "action_required": "none"}'
        result = RetrievalQA._parse_llm_json(json_str)

        assert result["answer"] == "test answer"
        assert result["action_required"] == "none"

    def test_parse_llm_json_incomplete(self):
        """Test parsing incomplete JSON (missing closing brace)."""
        json_str = '{"answer": "test", "action_required": "none"'
        result = RetrievalQA._parse_llm_json(json_str)

        assert result["answer"] == "test"

    def test_parse_llm_json_invalid(self):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(RuntimeError, match="No JSON object found in output"):
            RetrievalQA._parse_llm_json("invalid json")

    @pytest.mark.asyncio
    async def test_warmup(self, mock_components):
        """Test LLM warmup."""
        vectorstore, llm = mock_components
        rag = RetrievalQA(vectorstore, llm)

        await rag.warmup()

        llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_empty_query(self, mock_components):
        """Test empty query validation."""
        vectorstore, llm = mock_components
        rag = RetrievalQA(vectorstore, llm)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await rag.answer("")

    @pytest.mark.asyncio
    async def test_answer_success(self, mock_components):
        """Test successful answer generation."""
        vectorstore, llm = mock_components
        rag = RetrievalQA(vectorstore, llm)

        # Mock search results
        vectorstore.search.return_value = [
            {"text": "relevant info", "metadata": {"source": "doc1"}}
        ]

        # Mock LLM response
        llm.generate.return_value = {
            "choices": [
                {"text": '{"answer": "Generated answer", "action_required": "none"}'}
            ]
        }

        with (
            patch(
                "src.core.rag_pipeline.prompt_builder.build_mcp_prompt"
            ) as mock_prompt,
            patch(
                "src.core.rag_pipeline.json_schemas.enforce_ticket_schema"
            ) as mock_schema,
        ):
            mock_prompt.return_value = "test prompt"
            mock_schema.return_value = {
                "answer": "Generated answer",
                "action_required": "none",
            }

            result = await rag.answer("test query")

            assert result["answer"] == "Generated answer"
            vectorstore.search.assert_called_once_with("test query", top_k=2)
            llm.generate.assert_called_once()


class TestModuleFunctions:
    """Test module-level helper functions."""

    def test_get_vectorstore(self):
        """Test vectorstore creation."""
        with patch("src.core.rag_pipeline.retrieval.VectorStore") as mock_vs:
            # Clear singleton
            import src.core.rag_pipeline as rag_pipeline

            rag_pipeline._vectorstore = None

            mock_instance = MagicMock()
            mock_vs.return_value = mock_instance

            result = get_vectorstore()

            assert result == mock_instance
            mock_vs.assert_called_once()

    def test_get_llm_with_path(self):
        """Test LLM creation with local model path."""
        with (
            patch("src.core.rag_pipeline.llm_wrapper.LLMWrapper") as mock_llm,
            patch("src.core.rag_pipeline.validate_helper.validate_model_arguments"),
        ):
            # Clear singleton
            import src.core.rag_pipeline as rag_pipeline

            rag_pipeline._llm = None

            mock_instance = MagicMock()
            mock_llm.return_value = mock_instance

            model_path = Path("test_model.gguf")
            result = get_llm(model_path=model_path)

            assert result == mock_instance
            mock_llm.assert_called_once_with(model_path=model_path, n_gpu_layers=-1)

    def test_initialize_rag(self):
        """Test RAG system initialization."""
        with (
            patch("src.core.rag_pipeline.get_vectorstore") as mock_get_vs,
            patch("src.core.rag_pipeline.get_llm") as mock_get_llm,
        ):
            mock_vs = MagicMock()
            mock_llm = MagicMock()
            mock_get_vs.return_value = mock_vs
            mock_get_llm.return_value = mock_llm

            vs, llm = initialize_rag(n_gpu_layers=8, model_path=Path("test.gguf"))

            assert vs == mock_vs
            assert llm == mock_llm
            mock_get_vs.assert_called_once()
            mock_get_llm.assert_called_once()


class TestErrorHandling:
    """Test basic error scenarios."""

    @pytest.mark.asyncio
    async def test_retrieval_error(self):
        """Test handling of retrieval errors."""
        vectorstore = MagicMock()
        vectorstore.search = AsyncMock(side_effect=Exception("Search failed"))
        llm = MagicMock()

        rag = RetrievalQA(vectorstore, llm)

        with pytest.raises(RuntimeError, match="Failed to retrieve context"):
            await rag.answer("test query")

    @pytest.mark.asyncio
    async def test_llm_error(self):
        """Test handling of LLM generation errors."""
        vectorstore = MagicMock()
        vectorstore.search = AsyncMock(return_value=[{"text": "chunk"}])

        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM failed"))

        rag = RetrievalQA(vectorstore, llm)

        with patch("src.core.rag_pipeline.prompt_builder.build_mcp_prompt"):
            with pytest.raises(RuntimeError, match="Failed to generate response"):
                await rag.answer("test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
