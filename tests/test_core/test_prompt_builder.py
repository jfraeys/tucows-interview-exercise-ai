"""
Focused test suite for prompt_builder.py
Tests build_mcp_prompt with correct template structure
"""

import pytest
from jinja2 import TemplateError

from src import models
from src.core.prompt_builder import build_mcp_prompt


class TestBuildMcpPrompt:
    """Test the main prompt building function"""

    def test_basic_prompt_with_chunks(self):
        """Test building a prompt with retrieved chunks"""
        chunk = models.Chunks(
            id="1",
            text="The return policy is 30 days for all items",
            metadata=models.ChunksMetadata(
                source="policy.txt", section="Returns", subsection="General Policy"
            ),
        )
        chunks = [chunk]  # Pass Chunks objects directly

        result = build_mcp_prompt("What is the return policy?", chunks)

        # Should contain the query
        assert "What is the return policy?" in result

        # Should contain the chunk content
        assert "The return policy is 30 days for all items" in result
        assert "Returns" in result

        # Should have JSON output structure
        assert '"answer":' in result
        assert '"references":' in result
        assert '"action_required":' in result

    def test_multiple_chunks(self):
        """Test with multiple retrieved chunks"""
        chunk1 = models.Chunks(
            id="1",
            text="Return within 30 days",
            metadata=models.ChunksMetadata(source="policy.txt", section="Returns"),
        )
        chunk2 = models.Chunks(
            id="2",
            text="Contact support@company.com",
            metadata=models.ChunksMetadata(source="contact.txt", section="Support"),
        )
        chunks = [chunk1, chunk2]  # Pass Chunks objects directly

        result = build_mcp_prompt("How do I return an item?", chunks)

        assert "Return within 30 days" in result
        assert "Contact support@company.com" in result
        assert "Context 1" in result
        assert "Context 2" in result

    def test_no_chunks_retrieved(self):
        """Test when no chunks are retrieved"""
        result = build_mcp_prompt("What is the policy?", [])

        assert "What is the policy?" in result
        assert "No context retrieved" in result
        assert '"answer":' in result  # Should still have JSON structure

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError"""
        with pytest.raises(ValueError, match="User query cannot be empty"):
            build_mcp_prompt("", [])

        # Whitespace-only should also raise error
        with pytest.raises(ValueError, match="User query cannot be empty"):
            build_mcp_prompt("   ", [])

    def test_chunks_with_minimal_metadata(self):
        """Test chunks with only source metadata"""
        chunk = models.Chunks(
            id="1",
            text="Basic content here",
            metadata=models.ChunksMetadata(
                source="basic.txt", section="Test"
            ),  # No subsection
        )
        chunks = [chunk]  # Pass Chunks object directly

        result = build_mcp_prompt("Test question", chunks)

        assert "Basic content here" in result
        assert "basic.txt" in result
        assert "Test question" in result

    def test_custom_template(self):
        """Test using a custom template string"""
        custom_template = """
Question: {{ user_query }}
{% for chunk in context_chunks %}
Content: {{ chunk.text }}
{% endfor %}
Answer the question above.
"""

        chunks = [
            models.Chunks(
                id="1",
                text="Test content",
                metadata=models.ChunksMetadata(source="test.txt", section="Test"),
            )
        ]

        result = build_mcp_prompt("What is this?", chunks, custom_template)

        assert "Question: What is this?" in result
        assert "Content: Test content" in result
        assert "Answer the question above." in result

    def test_chunks_as_dicts(self):
        """Test using dictionary format for chunks"""
        chunk_dict = {
            "id": "1",
            "text": "Dictionary chunk content",
            "metadata": {"source": "dict.txt", "section": "Dict Section"},
        }
        chunks = [chunk_dict]

        result = build_mcp_prompt("Test with dict?", chunks)

        assert "Dictionary chunk content" in result
        assert "Test with dict?" in result

    def test_invalid_template_raises_error(self):
        """Test that malformed Jinja2 template raises TemplateError"""
        bad_template = "{{ user_query.nonexistent_method() }}"

        with pytest.raises(TemplateError, match="Failed to render prompt template"):
            build_mcp_prompt("Test", [], bad_template)

    def test_chunks_without_metadata_sections(self):
        """Test chunks where metadata sections are None"""
        chunk = models.Chunks(
            id="1",
            text="Content without sections",
            metadata=models.ChunksMetadata(
                source="test.txt", section="Test", subsection=None
            ),
        )
        chunks = [chunk]

        result = build_mcp_prompt("Question?", chunks)

        assert "Content without sections" in result
        assert "test.txt" in result

    def test_empty_template_uses_default(self):
        """Test that empty template string uses the default template"""
        chunk = models.Chunks(
            id="1",
            text="Test",
            metadata=models.ChunksMetadata(source="test.txt", section="Test"),
        )
        chunks = [chunk]

        # Empty string should use default
        result1 = build_mcp_prompt("Question", chunks, "")
        result2 = build_mcp_prompt("Question", chunks)  # No template

        # Both should use default template (contain JSON structure)
        assert '"answer":' in result1
        assert '"answer":' in result2

        # Whitespace-only should also use default
        result3 = build_mcp_prompt("Question", chunks, "   ")
        assert '"answer":' in result3
