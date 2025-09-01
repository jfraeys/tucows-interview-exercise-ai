import os

import pytest

from src.data_handling.parser import parse_file

DATA_DIR = os.path.join("tests", "data")


@pytest.mark.parametrize(
    "filepath",
    [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith((".md", ".txt", ".pdf"))
    ],
)
def test_parsers(filepath):
    """Test parsing of all supported files in tests/data directory."""
    chunks = parse_file(filepath)
    assert len(chunks) > 0, f"No chunks parsed from {filepath}"

    first_chunk = chunks[0]
    # Basic metadata checks
    assert "id" in first_chunk
    assert "text" in first_chunk and len(first_chunk["text"]) > 0
    assert "metadata" in first_chunk
    assert "source" in first_chunk["metadata"]
    assert first_chunk["metadata"]["source"] == os.path.basename(filepath)
    assert "section" in first_chunk["metadata"] or "question" in first_chunk["metadata"]

    # Optional: check subsections exist for Markdown/PDF
    subsections = [c for c in chunks if c["metadata"].get("subsection")]
    if filepath.endswith((".md", ".pdf")):
        assert len(subsections) > 0, f"No subsections detected in {filepath}"


def test_unsupported_file_type(tmp_path):
    """
    Test that ParserFactory raises ValueError for unsupported file types.
    """

    # Create a file with an unsupported extension
    unsupported_file = tmp_path / "dummy.unsupported"
    unsupported_file.write_text("This is some dummy content")

    # Expect a ValueError when trying to parse it
    with pytest.raises(ValueError) as exc_info:
        parse_file(str(unsupported_file))

    assert "No parser for file type" in str(exc_info.value)
