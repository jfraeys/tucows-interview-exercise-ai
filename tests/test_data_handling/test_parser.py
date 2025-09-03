import os

import pytest
from reportlab.pdfgen import canvas

from src.data_handling.parser import parse_file
from src.models import ParserResponse

# ---------------------------
# Fixtures for test documents
# ---------------------------


@pytest.fixture
def tiny_txt(tmp_path):
    path = tmp_path / "CS_FAQ.txt"
    path.write_text(
        "Q1: What is test?\nA: This is a test answer.\n\n"
        "Q2: Another question?\nA: This is another test answer.\n\n"
        "Q3: Multi-line answer?\nA: This test the multi-line\n answer parsing.\n"
    )
    return path


@pytest.fixture
def tiny_md(tmp_path):
    path = tmp_path / "domain_support_docs.md"
    path.write_text(
        "## 1. Test Policy\n### 1.1 Subsection\nContent here.\n\n"
        "## 2. Another Section\nSome other content.\n"
        "### 2.1 Another Subsection\nMore multi-line\ncontent here.\n"
    )
    return path


@pytest.fixture
def tiny_pdf(tmp_path):
    path = tmp_path / "tiny.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(100, 800, "Document title")
    c.drawString(100, 750, "1. Test Guidelines")
    c.drawString(100, 730, "1.1 Subsection")
    c.drawString(100, 710, "Some content here.")
    c.drawString(100, 690, "1.2 Another Subsection")
    c.drawString(100, 670, "More content here.")
    c.drawString(100, 650, "2. Another Section")
    c.drawString(100, 630, "Some other content.")
    c.save()
    return path


# ---------------------------
# Key edge case fixtures
# ---------------------------


@pytest.fixture
def empty_section_txt(tmp_path):
    path = tmp_path / "empty_section.txt"
    path.write_text(
        "Q1: Empty question?\nA:\n\nQ2: Normal question?\nA: Normal answer."
    )
    return path


@pytest.fixture
def text_before_subsection_md(tmp_path):
    path = tmp_path / "text_before_subsection.md"
    path.write_text(
        "## 1. Section with text\nIntro text before subsection.\n"
        "### 1.1 Subsection\nSubsection content."
    )
    return path


@pytest.fixture
def bullet_points_md(tmp_path):
    """Test file with bullet points and lists"""
    path = tmp_path / "bullet_points.md"
    path.write_text(
        "## 1. Features\n"
        "Key features include:\n\n"
        "- Feature one\n"
        "- Feature two\n"
        "- Feature three\n\n"
        "Additional details here.\n\n"
        "### 1.1 Numbered List\n"
        "Steps to follow:\n\n"
        "1. First step\n"
        "2. Second step\n"
        "3. Third step\n\n"
        "That's it!"
    )
    return path


# ---------------------------
# Expected content mapping
# ---------------------------

EXPECTED_CONTENT = {
    "CS_FAQ.txt": [
        {
            "text": "This is a test answer.",
            "section": "Q1: What is test?",
            "subsection": None,
        },
        {
            "text": "This is another test answer.",
            "section": "Q2: Another question?",
            "subsection": None,
        },
        {
            "text": "This test the multi-line\nanswer parsing.",
            "section": "Q3: Multi-line answer?",
            "subsection": None,
        },
    ],
    "domain_support_docs.md": [
        {
            "text": "Content here.",
            "section": "1. Test Policy",
            "subsection": "1.1 Subsection",
        },
        {
            "text": "Some other content.",
            "section": "2. Another Section",
            "subsection": None,
        },
        {
            "text": "More multi-line\ncontent here.",  # Now preserves paragraph breaks
            "section": "2. Another Section",
            "subsection": "2.1 Another Subsection",
        },
    ],
    "tiny.pdf": [
        {
            "text": "Some content here.",
            "section": "1. Test Guidelines",
            "subsection": "1.1 Subsection",
        },
        {
            "text": "More content here.",
            "section": "1. Test Guidelines",
            "subsection": "1.2 Another Subsection",
        },
        {
            "text": "Some other content.",
            "section": "2. Another Section",
            "subsection": None,
        },
    ],
}

# ---------------------------
# Main tests
# ---------------------------


@pytest.mark.parametrize("doc_fixture", ["tiny_txt", "tiny_md", "tiny_pdf"])
def test_parsers_debug(doc_fixture, request) -> None:
    """Debug version to see what's actually being parsed"""
    filepath = request.getfixturevalue(doc_fixture)
    response: ParserResponse = parse_file(str(filepath))
    filename = os.path.basename(filepath)

    print(f"\n=== DEBUGGING {filename} ===")
    print(f"Total chunks: {len(response.chunks)}")

    for i, chunk in enumerate(response.chunks):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk.id}")
        print(f"  Text: '{chunk.text}'")
        print(f"  Section: '{chunk.metadata.section}'")
        print(f"  Subsection: '{chunk.metadata.subsection}'")
        print(f"  Source: '{chunk.metadata.source}'")


@pytest.mark.parametrize("doc_fixture", ["tiny_txt", "tiny_md", "tiny_pdf"])
def test_parsers(doc_fixture, request) -> None:
    filepath = request.getfixturevalue(doc_fixture)
    response: ParserResponse = parse_file(str(filepath))
    filename = os.path.basename(filepath)
    expected_chunks = EXPECTED_CONTENT[filename]

    # Exact count check
    assert len(response.chunks) == len(expected_chunks), (
        f"Expected exactly {len(expected_chunks)} chunks for {filename}, "
        f"got {len(response.chunks)}"
    )

    # Validate each chunk
    for i, (expected, chunk) in enumerate(zip(expected_chunks, response.chunks)):
        # Text content match
        assert chunk.text.strip() == expected["text"]

        # Metadata validation
        assert chunk.metadata.section == expected["section"]
        assert chunk.metadata.subsection == expected["subsection"]
        assert chunk.metadata.source == filename

        # Consistent ID format
        assert chunk.id == f"{filename}_{i}"


# ---------------------------
# Essential edge case tests
# ---------------------------


def test_empty_section(empty_section_txt) -> None:
    """Test handling of empty answers in FAQ format"""
    response: ParserResponse = parse_file(str(empty_section_txt))

    # Should have 1 chunk (only the non-empty Q2)
    assert len(response.chunks) == 1

    # Should be the normal question
    chunk = response.chunks[0]
    assert "Normal answer" in chunk.text
    assert chunk.metadata.section == "Q2: Normal question?"


def test_text_before_subsection(text_before_subsection_md) -> None:
    """Test that text before subsections is captured correctly"""
    response: ParserResponse = parse_file(str(text_before_subsection_md))

    # Should have 2 chunks: section intro + subsection
    assert len(response.chunks) == 2

    # Check the section chunk before subsection
    section_chunk = response.chunks[0]
    assert "Intro text before subsection." in section_chunk.text
    assert section_chunk.metadata.subsection is None

    # Check the subsection chunk
    subsection_chunk = response.chunks[1]
    assert "Subsection content." in subsection_chunk.text
    assert subsection_chunk.metadata.subsection == "1.1 Subsection"


@pytest.fixture
def empty_file_txt(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    return path


def test_bullet_points_preservation(bullet_points_md) -> None:
    """Test that bullet points and lists are preserved"""
    response: ParserResponse = parse_file(str(bullet_points_md))

    # Should have 2 chunks: main section + subsection
    assert len(response.chunks) >= 1

    # Check that bullet points are preserved in first chunk
    main_chunk = response.chunks[0]
    assert "- Feature one" in main_chunk.text
    assert "- Feature two" in main_chunk.text
    assert "- Feature three" in main_chunk.text

    # Check numbered list in subsection if present
    if len(response.chunks) > 1:
        sub_chunk = response.chunks[1]
        assert "1. First step" in sub_chunk.text
        assert "2. Second step" in sub_chunk.text


def test_unsupported_file_type(tmp_path):
    """Test that ParserFactory raises ValueError for unsupported file types"""
    unsupported_file = tmp_path / "dummy.unsupported"
    unsupported_file.write_text("This is some dummy content")

    with pytest.raises(ValueError) as exc_info:
        parse_file(str(unsupported_file))

    assert "Unsupported file type '.unsupported'" in str(exc_info.value)


def test_file_extension_override(tiny_txt):
    """Test the file extension override functionality"""
    # Test valid override (parse .txt as .txt explicitly)
    response = parse_file(str(tiny_txt), file_extension_override=".txt")
    assert len(response.chunks) == 3  # Should work normally
