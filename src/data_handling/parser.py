import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pypdf

from src.models import Chunks, ChunksMetadata, ParserResponse


class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    Defines the interface that all document parsers must implement.
    Each parser is responsible for extracting structured chunks from
    a specific file format (PDF, Markdown, TXT, etc.).
    """

    @abstractmethod
    def parse(self, filepath: str) -> ParserResponse:
        """
        Parse a document file into structured chunks.

        Args:
            filepath: Path to the document file to parse

        Returns:
            ParserResponse containing list of chunks with text and metadata

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            PermissionError: If unable to read the file
            ValueError: If file format is unsupported or corrupted
        """
        ...


class BasePolicyParser(BaseParser):
    """
    Base class for policy document parsers with shared functionality.

    Provides common methods for managing parsing state, section tracking,
    and chunk creation. Subclasses implement format-specific parsing logic
    while using these shared utilities for consistent output structure.

    Attributes:
        filepath: Path object for current file being parsed
        paragraph_lines: Temporary storage for lines of current paragraph
        chunks: List of completed chunks from parsing
        counter: Sequential counter for generating unique chunk IDs
        current_section: Currently active section name/number
        current_subsection: Currently active subsection name/number
    """

    def __init__(self):
        """Initialize parser with empty state."""
        self.filepath = None
        self.paragraph_lines = []
        self.chunks = []
        self.counter = 0
        self.current_section = None
        self.current_subsection = None

    def _initialize_parsing(self, filepath: str) -> None:
        """
        Initialize parsing state for a new file.

        Resets all internal state variables and sets up the parser
        for processing a new document. Must be called before parsing.

        Args:
            filepath: Path to the file being parsed
        """
        self.filepath = Path(filepath)
        self.paragraph_lines = []
        self.chunks = []
        self.counter = 0
        self.current_section = None
        self.current_subsection = None

    def _flush_paragraph(self) -> None:
        """
        Convert accumulated paragraph lines into a chunk.

        Takes the lines stored in paragraph_lines, joins them with newlines,
        and creates a new Chunks object with appropriate metadata. Only creates
        a chunk if we're currently in a section and have non-empty content.
        Resets paragraph_lines after flushing.

        Raises:
            ValueError: If parser hasn't been initialized with a file
        """
        if not self.filepath:
            raise ValueError(
                "Parser not initialized with a file. Call _initialize_parsing first."
            )
        if not self.current_section:
            return  # Skip if no section defined, no action needed

        if self.paragraph_lines:
            # Join with newlines to preserve structure
            text = "\n".join(self.paragraph_lines).strip()
            if text:
                chunk_id = f"{self.filepath.name}_{self.counter}"
                self.chunks.append(
                    Chunks(
                        id=chunk_id,
                        text=text,
                        metadata=ChunksMetadata(
                            source=self.filepath.name,
                            section=self.current_section,
                            subsection=self.current_subsection,
                        ),
                    )
                )
                self.counter += 1
            self.paragraph_lines = []

    def _finalize_parsing(self) -> ParserResponse:
        """
        Complete the parsing process and return results.

        Flushes any remaining paragraph content and wraps all
        accumulated chunks in a ParserResponse object.

        Returns:
            ParserResponse containing all parsed chunks
        """
        self._flush_paragraph()
        return ParserResponse(chunks=self.chunks)


class TxtFAQParser(BaseParser):
    """
    Parser for FAQ text files with Q/A structure.

    Expects text files formatted with questions starting with "Q" and
    answers starting with "A:" or as continuation lines. Each Q/A pair
    becomes a separate chunk with the question as the section.

    Example format:
        Q: What is the return policy?
        A: Items can be returned within 30 days.
        Additional details here.

        Q: How do I contact support?
        A: Email us at support@company.com
    """

    def parse(self, filepath: str) -> ParserResponse:
        """
        Parse FAQ text file into question/answer chunks.

        Args:
            filepath: Path to the FAQ text file

        Returns:
            ParserResponse with chunks where each chunk represents one Q/A pair

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If unable to read the file
            UnicodeDecodeError: If file encoding is not UTF-8 compatible
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"FAQ file not found: {filepath}")
        except PermissionError:
            raise PermissionError(f"Cannot read FAQ file: {filepath}")

        response = ParserResponse(chunks=[])
        question = None
        answer_lines: list = []

        def _flush_answer():
            """Helper to flush current Q/A pair"""
            if question and answer_lines:
                # Filter out empty lines and join
                text_lines = [line for line in answer_lines if line.strip()]
                if text_lines:
                    response.chunks.append(
                        Chunks(
                            id=f"{os.path.basename(filepath)}_{len(response.chunks)}",
                            text="\n".join(text_lines),
                            metadata=ChunksMetadata(
                                source=os.path.basename(filepath),
                                section=question,
                                subsection=None,
                            ),
                        )
                    )

        for line in lines:
            line = line.strip()
            if line.startswith("Q"):
                _flush_answer()
                question = line
                answer_lines = []
            else:
                # Handle both "A:" prefixed lines and continuation lines
                cleaned_line = line.lstrip("A:").strip()
                if cleaned_line:  # Only add non-empty lines
                    answer_lines.append(cleaned_line)

        # Flush the last Q/A pair
        _flush_answer()
        return response


class MdPolicyParser(BasePolicyParser):
    """
    Parser for Markdown policy documents with hierarchical structure.

    Parses Markdown files with specific heading patterns:
    - Level 2 headings (##) with numbering: "## 1. Section Title"
    - Level 3 headings (###) with sub-numbering: "### 1.1 Subsection Title"

    Each section/subsection becomes a separate chunk with hierarchical metadata.
    Content between headings is grouped into logical paragraphs.
    """

    # Regex patterns for identifying sections and subsections
    SECTION_PATTERN = re.compile(r"^##\s*(?P<num>\d+\.)\s*(?P<title>.*)", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(
        r"^###\s*(?P<num>\d+\.\d+)\s*(?P<title>.*)", re.MULTILINE
    )

    def parse(self, filepath: str) -> ParserResponse:
        """
        Parse Markdown policy document into structured chunks.

        Args:
            filepath: Path to the Markdown file

        Returns:
            ParserResponse with chunks representing sections and subsections

        Raises:
            FileNotFoundError: If the Markdown file doesn't exist
            PermissionError: If unable to read the file
            UnicodeDecodeError: If file encoding is not UTF-8 compatible
        """
        self._initialize_parsing(filepath)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Markdown file not found: {filepath}")
        except PermissionError:
            raise PermissionError(f"Cannot read Markdown file: {filepath}")

        # Find all main sections in the document
        section_matches = list(self.SECTION_PATTERN.finditer(text))

        for i, sec_match in enumerate(section_matches):
            sec_num = sec_match.group("num")
            sec_title = sec_match.group("title")
            self.current_section = f"{sec_num} {sec_title}"
            self.current_subsection = None

            # Extract text between this section and the next
            start = sec_match.end()
            end = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(text)
            )
            section_text = text[start:end].strip()

            # Look for subsections within this section
            subsections = list(self.SUBSECTION_PATTERN.finditer(section_text))

            if subsections:
                # Handle content before first subsection
                first_sub_start = subsections[0].start()
                pre_sub_text = section_text[:first_sub_start].strip()
                if pre_sub_text:
                    self.paragraph_lines = [
                        line.strip()
                        for line in pre_sub_text.split("\n")
                        if line.strip()
                    ]
                    self._flush_paragraph()

                # Process each subsection
                for j, sub_match in enumerate(subsections):
                    sub_num = sub_match.group("num")
                    sub_title = sub_match.group("title")
                    self.current_subsection = f"{sub_num} {sub_title}"

                    # Extract subsection content
                    sub_start = sub_match.end()
                    sub_end = (
                        subsections[j + 1].start()
                        if j + 1 < len(subsections)
                        else len(section_text)
                    )
                    sub_text = section_text[sub_start:sub_end].strip()
                    if sub_text:
                        self.paragraph_lines = [
                            line.strip()
                            for line in sub_text.split("\n")
                            if line.strip()
                        ]
                        self._flush_paragraph()
            else:
                # No subsections, process entire section as one chunk
                if section_text:
                    self.paragraph_lines = [
                        line.strip()
                        for line in section_text.split("\n")
                        if line.strip()
                    ]
                    self._flush_paragraph()

        return self._finalize_parsing()


class PdfPolicyParser(BasePolicyParser):
    """
    Parser for PDF policy documents with numbered sections.

    Extracts text from PDF files and identifies sections/subsections based on
    numbered headings. Handles multi-page documents and preserves document
    structure while creating searchable chunks.

    Expected format:
    - Sections: "1. Section Title"
    - Subsections: "1.1 Subsection Title"
    """

    # Regex patterns for identifying numbered sections
    SECTION_PATTERN = re.compile(r"^(?P<num>\d+\.)\s+(?P<title>.*)", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(r"^(?P<num>\d+\.\d+)\s*(?P<title>.*)", re.MULTILINE)

    def parse(self, filepath: str) -> ParserResponse:
        """
        Parse PDF policy document into structured chunks.

        Extracts text from all pages, identifies sections and subsections
        based on numbered headings, and creates chunks with hierarchical metadata.

        Args:
            filepath: Path to the PDF file

        Returns:
            ParserResponse with chunks representing document sections

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            PermissionError: If unable to read the file
            pypdf.errors.PdfReadError: If PDF is corrupted or encrypted
        """
        self._initialize_parsing(filepath)

        try:
            reader = pypdf.PdfReader(filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        except PermissionError:
            raise PermissionError(f"Cannot read PDF file: {filepath}")
        except pypdf.errors.PdfReadError as e:
            raise ValueError(f"Cannot read PDF (corrupted or encrypted): {e}")

        lines = []

        # Extract text from all pages
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    # Split on newlines and clean up
                    page_lines = [line.strip() for line in text.split("\n")]
                    lines.extend([line for line in page_lines if line])
            except Exception:
                # Skip pages that can't be extracted (images, corrupted, etc.)
                continue

        # Process extracted lines to identify structure
        for line in lines:
            if self.SECTION_PATTERN.match(line):
                self._flush_paragraph()
                match = self.SECTION_PATTERN.match(line)
                if not match:
                    continue

                sec_num = match.group("num")
                sec_title = match.group("title")
                self.current_section = f"{sec_num} {sec_title}"
                self.current_subsection = None
                continue
            elif self.SUBSECTION_PATTERN.match(line):
                self._flush_paragraph()
                match = self.SUBSECTION_PATTERN.match(line)
                if not match:
                    continue

                sub_num = match.group("num")
                sub_title = match.group("title")
                self.current_subsection = f"{sub_num} {sub_title}"
                continue

            # Add content lines only if we're inside a section
            if self.current_section:
                self.paragraph_lines.append(line)

        return self._finalize_parsing()


class ParserFactory:
    """
    Factory class for creating appropriate document parsers.

    Maintains a registry of parsers for different file types and provides
    a unified interface for getting the right parser based on file extension.
    """

    # Registry mapping file extensions to parser instances
    parsers = {
        ".txt": TxtFAQParser(),
        ".md": MdPolicyParser(),
        ".pdf": PdfPolicyParser(),
    }

    @staticmethod
    def get_parser(
        filepath: str, file_extension_override: str | None = None
    ) -> BaseParser:
        """
        Get the appropriate parser for a given file.

        Selects parser based on file extension or override parameter.
        Useful for testing with different parsers or handling files
        with non-standard extensions.

        Args:
            filepath: Path to the file to be parsed
            file_extension_override: Override file extension for testing
                                   (e.g., "pdf" or ".pdf")

        Returns:
            BaseParser instance appropriate for the file type

        Raises:
            ValueError: If file extension is not supported
        """
        if file_extension_override:
            ext = file_extension_override.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
        else:
            ext = os.path.splitext(filepath)[1].lower()

        parser = ParserFactory.parsers.get(ext)
        if not parser:
            supported_types = ", ".join(ParserFactory.parsers.keys())
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported types: {supported_types}"
            )
        return parser


def parse_file(
    filepath: str, file_extension_override: str | None = None
) -> ParserResponse:
    """
    Parse a document file using the appropriate parser.

    Convenience function that automatically selects the right parser
    based on file extension and handles the parsing process. This is
    the main entry point for parsing individual files.

    Args:
        filepath: Path to the file to parse
        file_extension_override: Optional override for file extension
                               (useful for testing or non-standard extensions)

    Returns:
        ParserResponse containing parsed chunks with text and metadata

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If the specified file doesn't exist
        PermissionError: If unable to read the file

    Examples:
        >>> response = parse_file("docs/policy.pdf")
        >>> response = parse_file("data.txt", file_extension_override="md")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    parser = ParserFactory.get_parser(filepath, file_extension_override)
    return parser.parse(filepath)
