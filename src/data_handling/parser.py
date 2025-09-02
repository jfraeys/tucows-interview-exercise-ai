import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pypdf

from src.models import Chunks, ChunksMetadata, ParserResponse


# Base Parser
class BaseParser(ABC):
    @abstractmethod
    def parse(self, filepath: str) -> ParserResponse:
        """Return list of chunks with text and metadata"""
        ...


# Base Policy Parser with shared functionality
class BasePolicyParser(BaseParser):
    def __init__(self):
        self.filepath = None
        self.paragraph_lines = []
        self.chunks = []
        self.counter = 0
        self.current_section = None
        self.current_subsection = None

    def _initialize_parsing(self, filepath: str):
        """Initialize parsing state for a new file"""
        self.filepath = Path(filepath)
        self.paragraph_lines = []
        self.chunks = []
        self.counter = 0
        self.current_section = None
        self.current_subsection = None

    def _flush_paragraph(self):
        """Flush current paragraph to chunks if it has content"""
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
        """Flush final paragraph and return response"""
        self._flush_paragraph()
        return ParserResponse(chunks=self.chunks)


# TXT FAQ Parser - Simplified
class TxtFAQParser(BaseParser):
    def parse(self, filepath: str) -> ParserResponse:
        response = ParserResponse(chunks=[])

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

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


# Markdown Policy Parser - Improved
class MdPolicyParser(BasePolicyParser):
    """
    Parse a .md policy document.

    Sections are identified by headings:
    - '## <number>. <title>' marks a section
    - '### <number>.<number> <title>' marks a subsection

    Returns a list of chunks with 'id', 'text', and 'metadata'.
    """

    # Improved regex patterns with named groups and more tolerance
    SECTION_PATTERN = re.compile(r"^##\s*(?P<num>\d+\.)\s*(?P<title>.*)", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(
        r"^###\s*(?P<num>\d+\.\d+)\s*(?P<title>.*)", re.MULTILINE
    )

    def parse(self, filepath: str) -> ParserResponse:
        self._initialize_parsing(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Find all sections
        section_matches = list(self.SECTION_PATTERN.finditer(text))

        for i, sec_match in enumerate(section_matches):
            sec_num = sec_match.group("num")
            sec_title = sec_match.group("title")
            self.current_section = f"{sec_num} {sec_title}"
            self.current_subsection = None

            start = sec_match.end()
            end = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(text)
            )
            section_text = text[start:end].strip()

            # Find all subsections in this section
            subsections = list(self.SUBSECTION_PATTERN.finditer(section_text))

            if subsections:
                # Capture section text before the first subsection
                first_sub_start = subsections[0].start()
                pre_sub_text = section_text[:first_sub_start].strip()
                if pre_sub_text:
                    # Split on double newlines to preserve paragraph structure
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

                    sub_start = sub_match.end()
                    sub_end = (
                        subsections[j + 1].start()
                        if j + 1 < len(subsections)
                        else len(section_text)
                    )
                    sub_text = section_text[sub_start:sub_end].strip()
                    if sub_text:
                        # Handle multi-line content more naturally
                        self.paragraph_lines = [
                            line.strip()
                            for line in sub_text.split("\n")
                            if line.strip()
                        ]
                        self._flush_paragraph()
            else:
                # No subsections, store whole section
                if section_text:
                    # Handle multi-line content more naturally
                    self.paragraph_lines = [
                        line.strip()
                        for line in section_text.split("\n")
                        if line.strip()
                    ]
                    self._flush_paragraph()

        return self._finalize_parsing()


# PDF Policy Parser - Improved
class PdfPolicyParser(BasePolicyParser):
    # Improved regex patterns with named groups
    SECTION_PATTERN = re.compile(r"^(?P<num>\d+\.)\s+(?P<title>.*)", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(r"^(?P<num>\d+\.\d+)\s*(?P<title>.*)", re.MULTILINE)

    def parse(self, filepath: str) -> ParserResponse:
        self._initialize_parsing(filepath)

        reader = pypdf.PdfReader(filepath)
        lines = []

        # Extract text from all pages with better line handling
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Split on single newlines and only strip leading/trailing spaces
                page_lines = [line.strip() for line in text.split("\n")]
                lines.extend(
                    [line for line in page_lines if line]
                )  # Remove empty lines

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

            # Only add content if we're in a section
            if self.current_section:
                self.paragraph_lines.append(line)

        return self._finalize_parsing()


# Parser Factory - Enhanced
class ParserFactory:
    parsers = {
        ".txt": TxtFAQParser(),
        ".md": MdPolicyParser(),
        ".pdf": PdfPolicyParser(),
    }

    @staticmethod
    def get_parser(filepath: str, file_extension_override: str = None) -> BaseParser:
        """
        Get appropriate parser for file.

        Args:
            filepath: Path to the file
            file_extension_override: Override file extension for testing
        """
        if file_extension_override:
            ext = file_extension_override.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
        else:
            ext = os.path.splitext(filepath)[1].lower()

        parser = ParserFactory.parsers.get(ext)
        if not parser:
            raise ValueError(f"Unsupported file type: {ext}")
        return parser


# Convenience function - Enhanced
def parse_file(filepath: str, file_extension_override: str = None) -> ParserResponse:
    """
    Parse a file using the appropriate parser.

    Args:
        filepath: Path to the file to parse
        file_extension_override: Override file extension for testing purposes
    """
    parser = ParserFactory.get_parser(filepath, file_extension_override)
    return parser.parse(filepath)
