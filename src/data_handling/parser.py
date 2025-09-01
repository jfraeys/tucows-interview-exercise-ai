import os
import re
from collections.abc import Collection

import pypdf


# Base Parser
class BaseParser:
    def parse(self, filepath: str) -> list[dict]:
        """Return list of chunks with text and metadata"""
        raise NotImplementedError


# TXT FAQ Parser
class TxtFAQParser(BaseParser):
    def parse(self, filepath: str) -> list[dict[str, Collection[str]]]:
        """
        Parse a .txt FAQ file with format:

        Q1: Question text
        A: Answer text

        Returns a list of dicts with:
            - id: unique identifier
            - text: answer text
            - metadata: source + question
        """
        chunks = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        question = None
        answer_lines: list[str] = []

        for line in lines:
            line = line.strip()
            if line.startswith("Q"):
                # save previous Q/A
                if question and answer_lines:
                    chunks.append(
                        {
                            "id": f"{os.path.basename(filepath)}_{question[:30]}",
                            "text": "\n".join(answer_lines),
                            "metadata": {
                                "source": os.path.basename(filepath),
                                "question": question,
                            },
                        }
                    )
                question = line
                answer_lines = []
            elif line.startswith("A:"):
                answer_lines.append(line[2:].strip())
            elif line:
                answer_lines.append(line)

        # add last Q/A
        if question and answer_lines:
            chunks.append(
                {
                    "id": f"{os.path.basename(filepath)}_{question[:30]}",
                    "text": "\n".join(answer_lines),
                    "metadata": {
                        "source": os.path.basename(filepath),
                        "question": question,
                    },
                }
            )

        return chunks


# Markdown Policy Parser
class MdPolicyParser(BaseParser):
    """
    Parse a .md policy document.

    Sections are identified by headings:
    - '## <number>. <title>' marks a section
    - '### <number>.<number> <title>' marks a subsection

    Returns a list of dicts with 'id', 'text', and 'metadata'.

    - 'id': unique identifier combining filename and section number
    - 'text': extracted section or subsection content
    - 'metadata': includes 'source' (filename) and 'section' (heading)
    """

    SECTION_PATTERN = re.compile(r"^##\s+(\d+)\.\s*(.*)", re.MULTILINE)
    SUBSECTION_PATTERN = re.compile(r"^###\s+(\d+\.\d+)\s*(.*)", re.MULTILINE)

    def parse(self, filepath: str) -> list[dict[str, Collection[str]]]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = []

        # Find all sections
        section_matches = list(self.SECTION_PATTERN.finditer(text))
        for i, sec_match in enumerate(section_matches):
            sec_num, sec_title = sec_match.groups()
            start = sec_match.end()
            end = (
                section_matches[i + 1].start()
                if i + 1 < len(section_matches)
                else len(text)
            )
            section_text = text[start:end].strip()

            # Find subsections inside this section
            subsections = list(self.SUBSECTION_PATTERN.finditer(section_text))
            if subsections:
                for j, sub_match in enumerate(subsections):
                    sub_num, sub_title = sub_match.groups()
                    sub_start = sub_match.end()
                    sub_end = (
                        subsections[j + 1].start()
                        if j + 1 < len(subsections)
                        else len(section_text)
                    )
                    sub_text = section_text[sub_start:sub_end].strip()

                    chunks.append(
                        {
                            "id": f"{os.path.basename(filepath)}_{sub_num}",
                            "text": sub_text,
                            "metadata": {
                                "source": os.path.basename(filepath),
                                "section": f"{sec_num} {sec_title}",
                                "subsection": f"{sub_num} {sub_title}",
                            },
                        }
                    )
            else:
                # No subsections, store whole section
                chunks.append(
                    {
                        "id": f"{os.path.basename(filepath)}_{sec_num}",
                        "text": section_text,
                        "metadata": {
                            "source": os.path.basename(filepath),
                            "section": f"{sec_num} {sec_title}",
                            "subsection": None,
                        },
                    }
                )

        return chunks


# PDF Policy Parser
class PdfPolicyParser(BaseParser):
    def parse(self, filepath: str) -> list[dict[str, Collection[str]]]:
        """
        Parse a structured policy PDF into chunks.

        The PDF is expected to follow a hierarchy:
            - Title (Heading1)
            - Sections (Heading2)
            - Subsections (Heading3)
            - Paragraphs (BodyText)

        Returns a list of dicts with:
            - id: unique identifier
            - text: paragraph text
            - metadata: source + section + subsection
        """
        reader = pypdf.PdfReader(filepath)
        text_blocks = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_blocks.append(page_text.strip())

        full_text = "\n".join(text_blocks)

        # Split into lines (ReportLab usually separates Paragraphs this way)
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]

        chunks = []
        current_section = None
        current_subsection = None

        for i, line in enumerate(lines):
            # Detect section (e.g., "1. Domain Suspension & Reactivation")
            if line[0].isdigit() and line[1] == "." and not line.startswith("1.1"):
                current_section = line
                current_subsection = None
                continue

            # Detect subsection (e.g., "1.1 Reasons for Suspension")
            if (
                len(line) > 3
                and line[0].isdigit()
                and line[1] == "."
                and line[2].isdigit()
            ):
                current_subsection = line
                continue

            # Treat everything else as paragraph text
            chunks.append(
                {
                    "id": f"{os.path.basename(filepath)}_chunk{i}",
                    "text": line,
                    "metadata": {
                        "source": os.path.basename(filepath),
                        "section": current_section,
                        "subsection": current_subsection,
                    },
                }
            )

        return chunks


# Parser Factory
class ParserFactory:
    parsers = {
        ".txt": TxtFAQParser(),
        ".md": MdPolicyParser(),
        ".pdf": PdfPolicyParser(),
    }

    @staticmethod
    def get_parser(filepath: str) -> BaseParser:
        ext = os.path.splitext(filepath)[1].lower()
        parser = ParserFactory.parsers.get(ext)
        if not parser:
            raise ValueError(f"No parser for file type: {ext}")
        return parser


# Convenience function
def parse_file(filepath: str) -> list[dict[str, str]]:
    parser = ParserFactory.get_parser(filepath)
    return parser.parse(filepath)
