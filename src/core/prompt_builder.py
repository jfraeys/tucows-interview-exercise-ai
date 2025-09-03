"""
Model Context Protocol (MCP) prompt builder for RAG systems.

Constructs structured prompts that combine user queries with retrieved
document context for optimal language model performance.
"""

from typing import Any, Dict, Iterable, List, Union

from jinja2 import Template, TemplateError

from src import models
from src.utils import logging_helper

logger = logging_helper.get_logger(__name__)

DEFAULT_MCP_TEMPLATE = """
You are an AI assistant specialized in answering support tickets with accuracy and clarity. Only answer the question using the provided context. Do not
create new context; only use what is provided. If the context doesn't contain sufficient information, clearly state what information is missing in the output JSON. Stop after the JSON output.
Do not ask any questions in your answer. Do not include any new tasks in your answer.

{% if context_chunks %}
## Retrieved Context:
{% for item in context_chunks %}
[Context {{ loop.index }}] {{ item.chunk.text }}
{% if item.chunk.metadata and (item.chunk.metadata.subsection or item.chunk.metadata.section or item.chunk.metadata.source) %}
Source: {{ item.chunk.metadata.subsection or item.chunk.metadata.section or item.chunk.metadata.source }}
{% endif %}

{% endfor %}
{% else %}
No context retrieved.
{% endif %}

## User Question:
{{ user_query }}

Using this context, provide a detailed and accurate answer to the user's question.

DO NOT include the context or the questions in your answer.

ALL that is needed is the answer to the question. Produce one single, comprehensive response.

Only output a JSON object in the following format. Any additional text will cause issues downstream.

Pick the most appropriate value for "action_required" based on the definitions:
- "none": fully resolved, no further action needed
- "escalate": requires human intervention
- "follow_up": needs additional steps to resolve
- "more_info": requires more details from the user

## Output Format Requirements:
You MUST respond with ONLY a valid JSON object in this exact format, nothing else:
{
  "answer": "Your detailed answer here",
  "references": [
    "List of sources actually used to answer the question. In the references field only refer to source, section or subsections. If no sources were used, return an empty list."
  ],
  "action_required": "none|escalate|follow_up|more_info"
}
"""


def build_mcp_prompt(
    user_query: str,
    retrieved_chunks: Iterable[Union[models.Chunks, Dict[str, Any]]],
    template_str: str = "",
) -> str:
    """
    Build MCP compliant prompt for RAG systems.

    Args:
        user_query: The user's question
        retrieved_chunks: Iterable of Chunks objects with metadata
        template_str: Custom template (uses default if empty)

    Returns:
        Formatted prompt string ready for LLM input
    """
    if not user_query.strip():
        raise ValueError("User query cannot be empty")

    chunks_list = list(retrieved_chunks)
    template_source = template_str.strip() or DEFAULT_MCP_TEMPLATE

    try:
        template = Template(template_source)
        prompt = template.render(
            user_query=user_query, context_chunks=chunks_list
        ).strip()
        return prompt
    except TemplateError as e:
        raise TemplateError(f"Failed to render prompt template: {e}") from e


def extract_references_from_chunks(
    retrieved_chunks: Iterable[Union[models.Chunks, Dict[str, Any]]],
) -> List[str]:
    """
    Extract reference information from chunk metadata.

    Args:
        retrieved_chunks: Chunks objects or dicts with metadata

    Returns:
        List of formatted reference strings
    """
    references = []

    for chunk in retrieved_chunks:
        try:
            if isinstance(chunk, models.Chunks):
                # Pydantic Chunks object
                ref_parts = [chunk.metadata.source]
                if chunk.metadata.section:
                    ref_parts.append(chunk.metadata.section)
                if chunk.metadata.subsection:
                    ref_parts.append(chunk.metadata.subsection)
                reference = " - ".join(ref_parts)

            elif isinstance(chunk, dict) and "metadata" in chunk:
                # Dictionary with metadata
                metadata = chunk["metadata"]
                ref_parts = [metadata.get("source", "Unknown")]
                if metadata.get("section"):
                    ref_parts.append(metadata["section"])
                if metadata.get("subsection"):
                    ref_parts.append(metadata["subsection"])
                reference = " - ".join(ref_parts)

            else:
                reference = "Unknown source"

            if reference not in references:
                references.append(reference)

        except Exception as e:
            logger.warning(f"Failed to extract reference: {e}")
            references.append("Unknown source")

    return references
