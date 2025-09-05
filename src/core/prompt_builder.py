"""
Model Context Protocol (MCP) prompt builder for RAG systems.

Constructs structured prompts that combine user queries with retrieved
document context for optimal language model performance.
"""

from typing import Any, Dict, Iterable, Union

from jinja2 import Template, TemplateError

from src import models
from src.utils import logging_helper

logger = logging_helper.get_logger(__name__)

DEFAULT_MCP_TEMPLATE = """
You are an AI assistant specialized in answering support tickets with accuracy and clarity. Answer the question using the provided context. If the context doesn't contain sufficient information, clearly state what information is missing and determine the appropriate escalation.

{% if context_chunks %}
## Retrieved Context:
{% for item in context_chunks %}
[Context {{ loop.index }}] {{ item.text }}
Source: {{ item.metadata.source }}{% if item.metadata.section %} - {{ item.metadata.section }}{% endif %}{% if item.metadata.subsection %} - {{ item.metadata.subsection }}{% endif %}

{% endfor %}
{% else %}
No context retrieved.
{% endif %}

## User Question:
{{ user_query }}

Using this context, provide a detailed and accurate answer to the user's question. Carefully evaluate whether the issue requires escalation based on these criteria:

**Escalation Guidelines:**
- Use "escalate_to_abuse_team" for: policy violations, spam, malicious activity, account suspensions, security concerns
- Use "escalate_to_billing" for: payment issues, refunds, billing disputes, account charges, subscription problems
- Use "escalate_to_technical" for: server issues, DNS problems, technical configurations, system outages, complex technical troubleshooting
- Use "follow_up_required" for: issues that need monitoring, pending actions, or multi-step resolution processes
- Use "more_info_needed" when: critical information is missing, user needs to provide additional details, or clarification is required
- Use "none" only when: the issue is completely resolved with the provided information and no further action is needed

You MUST respond with ONLY a valid JSON object in this exact format:

{% raw %}
{
  "answer": "Your detailed answer here",
  "references": ["List of sources used - include document name and section/subsection if available. Use lowest-level hierarchy. Empty list if no sources."],
  "action_required": "none|escalate_to_abuse_team|escalate_to_billing|escalate_to_technical|follow_up_required|more_info_needed"
}
{% endraw %}
"""


def build_prompt(
    user_query: str,
    context_items: Iterable[Dict[str, Any]],
    template_str: str,
    default_template: str = "",
) -> str:
    """
    Generic prompt builder using Jinja2 templates.

    Args:
        user_query: The user's question.
        context_items: Iterable of dictionaries with context data.
        template_str: Custom template (uses default_template if empty).
        default_template: Fallback template if none is provided.

    Returns:
        Rendered prompt string.
    """
    if not user_query.strip():
        raise ValueError("User query cannot be empty")

    items = list(context_items)
    template_source = template_str.strip() or default_template

    try:
        template = Template(template_source)
        return template.render(user_query=user_query, context_chunks=items).strip()
    except TemplateError as e:
        raise TemplateError(f"Failed to render prompt template: {e}") from e


def build_mcp_prompt(
    user_query: str,
    retrieved_chunks: Iterable[Union[models.Chunks, Dict[str, Any]]],
    template_str: str = "",
) -> str:
    """
    Build MCP compliant prompt for RAG systems.

    Args:
        user_query: The user's question.
        retrieved_chunks: Iterable of Chunks or dict-like objects with metadata.
        template_str: Optional custom template.

    Returns:
        Formatted MCP prompt string ready for LLM input.
    """
    # Normalize chunks -> dict for template rendering
    chunks_list: list[Dict[str, Any]] = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, models.Chunks):
            chunks_list.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {
                        "source": chunk.metadata.source,
                        "section": chunk.metadata.section,
                        "subsection": chunk.metadata.subsection,
                    },
                }
            )
        elif isinstance(chunk, dict):
            # Handle retrieval system format: {"chunk": {...}, "score": ..., "rank": ...}
            if "chunk" in chunk:
                chunks_list.append(chunk["chunk"])
            else:
                chunks_list.append(chunk)
        else:
            raise TypeError(f"Unsupported chunk type: {type(chunk)}")

    return build_prompt(
        user_query=user_query,
        context_items=chunks_list,
        template_str=template_str,
        default_template=DEFAULT_MCP_TEMPLATE,
    )


# def extract_references_from_chunks(
#     retrieved_chunks: Iterable[Union[models.Chunks, Dict[str, Any]]],
# ) -> List[str]:
#     """
#     Extract reference information from chunk metadata.
#
#     Args:
#         retrieved_chunks: Chunks objects or dicts with metadata
#
#     Returns:
#         List of formatted reference strings
#     """
#     references = []
#
#     for chunk in retrieved_chunks:
#         try:
#             if isinstance(chunk, models.Chunks):
#                 # Pydantic Chunks object
#                 ref_parts = [chunk.metadata.source]
#                 if chunk.metadata.section:
#                     ref_parts.append(chunk.metadata.section)
#                 if chunk.metadata.subsection:
#                     ref_parts.append(chunk.metadata.subsection)
#                 reference = " - ".join(ref_parts)
#
#             elif isinstance(chunk, dict) and "metadata" in chunk:
#                 # Dictionary with metadata
#                 metadata = chunk["metadata"]
#                 ref_parts = [metadata.get("source", "Unknown")]
#                 if metadata.get("section"):
#                     ref_parts.append(metadata["section"])
#                 if metadata.get("subsection"):
#                     ref_parts.append(metadata["subsection"])
#                 reference = " - ".join(ref_parts)
#
#             else:
#                 reference = "Unknown source"
#
#             if reference not in references:
#                 references.append(reference)
#
#         except Exception as e:
#             logger.warning(f"Failed to extract reference: {e}")
#             references.append("Unknown source")
#
#     return references
