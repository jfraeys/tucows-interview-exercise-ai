from typing import Any

REQUIRED_KEYS = {
    "answer": "No answer generated.",
    "references": [],
    "action_required": "none",
}


def enforce_schema(parsed: dict[str, Any], required: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and parse JSON output from the LLM for support tickets.
    Ensures required keys exist and sets defaults when missing.

    Args:
        output: dict parsed from LLM JSON output.

    Returns:
        Dictionary with keys 'answer', 'references', and 'action_required'.
    """
    # Apply defaults for missing keys
    for key, default in required.items():
        if key not in parsed:
            parsed[key] = default

    return parsed


def enforce_ticket_schema(output: str) -> dict[str, Any]:
    return enforce_schema(output, REQUIRED_KEYS)
