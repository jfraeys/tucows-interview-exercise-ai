from typing import Any

REQUIRED_KEYS = {
    "answer": "No answer generated.",
    "references": [],
    "action_required": "none",
}

VALID_ACTION_REQUIRED_VALUES = {
    "none",
    "escalate_to_abuse_team", 
    "escalate_to_billing",
    "escalate_to_technical",
    "follow_up_required",
    "more_info_needed"
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


def enforce_ticket_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce the ticket schema and validate action_required values.
    
    Args:
        parsed: Dictionary parsed from LLM JSON output
        
    Returns:
        Dictionary with validated schema
    """
    result = enforce_schema(parsed, REQUIRED_KEYS)
    
    # Validate action_required value
    if result["action_required"] not in VALID_ACTION_REQUIRED_VALUES:
        result["action_required"] = "none"  # Default to safe value
        
    return result
