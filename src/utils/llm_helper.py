import json

from llama_cpp import CreateCompletionResponse


def detect_device() -> str:
    """Detects CUDA, MPS (Apple), or falls back to CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def parse_llama_response(response: CreateCompletionResponse) -> dict:
    """
    Convert a Llama CreateCompletionResponse into a structured dict
    according to the MCP schema: {"answer": str, "references": list, "action_required": str}.
    """
    raw_text = response["choices"][0].text if response.choices else ""

    # Attempt to parse as JSON
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON")

    # Optional: validate the keys exist
    for key in ("answer", "references", "action_required"):
        if key not in parsed:
            parsed[key] = None  # or raise an error

    return parsed
