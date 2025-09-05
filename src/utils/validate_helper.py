from pathlib import Path


def validate_model_arguments(
    model_path: Path | None, model_repo: str | None, model_filename: str | None
) -> None:
    """
    Validate model configuration arguments for logical consistency.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If argument combinations are invalid
    """
    # Ensure mutually exclusive model source options
    has_local_path = True if model_path else False
    has_hf_repo = True if (model_repo or model_filename) else False

    if has_local_path and has_hf_repo:
        raise ValueError(
            "Conflicting model sources: use either --model-path OR "
            "--model-repo + --model-filename, not both"
        )

    # Ensure complete Hugging Face configuration
    if bool(model_repo) != bool(model_filename):
        raise ValueError(
            "Incomplete Hugging Face configuration: both --model-repo and "
            "--model-filename are required when using HF models"
        )

    # Ensure at least one model source is specified
    if not has_local_path and not has_hf_repo:
        raise ValueError(
            "No model specified: provide either --model-path or "
            "--model-repo + --model-filename"
        )
