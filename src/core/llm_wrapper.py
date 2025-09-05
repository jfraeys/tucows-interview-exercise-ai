"""
LLM wrapper for llama-cpp-python with async support and device management.

Clean interface for loading and running GGUF models with proper async handling.
"""

import asyncio
import errno
import os
from pathlib import Path
from typing import Literal

from llama_cpp import CreateCompletionResponse, Llama

from src.utils import logging_helper, validate_helper

logger = logging_helper.get_logger(__name__)

DEFAULT_MODEL_DIR = "./models"
DeviceMode = Literal["cpu", "gpu", "auto"]


N_CTX_DEFAULT = 4096
MAX_TOKENS_DEFAULT = 300
TEMPERATURE_DEFAULT = 0.2
TOP_P_DEFAULT = 1.0


class LLMWrapper:
    """
    Async wrapper for llama-cpp-python with device management.

    Supports loading from local files or Hugging Face Hub.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = N_CTX_DEFAULT,
        n_gpu_layers: int = 0,
        device_mode: DeviceMode = "auto",
        local_dir: str = DEFAULT_MODEL_DIR,
        verbose: bool = False,
    ) -> None:
        """
        Initialize LLM wrapper.

        Args:
            model_path: Path to local GGUF model file
            repo_id: Hugging Face repository ID
            filename: Filename pattern for GGUF file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            device_mode: Device preference ('cpu', 'gpu', 'auto')
            local_dir: Directory for storing downloaded models
            verbose: Enable detailed logging
        """
        validate_helper.validate_model_arguments(model_path, repo_id, filename)
        self.device_mode = self._resolve_device_mode(device_mode, n_gpu_layers)
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        self.model = self._load_model(
            model_path, repo_id, filename, n_ctx, n_gpu_layers, verbose
        )

        logger.info(f"LLM initialized on {self.device_mode} mode")

    def _resolve_device_mode(
        self, device_mode: DeviceMode, n_gpu_layers: int
    ) -> DeviceMode:
        """Resolve device mode based on configuration."""
        if n_gpu_layers == 0:
            return "cpu"
        return device_mode

    def _ensure_local_directory(self) -> None:
        """Ensure the local directory exists for model storage."""
        try:
            os.makedirs(self.local_dir, exist_ok=True)
            logger.info(f"Ensured local model directory exists: {self.local_dir}")
        except OSError as e:
            if e.errno == errno.EACCES:
                logger.error(
                    f"Permission denied while creating directory: {self.local_dir}"
                )
                raise PermissionError(
                    f"Cannot create local model directory due to permissions: {self.local_dir}"
                ) from e
            else:
                logger.error(
                    f"Failed to create local model directory: {self.local_dir} - {e}"
                )
                raise RuntimeError(
                    f"Could not create local model directory: {self.local_dir}"
                ) from e

    def _load_model(
        self,
        model_path: Path | None,
        repo_id: str | None,
        filename: str | None,
        n_ctx: int,
        n_gpu_layers: int,
        verbose: bool,
    ) -> Llama:
        """Load model using appropriate method."""
        # Always ensure directory exists first (even for local models)
        self._ensure_local_directory()

        # Load local model if path provided
        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

        # Handle HuggingFace models
        if not filename or not repo_id:
            raise ValueError("Both repo_id and filename are required for HF models")

        # Check for cached model
        cached_path = os.path.join(self.local_dir, filename)
        if os.path.exists(cached_path):
            logger.info(f"Loading cached model: {cached_path}")
            return Llama(
                model_path=str(cached_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

        # Download from Hugging Face
        logger.info(f"Downloading model {repo_id}/{filename}")
        return Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self.local_dir),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = MAX_TOKENS_DEFAULT,
        temperature: float = TEMPERATURE_DEFAULT,
        top_p: float = TOP_P_DEFAULT,
        stop_sequences: list[str] | None = None,
        echo: bool = False,
        stream: bool = False,  # ensures the return type is CreateCompletionResponse
    ) -> CreateCompletionResponse:
        """
        Generate text from prompt with async handling.

        Args:
            prompt: Input prompt string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nuclear sampling parameter
            stop_sequences: Stop generation before sequences

        Returns:
            llama-cpp CreateCompletionResponse object
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        return await asyncio.to_thread(
            self.model.create_completion,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
            echo=echo,
            stream=stream,
        )
