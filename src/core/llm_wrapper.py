"""
LLM wrapper for llama-cpp-python with async support and device management.

Clean interface for loading and running GGUF models with proper async handling.
"""

import asyncio
import os
from pathlib import Path
from typing import Literal, Optional

from llama_cpp import CreateCompletionResponse, Llama

from src.utils import logging_helper, validate_helper

logger = logging_helper.get_logger(__name__)

DEFAULT_MODEL_DIR = "./models"
DeviceMode = Literal["cpu", "gpu", "auto"]


class LLMWrapper:
    """
    Async wrapper for llama-cpp-python with device management.

    Supports loading from local files or Hugging Face Hub.
    """

    def __init__(
        self,
        model_path: str | None = None,
        repo_id: str | None = None,
        filename: str | None = None,
        n_ctx: int = 4096,
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

    def _load_model(
        self,
        model_path: str | None,
        repo_id: str | None,
        filename: str | None,
        n_ctx: int,
        n_gpu_layers: int,
        verbose: bool,
    ) -> Llama:
        """Load model using appropriate method."""
        kwargs = {"n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers, "verbose": verbose}

        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return Llama(model_path=model_path, **kwargs)

        if not filename or not repo_id:
            raise ValueError("Both repo_id and filename are required for HF models")

        # Check for cached model
        cached_path = os.path.join(self.local_dir, filename)
        if os.path.exists(cached_path):
            logger.info(f"Loading cached model: {cached_path}")
            return Llama(model_path=str(cached_path), **kwargs)

        # Download from Hugging Face
        logger.info(f"Downloading model {repo_id}/{filename}")
        return Llama.from_pretrained(
            repo_id=repo_id, filename=filename, local_dir=str(self.local_dir), **kwargs
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.2,
        top_p: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
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

        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop_sequences or [],
            "echo": False,
        }

        return await asyncio.to_thread(self.model, **kwargs)
