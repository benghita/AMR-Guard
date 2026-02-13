
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from .config import get_settings


logger = logging.getLogger(__name__)

TextBackend = Literal["vertex", "local"]
TextModelName = Literal["medgemma_4b", "medgemma_27b", "txgemma_9b", "txgemma_2b"]


def _resolve_backend(
    requested: Optional[TextBackend],
) -> TextBackend:
    settings = get_settings()
    backend = requested or settings.default_backend  # type: ignore[assignment]
    if backend == "vertex" and not settings.use_vertex:
        logger.info("Vertex disabled in settings, falling back to local backend.")
        return "local"
    return backend


@lru_cache(maxsize=8)
def _get_vertex_chat_model(model_name: TextModelName):
    """
    Lazily construct a Vertex AI chat model via langchain-google-vertexai.

    Returns an object with an .invoke(str) method; we wrap this in a simple
    callable for downstream use.
    """

    try:
        from langchain_google_vertexai import ChatVertexAI
    except Exception as exc:  # pragma: no cover - import-time failure
        raise RuntimeError(
            "langchain-google-vertexai is not available; "
            "install it or switch MEDIC_DEFAULT_BACKEND=local."
        ) from exc

    settings = get_settings()

    if settings.vertex_project_id is None:
        raise RuntimeError(
            "MEDIC_VERTEX_PROJECT_ID is not set. "
            "Set it in your environment or .env to use Vertex AI."
        )

    model_id_map: Dict[TextModelName, str] = {
        "medgemma_4b": settings.vertex_medgemma_4b_model,
        "medgemma_27b": settings.vertex_medgemma_27b_model,
        "txgemma_9b": settings.vertex_txgemma_9b_model,
        "txgemma_2b": settings.vertex_txgemma_2b_model,
    }
    model_id = model_id_map[model_name]

    llm = ChatVertexAI(
        model=model_id,
        project=settings.vertex_project_id,
        location=settings.vertex_location,
        temperature=0.2,
    )

    def _call(prompt: str, **kwargs: Any) -> str:
        """Thin wrapper returning plain text from ChatVertexAI."""

        result = llm.invoke(prompt, **kwargs)
        # langchain BaseMessage or plain string
        content = getattr(result, "content", result)
        return str(content)

    return _call


@lru_cache(maxsize=8)
def _get_local_causal_lm(model_name: TextModelName):
    """
    Lazily load a local transformers model for offline / Kaggle usage.

    Assumes model paths are provided via MEDIC_LOCAL_* env vars and that
    the appropriate model weights are available in the environment.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    settings = get_settings()

    model_path_map: Dict[TextModelName, Optional[str]] = {
        "medgemma_4b": settings.local_medgemma_4b_model,
        "medgemma_27b": settings.local_medgemma_27b_model,
        "txgemma_9b": settings.local_txgemma_9b_model,
        "txgemma_2b": settings.local_txgemma_2b_model,
    }

    model_path = model_path_map[model_name]
    if not model_path:
        raise RuntimeError(
            f"No local model path configured for {model_name}. "
            f"Set MEDIC_LOCAL_*_MODEL or use the Vertex backend."
        )

    load_kwargs: Dict[str, Any] = {
        "device_map": "auto",
    }

    # Optional 4-bit quantization via bitsandbytes
    if get_settings().quantization == "4bit":
        load_kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    def _call(
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        **generate_kwargs: Any,
    ) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = temperature > 0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else 0.0,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )

        # Drop the prompt tokens and decode only the completion
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    return _call


@lru_cache(maxsize=32)
def get_text_model(
    model_name: TextModelName = "medgemma_4b",
    backend: Optional[TextBackend] = None,
) -> Callable[..., str]:
    """
    Return a cached text-generation callable.

    Example:

        from src.loader import get_text_model
        model = get_text_model("medgemma_4b")
        answer = model("Explain ESBL in simple terms.")
    """

    resolved_backend = _resolve_backend(backend)

    if resolved_backend == "vertex":
        return _get_vertex_chat_model(model_name)
    else:
        return _get_local_causal_lm(model_name)


def run_inference(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    backend: Optional[TextBackend] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """
    Convenience wrapper around `get_text_model`.

    This is the simplest entry point to use inside agents:

        from src.loader import run_inference
        text = run_inference(prompt, model_name="medgemma_4b")
    """

    model = get_text_model(model_name=model_name, backend=backend)
    return model(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )


__all__ = [
    "TextBackend",
    "TextModelName",
    "get_text_model",
    "run_inference",
]

