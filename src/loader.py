
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional

from .config import get_settings

logger = logging.getLogger(__name__)

TextBackend = Literal["vertex", "local"]
TextModelName = Literal["medgemma_4b", "medgemma_27b", "txgemma_9b", "txgemma_2b"]


def _resolve_backend(requested: Optional[TextBackend]) -> TextBackend:
    settings = get_settings()
    backend = requested or settings.default_backend  # type: ignore[assignment]
    if backend == "vertex" and not settings.use_vertex:
        logger.info("Vertex disabled in settings, falling back to local backend.")
        return "local"
    return backend


@lru_cache(maxsize=8)
def _get_vertex_chat_model(model_name: TextModelName):
    """Load a Vertex AI chat model and return a callable that takes a prompt string."""
    try:
        from langchain_google_vertexai import ChatVertexAI
    except Exception as exc:
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

    llm = ChatVertexAI(
        model=model_id_map[model_name],
        project=settings.vertex_project_id,
        location=settings.vertex_location,
        temperature=0.2,
    )

    def _call(prompt: str, **kwargs: Any) -> str:
        result = llm.invoke(prompt, **kwargs)
        return str(getattr(result, "content", result))

    return _call


@lru_cache(maxsize=8)
def _get_local_causal_lm(model_name: TextModelName):
    """Load a local HuggingFace causal LM and return a generation callable."""
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
            "Set MEDIC_LOCAL_*_MODEL or use the Vertex backend."
        )

    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if settings.quantization == "4bit":
        load_kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    def _call(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, **generate_kwargs: Any) -> str:
        inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else 0.0,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        # Decode only the newly generated tokens, not the input prompt
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return _call


@lru_cache(maxsize=32)
def get_text_model(
    model_name: TextModelName = "medgemma_4b",
    backend: Optional[TextBackend] = None,
) -> Callable[..., str]:
    """Return a cached callable for the requested model and backend."""
    resolved = _resolve_backend(backend)
    return _get_vertex_chat_model(model_name) if resolved == "vertex" else _get_local_causal_lm(model_name)


def run_inference(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    backend: Optional[TextBackend] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """Run inference with the specified model. This is the primary entry point for agents."""
    model = get_text_model(model_name=model_name, backend=backend)
    return model(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)

