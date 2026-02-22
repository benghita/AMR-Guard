
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional

from .config import get_settings

logger = logging.getLogger(__name__)

TextModelName = Literal["medgemma_4b", "medgemma_27b", "txgemma_9b", "txgemma_2b"]


@lru_cache(maxsize=8)
def _get_local_causal_lm(model_name: TextModelName):
    """Load a local HuggingFace causal LM and return a generation callable."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    settings = get_settings()
    model_path_map: Dict[TextModelName, Optional[str]] = {
        "medgemma_4b": settings.medgemma_4b_model,
        "medgemma_27b": settings.medgemma_27b_model,
        "txgemma_9b": settings.txgemma_9b_model,
        "txgemma_2b": settings.txgemma_2b_model,
    }

    model_path = model_path_map[model_name]
    if not model_path:
        raise RuntimeError(
            f"No local model path configured for {model_name}. "
            f"Set MEDIC_LOCAL_*_MODEL in your environment or .env."
        )

    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if settings.quantization == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

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
) -> Callable[..., str]:
    """Return a cached callable for the requested model."""
    return _get_local_causal_lm(model_name)


def run_inference(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """Run inference with the specified model. This is the primary entry point for agents."""
    model = get_text_model(model_name=model_name)
    return model(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
