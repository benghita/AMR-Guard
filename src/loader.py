
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional

from .config import get_settings

# ── ZeroGPU registration (HF Spaces only) ────────────────────────────────────
# Calling spaces.GPU() at import time establishes the connection with the
# ZeroGPU daemon.  The actual @spaces.GPU decorator for the pipeline lives in
# app.py and wraps the *entire* multi-agent run so all agents share one GPU
# session (avoids lru_cache + freed-CUDA-memory hangs between agent calls).
if os.environ.get("SPACE_ID"):
    try:
        import spaces as _spaces
        try:
            _spaces.GPU(duration=600)  # register with ZeroGPU daemon at startup
        except TypeError:
            pass  # older spaces without duration param — registration still happens
    except ImportError:
        pass

logger = logging.getLogger(__name__)

TextModelName = Literal["medgemma_4b", "medgemma_27b", "txgemma_9b", "txgemma_2b"]

# MedGemma 4B IT is a vision-language model (Gemma3ForConditionalGeneration).
# It must be loaded with AutoModelForImageTextToText + AutoProcessor.
# All other models (medgemma-27b-text-it, txgemma-*) are causal LMs.
# On Kaggle T4, medgemma_27b is substituted with medgemma-4b-it (also multimodal),
# so we detect the architecture dynamically from the model config.
_MULTIMODAL_ARCHITECTURES = {"Gemma3ForConditionalGeneration"}


def _get_model_path(model_name: TextModelName) -> str:
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
    return model_path


def _get_load_kwargs() -> Dict[str, Any]:
    import torch
    settings = get_settings()
    has_cuda = torch.cuda.is_available()
    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if settings.quantization == "4bit" and has_cuda:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif not has_cuda:
        logger.warning("No CUDA GPU detected — loading model in float32 on CPU (inference will be slow)")
    return load_kwargs


@lru_cache(maxsize=8)
def _get_local_multimodal(model_name: TextModelName):
    """Load a multimodal model (e.g. MedGemma 4B IT) and return a text generation callable."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    model_path = _get_model_path(model_name)
    load_kwargs = _get_load_kwargs()

    logger.info(f"Loading multimodal model: {model_path} with kwargs: {load_kwargs}")
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info(f"Processor loaded for {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    logger.info(f"Model loaded successfully: {model_path}")

    def _call(
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        image=None,  # optional PIL.Image.Image for vision-language inference
        **generate_kwargs: Any,
    ) -> str:
        # Build chat content; prepend image token when an image is provided
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        # Decode only the newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids, skip_special_tokens=True).strip()

    return _call


@lru_cache(maxsize=8)
def _get_local_causal_lm(model_name: TextModelName):
    """Load a causal LM (e.g. TxGemma, MedGemma 27B text) and return a generation callable."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = _get_model_path(model_name)
    load_kwargs = _get_load_kwargs()

    logger.info(f"Loading causal LM: {model_path} with kwargs: {load_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"Tokenizer loaded for {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    logger.info(f"Model loaded successfully: {model_path}")

    def _call(prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, **generate_kwargs: Any) -> str:
        inputs = {k: v.to(model.device) for k, v in tokenizer(prompt, return_tensors="pt").items()}
        do_sample = temperature > 0
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                **generate_kwargs,
            )
        # Decode only the newly generated tokens, not the input prompt
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return _call


@lru_cache(maxsize=8)
def _is_multimodal(model_path: str) -> bool:
    """Check if a model uses a multimodal architecture by inspecting its config."""
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_path)
        architectures = getattr(config, "architectures", []) or []
        return bool(set(architectures) & _MULTIMODAL_ARCHITECTURES)
    except Exception:
        return False


@lru_cache(maxsize=32)
def get_text_model(
    model_name: TextModelName = "medgemma_4b",
) -> Callable[..., str]:
    """Return a cached callable for the requested model."""
    model_path = _get_model_path(model_name)
    if _is_multimodal(model_path):
        return _get_local_multimodal(model_name)
    return _get_local_causal_lm(model_name)



def _inference_core(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """Core text inference — no GPU decorator, runs on whatever device is available."""
    model = get_text_model(model_name=model_name)
    logger.info(f"Model {model_name} ready")
    result = model(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
    logger.info(f"Inference complete, response length: {len(result)} chars")
    return result


def _inference_with_image_core(
    prompt: str,
    image: Any,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """Core vision inference — no GPU decorator, runs on whatever device is available."""
    model_path = _get_model_path(model_name)
    if not _is_multimodal(model_path):
        logger.warning(
            f"{model_name} ({model_path}) is not a multimodal model; "
            "falling back to text-only inference."
        )
        return _inference_core(prompt, model_name, max_new_tokens, temperature, **kwargs)
    model_fn = _get_local_multimodal(model_name)
    result = model_fn(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature, image=image, **kwargs
    )
    logger.info(f"Vision inference complete, response length: {len(result)} chars")
    return result


def run_inference(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """Run inference with the specified model.

    Must be called from within an active @spaces.GPU context (e.g. the
    pipeline wrapper in app.py).  All agents share one GPU session so that
    the lru_cache'd model stays valid across the full pipeline.
    """
    logger.info(f"Running inference with {model_name}, max_tokens={max_new_tokens}, temp={temperature}")
    return _inference_core(prompt, model_name, max_new_tokens, temperature, **kwargs)


def run_inference_with_image(
    prompt: str,
    image: Any,  # PIL.Image.Image
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """Run vision-language inference passing a PIL image alongside the text prompt.

    Falls back to text-only if the resolved model is not multimodal.
    Must be called from within an active @spaces.GPU context.
    """
    logger.info(f"Running vision inference with {model_name}, max_tokens={max_new_tokens}")
    return _inference_with_image_core(prompt, image, model_name, max_new_tokens, temperature, **kwargs)


