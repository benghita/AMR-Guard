
import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Literal, Optional

from .config import get_settings

# ── ZeroGPU decorator (HF Spaces only) ────────────────────────────────────────
# ZeroGPU default duration is 60 s — too short for 4B model load + inference.
# We request 120 s; fall back gracefully if the spaces version lacks `duration`.
if os.environ.get("SPACE_ID"):
    try:
        import spaces as _spaces
        try:
            _gpu = _spaces.GPU(duration=120)
        except TypeError:
            _gpu = _spaces.GPU  # older spaces API without duration param
    except ImportError:
        _gpu = lambda f: f  # noqa: E731
else:
    _gpu = lambda f: f  # noqa: E731

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


def _is_zerogpu_error(e: Exception) -> bool:
    """Return True for errors that indicate ZeroGPU failed to allocate / init a GPU.

    The spaces package raises ZeroGPUException (not RuntimeError) in newer versions,
    and re-wraps the original CUDA RuntimeError as RuntimeError('RuntimeError') in
    older versions, so we check for multiple patterns.
    """
    import traceback as _tb
    # Check exception class name — spaces raises ZeroGPUException in newer versions
    cls_name = type(e).__name__
    if "ZeroGPU" in cls_name or "GPU" in cls_name:
        return True
    msg = str(e)
    if "No CUDA GPUs are available" in msg or "CUDA" in msg:
        return True
    # spaces re-wraps with the type name: RuntimeError("'RuntimeError'") or RuntimeError("RuntimeError")
    if "RuntimeError" in msg:
        return True
    # Inspect traceback for ZeroGPU stack frames
    full_tb = "".join(_tb.format_exception(type(e), e, e.__traceback__))
    return "spaces/zero" in full_tb or "device-api.zero" in full_tb


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


@_gpu
def _run_inference_gpu(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    return _inference_core(prompt, model_name, max_new_tokens, temperature, **kwargs)


@_gpu
def _run_inference_with_image_gpu(
    prompt: str,
    image: Any,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    return _inference_with_image_core(prompt, image, model_name, max_new_tokens, temperature, **kwargs)


def run_inference(
    prompt: str,
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    **kwargs: Any,
) -> str:
    """Run inference with the specified model. Tries ZeroGPU first, falls back to CPU."""
    logger.info(f"Running inference with {model_name}, max_tokens={max_new_tokens}, temp={temperature}")
    try:
        return _run_inference_gpu(prompt, model_name, max_new_tokens, temperature, **kwargs)
    except Exception as e:
        if _is_zerogpu_error(e):
            logger.warning("ZeroGPU unavailable (%s: %s) — retrying on CPU", type(e).__name__, e)
            try:
                return _inference_core(prompt, model_name, max_new_tokens, temperature, **kwargs)
            except Exception as cpu_err:
                logger.error(f"CPU fallback also failed for {model_name}: {cpu_err}", exc_info=True)
                raise
        logger.error(f"Inference failed for {model_name}: {e}", exc_info=True)
        raise


def run_inference_with_image(
    prompt: str,
    image: Any,  # PIL.Image.Image
    model_name: TextModelName = "medgemma_4b",
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    **kwargs: Any,
) -> str:
    """
    Run vision-language inference passing a PIL image alongside the text prompt.

    Falls back to text-only inference if the resolved model is not multimodal.
    Tries ZeroGPU first, falls back to CPU on ZeroGPU init failure.
    """
    logger.info(f"Running vision inference with {model_name}, max_tokens={max_new_tokens}")
    try:
        return _run_inference_with_image_gpu(prompt, image, model_name, max_new_tokens, temperature, **kwargs)
    except Exception as e:
        if _is_zerogpu_error(e):
            logger.warning("ZeroGPU unavailable (%s: %s) — retrying vision inference on CPU", type(e).__name__, e)
            try:
                return _inference_with_image_core(prompt, image, model_name, max_new_tokens, temperature, **kwargs)
            except Exception as cpu_err:
                logger.error(f"CPU vision fallback also failed for {model_name}: {cpu_err}", exc_info=True)
                raise
        logger.error(f"Vision inference failed for {model_name}: {e}", exc_info=True)
        raise


