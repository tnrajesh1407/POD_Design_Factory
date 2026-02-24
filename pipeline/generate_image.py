# pipeline/generate_image.py
from __future__ import annotations

import os
import time
import base64
from typing import Optional, Dict, Any, Tuple

import requests
from PIL import Image

import replicate
from replicate.exceptions import ReplicateError

from pipeline.replicate_rate_limiter import wait_for_slot
from config import settings


# ----------------------------
# POD-safe defaults (still env-driven but validated via settings)
# ----------------------------
DEFAULT_ASPECT_RATIO = os.getenv("GEN_ASPECT_RATIO", "1:1")
DEFAULT_PROMPT_UPSAMPLING = os.getenv("GEN_PROMPT_UPSAMPLING", "false").lower() == "true"
DEFAULT_DISABLE_SAFETY = os.getenv("GEN_DISABLE_SAFETY", "false").lower() == "true"
DEFAULT_W = int(os.getenv("GEN_IMG_W", "1024"))
DEFAULT_H = int(os.getenv("GEN_IMG_H", "1024"))
DEFAULT_LORA_WEIGHTS = os.getenv("GEN_LORA_WEIGHTS", "").strip() or None
DEFAULT_LORA_SCALE = float(os.getenv("GEN_LORA_SCALE", "0.5"))


class ImageGenTransientError(RuntimeError):
    """Retryable/transient error."""


def _multiple_of_16(x: int) -> int:
    x = int(x)
    return max(256, ((x + 15) // 16) * 16)


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _download_to_path(url: str, output_path: str, timeout: int = 60) -> None:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    _ensure_dir_for_file(output_path)
    with open(output_path, "wb") as f:
        f.write(r.content)
    Image.open(output_path).verify()


def _get_image_url(output) -> str:
    if output is None:
        raise RuntimeError("Replicate returned empty output")
    if hasattr(output, "url"):
        return output.url
    if isinstance(output, str) and output.startswith("http"):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0:
        first = output[0]
        if hasattr(first, "url"):
            return first.url
        if isinstance(first, str) and first.startswith("http"):
            return first
    raise RuntimeError(f"Unsupported Replicate output type: {type(output)}")


def _map_aspect_ratio_to_openai_size(aspect_ratio: str) -> str:
    # explicit override
    if settings.openai_image_size:
        return settings.openai_image_size

    ar = (aspect_ratio or "1:1").strip()
    if ar in ("1:1", "square"):
        return "1024x1024"
    if ar in ("2:3", "3:4", "4:5", "9:16", "portrait"):
        return "1024x1536"
    if ar in ("3:2", "4:3", "16:9", "landscape"):
        return "1536x1024"
    return "1024x1024"


def _openai_client():
    # Lazy import so Replicate-only users don't need openai installed
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI provider selected but package 'openai' not installed. Run: pip install openai") from e
    return OpenAI(api_key=settings.openai_api_key)


def _save_b64_png_to_path(b64_data: str, output_path: str) -> None:
    _ensure_dir_for_file(output_path)
    img_bytes = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    Image.open(output_path).verify()


def _is_transient_openai_error(msg: str) -> bool:
    m = msg.lower()
    return any(k in m for k in ("rate limit", "429", "timeout", "temporar", "overloaded", "try again", "server error"))


def _generate_openai(prompt: str, output_path: str, *, aspect_ratio: str, seed: Optional[int]) -> str:
    client = _openai_client()
    size = _map_aspect_ratio_to_openai_size(aspect_ratio)

    params: Dict[str, Any] = {
        "model": settings.openai_image_model,
        "prompt": prompt,
        "size": size,
        "quality": settings.openai_image_quality,
    }
    if seed is not None:
        params["seed"] = int(seed)

    delay = float(settings.openai_base_delay_sec)
    last_err: Optional[Exception] = None

    for attempt in range(1, settings.openai_max_retries + 1):
        try:
            print(f"Calling OpenAI images model='{settings.openai_image_model}' size='{size}' (attempt {attempt}/{settings.openai_max_retries})...")
            resp = client.images.generate(**params)

            if not resp.data or not getattr(resp.data[0], "b64_json", None):
                raise RuntimeError("OpenAI returned no image data (missing b64_json).")

            _save_b64_png_to_path(resp.data[0].b64_json, output_path)
            print("Image saved:", output_path)
            return output_path

        except Exception as e:
            last_err = e
            msg = str(e)

            # If seed is rejected, retry once without it
            if seed is not None and "seed" in msg.lower():
                print("OpenAI rejected 'seed' parameter. Retrying without seed...")
                params.pop("seed", None)
                seed = None
                time.sleep(min(delay, 15))
                continue

            if _is_transient_openai_error(msg) and attempt < settings.openai_max_retries:
                print(f"Transient OpenAI error. Backing off {delay:.1f}s... ({msg[:160]})")
                time.sleep(min(delay, 20))
                delay *= 1.7
                continue

            # Non-transient
            raise

    raise RuntimeError(f"OpenAI image generation failed after {settings.openai_max_retries} retries. Last error: {last_err}")


def _generate_replicate(
    prompt: str,
    output_path: str,
    *,
    aspect_ratio: str,
    width: int,
    height: int,
    seed: Optional[int],
    prompt_upsampling: bool,
    disable_safety_checker: bool,
    lora_weights: Optional[str],
    lora_scale: float,
) -> str:
    delay = float(settings.replicate_base_delay_sec)
    last_err: Optional[Exception] = None

    for attempt in range(1, settings.replicate_max_retries + 1):
        try:
            wait_for_slot()
            print(f"Calling Replicate model='{settings.replicate_model}' (attempt {attempt}/{settings.replicate_max_retries})...")

            inputs: Dict[str, Any] = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "prompt_upsampling": bool(prompt_upsampling),
                "disable_safety_checker": bool(disable_safety_checker),
            }

            if seed is not None:
                inputs["seed"] = int(seed)

            if aspect_ratio == "custom":
                inputs["width"] = _multiple_of_16(width)
                inputs["height"] = _multiple_of_16(height)

            if lora_weights:
                inputs["lora_weights"] = str(lora_weights)
                inputs["lora_scale"] = float(lora_scale)

            output = replicate.run(settings.replicate_model, input=inputs)
            url = _get_image_url(output)
            _download_to_path(url, output_path)

            print("Image saved:", output_path)
            return output_path

        except ReplicateError as e:
            last_err = e
            msg = str(e)
            if "429" in msg and attempt < settings.replicate_max_retries:
                print(f"429 detected. Backing off {delay:.1f}s...")
                time.sleep(min(delay, 30))
                delay *= 1.6
                continue
            raise

        except (requests.RequestException, OSError, RuntimeError) as e:
            last_err = e
            if attempt < settings.replicate_max_retries:
                print(f"Download/IO error: {e}. Backing off {delay:.1f}s...")
                time.sleep(min(delay, 30))
                delay *= 1.4
                continue
            raise RuntimeError(f"Replicate failed after retries. Last error: {last_err}") from e

    raise RuntimeError(f"Failed after {settings.replicate_max_retries} retries. Last error: {last_err}")


def _run_provider(
    provider: str,
    prompt: str,
    output_path: str,
    *,
    aspect_ratio: str,
    width: int,
    height: int,
    seed: Optional[int],
    prompt_upsampling: bool,
    disable_safety_checker: bool,
    lora_weights: Optional[str],
    lora_scale: float,
) -> str:
    provider = (provider or "").lower().strip()
    if provider == "openai":
        return _generate_openai(prompt, output_path, aspect_ratio=aspect_ratio, seed=seed)
    # default replicate
    return _generate_replicate(
        prompt,
        output_path,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        seed=seed,
        prompt_upsampling=prompt_upsampling,
        disable_safety_checker=disable_safety_checker,
        lora_weights=lora_weights,
        lora_scale=lora_scale,
    )


def generate_image(
    prompt: str,
    output_path: str,
    *,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    width: int = DEFAULT_W,
    height: int = DEFAULT_H,
    seed: Optional[int] = None,
    prompt_upsampling: bool = DEFAULT_PROMPT_UPSAMPLING,
    disable_safety_checker: bool = DEFAULT_DISABLE_SAFETY,
    lora_weights: Optional[str] = DEFAULT_LORA_WEIGHTS,
    lora_scale: float = DEFAULT_LORA_SCALE,
    provider_override: Optional[str] = None,
) -> str:
    """
    Unified image generation entrypoint.

    - Uses settings.image_provider by default
    - If provider_override is provided ("openai" or "replicate"), it wins
    - If settings.enable_fallback=True, it will attempt fallback provider on failure
    """
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt is empty")

    aspect_ratio = (aspect_ratio or "1:1").strip()
    _ensure_dir_for_file(output_path)

    primary = (provider_override or settings.image_provider).lower().strip()
    fallback = settings.fallback_provider.lower().strip()

    try:
        return _run_provider(
            primary,
            str(prompt).strip(),
            output_path,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            seed=seed,
            prompt_upsampling=prompt_upsampling,
            disable_safety_checker=disable_safety_checker,
            lora_weights=lora_weights,
            lora_scale=lora_scale,
        )
    except Exception as e:
        if not settings.enable_fallback:
            raise

        print(f"[WARN] Primary provider '{primary}' failed: {e}")
        print(f"[WARN] Falling back to '{fallback}'...")

        return _run_provider(
            fallback,
            str(prompt).strip(),
            output_path,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            seed=seed,
            prompt_upsampling=prompt_upsampling,
            disable_safety_checker=disable_safety_checker,
            lora_weights=lora_weights,
            lora_scale=lora_scale,
        )