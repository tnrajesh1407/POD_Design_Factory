# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env (ignored in prod if not present)
load_dotenv()


class ConfigError(RuntimeError):
    pass


def _req(name: str) -> str:
    v = os.getenv(name)
    if not v or not str(v).strip():
        raise ConfigError(f"Missing required environment variable: {name}")
    return str(v).strip()


def _opt(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def _opt_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _opt_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        raise ConfigError(f"Invalid integer for {name}: {v!r}")


def _opt_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        raise ConfigError(f"Invalid float for {name}: {v!r}")


@dataclass(frozen=True)
class Settings:
    # Provider selection
    image_provider: str  # "openai" or "replicate"
    enable_fallback: bool
    fallback_provider: str  # "openai" or "replicate"

    # OpenAI image generation
    openai_api_key: Optional[str]
    openai_image_model: str
    openai_image_quality: str  # "high" | "standard"
    openai_image_size: Optional[str]  # e.g. "1024x1024"
    openai_max_retries: int
    openai_base_delay_sec: float

    # Replicate generation
    replicate_api_token: Optional[str]
    replicate_model: str
    replicate_max_retries: int
    replicate_base_delay_sec: float


def load_settings() -> Settings:
    provider = (_opt("IMAGE_PROVIDER", "replicate") or "replicate").lower().strip()
    if provider not in ("openai", "replicate"):
        raise ConfigError("IMAGE_PROVIDER must be 'openai' or 'replicate'")

    enable_fallback = _opt_bool("IMAGE_ENABLE_FALLBACK", False)
    fallback_provider = (_opt("IMAGE_FALLBACK_PROVIDER", "replicate") or "replicate").lower().strip()
    if fallback_provider not in ("openai", "replicate"):
        raise ConfigError("IMAGE_FALLBACK_PROVIDER must be 'openai' or 'replicate'")
    if fallback_provider == provider and enable_fallback:
        raise ConfigError("IMAGE_FALLBACK_PROVIDER must differ from IMAGE_PROVIDER when fallback is enabled")

    # OpenAI
    openai_api_key = _opt("OPENAI_API_KEY")
    openai_model = _opt("OPENAI_IMAGE_MODEL", "gpt-image-1") or "gpt-image-1"
    openai_quality = _opt("OPENAI_IMAGE_QUALITY", "high") or "high"
    if openai_quality not in ("high", "standard"):
        raise ConfigError("OPENAI_IMAGE_QUALITY must be 'high' or 'standard'")
    openai_size = _opt("OPENAI_IMAGE_SIZE", None)
    openai_max_retries = _opt_int("OPENAI_MAX_RETRIES", 6)
    openai_base_delay = _opt_float("OPENAI_BASE_DELAY_SEC", 2.0)

    # Replicate
    replicate_token = _opt("REPLICATE_API_TOKEN")
    replicate_model = _opt("REPLICATE_MODEL", "prunaai/p-image") or "prunaai/p-image"
    replicate_max_retries = _opt_int("REPLICATE_MAX_RETRIES", 6)
    replicate_base_delay = _opt_float("REPLICATE_BASE_DELAY_SEC", 8.0)

    # Validate required keys for selected provider (and fallback if enabled)
    def require_openai():
        if not openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when using OpenAI image provider.")

    def require_replicate():
        if not replicate_token:
            raise ConfigError("REPLICATE_API_TOKEN is required when using Replicate image provider.")

    if provider == "openai":
        require_openai()
    else:
        require_replicate()

    if enable_fallback:
        if fallback_provider == "openai":
            require_openai()
        else:
            require_replicate()

    return Settings(
        image_provider=provider,
        enable_fallback=enable_fallback,
        fallback_provider=fallback_provider,
        openai_api_key=openai_api_key,
        openai_image_model=openai_model,
        openai_image_quality=openai_quality,
        openai_image_size=openai_size,
        openai_max_retries=openai_max_retries,
        openai_base_delay_sec=openai_base_delay,
        replicate_api_token=replicate_token,
        replicate_model=replicate_model,
        replicate_max_retries=replicate_max_retries,
        replicate_base_delay_sec=replicate_base_delay,
    )


# Fail fast at import time
settings = load_settings()