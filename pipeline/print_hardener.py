# pipeline/print_hardener.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

try:
    import cv2  # optional
except Exception:
    cv2 = None


@dataclass
class HardenConfig:
    alpha_threshold: int = 200          # 180-220 typical
    quantize_colors: int = 24           # 12-32 typical
    enable_quantize: bool = True
    enable_thicken_lineart: bool = True
    lineart_unique_colors_max: int = 24 # heuristic trigger
    thicken_kernel: int = 2
    thicken_iterations: int = 1


def _harden_alpha(img: Image.Image, threshold: int) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    a = data[:, :, 3]
    a = np.where(a >= threshold, 255, 0).astype(np.uint8)
    data[:, :, 3] = a
    return Image.fromarray(data)


def _purge_matte(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img, dtype=np.uint8)
    a = data[:, :, 3]
    mask = (a == 0)
    data[mask] = [0, 0, 0, 0]  # purge RGB in fully-transparent pixels
    return Image.fromarray(data)


def _quantize_rgba(img: Image.Image, colors: int) -> Image.Image:
    """Quantize RGB only; preserve alpha channel."""
    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb = Image.fromarray(rgba[:, :, :3]).convert("RGB")

    # Adaptive palette quantization
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=colors).convert("RGB")
    pal_np = np.array(pal, dtype=np.uint8)

    out = np.dstack([pal_np, rgba[:, :, 3]]).astype(np.uint8)
    return Image.fromarray(out)


def _estimate_unique_colors(img: Image.Image, sample: int = 512) -> int:
    """Fast-ish unique color count estimate by downscaling."""
    rgb = img.convert("RGBA")
    arr = np.array(rgb, dtype=np.uint8)[:, :, :3]
    small = Image.fromarray(arr).resize((sample, sample), Image.NEAREST)
    sarr = np.array(small, dtype=np.uint8).reshape(-1, 3)
    return int(np.unique(sarr, axis=0).shape[0])


def _thicken_alpha(img: Image.Image, kernel: int, iterations: int) -> Image.Image:
    if cv2 is None:
        return img  # no-op if opencv not available

    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    a = rgba[:, :, 3]

    k = np.ones((kernel, kernel), np.uint8)
    a2 = cv2.dilate(a, k, iterations=iterations)

    rgba[:, :, 3] = a2
    return Image.fromarray(rgba)


def harden_print_png(
    in_path: str,
    out_path: str,
    cfg: Optional[HardenConfig] = None,
) -> Dict[str, Any]:
    """
    Production hardening:
      1) binary alpha
      2) purge matte
      3) optional thicken (line-art heuristic)
      4) optional quantize (RGB)
      5) binary alpha again + purge matte
    """
    cfg = cfg or HardenConfig()

    img = Image.open(in_path).convert("RGBA")

    # 1) harden + purge
    img = _harden_alpha(img, cfg.alpha_threshold)
    img = _purge_matte(img)

    # 2) line-art thickening if triggered
    uniq = _estimate_unique_colors(img)
    did_thicken = False
    if cfg.enable_thicken_lineart and uniq <= cfg.lineart_unique_colors_max:
        img = _thicken_alpha(img, cfg.thicken_kernel, cfg.thicken_iterations)
        img = _harden_alpha(img, cfg.alpha_threshold)
        img = _purge_matte(img)
        did_thicken = True

    # 3) quantize (RGB only)
    did_quantize = False
    if cfg.enable_quantize and cfg.quantize_colors and cfg.quantize_colors > 0:
        img = _quantize_rgba(img, cfg.quantize_colors)
        img = _harden_alpha(img, cfg.alpha_threshold)
        img = _purge_matte(img)
        did_quantize = True

    img.save(out_path, "PNG", optimize=True)

    return {
        "out_path": out_path,
        "unique_colors_est": uniq,
        "did_thicken": did_thicken,
        "did_quantize": did_quantize,
        "alpha_threshold": cfg.alpha_threshold,
        "quantize_colors": cfg.quantize_colors,
        "cv2_available": cv2 is not None,
    }