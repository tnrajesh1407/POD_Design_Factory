# pipeline/remove_bg.py
from __future__ import annotations

import os
import io

import numpy as np
from rembg import remove
from PIL import Image, ImageFilter


def remove_background(
    input_path: str,
    output_path: str,
    *,
    alpha_cutoff: int = 22,
    feather: float = 0.4,
    shrink_px: int = 1,
    defringe_px: int = 2,
) -> str:
    """
    Remove background using rembg and post-process alpha & colors to reduce halos.

    Steps:
    1) rembg -> alpha mask
    2) stabilize alpha ramps (cutoff + slight feather)
    3) optional shrink (erode) mask slightly to remove bright rims
    4) defringe colors near edge (reduce background color bleed)
    """
    img = Image.open(input_path).convert("RGBA")

    out = remove(img)
    if isinstance(out, (bytes, bytearray)):
        out = Image.open(io.BytesIO(out)).convert("RGBA")
    else:
        out = out.convert("RGBA")

    out = stabilize_alpha_edges(out, cutoff=alpha_cutoff, feather=feather)

    if shrink_px and shrink_px > 0:
        out = shrink_alpha(out, px=shrink_px)

    if defringe_px and defringe_px > 0:
        out = defringe_rgba(out, radius=defringe_px)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.save(output_path, "PNG")
    return output_path


def stabilize_alpha_edges(img: Image.Image, cutoff: int = 22, feather: float = 0.4) -> Image.Image:
    """
    Convert weak semi-transparent edge ramps into cleaner edges while keeping interior detail.
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    alpha = arr[:, :, 3].astype(np.float32)

    alpha[alpha < float(cutoff)] = 0.0

    if feather and feather > 0:
        alpha_img = Image.fromarray(alpha.astype(np.uint8), mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(float(feather)))
        alpha = np.array(alpha_img, dtype=np.float32)

    alpha[alpha > 235.0] = 255.0

    arr[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def shrink_alpha(img: Image.Image, px: int = 1) -> Image.Image:
    """
    Slightly erode the alpha mask to remove a thin bright rim.
    Implemented via MaxFilter on inverted alpha (fast, PIL-only).
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    a = Image.fromarray(arr[:, :, 3], mode="L")
    inv = Image.eval(a, lambda v: 255 - v)

    k = int(2 * px + 1)
    inv2 = inv.filter(ImageFilter.MaxFilter(size=k))
    a2 = Image.eval(inv2, lambda v: 255 - v)

    arr[:, :, 3] = np.array(a2, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def defringe_rgba(img: Image.Image, radius: int = 2, *, protect_opaque_above: int = 220) -> Image.Image:
    """
    Reduce halo by pulling edge RGB toward nearby interior RGB, but protect opaque pixels.

    - Works on pixels where 0 < alpha < 255
    - Only modifies pixels where alpha <= protect_opaque_above (keeps crisp opaque strokes)
    - Uses a small interior propagation (BoxBlur) but clamps effect to the edge band only
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8).astype(np.float32)

    a = arr[:, :, 3]
    edge = (a > 0) & (a < 255) & (a <= float(protect_opaque_above))

    if not edge.any():
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    # Define interior as strongly opaque pixels
    interior = (a >= 245).astype(np.uint8) * 255
    interior_img = Image.fromarray(interior, mode="L").filter(ImageFilter.MinFilter(size=2 * radius + 1))
    interior_mask = np.array(interior_img, dtype=np.uint8) > 0

    # If interior is too small, loosen interior threshold
    if not interior_mask.any():
        interior = (a >= 220).astype(np.uint8) * 255
        interior_img = Image.fromarray(interior, mode="L").filter(ImageFilter.MinFilter(size=2 * radius + 1))
        interior_mask = np.array(interior_img, dtype=np.uint8) > 0

    if not interior_mask.any():
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")

    rgb = arr[:, :, :3].copy()

    # Create reference RGB from interior only
    ref = np.zeros_like(rgb)
    ref[interior_mask] = rgb[interior_mask]

    # Propagate interior colors outward gently
    ref_img = Image.fromarray(ref.astype(np.uint8), mode="RGB")
    for _ in range(max(1, radius)):
        ref_img = ref_img.filter(ImageFilter.BoxBlur(1))
    ref_rgb = np.array(ref_img, dtype=np.float32)

    # Blend: stronger correction on lower alpha pixels, lighter on higher alpha
    alpha_norm = np.clip(a / 255.0, 0.0, 1.0)
    strength = (1.0 - alpha_norm)  # 1 at alpha=0, 0 at alpha=255
    strength = np.clip(strength * 0.85, 0.0, 0.85)  # cap to avoid over-wash

    # Apply only on edge band
    m = edge
    rgb[m] = (1.0 - strength[m, None]) * rgb[m] + strength[m, None] * ref_rgb[m]

    arr[:, :, :3] = rgb
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")