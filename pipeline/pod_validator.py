# pod_validator.py
"""
Print-on-Demand (POD) print-ready PNG validator + adaptive fixer.

Guarantees (when validation passes):
- Size: 4500x5400 (default; configurable)
- RGBA
- Transparent background (not fully opaque)
- Embedded sRGB ICC profile (auto-fixed)
- DPI metadata = 300 (auto-fixed)
- Centered within tolerance
- Respects safe margins
- Detects and optionally fixes "edge-band junk" (semi-transparent halo pixels near edges)

OpenCV is OPTIONAL. If it's missing, the validator still works;
some edge scoring / aggressive fixes are skipped gracefully.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
from PIL import Image, ImageFilter, ImageCms, ImageDraw

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class PODValidationError(Exception):
    """Raised when a print-ready PNG fails validation."""
    pass


# ---------------------------------------------------------
# DPI + ICC helpers (cached)
# ---------------------------------------------------------

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def _srgb_icc_bytes() -> bytes:
    return _SRGB_ICC_BYTES


def _save_png_with_profile(img: Image.Image, path: str, *, dpi: Tuple[int, int] = (300, 300)) -> None:
    """Always save print-ready PNGs with correct DPI + embedded sRGB ICC profile."""
    img = img.convert("RGBA")
    img.save(path, "PNG", dpi=dpi, icc_profile=_srgb_icc_bytes(), optimize=True)


def _check_and_fix_dpi_and_icc(
    image_path: str,
    *,
    expected_dpi: int = 300,
    auto_fix: bool = True,
    require_icc: bool = True,
) -> bool:
    """Ensures PNG has correct DPI metadata and an embedded ICC profile (sRGB)."""
    img = Image.open(image_path).convert("RGBA")

    dpi = img.info.get("dpi", None)
    dpi_ok = False
    if dpi is not None:
        try:
            dpi_ok = (int(dpi[0]) == int(expected_dpi)) and (int(dpi[1]) == int(expected_dpi))
        except Exception:
            dpi_ok = False

    icc = img.info.get("icc_profile", None)
    icc_ok = (icc is not None) if require_icc else True

    if dpi_ok and icc_ok:
        return True

    if not auto_fix:
        if not dpi_ok and not icc_ok:
            raise PODValidationError(
                f"Missing/invalid DPI and missing ICC profile (expected {expected_dpi} DPI + sRGB ICC)."
            )
        if not dpi_ok:
            raise PODValidationError(f"Invalid or missing DPI metadata (expected {expected_dpi}).")
        if not icc_ok:
            raise PODValidationError("Missing ICC profile (expected sRGB).")
        return True

    _save_png_with_profile(img, image_path, dpi=(expected_dpi, expected_dpi))
    return True


# ---------------------------------------------------------
# Production hardener: binary alpha + matte purge (+ optional)
# ---------------------------------------------------------

def _estimate_unique_colors(img: Image.Image, sample: int = 512) -> int:
    rgb = img.convert("RGBA")
    arr = np.array(rgb, dtype=np.uint8)[:, :, :3]
    small = Image.fromarray(arr).resize((sample, sample), Image.NEAREST)
    sarr = np.array(small, dtype=np.uint8).reshape(-1, 3)
    return int(np.unique(sarr, axis=0).shape[0])


def _harden_alpha_binary(img: Image.Image, threshold: int = 200) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3]
    a = np.where(a >= int(threshold), 255, 0).astype(np.uint8)
    arr[:, :, 3] = a
    return Image.fromarray(arr)


def _purge_matte_rgb_where_transparent(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3]
    mask = (a == 0)
    arr[mask] = [0, 0, 0, 0]
    return Image.fromarray(arr)


def _thicken_alpha_if_lineart(img: Image.Image, kernel: int = 2, iterations: int = 1) -> Image.Image:
    # Only if cv2 available
    if cv2 is None:
        return img
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    a = arr[:, :, 3]
    k = np.ones((int(kernel), int(kernel)), np.uint8)
    a2 = cv2.dilate(a, k, iterations=int(iterations))
    arr[:, :, 3] = a2
    return Image.fromarray(arr)


def _quantize_rgb_keep_alpha(img: Image.Image, colors: int = 24) -> Image.Image:
    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb = Image.fromarray(rgba[:, :, :3]).convert("RGB")
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=int(colors)).convert("RGB")
    pal_np = np.array(pal, dtype=np.uint8)
    out = np.dstack([pal_np, rgba[:, :, 3]]).astype(np.uint8)
    return Image.fromarray(out)


def harden_print_ready_png(
    image_path: str,
    *,
    out_suffix: str = "_hardened",
    alpha_threshold: int = 200,
    quantize_colors: int = 24,
    enable_quantize: bool = True,
    enable_thicken_lineart: bool = True,
    lineart_unique_colors_max: int = 24,
    thicken_kernel: int = 2,
    thicken_iterations: int = 1,
    expected_dpi: int = 300,
) -> str:
    """
    Deterministic hardening:
      - binary alpha (kills edge softness)
      - purge RGB in transparent pixels (prevents halos)
      - optional: thicken line art (if palette small)
      - optional: quantize RGB palette
    """
    img = Image.open(image_path).convert("RGBA")

    uniq = _estimate_unique_colors(img)
    do_thicken = bool(enable_thicken_lineart and uniq <= int(lineart_unique_colors_max))

    img = _harden_alpha_binary(img, threshold=int(alpha_threshold))
    img = _purge_matte_rgb_where_transparent(img)

    if do_thicken:
        img = _thicken_alpha_if_lineart(img, kernel=int(thicken_kernel), iterations=int(thicken_iterations))
        img = _harden_alpha_binary(img, threshold=int(alpha_threshold))
        img = _purge_matte_rgb_where_transparent(img)

    if enable_quantize and int(quantize_colors) > 0:
        img = _quantize_rgb_keep_alpha(img, colors=int(quantize_colors))
        img = _harden_alpha_binary(img, threshold=int(alpha_threshold))
        img = _purge_matte_rgb_where_transparent(img)

    root, ext = os.path.splitext(image_path)
    out_path = root + out_suffix + ext
    _save_png_with_profile(img, out_path, dpi=(int(expected_dpi), int(expected_dpi)))
    return out_path


# ---------------------------------------------------------
# Edge quality scoring
# ---------------------------------------------------------

def _edge_band_mask_from_alpha(a_u8: np.ndarray, alpha_threshold: int, band_px: int) -> np.ndarray:
    """
    Thin band *inside* the solid alpha region near boundary.
    Requires OpenCV distanceTransform; if cv2 missing, returns empty mask.
    """
    a = a_u8.astype(np.uint8)
    alpha_threshold = int(alpha_threshold)
    band_px = int(band_px)

    if band_px <= 0:
        return np.zeros_like(a, dtype=bool)

    solid = (a >= alpha_threshold).astype(np.uint8) * 255
    if solid.max() == 0:
        return np.zeros_like(a, dtype=bool)

    if cv2 is None:
        return np.zeros_like(a, dtype=bool)

    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3).astype(np.float32)
    band = (dist > 0.0) & (dist <= float(band_px))
    return band.astype(bool)


def score_edge_quality_v2(
    image_path: str,
    *,
    alpha_threshold: int = 20,
    band_px: int = 6,
    inner_px: int = 3,
) -> Dict[str, Any]:
    """
    Scores edge quality (0..100). Higher is better.
    Metrics:
      - soft_ratio: mid-alpha pixels among present pixels
      - edge_soft_ratio: mid-alpha pixels in an edge band
      - halo_drift: RGB drift between edge band and inner band (matte proxy)
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)

    rgb = arr[:, :, :3].astype(np.int32)
    a = arr[:, :, 3].astype(np.uint8)

    if a.max() == 0:
        return {"score": 0.0, "reason": "empty_alpha"}

    solid = a >= int(alpha_threshold)
    solid_cnt = int(np.sum(solid))
    if solid_cnt == 0:
        return {"score": 0.0, "reason": "no_solid_pixels"}

    edge_band = _edge_band_mask_from_alpha(a, alpha_threshold=int(alpha_threshold), band_px=int(band_px))
    edge_band_cnt = int(np.sum(edge_band))

    present = (a > 0)
    present_cnt = int(np.sum(present))

    soft = (a >= 10) & (a <= 240)
    soft_cnt = int(np.sum(soft))
    soft_ratio = float(soft_cnt) / float(max(1, present_cnt))

    edge_soft = edge_band & soft
    edge_soft_cnt = int(np.sum(edge_soft))
    edge_soft_ratio = float(edge_soft_cnt) / float(max(1, edge_band_cnt))

    halo_drift = 0.0
    if cv2 is not None and edge_band_cnt > 0:
        solid_u8 = (solid.astype(np.uint8) * 255)
        dist_in = cv2.distanceTransform(solid_u8, cv2.DIST_L2, 3).astype(np.float32)

        inner_band = (dist_in >= float(inner_px)) & (dist_in <= float(inner_px + band_px))
        if np.any(inner_band):
            rgbf = rgb.astype(np.float32)
            edge_mean = np.mean(rgbf[edge_band & (a > 0)], axis=0)
            inner_mean = np.mean(rgbf[inner_band & (a > 0)], axis=0)
            halo_drift = float(np.linalg.norm(edge_mean - inner_mean))

    p_soft = min(35.0, max(0.0, (soft_ratio - 0.06)) * 300.0)
    p_edge_soft = min(45.0, max(0.0, (edge_soft_ratio - 0.12)) * 180.0)
    p_halo = min(40.0, max(0.0, (halo_drift - 12.0)) * (40.0 / 35.0))

    score = 100.0 - (p_soft + p_edge_soft + p_halo)
    score = float(max(0.0, min(100.0, score)))

    return {
        "score": round(score, 2),
        "metrics": {
            "soft_ratio": round(soft_ratio, 6),
            "edge_soft_ratio": round(edge_soft_ratio, 6),
            "halo_drift": round(halo_drift, 3),
            "solid_pixels": solid_cnt,
            "present_pixels": present_cnt,
            "soft_pixels": soft_cnt,
            "edge_soft_pixels": edge_soft_cnt,
            "edge_band_pixels": edge_band_cnt,
            "cv2_available": bool(cv2 is not None),
        },
        "penalties": {
            "p_soft": round(p_soft, 2),
            "p_edge_soft": round(p_edge_soft, 2),
            "p_halo": round(p_halo, 2),
        },
    }


def detect_vector_mode(
    image_path: str,
    *,
    alpha_threshold: int = 20,
    soft_ratio_max: float = 0.04,
    max_palette: int = 48,
    quant_step: int = 16,
) -> Dict[str, Any]:
    """Heuristic vector-like detector."""
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3]
    a = arr[:, :, 3]

    solid = a >= int(alpha_threshold)
    if not np.any(solid):
        return {"is_vector": False, "reasons": {"empty": True}}

    soft = (a > 0) & (a < 255)
    solid_cnt = int(np.sum(solid))
    soft_cnt = int(np.sum(soft))
    soft_ratio = float(soft_cnt) / float(max(1, solid_cnt))

    q = int(quant_step)
    rgb_q = (rgb[solid] // q) * q
    uniq = int(np.unique(rgb_q.reshape(-1, 3), axis=0).shape[0]) if rgb_q.size else 0

    is_vector = (soft_ratio <= float(soft_ratio_max)) and (uniq <= int(max_palette))
    return {
        "is_vector": bool(is_vector),
        "reasons": {
            "soft_ratio": round(soft_ratio, 6),
            "unique_colors_est": uniq,
            "soft_ratio_max": soft_ratio_max,
            "max_palette": max_palette,
        },
    }


# ---------------------------------------------------------
# Strong fix (cv2 required; skip if missing)
# ---------------------------------------------------------

def _harden_edges_matte_purge(
    image_path: str,
    *,
    alpha_low_cut: int = 22,
    alpha_snap_opaque: int = 245,
    edge_harden_mid: int = 140,
    erode_px: int = 1,
    dilate_px: int = 0,
    inpaint_radius: int = 3,
    out_suffix: str = "_edgefixed",
    expected_dpi: int = 300,
) -> str:
    """
    Aggressive fix (requires OpenCV):
      - clamp alpha
      - harden mid alpha
      - morphology on solid mask
      - inpaint RGB in soft zone
    """
    if cv2 is None:
        # OpenCV not available -> cannot do morphology/inpaint safely
        return image_path

    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3].copy()
    a = arr[:, :, 3].copy()

    a[a < int(alpha_low_cut)] = 0
    a[a > int(alpha_snap_opaque)] = 255
    a[(a >= int(edge_harden_mid)) & (a < 255)] = 255

    solid = (a >= 255).astype(np.uint8) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid = cv2.erode(solid, k, iterations=int(erode_px))
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid = cv2.dilate(solid, k, iterations=int(dilate_px))

    a = np.where(solid > 0, 255, a).astype(np.uint8)

    soft_mask = ((a > 0) & (a < 255)).astype(np.uint8) * 255
    if soft_mask.max() > 0:
        rgb_out = cv2.inpaint(rgb.astype(np.uint8), soft_mask, inpaintRadius=int(inpaint_radius), flags=cv2.INPAINT_TELEA)
        rgb = rgb_out

    arr[:, :, :3] = rgb
    arr[:, :, 3] = a

    root, ext = os.path.splitext(image_path)
    out_path = root + out_suffix + ext
    _save_png_with_profile(Image.fromarray(arr, "RGBA"), out_path, dpi=(int(expected_dpi), int(expected_dpi)))
    return out_path


def maybe_fix_edges(
    image_path: str,
    *,
    alpha_threshold: int = 20,
    band_px: int = 6,
    expected_dpi: int = 300,
    min_ok_score: float = 90.0,
    halo_drift_trigger: float = 20.0,
    edge_soft_ratio_trigger: float = 0.18,
) -> Tuple[str, Dict[str, Any]]:
    """Metric-driven decision to apply strong edge fix."""
    rep = score_edge_quality_v2(image_path, alpha_threshold=int(alpha_threshold), band_px=int(band_px), inner_px=3)
    rep["vector_mode"] = detect_vector_mode(image_path, alpha_threshold=int(alpha_threshold))

    score = float(rep.get("score", 0.0))
    if not math.isfinite(score):
        score = 0.0
        rep["score"] = score

    metrics = rep.get("metrics", {}) or {}
    halo_drift = float(metrics.get("halo_drift", 0.0)) if math.isfinite(float(metrics.get("halo_drift", 0.0))) else 0.0
    edge_soft_ratio = float(metrics.get("edge_soft_ratio", 0.0)) if math.isfinite(float(metrics.get("edge_soft_ratio", 0.0))) else 0.0
    soft_ratio = float(metrics.get("soft_ratio", 0.0)) if math.isfinite(float(metrics.get("soft_ratio", 0.0))) else 0.0

    if score >= float(min_ok_score):
        rep["action"] = "skip_ok"
        rep["output"] = image_path
        return image_path, rep

    needs = (
        halo_drift >= float(halo_drift_trigger)
        or edge_soft_ratio >= float(edge_soft_ratio_trigger)
        or soft_ratio >= 0.12
    )
    if not needs:
        rep["action"] = "skip_not_needed"
        rep["output"] = image_path
        return image_path, rep

    out = _harden_edges_matte_purge(image_path, expected_dpi=int(expected_dpi))
    if out == image_path and cv2 is None:
        rep["action"] = "skip_cv2_missing"
        rep["output"] = image_path
        return image_path, rep

    rep["action"] = "harden_matte_purge"
    rep["output"] = out
    return out, rep


# ---------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------

def _write_validator_debug(
    image_path: str,
    *,
    out_path: str,
    alpha_threshold: int = 20,
    safe_margin_px: int = 250,
) -> None:
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    a = arr[:, :, 3]

    mask = a >= int(alpha_threshold)
    if np.any(mask):
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
    else:
        left = right = top = bottom = 0

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    d.rectangle(
        [safe_margin_px, safe_margin_px, w - safe_margin_px, h - safe_margin_px],
        outline=(0, 255, 0, 180),
        width=10,
    )
    d.rectangle([left, top, right, bottom], outline=(255, 0, 0, 220), width=8)

    out = Image.alpha_composite(img, overlay)
    out.save(out_path, "PNG", dpi=(300, 300))


# ---------------------------------------------------------
# Alpha cleanup + edge-band junk detection
# ---------------------------------------------------------

def _tighten_alpha_edges(
    img: Image.Image,
    *,
    low: float = 22,
    high: float = 210,
    blur: float = 0.6,
    snap_opaque: Optional[float] = 235,
) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3].astype(np.float32)

    if blur and blur > 0:
        a_img = Image.fromarray(a.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(float(blur)))
        a = np.array(a_img, dtype=np.float32)

    low = float(low)
    high = float(high)
    if high <= low:
        a = np.clip(a, 0, 255)
    else:
        a = (a - low) * (255.0 / (high - low))
        a = np.clip(a, 0, 255)

    if snap_opaque is not None:
        a[a >= float(snap_opaque)] = 255.0

    arr[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _edge_band_junk_count(alpha: np.ndarray, *, alpha_threshold: int = 20, band_px: int = 6) -> int:
    a = alpha.astype(np.uint8)
    alpha_threshold = int(alpha_threshold)
    band_px = int(band_px)

    if band_px <= 0:
        return 0

    solid = (a >= alpha_threshold).astype(np.uint8) * 255
    if solid.max() == 0:
        return 0

    # Fast path (cv2)
    if cv2 is not None:
        k = band_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        dil = cv2.dilate(solid, kernel, iterations=1)
        ero = cv2.erode(solid, kernel, iterations=1)
        edge_band = (dil > 0) & (ero == 0)
        junk = np.sum(edge_band & (a > 0) & (a < alpha_threshold))
        return int(junk)

    # Fallback: pure numpy
    solid01 = (solid > 0).astype(np.uint8)
    k = band_px
    pad = k
    solid_p = np.pad(solid01, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    dil = np.zeros_like(solid01, dtype=np.uint8)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            dil = np.maximum(dil, solid_p[pad + dy:pad + dy + solid01.shape[0], pad + dx:pad + dx + solid01.shape[1]])

    ero = np.ones_like(solid01, dtype=np.uint8)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            ero = np.minimum(ero, solid_p[pad + dy:pad + dy + solid01.shape[0], pad + dx:pad + dx + solid01.shape[1]])

    edge_band = (dil == 1) & (ero == 0)
    junk = np.sum(edge_band & (a > 0) & (a < alpha_threshold))
    return int(junk)


# ---------------------------------------------------------
# Safe margin logic
# ---------------------------------------------------------

def _bbox_intersects_safe_margin(
    bbox_left: int,
    bbox_top: int,
    bbox_right: int,
    bbox_bottom: int,
    canvas_w: int,
    canvas_h: int,
    safe_margin_px: int,
) -> bool:
    safe_margin_px = int(safe_margin_px)
    if bbox_left < safe_margin_px:
        return True
    if bbox_top < safe_margin_px:
        return True
    if bbox_right > (int(canvas_w) - safe_margin_px):
        return True
    if bbox_bottom > (int(canvas_h) - safe_margin_px):
        return True
    return False


def _build_final_report(final_path: str, *, alpha_threshold: int, band_px: int) -> Dict[str, Any]:
    rep = score_edge_quality_v2(final_path, alpha_threshold=int(alpha_threshold), band_px=int(band_px), inner_px=3)
    rep["vector_mode"] = detect_vector_mode(final_path, alpha_threshold=int(alpha_threshold))
    return rep


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def validate_print_ready(
    image_path: str,
    *,
    expected_size: Tuple[int, int] = (4500, 5400),
    alpha_threshold: int = 20,
    center_tolerance: float = 0.02,
    min_margin_px: int = 250,
    max_file_mb: float = 25,
    band_px: int = 6,
    max_edge_junk: int = 25000,
    auto_fix: bool = True,
    auto_fix_passes: int = 1,
    persist_fix: bool = True,
    expected_dpi: int = 300,
    require_icc: bool = True,
) -> bool:
    if not os.path.exists(image_path):
        raise PODValidationError(f"File not found: {image_path}")

    try:
        img = Image.open(image_path)
        if img.size != expected_size:
            raise PODValidationError(f"Invalid size {img.size}. Expected {expected_size}.")

        _check_and_fix_dpi_and_icc(
            image_path,
            expected_dpi=int(expected_dpi),
            auto_fix=True,
            require_icc=bool(require_icc),
        )

        img = Image.open(image_path).convert("RGBA")

        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if size_mb > float(max_file_mb):
            raise PODValidationError(f"File too large: {size_mb:.2f}MB (limit {max_file_mb}MB)")

        rgba = np.array(img, dtype=np.uint8)
        alpha = rgba[:, :, 3]

        if alpha.max() == 255 and alpha.min() == 255:
            raise PODValidationError("Background is not transparent.")

        alpha_threshold_i = int(alpha_threshold)
        ys, xs = np.where(alpha >= alpha_threshold_i)
        if xs.size == 0:
            raise PODValidationError("Design appears empty.")

        bbox_left = int(xs.min())
        bbox_right = int(xs.max())
        bbox_top = int(ys.min())
        bbox_bottom = int(ys.max())

        design_center_x = (bbox_left + bbox_right) / 2.0
        canvas_center_x = float(expected_size[0]) / 2.0
        center_diff_ratio = abs(design_center_x - canvas_center_x) / float(expected_size[0])
        if center_diff_ratio > float(center_tolerance):
            raise PODValidationError(f"Design not centered (offset {center_diff_ratio * 100:.2f}%)")

        canvas_w, canvas_h = expected_size
        if _bbox_intersects_safe_margin(
            bbox_left=bbox_left,
            bbox_top=bbox_top,
            bbox_right=bbox_right,
            bbox_bottom=bbox_bottom,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            safe_margin_px=int(min_margin_px),
        ):
            raise PODValidationError(
                f"Design intersects safe margin ({min_margin_px}px). "
                f"bbox=({bbox_left},{bbox_top},{bbox_right},{bbox_bottom}) canvas=({canvas_w},{canvas_h})"
            )

        def edge_junk(a: np.ndarray) -> int:
            return _edge_band_junk_count(a, alpha_threshold=alpha_threshold_i, band_px=int(band_px))

        junk = edge_junk(alpha)

        if junk > int(max_edge_junk) and bool(auto_fix):
            fixed = img
            for _ in range(int(auto_fix_passes)):
                fixed = _tighten_alpha_edges(fixed, low=22, high=210, blur=0.6, snap_opaque=235)

            fixed_alpha = np.array(fixed, dtype=np.uint8)[:, :, 3]
            junk2 = edge_junk(fixed_alpha)

            if junk2 > int(max_edge_junk):
                raise PODValidationError(
                    f"Too many semi-transparent edge pixels (edge-band): {junk} "
                    f"(max {max_edge_junk}, band_px={band_px}, alpha_threshold={alpha_threshold_i})"
                )

            if bool(persist_fix):
                _save_png_with_profile(fixed, image_path, dpi=(int(expected_dpi), int(expected_dpi)))

            return True

        if junk > int(max_edge_junk):
            raise PODValidationError(f"Too many semi-transparent edge pixels (edge-band): {junk}")

        return True

    except PODValidationError:
        debug_path = image_path.replace(".png", "_validator_debug.png")
        try:
            _write_validator_debug(
                image_path,
                out_path=debug_path,
                alpha_threshold=int(alpha_threshold),
                safe_margin_px=int(min_margin_px),
            )
        except Exception:
            pass
        raise


def validate_or_fix_print_ready(
    image_path: str,
    *,
    fixed_suffix: str = "_fixed",
    expected_size: Tuple[int, int] = (4500, 5400),
    alpha_threshold: int = 20,
    center_tolerance: float = 0.02,
    min_margin_px: int = 250,
    max_file_mb: float = 25,
    band_px: int = 6,
    max_edge_junk: int = 25000,
    auto_fix_passes: int = 1,
    expected_dpi: int = 300,
    require_icc: bool = True,
) -> str:
    """
    A) Pre-harden ALWAYS (binary alpha + matte purge + optional quantize/thicken)
    B) Strict validate
    C) If edge metrics still bad, apply maybe_fix_edges() (strong fix if cv2 available)
    D) Strict validate again
    """
    hardened_path = harden_print_ready_png(
        image_path,
        out_suffix="_hardened",
        alpha_threshold=200,
        enable_quantize=True,
        quantize_colors=24,
        enable_thicken_lineart=True,
        lineart_unique_colors_max=24,
        thicken_kernel=2,
        thicken_iterations=1,
        expected_dpi=int(expected_dpi),
    )

    try:
        validate_print_ready(
            hardened_path,
            expected_size=expected_size,
            alpha_threshold=alpha_threshold,
            center_tolerance=center_tolerance,
            min_margin_px=min_margin_px,
            max_file_mb=max_file_mb,
            band_px=band_px,
            max_edge_junk=max_edge_junk,
            auto_fix=False,
            auto_fix_passes=auto_fix_passes,
            persist_fix=False,
            expected_dpi=expected_dpi,
            require_icc=require_icc,
        )
        return hardened_path
    except PODValidationError as e:
        msg = str(e).lower()
        if ("semi-transparent" not in msg) and ("edge-band" not in msg) and ("edge pixels" not in msg):
            raise

    best_path, _rep = maybe_fix_edges(
        hardened_path,
        alpha_threshold=int(alpha_threshold),
        band_px=int(band_px),
        expected_dpi=int(expected_dpi),
        min_ok_score=90.0,
        halo_drift_trigger=20.0,
        edge_soft_ratio_trigger=0.18,
    )

    root, ext = os.path.splitext(image_path)
    final_path = root + fixed_suffix + ext
    if best_path != final_path:
        img = Image.open(best_path).convert("RGBA")
        _save_png_with_profile(img, final_path, dpi=(int(expected_dpi), int(expected_dpi)))

    validate_print_ready(
        final_path,
        expected_size=expected_size,
        alpha_threshold=alpha_threshold,
        center_tolerance=center_tolerance,
        min_margin_px=min_margin_px,
        max_file_mb=max_file_mb,
        band_px=band_px,
        max_edge_junk=max_edge_junk,
        auto_fix=False,
        auto_fix_passes=auto_fix_passes,
        persist_fix=False,
        expected_dpi=expected_dpi,
        require_icc=require_icc,
    )
    return final_path


def validate_or_fix_print_ready_with_report(
    image_path: str,
    *,
    fixed_suffix: str = "_fixed",
    expected_size: Tuple[int, int] = (4500, 5400),
    alpha_threshold: int = 20,
    center_tolerance: float = 0.02,
    min_margin_px: int = 250,
    max_file_mb: float = 25,
    band_px: int = 6,
    max_edge_junk: int = 25000,
    auto_fix_passes: int = 1,
    expected_dpi: int = 300,
    require_icc: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (final_path, report)
    report includes edge score metrics + penalties + vector_mode + action + output
    """

    hardened_path = harden_print_ready_png(
        image_path,
        out_suffix="_hardened",
        alpha_threshold=200,
        enable_quantize=True,
        quantize_colors=24,
        enable_thicken_lineart=True,
        lineart_unique_colors_max=24,
        thicken_kernel=2,
        thicken_iterations=1,
        expected_dpi=int(expected_dpi),
    )

    try:
        validate_print_ready(
            hardened_path,
            expected_size=expected_size,
            alpha_threshold=alpha_threshold,
            center_tolerance=center_tolerance,
            min_margin_px=min_margin_px,
            max_file_mb=max_file_mb,
            band_px=band_px,
            max_edge_junk=max_edge_junk,
            auto_fix=False,
            auto_fix_passes=auto_fix_passes,
            persist_fix=False,
            expected_dpi=expected_dpi,
            require_icc=require_icc,
        )
        rep = _build_final_report(hardened_path, alpha_threshold=alpha_threshold, band_px=band_px)
        rep["action"] = "pre_harden_only"
        rep["output"] = hardened_path
        return hardened_path, rep

    except PODValidationError as e:
        msg = str(e).lower()
        if ("semi-transparent" not in msg) and ("edge-band" not in msg) and ("edge pixels" not in msg):
            rep = _build_final_report(hardened_path, alpha_threshold=alpha_threshold, band_px=band_px)
            rep["action"] = "fail_non_edge"
            rep["error"] = str(e)
            rep["output"] = hardened_path
            raise

    best_path, rep = maybe_fix_edges(
        hardened_path,
        alpha_threshold=int(alpha_threshold),
        band_px=int(band_px),
        expected_dpi=int(expected_dpi),
        min_ok_score=90.0,
        halo_drift_trigger=20.0,
        edge_soft_ratio_trigger=0.18,
    )

    root, ext = os.path.splitext(image_path)
    final_path = root + fixed_suffix + ext
    if best_path != final_path:
        img = Image.open(best_path).convert("RGBA")
        _save_png_with_profile(img, final_path, dpi=(int(expected_dpi), int(expected_dpi)))

    validate_print_ready(
        final_path,
        expected_size=expected_size,
        alpha_threshold=alpha_threshold,
        center_tolerance=center_tolerance,
        min_margin_px=min_margin_px,
        max_file_mb=max_file_mb,
        band_px=band_px,
        max_edge_junk=max_edge_junk,
        auto_fix=False,
        auto_fix_passes=auto_fix_passes,
        persist_fix=False,
        expected_dpi=expected_dpi,
        require_icc=require_icc,
    )

    final_rep = _build_final_report(final_path, alpha_threshold=alpha_threshold, band_px=band_px)
    final_rep["action"] = rep.get("action", "harden_matte_purge")
    final_rep["output"] = final_path
    return final_path, final_rep


__all__ = [
    "PODValidationError",
    "validate_print_ready",
    "validate_or_fix_print_ready",
    "validate_or_fix_print_ready_with_report",
    "harden_print_ready_png",
    "score_edge_quality_v2",
    "detect_vector_mode",
    "maybe_fix_edges",
]