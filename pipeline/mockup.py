# finalmockup.py
import os
import json
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Expected project layout:
#   <project>/pipeline/mockup.py   (this file)
#   <project>/assets/tshirt_mockup.jpg (optional legacy default)
#   <project>/assets/tshirt_white.jpg
#   <project>/assets/tshirt_black.jpg
#   <project>/assets/tshirt_lifestyle.jpg
#   <project>/assets/mockup_config.json
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "assets", "mockup_config.json")


# -------------------------------
# Config helpers
# -------------------------------

def _load_cfg(mockup_filename: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get(mockup_filename)


def _estimate_shirt_center_x(
    mockup_bgr: np.ndarray,
    y1_norm: float = 0.25,
    y2_norm: float = 0.80,
    bg_patch: int = 40,
    diff_thresh: float = 18.0,
) -> Optional[float]:
    """
    Estimate the shirt's visual centerline (x) by segmenting foreground (shirt/person)
    from background using corner background sampling.

    Returns x in pixel coordinates, or None if it can't be estimated robustly.
    """
    H, W = mockup_bgr.shape[:2]

    # Clamp band
    y1 = int(np.clip(y1_norm, 0.0, 1.0) * H)
    y2 = int(np.clip(y2_norm, 0.0, 1.0) * H)
    if y2 <= y1 + 10:
        return None

    # Sample background color from corners (median over small patches)
    p = int(max(10, bg_patch))
    corners = [
        mockup_bgr[0:p, 0:p],
        mockup_bgr[0:p, W - p:W],
        mockup_bgr[H - p:H, 0:p],
        mockup_bgr[H - p:H, W - p:W],
    ]
    bg = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    bg_med = np.median(bg, axis=0).astype(np.float32)  # BGR median

    band = mockup_bgr[y1:y2, :, :].astype(np.float32)

    # Foreground mask by color distance from background
    diff = np.linalg.norm(band - bg_med[None, None, :], axis=2)
    mask = (diff > float(diff_thresh)).astype(np.uint8) * 255

    # Clean mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    xs = np.where(mask > 0)[1]
    if xs.size < W * (y2 - y1) * 0.02:  # not enough fg pixels
        return None

    # Use median for robustness
    return float(np.median(xs))


def _quad_center_x(quad: List[List[int]]) -> float:
    return float(sum(p[0] for p in quad) / 4.0)


def _apply_quad_autocenter(
    mockup_bgr: np.ndarray,
    quad: List[List[int]],
    W: int,
    H: int,
    band_norm: Tuple[float, float] = (0.25, 0.80),
    strength: float = 1.0,
    max_shift_norm: float = 0.03,
    diff_thresh: float = 18.0,
) -> List[List[int]]:
    """
    Shift quad in X so its center matches the shirt's detected visual centerline.
    """
    cx = _estimate_shirt_center_x(
        mockup_bgr,
        y1_norm=band_norm[0],
        y2_norm=band_norm[1],
        diff_thresh=diff_thresh,
    )
    if cx is None:
        return quad

    qcx = _quad_center_x(quad)
    dx = (cx - qcx) * float(strength)

    # Clamp shift so we don't over-correct
    max_dx = float(max_shift_norm) * W
    dx = float(np.clip(dx, -max_dx, max_dx))

    return _apply_quad_offset(quad, W, H, dx_px=int(round(dx)), dy_px=0)


def _quad_from_norm(quad_norm: List[List[float]], W: int, H: int) -> List[List[int]]:
    return [[int(round(x * W)), int(round(y * H))] for x, y in quad_norm]


def _offset_px_from_cfg(mode_cfg: Dict[str, Any], W: int, H: int) -> Tuple[int, int]:
    """Support either offset_norm ([dx,dy] in 0..1) or offset_px."""
    if not isinstance(mode_cfg, dict):
        return 0, 0

    off_norm = mode_cfg.get("offset_norm")
    if isinstance(off_norm, (list, tuple)) and len(off_norm) == 2:
        try:
            dx = int(round(float(off_norm[0]) * W))
            dy = int(round(float(off_norm[1]) * H))
            return dx, dy
        except Exception:
            pass

    off_px = mode_cfg.get("offset_px")
    if isinstance(off_px, (list, tuple)) and len(off_px) == 2:
        try:
            dx = int(round(float(off_px[0])))
            dy = int(round(float(off_px[1])))
            return dx, dy
        except Exception:
            pass

    return 0, 0


def _apply_quad_offset(quad_xy: List[List[int]], W: int, H: int, dx_px: int = 0, dy_px: int = 0) -> List[List[int]]:
    if not quad_xy:
        return quad_xy
    out = []
    for x, y in quad_xy:
        xx = int(round(x + dx_px))
        yy = int(round(y + dy_px))
        xx = max(0, min(W - 1, xx))
        yy = max(0, min(H - 1, yy))
        out.append([xx, yy])
    return out


def _quad_ok(quad: List[List[int]], W: int, H: int) -> bool:
    if not quad or len(quad) != 4:
        return False
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    if min(xs) < 0 or min(ys) < 0 or max(xs) >= W or max(ys) >= H:
        return False
    # polygon area (shoelace)
    area = 0.0
    for i in range(4):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    return area > (W * H * 0.005)  # >=0.5% of mockup area


def _strip_soft_edge_glow(
    img: Image.Image,
    enabled: bool = True,
    band_px: int = 10,
    feather_px: int = 3,
    min_alpha: int = 40,
    white_min: int = 235,
    sat_max: int = 45,
    dark_max: int = 40,
    mode: str = "white",  # "white", "dark", "both"
) -> Image.Image:
    """
    Strip soft halos near the alpha edge.

    Typical POD issue: near-white, low-saturation halo from background removal/upscaling.
    This function can strip:
      - white halos (default): gray >= white_min and saturation <= sat_max
      - dark shadows: gray <= dark_max
      - both
    """
    if not enabled:
        return img

    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    a = rgba[:, :, 3]

    if a.max() == 0:
        return img

    mask = (a >= int(min_alpha)).astype(np.uint8)

    # Distance from edge
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    band = (dist > 0) & (dist <= float(band_px))

    # RGB -> HSV for saturation checks
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    white_halo = band & (gray >= int(white_min)) & (sat <= int(sat_max))
    dark_halo = band & (gray <= int(dark_max))

    mode = (mode or "white").lower().strip()
    if mode == "dark":
        glow = dark_halo
    elif mode == "both":
        glow = white_halo | dark_halo
    else:
        glow = white_halo  # default: white halo

    if not np.any(glow):
        return img

    if feather_px > 0:
        d = dist.astype(np.float32)
        fade = np.clip((float(band_px) - d) / max(1.0, float(feather_px)), 0.0, 1.0)
        new_a = a.astype(np.float32)
        new_a[glow] = new_a[glow] * (1.0 - fade[glow])
        rgba[:, :, 3] = np.clip(new_a, 0, 255).astype(np.uint8)
    else:
        rgba[:, :, 3][glow] = 0

    return Image.fromarray(rgba, mode="RGBA")


def _shirt_is_dark(mockup_bgr: np.ndarray, threshold: float = 110.0) -> bool:
    gray = cv2.cvtColor(mockup_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < float(threshold)


# -------------------------------
# Image helpers
# -------------------------------

def _crop_to_alpha(img: Image.Image, pad_pct: float = 0.02) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Crop to alpha bbox; return (cropped, bbox). bbox is in original image coords."""
    img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return img, (0, 0, img.width, img.height)

    cropped = img.crop(bbox)

    if pad_pct and pad_pct > 0:
        pad_x = int(round(cropped.width * pad_pct))
        pad_y = int(round(cropped.height * pad_pct))
        out = Image.new("RGBA", (cropped.width + 2 * pad_x, cropped.height + 2 * pad_y), (0, 0, 0, 0))
        out.paste(cropped, (pad_x, pad_y), cropped)
        return out, bbox

    return cropped, bbox


def _clean_alpha_strong(
    img: Image.Image,
    alpha_cutoff: int = 32,
    blur: float = 0.8,
    shrink_px: int = 1,
) -> Image.Image:
    """
    Stronger alpha cleanup to remove asymmetric semi-transparent halos.
    """
    img = img.convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    a = rgba[:, :, 3]

    # Hard cutoff
    a = np.where(a < int(alpha_cutoff), 0, a)

    # Optional shrink to remove fringe bias
    if int(shrink_px) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        a = cv2.erode(a, kernel, iterations=int(shrink_px))

    # Light blur to smooth edge
    if float(blur) > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(blur))

    rgba[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _add_soft_shadow(
    design: Image.Image,
    blur: int = 10,
    opacity: int = 55,
    offset: Tuple[int, int] = (3, 4),
) -> Image.Image:
    design = design.convert("RGBA")
    alpha = design.split()[-1]

    shadow = Image.new("RGBA", design.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(int(blur)))

    r, g, b, a = shadow.split()
    a = a.point(lambda p: int(p * (int(opacity) / 255.0)))
    shadow = Image.merge("RGBA", (r, g, b, a))

    dx, dy = int(offset[0]), int(offset[1])
    pad_x = abs(dx)
    pad_y = abs(dy)

    out = Image.new("RGBA", (design.width + 2 * pad_x, design.height + 2 * pad_y), (0, 0, 0, 0))

    # Keep design centered
    design_pos = (pad_x, pad_y)
    shadow_pos = (pad_x + dx, pad_y + dy)

    out.paste(shadow, shadow_pos, shadow)
    out.paste(design, design_pos, design)
    return out


def _pil_rgba_to_cv_bgra(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGBA"))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def _warp_to_quad(mockup_bgr: np.ndarray, design_bgra: np.ndarray, quad_xy: List[List[int]]) -> np.ndarray:
    """Return warped BGRA overlay, same size as mockup."""
    H, W = mockup_bgr.shape[:2]
    dh, dw = design_bgra.shape[:2]

    src = np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]])
    dst = np.float32(quad_xy)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        design_bgra,
        M,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def _alpha_comp(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    base = base_bgr.astype(np.float32)
    over = overlay_bgra.astype(np.float32)

    alpha = over[:, :, 3:4] / 255.0
    rgb = over[:, :, :3]

    out = rgb * alpha + base * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _is_logo_like_from_bbox(bbox: Tuple[int, int, int, int], canvas_w: int, canvas_h: int) -> bool:
    """Heuristic on the *visible* alpha bbox inside a print-ready canvas."""
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # if it's small relative to the canvas, treat as logo
    return (bw < canvas_w * 0.30) and (bh < canvas_h * 0.30)


def _debug_paths(output_path: str) -> Tuple[str, str, str]:
    root, ext = os.path.splitext(output_path)
    ext = ext if ext else ".jpg"
    return root + "_area_debug" + ext, root + "_quad_outline" + ext, root + "_warp_debug" + ext


# -------------------------------
# Fabric blend (wrinkle/shading integration)
# -------------------------------

def _fabric_blend(
    base_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
    strength: float = 0.65,
    detail_strength: float = 0.35,
    shading_blur: float = 22.0,
    ink_bleed: float = 0.06,
    desat: float = 0.08,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Make the design look printed on fabric by modulating it with shirt shading/wrinkles.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    detail_strength = float(np.clip(detail_strength, 0.0, 1.0))
    ink_bleed = float(np.clip(ink_bleed, 0.0, 0.25))
    desat = float(np.clip(desat, 0.0, 0.25))
    shading_blur = float(max(0.0, shading_blur))
    gamma = float(max(0.1, gamma))

    base = base_bgr.astype(np.float32) / 255.0
    over = overlay_bgra.astype(np.float32) / 255.0

    alpha = over[:, :, 3:4]  # 0..1
    if float(alpha.max()) <= 0.0 or strength <= 0.0:
        return overlay_bgra

    over_rgb = over[:, :, :3]

    # broad shading (low frequency)
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shading = cv2.GaussianBlur(gray, (0, 0), shading_blur) if shading_blur > 0 else gray

    # wrinkle/detail (high frequency)
    eps = 1e-6
    detail = gray / (shading + eps)
    detail = np.clip(detail, 0.75, 1.25)

    detail_n = (detail - 0.75) / (1.25 - 0.75)
    detail_n = np.clip(detail_n, 0.0, 1.0)

    mod = (1.0 - detail_strength) * shading + detail_strength * detail_n

    # remap to pleasant range
    mod = 0.78 + 0.30 * mod
    if gamma != 1.0:
        mod = np.power(np.clip(mod, 0.0, 2.0), gamma)

    mod3 = mod[:, :, None]

    over_rgb_mod = over_rgb * ((1.0 - strength) + strength * mod3)

    if desat > 0:
        lum = (0.114 * over_rgb_mod[:, :, 0] + 0.587 * over_rgb_mod[:, :, 1] + 0.299 * over_rgb_mod[:, :, 2])[:, :, None]
        over_rgb_mod = over_rgb_mod * (1.0 - desat) + lum * desat

    if ink_bleed > 0:
        over_rgb_mod = over_rgb_mod * (1.0 - ink_bleed) + base * ink_bleed

    out = over.copy()
    out[:, :, :3] = np.clip(over_rgb_mod, 0.0, 1.0)
    out[:, :, 3:4] = alpha
    return (out * 255.0).astype(np.uint8)


# -------------------------------
# Public API
# -------------------------------

def create_mockup(
    print_ready_path: str,
    output_path: str,
    mode: Optional[str] = None,
    mockup_path: Optional[str] = None,
    write_debug: bool = True,
):
    """
    Create a warped t-shirt mockup.

    - Uses assets/<mockup_path> (defaults to assets/tshirt_mockup.jpg)
    - Uses assets/mockup_config.json keyed by the mockup filename
    - Supports modes.*.quad_norm (and optional offsets, fabric presets, shadow, alpha cleanup)

    Typical usage for multiple mockups:
      create_mockup(pr, out, mode="front", mockup_path="assets/tshirt_white.jpg")
      create_mockup(pr, out, mode="front", mockup_path="assets/tshirt_black.jpg")
      create_mockup(pr, out, mode="front", mockup_path="assets/tshirt_lifestyle.jpg")
    """
    if mockup_path is None:
        # legacy default if user doesn't pass a path
        mockup_path = os.path.join(BASE_DIR, "assets", "tshirt_mockup.png")
    else:
        # allow passing relative path like "assets/tshirt_white.jpg"
        if os.path.isabs(mockup_path) and not os.path.exists(mockup_path):
            candidate = os.path.join(BASE_DIR, "assets", os.path.basename(mockup_path))
            if os.path.exists(candidate):
                mockup_path = candidate


    mockup_filename = os.path.basename(mockup_path)

    cfg = _load_cfg(mockup_filename)
    # fallback to legacy entry if a new filename doesn't exist yet
    if not cfg and mockup_filename != "tshirt_mockup.png":
        cfg = _load_cfg("tshirt_mockup.png")

    if not cfg:
        raise RuntimeError(
            f"Missing config for '{mockup_filename}'. Add an entry in {CONFIG_PATH} keyed by the mockup filename "
            f"(e.g., '{mockup_filename}') with modes.*.quad_norm."
        )

    mockup_bgr = cv2.imread(mockup_path, cv2.IMREAD_COLOR)
    if mockup_bgr is None:
        raise FileNotFoundError(mockup_path)

    H, W = mockup_bgr.shape[:2]

    # Load print-ready input
    pr = Image.open(print_ready_path).convert("RGBA")

    # --- outline strip (config-driven) ---
    outline_cfg = cfg.get("outline_strip", {}) if isinstance(cfg.get("outline_strip"), dict) else {}
    pr = _strip_soft_edge_glow(
        pr,
        enabled=bool(outline_cfg.get("enabled", True)),
        band_px=int(outline_cfg.get("band_px", 10)),
        feather_px=int(outline_cfg.get("feather_px", 3)),
        min_alpha=int(outline_cfg.get("min_alpha", 40)),
        white_min=int(outline_cfg.get("white_min", 235)),
        sat_max=int(outline_cfg.get("sat_max", 45)),
        dark_max=int(outline_cfg.get("dark_max", 40)),
        mode=str(outline_cfg.get("mode", "white")).lower(),
    )

    # --- alpha cleanup (config-driven) ---
    ac = cfg.get("alpha_cleanup", {}) if isinstance(cfg.get("alpha_cleanup"), dict) else {}
    pr = _clean_alpha_strong(
        pr,
        alpha_cutoff=int(ac.get("alpha_cutoff", 32)),
        blur=float(ac.get("blur", 0.8)),
        shrink_px=int(ac.get("shrink_px", 1)),
    )

    # bbox only (used to infer mode)
    _, bbox = _crop_to_alpha(pr, pad_pct=0.0)

    modes = cfg.get("modes", {}) if isinstance(cfg.get("modes", {}), dict) else {}

    if mode is None:
        mode = "left_chest" if _is_logo_like_from_bbox(bbox, pr.width, pr.height) else "front"

    if mode not in modes:
        raise ValueError(f"Unknown mode '{mode}'. Available: {sorted(modes.keys())}")

    mode_cfg = modes.get(mode, {}) if isinstance(modes, dict) else {}

    # Build quad
    if isinstance(mode_cfg, dict) and "quad_norm" in mode_cfg:
        quad = _quad_from_norm(mode_cfg["quad_norm"], W, H)
    else:
        # backward compat
        quad = cfg.get("quad")

    # Fine-tune: optional quad offset
    dx_px, dy_px = _offset_px_from_cfg(mode_cfg, W, H)
    if dx_px or dy_px:
        quad = _apply_quad_offset(quad, W, H, dx_px=dx_px, dy_px=dy_px)

    # Optional: auto-center quad to shirt's visual centerline
    auto_center = False
    if isinstance(cfg.get("auto_center"), bool):
        auto_center = cfg.get("auto_center")
    if isinstance(mode_cfg, dict) and isinstance(mode_cfg.get("auto_center"), bool):
        auto_center = mode_cfg.get("auto_center")

    if auto_center:
        band = (0.25, 0.80)
        if isinstance(cfg.get("auto_center_band_norm"), (list, tuple)) and len(cfg["auto_center_band_norm"]) == 2:
            band = (float(cfg["auto_center_band_norm"][0]), float(cfg["auto_center_band_norm"][1]))
        if isinstance(mode_cfg.get("auto_center_band_norm"), (list, tuple)) and len(mode_cfg["auto_center_band_norm"]) == 2:
            band = (float(mode_cfg["auto_center_band_norm"][0]), float(mode_cfg["auto_center_band_norm"][1]))

        strength = float(mode_cfg.get("auto_center_strength", cfg.get("auto_center_strength", 1.0)))
        max_shift_norm = float(mode_cfg.get("auto_center_max_shift_norm", cfg.get("auto_center_max_shift_norm", 0.03)))
        diff_thresh = float(mode_cfg.get("auto_center_diff_thresh", cfg.get("auto_center_diff_thresh", 18.0)))

        quad = _apply_quad_autocenter(
            mockup_bgr=mockup_bgr,
            quad=quad,
            W=W,
            H=H,
            band_norm=band,
            strength=strength,
            max_shift_norm=max_shift_norm,
            diff_thresh=diff_thresh,
        )

    if not _quad_ok(quad, W, H):
        raise RuntimeError(f"Invalid quad for mockup '{mockup_filename}' mode '{mode}'. Check mockup_config.json.")

    # Extract design for warping
    preserve_canvas = bool(mode_cfg.get("preserve_canvas", False))
    if preserve_canvas:
        design = pr.copy()
    else:
        design, _ = _crop_to_alpha(pr, pad_pct=0.02)

    # Optional: add shadow (config-driven)
    shadow_cfg = cfg.get("shadow", {}) if isinstance(cfg.get("shadow"), dict) else {}
    if isinstance(mode_cfg, dict) and isinstance(mode_cfg.get("shadow"), dict):
        shadow_cfg = {**shadow_cfg, **mode_cfg.get("shadow")}

    if shadow_cfg:
        opacity = int(shadow_cfg.get("opacity", 0))
        if opacity > 0:
            off = shadow_cfg.get("offset", [3, 4])
            design = _add_soft_shadow(
                design,
                blur=int(shadow_cfg.get("blur", 10)),
                opacity=opacity,
                offset=(int(off[0]), int(off[1])) if isinstance(off, (list, tuple)) and len(off) == 2 else (3, 4),
            )

    # -------------------------------
    # Fabric blend settings:
    # - prefer cfg.fabric_presets (light/dark) chosen from mockup luminance
    # - fallback to cfg.fabric
    # - mode_cfg.fabric overrides always win
    # -------------------------------
    fabric_cfg: Dict[str, Any] = {}

    presets = cfg.get("fabric_presets") if isinstance(cfg.get("fabric_presets"), dict) else None
    detect_cfg = cfg.get("shirt_detect") if isinstance(cfg.get("shirt_detect"), dict) else {}
    detect_enabled = bool(detect_cfg.get("enabled", True))
    threshold = float(detect_cfg.get("threshold", 110))

    if presets and detect_enabled:
        preset_name = "dark" if _shirt_is_dark(mockup_bgr, threshold=threshold) else "light"
        if isinstance(presets.get(preset_name), dict):
            fabric_cfg.update(presets[preset_name])

    if not fabric_cfg and isinstance(cfg.get("fabric"), dict):
        fabric_cfg.update(cfg.get("fabric"))

    if isinstance(mode_cfg, dict) and isinstance(mode_cfg.get("fabric"), dict):
        fabric_cfg.update(mode_cfg.get("fabric"))

    # Warp and composite
    design_bgra = _pil_rgba_to_cv_bgra(design)
    warped = _warp_to_quad(mockup_bgr, design_bgra, quad)

    if fabric_cfg:
        warped = _fabric_blend(
            mockup_bgr,
            warped,
            strength=float(fabric_cfg.get("strength", 0.65)),
            detail_strength=float(fabric_cfg.get("detail_strength", 0.35)),
            shading_blur=float(fabric_cfg.get("shading_blur", 22.0)),
            ink_bleed=float(fabric_cfg.get("ink_bleed", 0.06)),
            desat=float(fabric_cfg.get("desat", 0.08)),
            gamma=float(fabric_cfg.get("gamma", 1.0)),
        )

    out = _alpha_comp(mockup_bgr, warped)

    # Ensure output directory exists and write robustly
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Failed to write mockup to {output_path}")

    # Debug outputs
    if write_debug:
        area_path, quad_path, warp_path = _debug_paths(output_path)

        overlay = mockup_bgr.copy()
        pts = np.array(quad, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        blend = cv2.addWeighted(mockup_bgr, 0.82, overlay, 0.18, 0)
        cv2.imwrite(area_path, blend)

        outline = mockup_bgr.copy()
        cv2.polylines(outline, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        for (x, y) in quad:
            cv2.circle(outline, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.imwrite(quad_path, outline)

        warp_dbg = mockup_bgr.copy()
        alpha = warped[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (0, 0), 1.2)
        colored = np.zeros_like(warp_dbg)
        colored[:, :, 1] = mask
        warp_dbg = cv2.addWeighted(warp_dbg, 0.9, colored, 0.35, 0)
        cv2.imwrite(warp_path, warp_dbg)


if __name__ == "__main__":
    # Minimal CLI usage example:
    #   python finalmockup.py /path/to/print_ready.png /path/to/output_mockup.jpg [mode] [mockup_filename]
    # Example:
    #   python finalmockup.py ./outputs/04_print_ready.png ./outputs/05_mockup_white.jpg front tshirt_white.jpg
    import sys

    if len(sys.argv) < 3:
        print("Usage: python finalmockup.py <print_ready.png> <output_mockup.jpg> [mode] [mockup_filename]")
        raise SystemExit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    mode_arg = sys.argv[3] if len(sys.argv) > 3 else None
    mockup_file = sys.argv[4] if len(sys.argv) > 4 else "tshirt_mockup.png"

    create_mockup(
        in_path,
        out_path,
        mode=mode_arg,
        mockup_path=os.path.join("assets", mockup_file),
        write_debug=True,
    )
    print("✅ Mockup created:", out_path)
