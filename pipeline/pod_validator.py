import os
from PIL import Image, ImageFilter, ImageCms, ImageDraw
import numpy as np


class PODValidationError(Exception):
    pass


# ---------------------------------------------------------
# DPI + ICC helpers (cached)
# ---------------------------------------------------------

# Create once at import time (fast + consistent)
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def _srgb_icc_bytes() -> bytes:
    """Return cached sRGB ICC profile bytes."""
    return _SRGB_ICC_BYTES


def _save_png_with_profile(img: Image.Image, path: str, *, dpi=(300, 300)):
    """
    Always save print-ready PNGs with correct DPI + embedded sRGB ICC profile.
    """
    img = img.convert("RGBA")
    img.save(path, "PNG", dpi=dpi, icc_profile=_srgb_icc_bytes())


from PIL import ImageDraw

def _write_validator_debug(
    image_path: str,
    *,
    out_path: str,
    alpha_threshold: int = 20,
    safe_margin_px: int = 250,
):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    a = arr[:, :, 3]

    # bbox from alpha threshold
    mask = a >= int(alpha_threshold)
    if np.any(mask):
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
    else:
        left = right = top = bottom = 0

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # safe zone
    d.rectangle(
        [safe_margin_px, safe_margin_px, w - safe_margin_px, h - safe_margin_px],
        outline=(0, 255, 0, 180),
        width=10,
    )

    # design bbox
    d.rectangle([left, top, right, bottom], outline=(255, 0, 0, 220), width=8)

    # composite + save (keep dpi/icc if you want, but debug can be plain)
    out = Image.alpha_composite(img, overlay)
    out.save(out_path, "PNG", dpi=(300, 300))


def _check_and_fix_dpi_and_icc(
    image_path: str,
    *,
    expected_dpi=300,
    auto_fix=True,
    require_icc=True,
):
    """
    Ensures PNG has correct DPI and (optionally) an embedded ICC profile.
    Auto-fixes by re-saving with DPI + sRGB ICC.
    """
    img = Image.open(image_path).convert("RGBA")

    # DPI check
    dpi = img.info.get("dpi", None)
    dpi_ok = False
    if dpi is not None:
        try:
            dpi_ok = (int(dpi[0]) == int(expected_dpi)) and (int(dpi[1]) == int(expected_dpi))
        except Exception:
            dpi_ok = False

    # ICC check
    icc = img.info.get("icc_profile", None)
    icc_ok = (icc is not None) if bool(require_icc) else True

    if dpi_ok and icc_ok:
        return True

    if not bool(auto_fix):
        if not dpi_ok and not icc_ok:
            raise PODValidationError(
                f"Missing/invalid DPI and missing ICC profile (expected {expected_dpi} DPI + sRGB ICC)."
            )
        if not dpi_ok:
            raise PODValidationError(f"Invalid or missing DPI metadata (expected {expected_dpi}).")
        if not icc_ok:
            raise PODValidationError("Missing ICC profile (expected sRGB).")
        return True

    # Auto-fix: rewrite with both DPI + sRGB ICC
    _save_png_with_profile(img, image_path, dpi=(expected_dpi, expected_dpi))
    return True


# ---------------------------------------------------------
# Alpha cleanup + edge-band junk detection
# ---------------------------------------------------------

def _tighten_alpha_edges(
    img: Image.Image,
    *,
    low=22,
    high=210,
    blur=0.6,
    snap_opaque=235,
) -> Image.Image:
    """
    Auto-fix: reduce semi-transparent edge ramps while keeping edges smooth.
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3].astype(np.float32)

    if blur and blur > 0:
        a_img = Image.fromarray(a.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(float(blur)))
        a = np.array(a_img, dtype=np.float32)

    low = float(low)
    high = float(high)
    if high <= low:
        # degenerate settings; do a safe clamp
        a = np.clip(a, 0, 255)
    else:
        a = (a - low) * (255.0 / (high - low))
        a = np.clip(a, 0, 255)

    if snap_opaque is not None:
        a[a >= float(snap_opaque)] = 255.0

    arr[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _edge_band_junk_count(alpha: np.ndarray, *, alpha_threshold=20, band_px=6) -> int:
    """
    Count semi-transparent pixels (1..alpha_threshold-1) only near the alpha edge.
    This targets halos/mattes rather than interior soft transparency.

    FAST PATH: uses OpenCV if installed.
    FALLBACK: pure numpy neighborhood ops.
    """
    a = alpha.astype(np.uint8)
    alpha_threshold = int(alpha_threshold)
    band_px = int(band_px)

    if band_px <= 0:
        return 0

    solid = (a >= alpha_threshold).astype(np.uint8) * 255  # 0/255
    if solid.max() == 0:
        return 0

    # ---- Fast path: OpenCV morphology (recommended, much faster) ----
    try:
        import cv2  # optional dependency
        k = band_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        dil = cv2.dilate(solid, kernel, iterations=1)
        ero = cv2.erode(solid, kernel, iterations=1)
        edge_band = (dil > 0) & (ero == 0)
        junk = np.sum(edge_band & (a > 0) & (a < alpha_threshold))
        return int(junk)
    except Exception:
        pass

    # ---- Fallback: pure numpy (slow for big band_px) ----
    solid01 = (solid > 0).astype(np.uint8)  # 0/1

    k = band_px
    pad = k
    solid_p = np.pad(solid01, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    # dilation (max filter)
    dil = np.zeros_like(solid01, dtype=np.uint8)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            dil = np.maximum(
                dil,
                solid_p[pad + dy:pad + dy + solid01.shape[0], pad + dx:pad + dx + solid01.shape[1]],
            )

    # erosion (min filter for binary)
    ero = np.ones_like(solid01, dtype=np.uint8)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            ero = np.minimum(
                ero,
                solid_p[pad + dy:pad + dy + solid01.shape[0], pad + dx:pad + dx + solid01.shape[1]],
            )

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
    """True if bbox violates safe margins."""
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


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def validate_print_ready(
    image_path: str,
    expected_size=(4500, 5400),
    alpha_threshold=20,
    center_tolerance=0.02,   # 2%
    min_margin_px=250,
    max_file_mb=25,
    # edge-band junk knobs
    band_px=6,
    max_edge_junk=25000,
    auto_fix=True,
    auto_fix_passes=1,
    # NEW: control whether auto-fix overwrites the original
    persist_fix=True,
):
    """
    Validates a POD print-ready PNG file.
    Raises PODValidationError if validation fails.

    If auto_fix=True and the only failure is edge-band junk,
    it will attempt to tighten alpha and (optionally) persist the fix.
    """

    try:

        if not os.path.exists(image_path):
            raise PODValidationError(f"File not found: {image_path}")

        img = Image.open(image_path)

        # Size
        if img.size != expected_size:
            raise PODValidationError(f"Invalid size {img.size}. Expected {expected_size}.")

        # Mode
        if img.mode != "RGBA":
            raise PODValidationError(f"Image mode must be RGBA, got {img.mode}")

        # DPI + ICC check (may rewrite file)
        _check_and_fix_dpi_and_icc(image_path, expected_dpi=300, auto_fix=True, require_icc=True)

        # IMPORTANT: reload after rewrite so alpha/arrays match the final file
        img = Image.open(image_path).convert("RGBA")

        # File size
        size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if size_mb > float(max_file_mb):
            raise PODValidationError(f"File too large: {size_mb:.2f}MB (limit {max_file_mb}MB)")

        rgba = np.array(img, dtype=np.uint8)
        alpha = rgba[:, :, 3]

        # Transparency integrity
        if alpha.max() == 255 and alpha.min() == 255:
            raise PODValidationError("Background is not transparent.")

        # Centering + bbox use same threshold
        alpha_threshold = int(alpha_threshold)
        ys, xs = np.where(alpha >= alpha_threshold)
        if xs.size == 0:
            raise PODValidationError("Design appears empty.")

        bbox_left = int(xs.min())
        bbox_right = int(xs.max())
        bbox_top = int(ys.min())
        bbox_bottom = int(ys.max())

        # Center check
        design_center_x = (bbox_left + bbox_right) / 2.0
        canvas_center_x = float(expected_size[0]) / 2.0
        center_diff_ratio = abs(design_center_x - canvas_center_x) / float(expected_size[0])

        if center_diff_ratio > float(center_tolerance):
            raise PODValidationError(f"Design not centered (offset {center_diff_ratio*100:.2f}%)")

        # Reject if bbox touches unsafe margins
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

        # Edge-band junk check (+ optional auto-fix)
        def edge_junk(a: np.ndarray) -> int:
            return _edge_band_junk_count(a, alpha_threshold=alpha_threshold, band_px=int(band_px))

        junk = edge_junk(alpha)

        if junk > int(max_edge_junk) and bool(auto_fix):
            fixed = img
            for _ in range(int(auto_fix_passes)):
                fixed = _tighten_alpha_edges(
                    fixed,
                    low=22,
                    high=210,
                    blur=0.6,
                    snap_opaque=235,
                )

            fixed_rgba = np.array(fixed, dtype=np.uint8)
            fixed_alpha = fixed_rgba[:, :, 3]
            junk2 = edge_junk(fixed_alpha)

            if junk2 > int(max_edge_junk):
                raise PODValidationError(
                    f"Too many semi-transparent edge pixels (edge-band): {junk} "
                    f"(max {max_edge_junk}, band_px={band_px}, alpha_threshold={alpha_threshold})"
                )

            if bool(persist_fix):
                _save_png_with_profile(fixed, image_path, dpi=(300, 300))

            return True

        if junk > int(max_edge_junk):
            raise PODValidationError(f"Too many semi-transparent edge pixels (edge-band): {junk}")

        return True
    except PODValidationError as e:
        # always write debug next to the file
        debug_path = image_path.replace(".png", "_validator_debug.png")
        try:
            _write_validator_debug(
                image_path,
                out_path=debug_path,
                alpha_threshold=alpha_threshold,
                safe_margin_px=min_margin_px,
            )
        except Exception:
            pass  # don't mask original error
        raise

def _write_validator_debug(
    image_path: str,
    *,
    out_path: str,
    alpha_threshold: int = 20,
    safe_margin_px: int = 250,
):
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    a = arr[:, :, 3]

    # bbox from alpha threshold
    mask = a >= int(alpha_threshold)
    if np.any(mask):
        ys, xs = np.where(mask)
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
    else:
        left = right = top = bottom = 0

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # safe zone
    d.rectangle(
        [safe_margin_px, safe_margin_px, w - safe_margin_px, h - safe_margin_px],
        outline=(0, 255, 0, 180),
        width=10,
    )

    # design bbox
    d.rectangle([left, top, right, bottom], outline=(255, 0, 0, 220), width=8)

    # composite + save (keep dpi/icc if you want, but debug can be plain)
    out = Image.alpha_composite(img, overlay)
    out.save(out_path, "PNG", dpi=(300, 300))    


def validate_or_fix_print_ready(
    image_path: str,
    *,
    fixed_suffix: str = "_fixed",
    **validate_kwargs,
) -> str:
    """
    Adaptive fixer:
    - Validate strict.
    - If only edge-band junk fails: try mild->medium->strong alpha tightening until it passes.
    - Writes <name>_fixed.png and returns that path.
    """

    # 1) strict validation first
    try:
        validate_print_ready(image_path, auto_fix=False, **validate_kwargs)
        return image_path
    except PODValidationError as e:
        msg = str(e).lower()
        if ("semi-transparent" not in msg) and ("edge-band" not in msg) and ("edge pixels" not in msg):
            raise

    # 2) progressive fix ladder (mild -> medium -> strong)
    fix_ladder = [
        dict(low=22, high=215, blur=0.45, snap_opaque=235),  # mild (best for thin lines)
        dict(low=24, high=210, blur=0.55, snap_opaque=235),  # medium
        dict(low=26, high=205, blur=0.65, snap_opaque=235),  # strong (kills halos, may eat thin strokes)
    ]

    img = Image.open(image_path).convert("RGBA")
    root, ext = os.path.splitext(image_path)
    fixed_path = root + fixed_suffix + ext

    for params in fix_ladder:
        fixed = img
        passes = int(validate_kwargs.get("auto_fix_passes", 1))
        for _ in range(passes):
            fixed = _tighten_alpha_edges(fixed, **params)

        _save_png_with_profile(fixed, fixed_path, dpi=(300, 300))

        # validate fixed strictly
        try:
            validate_print_ready(fixed_path, auto_fix=False, **validate_kwargs)
            return fixed_path
        except PODValidationError:
            continue

    # If nothing worked, raise the last failure
    validate_print_ready(fixed_path, auto_fix=False, **validate_kwargs)
    return fixed_path
