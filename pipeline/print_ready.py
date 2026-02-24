from PIL import Image, ImageFilter, ImageCms, ImageDraw, ImageEnhance, ImageChops
import numpy as np
import io


# Create once at import time
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


def get_srgb_icc_bytes() -> bytes:
    return _SRGB_ICC_BYTES


def _crop_to_alpha(img: Image.Image, pad_pct: float = 0.02) -> Image.Image:
    img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return img

    cropped = img.crop(bbox)

    if pad_pct and pad_pct > 0:
        pad_x = int(round(cropped.width * pad_pct))
        pad_y = int(round(cropped.height * pad_pct))
        out = Image.new("RGBA", (cropped.width + 2 * pad_x, cropped.height + 2 * pad_y), (0, 0, 0, 0))
        out.paste(cropped, (pad_x, pad_y), cropped)
        return out

    return cropped


def _tighten_alpha_edges(img: Image.Image, low=22, high=210, blur=0.5, snap_opaque=240) -> Image.Image:
    """
    Normalize alpha ramp to reduce semi-transparent junk without destroying soft edges.
    - below low  -> 0
    - above high -> 255
    - between    -> stretched
    - snap_opaque -> force near-opaque to 255
    """
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    a = arr[:, :, 3].astype(np.float32)

    if blur and blur > 0:
        a_img = Image.fromarray(a.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(float(blur)))
        a = np.array(a_img, dtype=np.float32)

    a = (a - float(low)) * (255.0 / max(1.0, float(high - low)))
    a = np.clip(a, 0.0, 255.0)

    if snap_opaque is not None:
        a[a >= float(snap_opaque)] = 255.0

    arr[:, :, 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _alpha_bbox(img: Image.Image, threshold: int = 20):
    arr = np.array(img.convert("RGBA"), dtype=np.uint8)
    alpha = arr[:, :, 3]
    mask = alpha >= int(threshold)
    if not np.any(mask):
        return None

    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _bbox_intersects_safe_margin(bbox, *, canvas_w: int, canvas_h: int, margin_px: int) -> bool:
    """
    True if bbox enters unsafe border region.
    bbox = (left, top, right, bottom) with right/bottom exclusive.
    """
    if bbox is None:
        return False
    left, top, right, bottom = bbox
    if left < margin_px:
        return True
    if top < margin_px:
        return True
    if right > (canvas_w - margin_px):
        return True
    if bottom > (canvas_h - margin_px):
        return True
    return False


def _create_bleed_debug_overlay(
    canvas: Image.Image,
    *,
    design_bbox=None,
    margin_px: int = 250,
):
    """
    Returns a debug image with safety guides overlayed.
    - Blue: full canvas border
    - Green: safe zone border
    - Red: design bbox
    """
    debug = canvas.copy().convert("RGBA")
    w, h = debug.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Safe zone (green)
    draw.rectangle(
        [margin_px, margin_px, w - margin_px, h - margin_px],
        outline=(0, 255, 0, 180),
        width=8,
    )

    # Full canvas border (blue)
    draw.rectangle(
        [0, 0, w - 1, h - 1],
        outline=(0, 150, 255, 150),
        width=6,
    )

    # Design bbox (red)
    if design_bbox is not None:
        left, top, right, bottom = design_bbox
        draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 200), width=6)

    return Image.alpha_composite(debug, overlay)


# ----------------------------
# NEW: Light-garment helpers
# ----------------------------
def _boost_color_for_light_garment(img: Image.Image, sat: float = 1.12, contrast: float = 1.08) -> Image.Image:
    """
    Small, safe boosts that help pastels pop on white garments.
    Operates only on RGB while preserving alpha.
    """
    img = img.convert("RGBA")
    a = img.split()[-1]
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(float(sat))
    rgb = ImageEnhance.Contrast(rgb).enhance(float(contrast))
    out = Image.merge("RGBA", (*rgb.split(), a))
    return out


def _apply_outer_keyline(img: Image.Image, outline_px: int = 12, outline_rgba=(42, 42, 42, 255)) -> Image.Image:
    """
    Create an outer stroke around non-transparent pixels (alpha-based).
    This is the main fix for 'blending into white shirts'.
    """
    if outline_px <= 0:
        return img.convert("RGBA")

    img = img.convert("RGBA")
    r, g, b, a = img.split()

    # MaxFilter size must be odd; k ~ stroke diameter
    k = max(3, int(outline_px) * 2 + 1)
    a_expanded = a.filter(ImageFilter.MaxFilter(k))

    # stroke alpha = expanded - original (outer ring only)
    stroke_alpha = ImageChops.subtract(a_expanded, a)

    stroke = Image.new("RGBA", img.size, outline_rgba)
    stroke.putalpha(stroke_alpha)

    # stroke behind original
    return Image.alpha_composite(stroke, img)


def place_on_pod_canvas(
    input_path: str,
    output_path: str,
    *,
    canvas_w: int = 4500,
    canvas_h: int = 5400,
    y_norm: float = 0.12,
    max_w_norm: float = 0.90,
    max_h_norm: float = 0.60,
    min_w_norm: float = 0.25,
    alpha_center_thresh: int = 20,
    safe_margin_px: int = 250,      # ✅ reject if bbox enters this margin
    reject_on_margin: bool = True,  # ✅ turn off if you want warning-only behavior
    debug_overlay: bool = False,    # ✅ writes *_debug.png

    # ----------------------------
    # NEW: Profile controls
    # ----------------------------
    profile: str = "dark",          # "dark" or "light"
    light_sat: float = 1.12,
    light_contrast: float = 1.08,
    light_outline_px: int = 12,
    light_outline_rgba=(42, 42, 42, 255),
):
    """
    Production-safe POD print-ready generator.

    NEW:
    - profile="light" produces a white-shirt friendly variant
      (outer keyline + small color boosts)
    """

    max_w = int(canvas_w * max_w_norm)
    max_h = int(canvas_h * max_h_norm)
    min_w = int(canvas_w * min_w_norm)

    # Load
    design = Image.open(input_path).convert("RGBA")

    # Tighten alpha BEFORE crop (reduces junk that biases bbox)
    design = _tighten_alpha_edges(design, low=20, high=220, blur=0.5, snap_opaque=240)

    # Crop to visible content
    design = _crop_to_alpha(design, pad_pct=0.02)

    w, h = max(1, design.width), max(1, design.height)

    # Scale to print zone
    scale = min(max_w / w, max_h / h)
    if w * scale < min_w:
        scale = min_w / w

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Safe resize: resize RGB and alpha separately (prevents fringes)
    r, g, b, a = design.split()
    rgb = Image.merge("RGB", (r, g, b)).resize((new_w, new_h), Image.LANCZOS)
    a = a.resize((new_w, new_h), Image.LANCZOS)
    design = Image.merge("RGBA", (*rgb.split(), a))

    # Tighten alpha AFTER resize (reduces edge-band semi-transparent ramps)
    design = _tighten_alpha_edges(design, low=22, high=210, blur=0.4, snap_opaque=240)

    # ----------------------------
    # NEW: Apply light-garment profile at final scale
    # ----------------------------
    if (profile or "").lower() == "light":
        design = _boost_color_for_light_garment(design, sat=light_sat, contrast=light_contrast)
        design = _apply_outer_keyline(design, outline_px=int(light_outline_px), outline_rgba=light_outline_rgba)

    # Compute bbox for centering (ignores faint junk)
    bbox = _alpha_bbox(design, threshold=alpha_center_thresh)

    # Compute X using bbox center
    if bbox is None:
        x = (canvas_w - new_w) // 2
    else:
        left, top, right, bottom = bbox
        bbox_center_x = (left + right) / 2.0
        x = int(round((canvas_w / 2.0) - bbox_center_x))

    y = int(round(canvas_h * y_norm))

    # Clamp placement
    x = max(0, min(canvas_w - new_w, x))
    y = max(0, min(canvas_h - new_h, y))

    # Paste on canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(design, (x, y), design)

    # Compute bbox on final canvas coordinates
    final_bbox = _alpha_bbox(canvas, threshold=alpha_center_thresh)

    # ✅ Reject if bbox intersects safe margin
    if reject_on_margin and _bbox_intersects_safe_margin(
        final_bbox, canvas_w=canvas_w, canvas_h=canvas_h, margin_px=safe_margin_px
    ):
        if debug_overlay:
            dbg = _create_bleed_debug_overlay(canvas, design_bbox=final_bbox, margin_px=safe_margin_px)
            debug_path = output_path.replace(".png", "_debug.png")
            dbg.save(debug_path, "PNG", dpi=(300, 300), icc_profile=_SRGB_ICC_BYTES)
            print("🟢 Debug safety guide created:", debug_path)

        raise ValueError(
            f"Design intersects safe margin ({safe_margin_px}px). "
            f"bbox={final_bbox} canvas=({canvas_w},{canvas_h})"
        )

    # Save main output with sRGB ICC + 300 DPI
    canvas.save(output_path, "PNG", dpi=(300, 300), icc_profile=_SRGB_ICC_BYTES)

    # Optional debug overlay output
    if debug_overlay:
        dbg = _create_bleed_debug_overlay(canvas, design_bbox=final_bbox, margin_px=safe_margin_px)
        debug_path = output_path.replace(".png", "_debug.png")
        dbg.save(debug_path, "PNG", dpi=(300, 300), icc_profile=_SRGB_ICC_BYTES)
        print("🟢 Debug safety guide created:", debug_path)

    print(f"✅ Print-ready file created ({profile}):", output_path)


# ----------------------------
# NEW: Export both variants
# ----------------------------
def export_pod_variants(
    input_path: str,
    out_dark_path: str,
    out_light_path: str,
    **kwargs,
):
    place_on_pod_canvas(input_path, out_dark_path, profile="dark", **kwargs)
    place_on_pod_canvas(input_path, out_light_path, profile="light", **kwargs)