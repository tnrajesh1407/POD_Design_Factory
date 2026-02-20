from PIL import Image, ImageFilter, ImageCms, ImageDraw
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
    safe_margin_px: int = 250,     # ✅ reject if bbox enters this margin
    reject_on_margin: bool = True, # ✅ turn off if you want warning-only behavior
    debug_overlay: bool = False,   # ✅ writes *_debug.png
):
    """
    Production-safe POD print-ready generator.

    Features:
    - alpha ramp tightening (pre + post resize)
    - bbox-based centering
    - safe resize (RGB & alpha resized separately)
    - sRGB ICC embed + 300 DPI
    - optional debug overlay
    - optional hard reject if bbox intersects safe margins
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
        # still write debug overlay if requested (helps you tune y_norm/scale)
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

    print("✅ Print-ready file created (production-safe):", output_path)
