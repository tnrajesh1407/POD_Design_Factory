from PIL import Image, ImageFilter


def _crop_to_alpha(img: Image.Image, pad_pct: float = 0.02) -> Image.Image:
    """Crop to non-transparent content (used BEFORE placing on the POD canvas)."""
    img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        return img

    cropped = img.crop(bbox)

    if pad_pct and pad_pct > 0:
        pad_x = int(cropped.width * pad_pct)
        pad_y = int(cropped.height * pad_pct)
        out = Image.new(
            "RGBA",
            (cropped.width + 2 * pad_x, cropped.height + 2 * pad_y),
            (0, 0, 0, 0),
        )
        out.paste(cropped, (pad_x, pad_y), cropped)
        return out

    return cropped


def _dehalo_alpha(img: Image.Image, alpha_cutoff: int = 18, blur: float = 0.6) -> Image.Image:
    """Removes low-alpha fringe that becomes a gray halo on shirts."""
    img = img.convert("RGBA")
    r, g, b, a = img.split()

    a = a.point(lambda p: 0 if p < alpha_cutoff else p)

    if blur and blur > 0:
        a2 = a.filter(ImageFilter.GaussianBlur(blur))
        a = a2.point(lambda p: 0 if p < alpha_cutoff else p)

    return Image.merge("RGBA", (r, g, b, a))


def place_on_pod_canvas(input_path: str, output_path: str):
    """Create a print-ready 4500x5400 transparent PNG with the design placed in a chest zone.

    NOTE:
    - We crop the source artwork to alpha BEFORE scaling.
    - We DO NOT crop AFTER placing; the canvas placement is intentional.
    """

    canvas_w, canvas_h = 4500, 5400

    # Print zone constraints
    max_w = int(canvas_w * 0.90)  # 90% width
    max_h = int(canvas_h * 0.60)  # 60% height
    min_w = int(canvas_w * 0.25)  # avoid tiny prints

    design = Image.open(input_path).convert("RGBA")

    # Clean halo + crop to visible content before scaling
    design = _dehalo_alpha(design, alpha_cutoff=18, blur=0.6)
    design = _crop_to_alpha(design, pad_pct=0.02)

    w, h = design.size
    w = max(1, w)
    h = max(1, h)

    # Scale constrained by both width and height
    scale_by_w = max_w / w
    scale_by_h = max_h / h
    scale = min(scale_by_w, scale_by_h)

    # enforce a minimum visible print width
    if w * scale < min_w:
        scale = min_w / w

    new_w = int(w * scale)
    new_h = int(h * scale)
    design = design.resize((new_w, new_h), Image.LANCZOS)

    # Create transparent POD canvas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Placement: centered horizontally, chest zone
    x = (canvas_w - new_w) // 2
    y = int(canvas_h * 0.12)

    canvas.paste(design, (x, y), design)
    canvas.save(output_path, "PNG")
    print("✅ Print-ready file created:", output_path)
