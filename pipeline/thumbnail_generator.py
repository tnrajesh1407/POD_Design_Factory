import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from PIL import ImageEnhance
from PIL import ImageChops


THUMB_W = 1280
THUMB_H = 769
SAFE_MARGIN = 80


# ----------------------------------------------------------
# FONT LOADER (Safe + Bold Priority)
# ----------------------------------------------------------

def _font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except:
        try:
            return ImageFont.truetype("Arial Bold.ttf", size)
        except:
            return ImageFont.load_default()


# ----------------------------------------------------------
# TEXT SIZE (Pillow 10+ Safe)
# ----------------------------------------------------------

def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ----------------------------------------------------------
# NICHE DETECTION
# ----------------------------------------------------------

def detect_niche(prompt: str):
    p = prompt.lower()
    if "pickleball" in p:
        return "pickleball"
    if "dog" in p:
        return "dog"
    return "generic"


# ----------------------------------------------------------
# HEADLINE PACK
# ----------------------------------------------------------

def headline_pack(niche):

    if niche == "pickleball":
        return (
            "PICKLEBALL BRAND",
            "PRO T-SHIRT DESIGNS",
            "Designed for Serious Players"
        )

    if niche == "dog":
        return (
            "DOG MERCH EXPERT",
            "VIRAL T-SHIRT DESIGNS",
            "Pet Brands Grow Faster"
        )

    return (
        "PRO MERCH DESIGNER",
        "HIGH-CONVERTING T-SHIRTS",
        "Built for Print-on-Demand"
    )


# ----------------------------------------------------------
# SMOOTH VERTICAL GRADIENT
# ----------------------------------------------------------

def gradient_bg(color1, color2):
    base = Image.new("RGB", (THUMB_W, THUMB_H), color1)
    top = Image.new("RGB", (THUMB_W, THUMB_H), color2)

    mask = Image.new("L", (1, THUMB_H))
    for y in range(THUMB_H):
        mask.putpixel((0, y), int(255 * (y / THUMB_H)))

    mask = mask.resize((THUMB_W, THUMB_H))
    base.paste(top, (0, 0), mask)
    return base


# ----------------------------------------------------------
# STROKE TEXT (Clean Outline)
# ----------------------------------------------------------

def stroke_text(draw, pos, text, font, fill, stroke_fill, stroke_w):
    x, y = pos
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke_w,
        stroke_fill=stroke_fill,
    )


# ----------------------------------------------------------
# BADGE
# ----------------------------------------------------------

def badge(canvas, text, x, y, bg_color):
    draw = ImageDraw.Draw(canvas)
    font = _font(44)
    pad = 20

    w, h = _text_size(draw, text, font)

    box = [x, y, x + w + pad * 2, y + h + pad * 2]

    draw.rounded_rectangle(box, radius=28, fill=bg_color)
    draw.text((x + pad, y + pad), text, fill="white", font=font)


# ----------------------------------------------------------
# SOCIAL PROOF
# ----------------------------------------------------------

def social_proof(canvas):
    draw = ImageDraw.Draw(canvas)
    font = _font(44)

    text = "★★★★★  300+ ORDERS  |  TOP RATED QUALITY"

    w, h = _text_size(draw, text, font)

    draw.text(
        (SAFE_MARGIN, THUMB_H - h - 40),
        text,
        fill=(255, 255, 0),
        font=font
    )



def trim_uniform_border(im: Image.Image, tol: int = 12) -> Image.Image:
    """
    Crops uniform-looking border from an image.
    Works for RGBA (uses alpha) and RGB (uses corner as background).
    """
    if im.mode == "RGBA":
        # If alpha exists, crop by non-transparent pixels first
        alpha = im.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            im = im.crop(bbox)
        return im

    # RGB/L: assume background = top-left pixel, remove near-uniform margins
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)

    if diff.mode != "L":
        diff = diff.convert("L")

    # Boost difference then threshold-ish via point()
    diff = diff.point(lambda p: 255 if p > tol else 0)

    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def fit_into_box(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize keeping aspect ratio to fit within box."""
    ratio = min(box_w / im.width, box_h / im.height)
    new_size = (max(1, int(im.width * ratio)), max(1, int(im.height * ratio)))
    return im.resize(new_size, Image.LANCZOS)

# ----------------------------------------------------------
# MAIN GENERATOR
# ----------------------------------------------------------

def generate_fiverr_domination_thumbnail(prompt, preview_path, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    niche = detect_niche(prompt)
    top, main, sub = headline_pack(niche)

    palettes = [
        ((0, 100, 255), (0, 30, 150)),
        ((0, 180, 120), (0, 90, 60)),
        ((255, 60, 60), (140, 0, 0)),
    ]

    outputs = {}

    for idx, (c1, c2) in enumerate(palettes):

        canvas = gradient_bg(c1, c2)
        draw = ImageDraw.Draw(canvas)

        font_top = _font(82)
        font_main = _font(118)
        font_sub = _font(52)

        # --- HEADLINES ---
        y_cursor = 260

        stroke_text(
            draw,
            (SAFE_MARGIN, y_cursor),
            top,
            font_top,
            fill="white",
            stroke_fill="black",
            stroke_w=6
        )

        _, h_top = _text_size(draw, top, font_top)
        y_cursor += h_top + 20

        stroke_text(
            draw,
            (SAFE_MARGIN, y_cursor),
            main,
            font_main,
            fill=(255, 255, 0),
            stroke_fill="black",
            stroke_w=8
        )

        _, h_main = _text_size(draw, main, font_main)
        y_cursor += h_main + 20

        draw.text(
            (SAFE_MARGIN, y_cursor),
            sub,
            fill=(240, 240, 240),
            font=font_sub
        )

        # --- VALUE STACK ---
        badge(canvas, "24H DELIVERY", SAFE_MARGIN, 80, (255, 0, 0))
        badge(canvas, "5 DESIGNS INCLUDED", SAFE_MARGIN, 150, (0, 0, 0))
        badge(canvas, "UNLIMITED REVISIONS", SAFE_MARGIN, 220, (30, 30, 30))

        # --- SOCIAL PROOF ---
        social_proof(canvas)

        # --- PREVIEW ---
        # --- PREVIEW (aligned + consistent) ---
        if os.path.exists(preview_path):
            preview = Image.open(preview_path)

            # Convert to RGBA for consistent pasting
            if preview.mode != "RGBA":
                preview = preview.convert("RGBA")

            # 1) Trim extra transparent padding (or uniform border if any)
            preview = trim_uniform_border(preview)

            # 2) Define a RIGHT-SIDE SAFE BOX where the preview must live
            #    (keeps it aligned across palettes and across different preview dimensions)
            right_box_x1 = int(THUMB_W * 0.58)          # left edge of right panel (tweak if needed)
            right_box_x2 = THUMB_W - SAFE_MARGIN        # right safe edge
            right_box_y1 = SAFE_MARGIN                  # top safe edge
            right_box_y2 = THUMB_H - SAFE_MARGIN        # bottom safe edge

            box_w = right_box_x2 - right_box_x1
            box_h = right_box_y2 - right_box_y1

            # 3) Fit preview into the safe box
            preview = fit_into_box(preview, box_w, box_h)

            # 4) Center it within the safe box (this is the key alignment fix)
            px = right_box_x1 + (box_w - preview.width) // 2
            py = right_box_y1 + (box_h - preview.height) // 2

            # Shadow (consistent offset relative to computed px/py)
            shadow = preview.copy().filter(ImageFilter.GaussianBlur(25))
            shadow = ImageEnhance.Brightness(shadow).enhance(0.30)

            canvas.paste(shadow, (px + 18, py + 18), shadow)
            canvas.paste(preview, (px, py), preview)

        out_path = os.path.join(out_dir, f"06_fiverr_domination_{idx+1}.png")
        canvas.save(out_path, "PNG")

        outputs[f"v{idx+1}"] = out_path

    return outputs