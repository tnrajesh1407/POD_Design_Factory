import os
import shutil
from typing import Optional

from PIL import Image


def _safe_copy(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _make_preview_png(src_path: str, dst_path: str, max_side: int = 1200):
    """
    Create a resized preview PNG for UI (fast to load).
    Keeps aspect ratio. max_side is max(width, height).
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with Image.open(src_path) as im:
        im = im.convert("RGBA")
        w, h = im.size
        scale = max_side / float(max(w, h))

        if scale >= 1.0:
            im.save(dst_path, "PNG", optimize=True)
            return

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), Image.LANCZOS)
        im.save(dst_path, "PNG", optimize=True)


def package_deliverables(design_dir: str):
    """
    Creates clean Fiverr-ready folder structure with renamed files:
    deliverables/
        print_files/print_dark.png, print_light.png, print_default.png (+ preview_print_*.png)
        mockups/mockup_black.png, mockup_blue.png, mockup_white.png
        marketplace_text/(etsy/redbubble/amazon/seo/readme/manifest)
    """
    deliverables_root = os.path.join(design_dir, "deliverables")
    print_dir = os.path.join(deliverables_root, "print_files")
    mockup_dir = os.path.join(deliverables_root, "mockups")
    text_dir = os.path.join(deliverables_root, "marketplace_text")

    os.makedirs(print_dir, exist_ok=True)
    os.makedirs(mockup_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    # ----------------------------
    # Print files (renamed)
    # ----------------------------
    src_dark = _first_existing(
        os.path.join(design_dir, "04_print_ready_dark.png"),
        os.path.join(design_dir, "04_print_ready_dark.PNG"),
    )
    src_light = _first_existing(
        os.path.join(design_dir, "04_print_ready_light.png"),
        os.path.join(design_dir, "04_print_ready_light.PNG"),
    )
    src_default = _first_existing(
        os.path.join(design_dir, "04_print_ready.png"),
        os.path.join(design_dir, "04_print_ready.PNG"),
    )

    dst_dark = os.path.join(print_dir, "print_dark.png")
    dst_light = os.path.join(print_dir, "print_light.png")
    dst_default = os.path.join(print_dir, "print_default.png")

    if src_dark:
        _safe_copy(src_dark, dst_dark)
    if src_light:
        _safe_copy(src_light, dst_light)
    if src_default:
        _safe_copy(src_default, dst_default)

    # ----------------------------
    # Create lightweight UI previews (NOT for POD upload)
    # ----------------------------
    preview_max_side = 1200
    if os.path.exists(dst_dark):
        _make_preview_png(dst_dark, os.path.join(print_dir, "preview_print_dark.png"), max_side=preview_max_side)
    if os.path.exists(dst_light):
        _make_preview_png(dst_light, os.path.join(print_dir, "preview_print_light.png"), max_side=preview_max_side)
    if os.path.exists(dst_default):
        _make_preview_png(dst_default, os.path.join(print_dir, "preview_print_default.png"), max_side=preview_max_side)

    # ----------------------------
    # Mockups (renamed)
    # ----------------------------
    mockup_map = {
        "05_mockup_black.png": "mockup_black.png",
        "05_mockup_blue.png": "mockup_blue.png",
        "05_mockup_white.png": "mockup_white.png",
        "05_mockup_black.jpg": "mockup_black.jpg",
        "05_mockup_blue.jpg": "mockup_blue.jpg",
        "05_mockup_white.jpg": "mockup_white.jpg",
        "05_mockup_black.jpeg": "mockup_black.jpeg",
        "05_mockup_blue.jpeg": "mockup_blue.jpeg",
        "05_mockup_white.jpeg": "mockup_white.jpeg",
    }

    for src_name, dst_name in mockup_map.items():
        src = os.path.join(design_dir, src_name)
        if os.path.exists(src):
            _safe_copy(src, os.path.join(mockup_dir, dst_name))

    # Also copy any other legacy mockups (future-proof), keep original filename
    for fname in os.listdir(design_dir):
        if fname.startswith("05_mockup_") and fname.lower().endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(design_dir, fname)
            dst = os.path.join(mockup_dir, fname)
            if not os.path.exists(dst):
                _safe_copy(src, dst)

    # ----------------------------
    # Marketplace text files
    # Copy from deliverables/marketplace_text if already generated there,
    # otherwise fallback to design_dir root (backward compatible)
    # ----------------------------
    text_files = [
        "etsy_listing.txt",
        "redbubble_tags.txt",
        "amazon_keywords.txt",
        "seo.json",
        "README_DELIVERABLES.txt",
        "deliverables_manifest.json",
    ]

    # Preferred source is the folder that create_marketplace_files() writes into
    preferred_text_src_dir = os.path.join(design_dir, "deliverables", "marketplace_text")
    for fname in text_files:
        src = os.path.join(preferred_text_src_dir, fname)
        if not os.path.exists(src):
            src = os.path.join(design_dir, fname)
        if os.path.exists(src):
            _safe_copy(src, os.path.join(text_dir, fname))