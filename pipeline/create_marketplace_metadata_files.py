import json
import os


def create_marketplace_files(job_dir: str):
    """
    One-pass marketplace metadata writer.

    If deliverables/marketplace_text/ exists, writes all metadata there.
    Otherwise writes into job_dir (backward compatible).
    """

    seo_path = os.path.join(job_dir, "seo.json")
    if not os.path.exists(seo_path):
        raise FileNotFoundError(f"seo.json not found at: {seo_path}")

    with open(seo_path, "r", encoding="utf-8") as f:
        seo = json.load(f)

    # If deliverables structure exists, write into it
    deliverables_root = os.path.join(job_dir, "deliverables")
    deliver_text_dir = os.path.join(deliverables_root, "marketplace_text")
    out_dir = deliver_text_dir if os.path.isdir(deliver_text_dir) else job_dir

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # Marketplace Text Outputs
    # ----------------------------
    etsy_text = f"""
TITLE:
{seo.get('title', '')}

DESCRIPTION:
{seo.get('description', '')}

TAGS:
{', '.join(seo.get('tags', []))}
""".strip()

    with open(os.path.join(out_dir, "etsy_listing.txt"), "w", encoding="utf-8") as f:
        f.write(etsy_text)

    with open(os.path.join(out_dir, "redbubble_tags.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(seo.get("tags", [])))

    with open(os.path.join(out_dir, "amazon_keywords.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(seo.get("keywords", [])))

    # Also copy seo.json into deliverables/marketplace_text if out_dir is deliverables
    if out_dir != job_dir:
        with open(os.path.join(out_dir, "seo.json"), "w", encoding="utf-8") as f:
            json.dump(seo, f, indent=2)

    # ----------------------------
    # Detect print files (prefer clean names)
    # ----------------------------
    clean_print_dir = os.path.join(deliverables_root, "print_files")

    clean_dark = os.path.join(clean_print_dir, "print_dark.png")
    clean_light = os.path.join(clean_print_dir, "print_light.png")
    clean_default = os.path.join(clean_print_dir, "print_default.png")

    # NEW: preview files (UI only)
    preview_dark = os.path.join(clean_print_dir, "preview_print_dark.png")
    preview_light = os.path.join(clean_print_dir, "preview_print_light.png")
    preview_default = os.path.join(clean_print_dir, "preview_print_default.png")

    canon_dark = os.path.join(job_dir, "04_print_ready_dark.png")
    canon_light = os.path.join(job_dir, "04_print_ready_light.png")
    canon_default = os.path.join(job_dir, "04_print_ready.png")

    def _pick(preferred: str, fallback: str):
        if preferred and os.path.exists(preferred):
            return preferred
        if fallback and os.path.exists(fallback):
            return fallback
        return None

    chosen_dark = _pick(clean_dark, canon_dark)
    chosen_light = _pick(clean_light, canon_light)
    chosen_default = _pick(clean_default, canon_default)

    # ----------------------------
    # Detect mockups (prefer clean names)
    # ----------------------------
    clean_mockup_black = os.path.join(deliverables_root, "mockups", "mockup_black.png")
    clean_mockup_blue = os.path.join(deliverables_root, "mockups", "mockup_blue.png")
    clean_mockup_white = os.path.join(deliverables_root, "mockups", "mockup_white.png")

    def _rel(p: str) -> str:
        return os.path.relpath(p, job_dir) if p else None

    # ----------------------------
    # Manifest
    # ----------------------------
    manifest = {
        "deliverables_root": "deliverables/" if os.path.isdir(deliverables_root) else None,
        "print_files": {
            "default": _rel(chosen_default),
            "dark_garments": _rel(chosen_dark),
            "light_garments": _rel(chosen_light),
        },
        # NEW: previews are optional; if present they help UIs and buyers
        "print_previews_ui_only": {
            "default": _rel(preview_default) if os.path.exists(preview_default) else None,
            "dark_garments": _rel(preview_dark) if os.path.exists(preview_dark) else None,
            "light_garments": _rel(preview_light) if os.path.exists(preview_light) else None,
            "note": "Preview files are for quick viewing only. Do NOT upload preview files to POD platforms.",
        },
        "recommended_usage": {
            "dark_shirts_black_navy_royal": "Upload the dark_garments file",
            "light_shirts_white_light_gray_pastel": "Upload the light_garments file",
        },
        "mockups": {
            "black": _rel(clean_mockup_black) if os.path.exists(clean_mockup_black) else None,
            "blue": _rel(clean_mockup_blue) if os.path.exists(clean_mockup_blue) else None,
            "white": _rel(clean_mockup_white) if os.path.exists(clean_mockup_white) else None,
            "note": "Mockups are generated using the correct print variant (white uses light_garments).",
        },
        "included_files_in_this_folder": [
            "etsy_listing.txt",
            "redbubble_tags.txt",
            "amazon_keywords.txt",
            "seo.json",
            "README_DELIVERABLES.txt",
            "deliverables_manifest.json",
        ],
    }

    with open(os.path.join(out_dir, "deliverables_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ----------------------------
    # README (uses clean deliverables paths if available)
    # ----------------------------
    def _pretty(path: str) -> str:
        return os.path.relpath(path, job_dir) if path else "NOT FOUND"

    readme_lines = [
        "DELIVERABLES GUIDE",
        "",
        "PRINT FILES (UPLOAD THESE TO POD PLATFORM):",
        f"- Default: {_pretty(chosen_default)}",
        f"- For DARK garments (black/navy/blue): {_pretty(chosen_dark)}",
        f"- For LIGHT garments (white/light gray): {_pretty(chosen_light)}",
        "",
        "IMPORTANT:",
        "- preview_print_*.png files are ONLY for quick viewing (smaller size).",
        "- DO NOT upload preview files to POD platforms. Upload print_dark/print_light/print_default instead.",
        "",
        "USAGE RECOMMENDATION:",
        "- Upload the DARK garment file for black/navy/dark shirts.",
        "- Upload the LIGHT garment file for white/light shirts to prevent the design from blending into fabric.",
        "",
        "MOCKUPS (PREVIEW IMAGES):",
        f"- Black: {_pretty(clean_mockup_black) if os.path.exists(clean_mockup_black) else 'NOT FOUND'}",
        f"- Blue:  {_pretty(clean_mockup_blue) if os.path.exists(clean_mockup_blue) else 'NOT FOUND'}",
        f"- White: {_pretty(clean_mockup_white) if os.path.exists(clean_mockup_white) else 'NOT FOUND'}",
        "",
        "MARKETPLACE TEXT FILES (THIS FOLDER):",
        "- etsy_listing.txt",
        "- redbubble_tags.txt",
        "- amazon_keywords.txt",
        "- seo.json",
        "",
    ]

    with open(os.path.join(out_dir, "README_DELIVERABLES.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))