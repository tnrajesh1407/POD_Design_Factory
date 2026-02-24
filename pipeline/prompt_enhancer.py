# pipeline/enhance_prompt.py
"""
Prompt enhancer tuned for POD workflows.

Key goals:
- Edge-safe, background-removal-friendly prompts (reduce halos/soft alpha edges)
- Pickleball niche templates (mascot / typography / icon)
- Single-string output compatible with generators that don't expose negative_prompt

Enhancement:
- Provider-aware prompt packing:
    * Replicate: keep "Negative prompt: ..." style, and can use white BG
    * OpenAI: avoid forcing white BG (causes halos during later background removal),
              prefer transparent or chroma-key background + short negatives
"""

from __future__ import annotations

import os
import re
import hashlib
from typing import Dict, Optional, Tuple


# -----------------------------
# Provider switch
# -----------------------------
from config import settings


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    return (s or "").strip()

def _lower(s: str) -> str:
    return _norm(s).lower()

def _contains_any(s: str, words) -> bool:
    s = _lower(s)
    return any(w in s for w in words)

def _extract_quoted_text(s: str) -> Optional[str]:
    m = re.search(r'"([^"]+)"', s)
    if m:
        return m.group(1).strip()
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1).strip()
    return None

def _detect_niche(user_prompt: str) -> str:
    p = _lower(user_prompt)
    if "pickleball" in p:
        return "pickleball"
    return "generic"

def _detect_variant(user_prompt: str) -> str:
    """
    Decide between:
      - mascot (default)
      - typography
      - icon (minimal)
    """
    p = _lower(user_prompt)

    if _contains_any(p, ["text", "slogan", "quote", "typography", "font", "letters"]) or _extract_quoted_text(user_prompt):
        return "typography"

    if _contains_any(p, ["icon", "minimal", "simple", "logo", "symbol", "pictogram", "flat icon"]):
        return "icon"

    return "mascot"


# -----------------------------
# Edge-safe POD constraints
# -----------------------------
# For Replicate style packer: "white background" is fine because you may not remove bg
EDGE_SAFE_POSITIVE_REPLICATE = (
    "clean vector illustration, bold flat colors, thick black outlines (minimum 4px stroke), "
    "high contrast, simple shapes, solid fills, minimal color palette (max 6 colors), "
    "no gradients, no glow, no shadow, no drop shadow, no blur, no texture, no grain, "
    "no transparency effects, no soft edges, "
    "isolated on a pure solid white background (#FFFFFF), plain white background with no shading, "
    "no floor shadow, no cast shadow, no vignette, "
    "centered composition, sticker style, t-shirt print ready, sharp crisp edges, 1:1 aspect ratio"
)

# For OpenAI style packer: avoid forcing white BG (halo risk during BG removal)
EDGE_SAFE_POSITIVE_OPENAI = (
    "clean vector illustration, bold flat colors, thick black outlines (minimum 4px stroke), "
    "high contrast, simple shapes, solid fills, minimal color palette (max 6 colors), "
    "no gradients, no glow, no shadow, no drop shadow, no blur, no texture, no grain, "
    "no transparency effects, no soft edges, sharp crisp edges, "
    "isolated subject, centered composition, sticker style, t-shirt print ready, 1:1 aspect ratio, "
    "background: transparent if possible; otherwise a single flat chroma key background (pure magenta #FF00FF) with no shading"
)

EDGE_SAFE_NEGATIVE = (
    "realistic, photo, 3d render, soft shadow, drop shadow, outer glow, bloom, haze, fog, smoke, "
    "blur, motion blur, depth of field, bokeh, vignette, watercolor, airbrush, "
    "texture overlay, grain, distressed fade, gradient shading, soft edges, thin lines, "
    "background scene, complex background, background objects, shadow on background"
)

NO_TEXT_POSITIVE = "no text, no letters, no numbers, no watermark, no logo text"
NO_TEXT_NEGATIVE = "text, typography, letters, numbers, watermark, signature, caption, label"


def _pack_prompt_replicate(positive_core: str, negative: str = EDGE_SAFE_NEGATIVE) -> str:
    """
    Single-string prompt for maximum compatibility across generators.
    Replicate prunaai/p-image has no negative_prompt field, so we embed negatives.
    """
    positive_core = _norm(positive_core)
    negative = _norm(negative)
    if not positive_core:
        positive_core = "t-shirt design"

    return f"{positive_core}, {EDGE_SAFE_POSITIVE_REPLICATE}. Negative prompt: {negative}."


def _pack_prompt_openai(positive_core: str, negative: str = EDGE_SAFE_NEGATIVE) -> str:
    """
    OpenAI-friendly single string:
    - Keep negatives SHORT
    - Avoid "pure white background" to reduce halo after BG removal
    """
    positive_core = _norm(positive_core)
    if not positive_core:
        positive_core = "t-shirt design"

    # Short, high-signal avoid list
    avoid = [
        "photo", "realistic", "3d",
        "gradients", "glow", "shadow", "drop shadow",
        "blur", "background scene",
        "watermark",
    ]

    # If template is a NO_TEXT case, ensure text is also avoided
    # (For typography variant you will intentionally include text, so don't force avoid-text there.)
    if "EXACT text:" not in positive_core:
        avoid.append("text")

    avoid_str = ", ".join(avoid)

    return (
        f"{positive_core}. {EDGE_SAFE_POSITIVE_OPENAI}. "
        f"Avoid: {avoid_str}."
    )


def _pack_prompt(positive_core: str, negative: str = EDGE_SAFE_NEGATIVE) -> str:
    if settings.image_provider == "openai":
        return _pack_prompt_openai(positive_core, negative=negative)
    return _pack_prompt_replicate(positive_core, negative=negative)


# -----------------------------
# Pickleball templates (score-optimized)
# -----------------------------
_PICKLEBALL_SLOGANS = [
    "EAT SLEEP PICKLEBALL REPEAT",
    "DINK RESPONSIBLY",
    "JUST ONE MORE GAME",
    "PICKLEBALL IS MY THERAPY",
    "KEEP CALM AND DINK ON",
    "LIFE IS BETTER WITH PICKLEBALL",
]

def _pick_slogan(user_prompt: str) -> str:
    """
    Deterministic slogan picker:
    - Same user_prompt -> same slogan every run (stable across restarts)
    - Different user_prompt -> likely different slogan
    """
    s = (user_prompt or "").strip().lower()
    if not s:
        return _PICKLEBALL_SLOGANS[0]

    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(_PICKLEBALL_SLOGANS)
    return _PICKLEBALL_SLOGANS[idx]


def _pickleball_mascot(user_prompt: str) -> str:
    p = _lower(user_prompt)

    subject = "cute cartoon pickleball mascot character holding a paddle and ball"
    if _contains_any(p, ["dog", "puppy"]):
        subject = "cute cartoon dog playing pickleball, holding a paddle and ball"
    elif _contains_any(p, ["cat", "kitten"]):
        subject = "cute cartoon cat playing pickleball, holding a paddle and ball"
    elif _contains_any(p, ["panda"]):
        subject = "cute cartoon panda playing pickleball, holding a paddle and ball"
    elif _contains_any(p, ["robot"]):
        subject = "cute cartoon robot playing pickleball, holding a paddle and ball"
    elif _contains_any(p, ["skull"]):
        subject = "cartoon skull playing pickleball, holding a paddle and ball (friendly, non-scary)"

    positive = f"{subject}, {NO_TEXT_POSITIVE}"
    negative = EDGE_SAFE_NEGATIVE + f", {NO_TEXT_NEGATIVE}"
    return _pack_prompt(positive, negative=negative)


def _pickleball_typography_with_meta(user_prompt: str) -> Tuple[str, Optional[str]]:
    quoted = _extract_quoted_text(user_prompt)
    selected = quoted or _pick_slogan(user_prompt)

    # Typography is especially sensitive: enforce exact text and forbid extra words
    positive = (
        f'pickleball typography design, EXACT text: "{selected}", '
        "heavy sans-serif block font, bold letters, solid fill letters, thick strokes, "
        "clean vector style, flat colors only, high contrast, "
        "no outline effects, no bevel/emboss, no neon, no glow, no shadow, "
        "no extra words, no additional text, no small print"
    )
    negative = EDGE_SAFE_NEGATIVE + ", neon, bevel, emboss, gradient fill, outline glow, letter shadow, glitter, extra words, additional text"
    return _pack_prompt(positive, negative=negative), (None if quoted else selected)


def _pickleball_typography(user_prompt: str) -> str:
    prompt, _selected = _pickleball_typography_with_meta(user_prompt)
    return prompt


def _pickleball_icon(user_prompt: str) -> str:
    positive = (
        "minimal pickleball paddle and ball icon, bold flat vector illustration, "
        "thick outlines, solid shapes, simple geometric design, high contrast, "
        f"{NO_TEXT_POSITIVE}"
    )
    negative = EDGE_SAFE_NEGATIVE + f", {NO_TEXT_NEGATIVE}"
    return _pack_prompt(positive, negative=negative)


# -----------------------------
# Generic fallback template
# -----------------------------
def _generic_template(user_prompt: str) -> str:
    subject = _norm(user_prompt) or "t-shirt design subject"
    return _pack_prompt(subject)


# -----------------------------
# Public API
# -----------------------------
def enhance_prompt(user_prompt: str) -> str:
    user_prompt = _norm(user_prompt)
    niche = _detect_niche(user_prompt)
    variant = _detect_variant(user_prompt)

    if niche == "pickleball":
        if variant == "typography":
            return _pickleball_typography(user_prompt)
        if variant == "icon":
            return _pickleball_icon(user_prompt)
        return _pickleball_mascot(user_prompt)

    return _generic_template(user_prompt)


def enhance_prompt_meta(user_prompt: str) -> Dict[str, str]:
    user_prompt = _norm(user_prompt)
    niche = _detect_niche(user_prompt)
    variant = _detect_variant(user_prompt)

    selected_slogan = None
    if niche == "pickleball" and variant == "typography":
        prompt, selected_slogan = _pickleball_typography_with_meta(user_prompt)
    else:
        prompt = enhance_prompt(user_prompt)

    return {
        "niche": niche,
        "variant": variant,
        "selected_slogan": selected_slogan or "",
        "prompt": prompt,
    }