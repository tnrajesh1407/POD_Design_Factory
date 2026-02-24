# pipeline/adaptive_tuning.py
import numpy as np
import cv2

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _luma(rgb_u8):
    # rgb_u8: HxWx3 uint8
    r = rgb_u8[..., 0].astype(np.float32)
    g = rgb_u8[..., 1].astype(np.float32)
    b = rgb_u8[..., 2].astype(np.float32)
    return 0.2126*r + 0.7152*g + 0.0722*b

def _edge_density_from_alpha(alpha_u8):
    # alpha_u8: HxW uint8
    mask = (alpha_u8 > 10).astype(np.uint8) * 255
    if mask.sum() == 0:
        return 0.0
    edges = cv2.Canny(mask, 50, 150)
    return float((edges > 0).sum()) / float(mask.size)

def _alpha_softness(alpha_u8):
    # proportion of mid-alpha pixels among "present" pixels
    present = alpha_u8 > 10
    if present.sum() == 0:
        return 0.0
    mid = (alpha_u8 >= 30) & (alpha_u8 <= 220)
    return float(mid.sum()) / float(present.sum())

def _texture_score(rgb_u8, alpha_u8):
    # Laplacian energy inside alpha region
    present = alpha_u8 > 10
    if present.sum() < 50:
        return 0.0
    gray = _luma(rgb_u8).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = np.mean((lap[present])**2)
    # normalize roughly to 0..1 (tuned empirically; safe clamp)
    return clamp(float(energy) / 800.0, 0.0, 1.0)

def analyze_design_rgba(design_rgba_u8):
    # design_rgba_u8: HxWx4 uint8
    rgb = design_rgba_u8[..., :3]
    alpha = design_rgba_u8[..., 3]

    coverage = float((alpha > 10).sum()) / float(alpha.size)
    edge_density = _edge_density_from_alpha(alpha)
    softness = _alpha_softness(alpha)
    texture = _texture_score(rgb, alpha)

    return {
        "coverage": coverage,
        "edge_density": edge_density,
        "alpha_softness": softness,
        "texture": texture,
    }

def analyze_shirt_bg(bgr_u8, shirt_mask_u8=None):
    # bgr_u8: HxWx3 uint8 (OpenCV)
    if shirt_mask_u8 is None:
        # fallback: whole image
        mask = np.ones(bgr_u8.shape[:2], dtype=np.uint8) * 255
    else:
        mask = (shirt_mask_u8 > 0).astype(np.uint8) * 255

    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    luma = _luma(rgb)
    region = mask > 0
    if region.sum() < 50:
        region = np.ones_like(region, dtype=bool)

    bg_brightness = float(np.mean(luma[region]))

    # fabric texture: laplacian energy on luma
    lap = cv2.Laplacian(luma.astype(np.float32), cv2.CV_32F, ksize=3)
    energy = float(np.mean((lap[region])**2))
    fabric = clamp(energy / 1200.0, 0.0, 1.0)

    return {
        "bg_brightness": bg_brightness,  # 0..255
        "fabric": fabric,                # 0..1
    }

def compute_adaptive_params(design_stats, bg_stats):
    C = clamp(design_stats["coverage"], 0.01, 0.95)
    E = clamp(design_stats["edge_density"], 0.0, 0.50)
    S = clamp(design_stats["alpha_softness"], 0.0, 0.60)
    T = clamp(design_stats["texture"], 0.0, 1.0)

    B = clamp(bg_stats["bg_brightness"], 0.0, 255.0)
    fabric = clamp(bg_stats["fabric"], 0.0, 1.0)

    bg_dark = clamp((140.0 - B) / 140.0, 0.0, 1.0)
    boldness = clamp((C - 0.18) / 0.55, 0.0, 1.0)
    sharpness = clamp(E / 0.25, 0.0, 1.0)
    distress = clamp(0.55*T + 0.45*S, 0.0, 1.0)

    alpha_cutoff = clamp(int(12 + 26*boldness + 10*sharpness + 10*bg_dark - 22*distress), 4, 52)
    alpha_blur   = clamp(0.15 + 0.90*sharpness + 0.35*boldness - 0.85*distress, 0.0, 1.4)
    shrink_px    = clamp(int(round(0 + 3.2*boldness + 1.2*bg_dark + 1.2*sharpness - 4.0*distress)), 0, 4)

    band_px    = clamp(int(round(3 + 10*boldness + 6*bg_dark + 2*sharpness - 6*distress)), 2, 16)
    feather_px = clamp(int(round(1 + 0.35*band_px)), 1, 8)
    min_alpha  = clamp(int(round(18 + 28*boldness + 12*bg_dark - 20*distress)), 6, 80)

    strength        = clamp(0.45 + 0.28*boldness + 0.22*fabric - 0.22*sharpness, 0.35, 0.85)
    detail_strength = clamp(0.18 + 0.30*sharpness + 0.14*fabric - 0.18*distress, 0.10, 0.55)
    shading_blur    = clamp(14.0 + 18.0*fabric + 6.0*boldness - 10.0*sharpness, 10.0, 30.0)
    ink_bleed       = clamp(0.00 + 0.07*boldness + 0.03*fabric - 0.07*sharpness, 0.0, 0.10)
    desat           = clamp(0.04 + 0.07*bg_dark + 0.05*boldness - 0.06*sharpness, 0.0, 0.14)
    gamma           = clamp(1.00 + 0.06*bg_dark - 0.04*boldness, 0.92, 1.10)

    return {
        "alpha_cleanup": {"alpha_cutoff": alpha_cutoff, "blur": alpha_blur, "shrink_px": shrink_px},
        "outline_strip": {"band_px": band_px, "feather_px": feather_px, "min_alpha": min_alpha},
        "fabric_presets": {"strength": strength, "detail_strength": detail_strength, "shading_blur": shading_blur,
                           "ink_bleed": ink_bleed, "desat": desat, "gamma": gamma},
        "signals": {"boldness": boldness, "sharpness": sharpness, "distress": distress, "bg_dark": bg_dark, "fabric": fabric},
    }