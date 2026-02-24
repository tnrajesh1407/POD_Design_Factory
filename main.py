# main.py (reviewed + fixed)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import json
import uuid
import zipfile
import traceback
from threading import Thread, Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional, List, Callable

# ----------------------------
# Central config (loads .env, validates, logs)
# ----------------------------
try:
    from config import settings  # config.py in project root
except Exception as e:
    print("CONFIG ERROR:", e)
    raise

# ----------------------------
# Pipeline imports
# ----------------------------
from pipeline.progress_store import init_job, update_job, get_job, update_design
from pipeline.prompt_enhancer import enhance_prompt
from pipeline.generate_image import generate_image
from pipeline.upscale import upscale_image
from pipeline.remove_bg import remove_background

# ✅ Use only export_pod_variants (dark/light deliverables)
from pipeline.print_ready import export_pod_variants

from pipeline.mockup import create_mockup
from pipeline.seo_generator import generate_seo_metadata
from pipeline.create_marketplace_metadata_files import create_marketplace_files

# ✅ Only WITH_REPORT is used in the pipeline below
from pipeline.pod_validator import validate_or_fix_print_ready_with_report

from pipeline.deliverable_packager import package_deliverables


# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()


@app.on_event("startup")
def _startup_validate_and_log():
    # settings import already validates; this is for clear logs
    print("=== POD DESIGN FACTORY CONFIG ===")
    print("IMAGE_PROVIDER:", settings.image_provider)
    print("FALLBACK ENABLED:", settings.enable_fallback)
    if settings.enable_fallback:
        print("FALLBACK_PROVIDER:", settings.fallback_provider)
    print("OPENAI_MODEL:", settings.openai_image_model)
    print("OPENAI_QUALITY:", settings.openai_image_quality)
    print("OPENAI_SIZE:", settings.openai_image_size or "(auto)")
    print("REPLICATE_MODEL:", settings.replicate_model)
    print("=================================")


os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DesignRequest(BaseModel):
    prompt: str
    num_designs: int = 1

    # bulk knobs
    max_workers: int = 3
    fail_fast: bool = True

    # external API throttling gate (recommended)
    api_concurrency: int = 2

    # mockups
    make_mockups: bool = True
    mockups: List[str] = ["tshirt_black.png", "tshirt_blue.png", "tshirt_white.png"]
    write_debug: bool = False


# ----------------------------
# Client-friendly helpers
# ----------------------------
def export_seo_txt_files(design_dir: str, seo: Dict[str, Any]) -> None:
    """
    Converts SEO JSON into client-friendly TXT files inside:
      <design_dir>/deliverables/marketplace_text/
    """
    out_dir = os.path.join(design_dir, "deliverables", "marketplace_text")
    os.makedirs(out_dir, exist_ok=True)

    title = (seo.get("title") or "").strip()
    description = (seo.get("description") or "").strip()
    alt_text = (seo.get("alt_text") or "").strip()

    tags = seo.get("tags") or []
    keywords = seo.get("keywords") or []

    tags = [str(t).strip() for t in tags if str(t).strip()]
    keywords = [str(k).strip() for k in keywords if str(k).strip()]

    with open(os.path.join(out_dir, "seo_title.txt"), "w", encoding="utf-8") as f:
        f.write(title + "\n")

    with open(os.path.join(out_dir, "seo_description.txt"), "w", encoding="utf-8") as f:
        f.write(description + "\n")

    with open(os.path.join(out_dir, "seo_alt_text.txt"), "w", encoding="utf-8") as f:
        f.write(alt_text + "\n")

    with open(os.path.join(out_dir, "seo_tags.txt"), "w", encoding="utf-8") as f:
        if tags:
            f.write("Top 13 Etsy Tags (recommended):\n")
            f.write("\n".join(tags[:13]) + "\n\n")
            f.write("All Tags:\n")
            f.write("\n".join(tags) + "\n\n")
            f.write("Comma-separated:\n")
            f.write(", ".join(tags) + "\n")

    with open(os.path.join(out_dir, "seo_keywords.txt"), "w", encoding="utf-8") as f:
        if keywords:
            f.write("\n".join(keywords) + "\n\n")
            f.write("Comma-separated:\n")
            f.write(", ".join(keywords) + "\n")

    # Bundle (single copy-paste file)
    with open(os.path.join(out_dir, "listing_bundle.txt"), "w", encoding="utf-8") as f:
        f.write("TITLE\n-----\n")
        f.write(title + "\n\n")

        f.write("DESCRIPTION\n-----------\n")
        f.write(description + "\n\n")

        if alt_text:
            f.write("ALT TEXT\n--------\n")
            f.write(alt_text + "\n\n")

        if tags:
            f.write("TAGS (Top 13 Etsy)\n------------------\n")
            f.write(", ".join(tags[:13]) + "\n\n")

            f.write("TAGS (All)\n----------\n")
            f.write(", ".join(tags) + "\n\n")

        if keywords:
            f.write("KEYWORDS\n--------\n")
            f.write(", ".join(keywords) + "\n")


def cleanup_extra_mockups(design_dir: str) -> None:
    """
    Keep only one mockup per garment color inside:
    deliverables/mockups/
    """
    mockup_dir = os.path.join(design_dir, "deliverables", "mockups")
    if not os.path.isdir(mockup_dir):
        return

    keep = {"mockup_black.png", "mockup_blue.png", "mockup_white.png"}

    for fn in os.listdir(mockup_dir):
        if fn not in keep:
            try:
                os.remove(os.path.join(mockup_dir, fn))
            except Exception:
                pass


def cleanup_marketplace_text(design_dir: str) -> None:
    """
    Keep only client-facing SEO txt files + listing_bundle + readme_deliverables
    inside deliverables/marketplace_text/ and delete everything else.
    """
    mdir = os.path.join(design_dir, "deliverables", "marketplace_text")
    if not os.path.isdir(mdir):
        return

    keep = {
        "seo_title.txt",
        "seo_description.txt",
        "seo_alt_text.txt",
        "seo_tags.txt",
        "seo_keywords.txt",
        "listing_bundle.txt",
        "readme_deliverables.txt",
    }

    for fn in os.listdir(mdir):
        if fn not in keep:
            try:
                os.remove(os.path.join(mdir, fn))
            except Exception:
                pass


def cleanup_preview_print_files(design_dir: str) -> None:
    """
    Removes internal preview assets from deliverables so clients don't see them.
    Keeps only final print_*.png files.
    """
    d = os.path.join(design_dir, "deliverables", "print_files")
    if not os.path.isdir(d):
        return

    for fn in os.listdir(d):
        lf = fn.lower()
        if lf.startswith("preview_print_") and lf.endswith(".png"):
            try:
                os.remove(os.path.join(d, fn))
            except Exception:
                pass


def _readme_text() -> str:
    return (
        "PRINT-ON-DEMAND DELIVERABLES\n"
        "============================\n\n"
        "FILE SPECIFICATIONS\n"
        "-------------------\n"
        "• Resolution: 4500 x 5400 pixels\n"
        "• DPI: 300\n"
        "• Color Profile: sRGB\n"
        "• Format: PNG with transparent background\n\n"
        "PRINT FILES\n"
        "-----------\n"
        "• for_dark_shirts.png  -> Use for black / navy / dark garments\n"
        "• for_light_shirts.png -> Use for white / pastel / light garments\n\n"
        "IMPORTANT\n"
        "---------\n"
        "• Do NOT add background color before uploading.\n"
        "• Upload the file that matches the shirt color.\n"
        "• These files are optimized for direct-to-garment (DTG) printing.\n\n"
        "MOCKUPS\n"
        "-------\n"
        "• Mockups are for preview/marketing use only.\n"
        "• Do NOT upload mockups to printing platforms.\n\n"
        "MARKETPLACE TEXT\n"
        "----------------\n"
        "• listing_bundle.txt -> Full copy-paste listing (Title, Description, Tags, Keywords)\n"
        "• Individual SEO files are also included.\n\n"
        "If you need alternate dimensions or placement adjustments, request a revision.\n"
    )


def write_readme_deliverables(design_dir: str) -> None:
    out_dir = os.path.join(design_dir, "deliverables", "marketplace_text")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "readme_deliverables.txt"), "w", encoding="utf-8") as f:
        f.write(_readme_text())


# ----------------------------
# API helpers
# ----------------------------
def _collect_preview_images(job_id: str) -> List[Dict[str, str]]:
    job_dir = f"outputs/{job_id}"
    if not os.path.exists(job_dir):
        return []

    images: List[Dict[str, str]] = []

    def _add(url: str, label: str):
        images.append({"url": url, "label": label})

    for i in range(1, 500):
        design_dir = os.path.join(job_dir, f"design_{i}")
        if not os.path.isdir(design_dir):
            break

        mock_dir = os.path.join(design_dir, "deliverables", "mockups")
        if os.path.isdir(mock_dir):
            mock_map = {
                "mockup_black.png": "Mockup - Black Shirt",
                "mockup_blue.png": "Mockup - Blue Shirt",
                "mockup_white.png": "Mockup - White Shirt",
            }
            for fname, label in mock_map.items():
                p = os.path.join(mock_dir, fname)
                if os.path.exists(p):
                    _add(f"/outputs/{job_id}/design_{i}/deliverables/mockups/{fname}", label)

        print_dir = os.path.join(design_dir, "deliverables", "print_files")
        if os.path.isdir(print_dir):
            print_map = {
                "print_dark.png": "Print File - Dark Shirts",
                "print_light.png": "Print File - Light Shirts",
            }
            for fname, label in print_map.items():
                p = os.path.join(print_dir, fname)
                if os.path.exists(p):
                    _add(f"/outputs/{job_id}/design_{i}/deliverables/print_files/{fname}", label)

    return images


def _variation_prompt(base_prompt: str, i: int) -> str:
    modifiers = [
        "minimalist vector style, bold shapes",
        "retro vintage distressed style",
        "clean outline illustration, high contrast",
        "flat modern illustration, simple geometry",
        "hand-drawn doodle style, playful",
        "sticker-like, thick outline, vibrant",
        "monochrome line art, minimal",
        "grunge texture, worn print look",
        "badge emblem style, centered composition",
        "simple icon + strong silhouette",
    ]
    mod = modifiers[(i - 1) % len(modifiers)]

    # ✅ Avoid white background in OpenAI mode to reduce rembg halos
    if settings.image_provider == "openai":
        bg = "background transparent if possible; otherwise a single flat chroma key background (pure magenta #FF00FF), no shading"
    else:
        bg = "isolated on pure solid white background (#FFFFFF), no shading"

    return f"{base_prompt}, {mod}, print-ready t-shirt design, {bg}, no watermark"


# ----------------------------
# Worker: one design pipeline
# ----------------------------
ProgressCB = Callable[[int, str, str, int], None]


def _process_one_design(
    *,
    job_id: str,
    job_dir: str,
    i: int,
    base_prompt: str,
    make_mockups: bool,
    mockups: Tuple[str, ...],
    write_debug: bool,
    progress_cb: ProgressCB,
    api_gate: Optional[Semaphore] = None,
) -> Dict[str, Any]:
    try:
        import shutil

        design_dir = os.path.join(job_dir, f"design_{i}")
        os.makedirs(design_dir, exist_ok=True)

        original = os.path.join(design_dir, "01_original.png")
        upscaled = os.path.join(design_dir, "02_upscaled.png")
        transparent = os.path.join(design_dir, "03_transparent.png")

        print_ready_dark = os.path.join(design_dir, "04_print_ready_dark.png")
        print_ready_light = os.path.join(design_dir, "04_print_ready_light.png")

        progress_cb(i, "prompt_variation", f"Variation {i}: creating variation prompt", 1)
        varied = _variation_prompt(base_prompt, i)

        progress_cb(i, "prompt_enhance", f"Variation {i}: enhancing prompt", 1)
        enhanced = enhance_prompt(varied)

        progress_cb(i, "generate_image", f"Variation {i}: generating image (provider={settings.image_provider})", 1)
        if api_gate:
            with api_gate:
                generate_image(enhanced, original)
        else:
            generate_image(enhanced, original)

        progress_cb(i, "upscale", f"Variation {i}: upscaling", 1)
        upscale_image(original, upscaled)

        progress_cb(i, "remove_bg", f"Variation {i}: removing background", 1)
        remove_background(upscaled, transparent)

        progress_cb(i, "print_ready_variants", f"Variation {i}: creating print-ready (dark + light)", 1)
        export_pod_variants(transparent, print_ready_dark, print_ready_light)

        progress_cb(i, "validate", f"Variation {i}: validating print-ready (dark + light)", 1)

        # Validate + fix + report (single pass, source of truth)
        best_dark, rep_dark = validate_or_fix_print_ready_with_report(
            print_ready_dark,
            alpha_threshold=20,
            band_px=6,
            max_edge_junk=40000,
            auto_fix_passes=1,
            expected_dpi=300,
        )
        best_light, rep_light = validate_or_fix_print_ready_with_report(
            print_ready_light,
            alpha_threshold=20,
            band_px=6,
            max_edge_junk=40000,
            auto_fix_passes=1,
            expected_dpi=300,
        )

        def _summ(rep: Dict[str, Any]) -> Dict[str, Any]:
            m = rep.get("metrics", {}) or {}
            v = rep.get("vector_mode", {}) or {}
            return {
                "score": rep.get("score"),
                "edge_soft_ratio": m.get("edge_soft_ratio"),
                "soft_ratio": m.get("soft_ratio"),
                "halo_drift": m.get("halo_drift"),
                "is_vector": (v.get("is_vector") if isinstance(v, dict) else None),
                "action": rep.get("action"),
                "output": rep.get("output"),
            }

        update_design(
            job_id,
            i,
            quality={
                "dark": _summ(rep_dark),
                "light": _summ(rep_light),
            },
        )

        # Persist best outputs back to canonical filenames expected by downstream steps
        if best_dark != print_ready_dark:
            shutil.copyfile(best_dark, print_ready_dark)
        if best_light != print_ready_light:
            shutil.copyfile(best_light, print_ready_light)

        # Write quality reports into deliverables (and keep them out of ZIP as you already do)
        qdir = os.path.join(design_dir, "deliverables", "_quality")
        os.makedirs(qdir, exist_ok=True)
        with open(os.path.join(qdir, "edge_report_dark.json"), "w", encoding="utf-8") as f:
            json.dump(rep_dark, f, indent=2)
        with open(os.path.join(qdir, "edge_report_light.json"), "w", encoding="utf-8") as f:
            json.dump(rep_light, f, indent=2)

        mockup_paths: List[str] = []
        if make_mockups and mockups:
            progress_cb(i, "mockup", f"Variation {i}: generating mockups", 1)
            for fname in mockups:
                out_name = "05_mockup_" + os.path.splitext(fname)[0].replace("tshirt_", "") + ".png"
                out_path = os.path.join(design_dir, out_name)

                src_print = print_ready_light if fname == "tshirt_white.png" else print_ready_dark

                create_mockup(
                    src_print,
                    out_path,
                    mode=None,
                    mockup_path=os.path.join("assets", fname),
                    write_debug=write_debug,
                )
                mockup_paths.append(out_path)
        else:
            progress_cb(i, "mockup_skip", f"Variation {i}: mockups skipped", 1)

        progress_cb(i, "seo", f"Variation {i}: generating SEO metadata", 1)
        seo = generate_seo_metadata(print_ready_dark)
        with open(os.path.join(design_dir, "seo.json"), "w", encoding="utf-8") as f:
            json.dump(seo, f, indent=2)
        export_seo_txt_files(design_dir, seo)

        progress_cb(i, "package", f"Variation {i}: packaging deliverables", 1)
        package_deliverables(design_dir)
        cleanup_preview_print_files(design_dir)
        cleanup_extra_mockups(design_dir)

        progress_cb(i, "marketplace", f"Variation {i}: generating marketplace files", 1)
        create_marketplace_files(design_dir)
        write_readme_deliverables(design_dir)
        cleanup_marketplace_text(design_dir)

        update_design(job_id, i, status="done", progress=100, step="done", message="Completed")

        return {
            "design_index": i,
            "design_dir": design_dir,
            "print_ready_dark": print_ready_dark,
            "print_ready_light": print_ready_light,
            "mockups": mockup_paths,
            "status": "ok",
        }

    except Exception as e:
        update_design(job_id, i, status="error", step="error", message=str(e), error=str(e))
        raise


# ----------------------------
# Bulk runner
# ----------------------------
def _run_pipeline(
    job_id: str,
    prompt: str,
    num_designs: int,
    *,
    max_workers: int = 3,
    fail_fast: bool = True,
    api_concurrency: int = 2,
    make_mockups: bool = True,
    mockups: Tuple[str, ...] = ("tshirt_black.png", "tshirt_blue.png", "tshirt_white.png"),
    write_debug: bool = False,
):
    job_dir = os.path.join("outputs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    steps_per_design = 11
    total_steps = num_designs * steps_per_design + 1
    done_steps = 0

    progress_lock = Lock()
    api_gate = Semaphore(max(1, int(api_concurrency)))

    def progress_cb(design_i: int, step_name: str, msg: str, inc: int = 1):
        nonlocal done_steps
        with progress_lock:
            done_steps += int(inc)
            pct = int((done_steps / total_steps) * 100)

        update_job(job_id, status="running", step=step_name, progress=pct, message=msg)
        update_design(job_id, design_i, step=step_name, message=msg, progress=min(100, pct), status="running")

    try:
        update_job(job_id, status="running", step="start", progress=1, message="Starting bulk pipeline...")

        failures: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            future_map = {}
            for i in range(1, num_designs + 1):
                update_job(job_id, step=f"design_{i}", message=f"Queued variation {i}/{num_designs}...")
                fut = ex.submit(
                    _process_one_design,
                    job_id=job_id,
                    job_dir=job_dir,
                    i=i,
                    base_prompt=prompt,
                    make_mockups=make_mockups,
                    mockups=mockups,
                    write_debug=write_debug,
                    progress_cb=progress_cb,
                    api_gate=api_gate,
                )
                future_map[fut] = i

            for fut in as_completed(future_map):
                i = future_map[fut]
                try:
                    fut.result()
                except Exception as e:
                    failures.append({"design_index": i, "error": str(e), "trace": traceback.format_exc()})
                    update_job(job_id, step=f"design_{i}", message=f"Variation {i} failed: {e}")

                    if fail_fast:
                        for f2 in future_map:
                            if not f2.done():
                                f2.cancel()
                        raise RuntimeError(f"Variation {i} failed: {e}")

        if failures:
            with open(os.path.join(job_dir, "failures.json"), "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2)

        update_job(job_id, status="running", step="zip", progress=95, message="Packaging ZIP...")

        zip_full = f"{job_dir}_full.zip"
        failures_fp = os.path.join(job_dir, "failures.json")

        def _norm(p: str) -> str:
            return p.replace("\\", "/")

        def iter_files_under_job():
            for root, _, files in os.walk(job_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    rel = _norm(os.path.relpath(fp, job_dir))
                    yield fp, rel, rel.rsplit("/", 1)[-1].lower()

        with zipfile.ZipFile(zip_full, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr("README.txt", _readme_text())

            if os.path.exists(failures_fp):
                zipf.write(failures_fp, "failures.json")

            for fp, rel, fname in iter_files_under_job():
                lrel = rel.lower()

                if "_debug" in lrel:
                    continue
                if "/deliverables/_quality/" in lrel:
                    continue
                if fname == "seo.json":
                    continue

                # print files renamed
                if "/deliverables/print_files/" in lrel:
                    if fname == "print_dark.png":
                        arc = rel.replace(
                            "/deliverables/print_files/print_dark.png",
                            "/deliverables/print_files/for_dark_shirts.png",
                        )
                        zipf.write(fp, arc)
                    elif fname == "print_light.png":
                        arc = rel.replace(
                            "/deliverables/print_files/print_light.png",
                            "/deliverables/print_files/for_light_shirts.png",
                        )
                        zipf.write(fp, arc)
                    continue

                # mockups
                if "/deliverables/mockups/" in lrel:
                    if fname.startswith("mockup_") and fname.endswith((".png", ".jpg", ".jpeg")):
                        zipf.write(fp, rel)
                    continue

                # marketplace text
                if "/deliverables/marketplace_text/" in lrel:
                    zipf.write(fp, rel)
                    continue

        update_job(
            job_id,
            status="done" if not failures else "done_with_errors",
            step="done",
            progress=100,
            message=(
                "All designs completed!"
                if not failures
                else f"Completed with {len(failures)} failures (see failures.json)."
            ),
        )

    except Exception as e:
        update_job(job_id, status="error", step="error", progress=100, message="Pipeline failed", error=str(e))


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/generate-pod-design")
def generate_pod_design(req: DesignRequest):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join("outputs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ✅ set total_designs so UI can show total vs completed
    init_job(job_id, total_designs=req.num_designs)

    update_job(
        job_id,
        status="queued",
        step="queued",
        progress=0,
        message=f"Queued (provider={settings.image_provider})",
    )

    t = Thread(
        target=_run_pipeline,
        args=(job_id, req.prompt, req.num_designs),
        kwargs=dict(
            max_workers=max(1, int(req.max_workers)),
            fail_fast=bool(req.fail_fast),
            api_concurrency=max(1, int(req.api_concurrency)),
            make_mockups=bool(req.make_mockups),
            mockups=tuple(req.mockups or []),
            write_debug=bool(req.write_debug),
        ),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


@app.get("/job/{job_id}/status")
def job_status(job_id: str):
    return get_job(job_id)


@app.get("/job/{job_id}")
def get_job_results(job_id: str):
    job_dir = f"outputs/{job_id}"
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    images = _collect_preview_images(job_id)

    zip_full_path = f"outputs/{job_id}_full.zip"
    zip_full_url = f"/outputs/{job_id}_full.zip" if os.path.exists(zip_full_path) else None

    failures_path = os.path.join(job_dir, "failures.json")
    failures_url = f"/outputs/{job_id}/failures.json" if os.path.exists(failures_path) else None

    return JSONResponse(
        {
            "job_id": job_id,
            "images": images,
            "zip": zip_full_url,
            "failures": failures_url,
        }
    )