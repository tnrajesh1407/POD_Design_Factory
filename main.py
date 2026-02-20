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
# Pipeline imports
# ----------------------------
from pipeline.progress_store import init_job, update_job, get_job

from pipeline.prompt_enhancer import enhance_prompt
from pipeline.generate_image import generate_image
from pipeline.upscale import upscale_image
from pipeline.remove_bg import remove_background
from pipeline.print_ready import place_on_pod_canvas
from pipeline.mockup import create_mockup
from pipeline.seo_generator import generate_seo_metadata
from pipeline.marketplace_files import create_marketplace_files
from pipeline.pod_validator import validate_or_fix_print_ready
from pipeline.progress_store import update_design
#from pipeline.thumbnail_generator import generate_fiverr_domination_thumbnail


# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI()

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
    # 1 = safest, 2-3 = faster if your providers allow it
    api_concurrency: int = 2

    # mockups
    make_mockups: bool = True
    mockups: List[str] = ["tshirt_black.png", "tshirt_blue.png", "tshirt_white.png"]
    #mockups: List[str] = ["tshirt_white.png"]
    write_debug: bool = False


# ----------------------------
# API helpers
# ----------------------------
def _collect_preview_images(job_id: str) -> List[str]:
    """
    Return preview images:
      - 01_original.png
      - 05_mockup_*.jpg (if present)
      - 06_fiverr_thumb_A/B/C.png (if present)
    """
    job_dir = f"outputs/{job_id}"
    images: List[str] = []
    if not os.path.exists(job_dir):
        return images

    for i in range(1, 500):
        design_dir = os.path.join(job_dir, f"design_{i}")
        if not os.path.isdir(design_dir):
            break

        # Always try original
        orig = os.path.join(design_dir, "01_original.png")
        if os.path.exists(orig):
            images.append(f"/outputs/{job_id}/design_{i}/01_original.png")

        # Mockups
        for fname in os.listdir(design_dir):
            if fname.startswith("05_mockup_") and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(f"/outputs/{job_id}/design_{i}/{fname}")

        # Fiverr thumbnails A/B/C
        #for v in ("A", "B", "C"):
            #tname = f"06_fiverr_thumb_{v}.png"
            #tpath = os.path.join(design_dir, tname)
            #if os.path.exists(tpath):
                #images.append(f"/outputs/{job_id}/design_{i}/{tname}")

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
    return f"{base_prompt}, {mod}, print-ready t-shirt design, transparent background, no watermark"


# ----------------------------
# Worker: one design pipeline with progress callback
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
    """
    One design end-to-end. Writes into outputs/<job_id>/design_<i>.
    Raises on failure.
    """
    try:
        design_dir = os.path.join(job_dir, f"design_{i}")
        os.makedirs(design_dir, exist_ok=True)

        original = os.path.join(design_dir, "01_original.png")
        upscaled = os.path.join(design_dir, "02_upscaled.png")
        transparent = os.path.join(design_dir, "03_transparent.png")
        print_ready = os.path.join(design_dir, "04_print_ready.png")

        # 1) prompt
        progress_cb(i, "prompt_variation", f"Variation {i}: creating variation prompt", 1)
        varied = _variation_prompt(base_prompt, i)

        progress_cb(i, "prompt_enhance", f"Variation {i}: enhancing prompt", 1)
        enhanced = enhance_prompt(varied)

        # 2) generate (gated)
        progress_cb(i, "generate_image", f"Variation {i}: generating image", 1)
        if api_gate:
            with api_gate:
                generate_image(enhanced, original)
        else:
            generate_image(enhanced, original)

        # 3) upscale (gated)
        progress_cb(i, "upscale", f"Variation {i}: upscaling", 1)
        if api_gate:
            with api_gate:
                upscale_image(original, upscaled)
        else:
            upscale_image(original, upscaled)

        # 4) remove bg (gated)
        progress_cb(i, "remove_bg", f"Variation {i}: removing background", 1)
        if api_gate:
            with api_gate:
                remove_background(upscaled, transparent)
        else:
            remove_background(upscaled, transparent)

        # 5) print-ready
        progress_cb(i, "print_ready", f"Variation {i}: placing on POD canvas", 1)
        place_on_pod_canvas(transparent, print_ready)

        # 6) validate/fix
        progress_cb(i, "validate", f"Variation {i}: validating print-ready", 1)
        best_print_ready = validate_or_fix_print_ready(
            print_ready,
            alpha_threshold=20,
            band_px=6,
            max_edge_junk=40000,
            auto_fix_passes=1,
        )

        # keep canonical output name for downstream
        if best_print_ready != print_ready:
            import shutil
            shutil.copyfile(best_print_ready, print_ready)
            best_print_ready = print_ready

        # 7) mockups
        mockup_paths: List[str] = []
        mockup_white_path: Optional[str] = None

        if make_mockups and mockups:
            progress_cb(i, "mockup", f"Variation {i}: generating mockups", 1)
            for fname in mockups:
                out_name = "05_mockup_" + os.path.splitext(fname)[0].replace("tshirt_", "") + ".png"
                out_path = os.path.join(design_dir, out_name)

                create_mockup(
                    best_print_ready,
                    out_path,
                    mode="front",
                    mockup_path=os.path.join("assets", fname),
                    write_debug=write_debug,
                )
                mockup_paths.append(out_path)

                if fname == "tshirt_white.png":
                    mockup_white_path = out_path
        else:
            progress_cb(i, "mockup_skip", f"Variation {i}: mockups skipped", 1)

        # 8) Fiverr thumbnails A/B/C
        #progress_cb(i, "thumbnails", f"Variation {i}: generating Fiverr thumbnails (A/B/C)", 1)
        #preview_for_thumb = mockup_white_path if (mockup_white_path and os.path.exists(mockup_white_path)) else best_print_ready

        #thumbs = generate_fiverr_domination_thumbnail(
            #prompt=base_prompt,          # niche detection uses the user's base prompt
            #preview_path=preview_for_thumb,
            #out_dir=design_dir,
        #)

        # 9) SEO
        progress_cb(i, "seo", f"Variation {i}: generating SEO metadata", 1)
        seo = generate_seo_metadata(best_print_ready)
        with open(os.path.join(design_dir, "seo.json"), "w", encoding="utf-8") as f:
            json.dump(seo, f, indent=2)

        # 10) marketplace files
        progress_cb(i, "marketplace", f"Variation {i}: generating marketplace files", 1)
        create_marketplace_files(design_dir)

        update_design(
            job_id,
            i,
            status="done",
            progress=100,
            step="done",
            message="Completed",
        )

        #return {
           # "design_index": i,
            #"design_dir": design_dir,
            #"print_ready": print_ready,
            #"mockups": mockup_paths,
            #"thumbnails": thumbs,  # {"A": "...", "B": "...", "C": "..."}
            #"status": "ok",
        #}

    except Exception as e:
        update_design(
            job_id,
            i,
            status="error",
            step="error",
            message=str(e),
            error=str(e),
        )
        raise

# ----------------------------
# Bulk runner with per-step progress
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
    mockups: Tuple[str, ...] = ("tshirt_white.jpg",),
    write_debug: bool = False,
):
    job_dir = os.path.join("outputs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Each design produces exactly 9 progress increments in the worker
    # (including mockup_skip if mockups disabled).
    steps_per_design = 10
    total_steps = num_designs * steps_per_design + 1  # + zip
    done_steps = 0

    progress_lock = Lock()
    last_msg_lock = Lock()

    # gate external-heavy steps to avoid 429s
    api_gate = Semaphore(max(1, int(api_concurrency)))

    def progress_cb(design_i: int, step_name: str, msg: str, inc: int = 1):
        nonlocal done_steps
        with progress_lock:
            done_steps += int(inc)
            pct = int((done_steps / total_steps) * 100)

        # Keep the job status readable but detailed
        update_job(
            job_id,
            status="running",
            step=step_name,
            progress=pct,
            message=msg,
        )

        # update per-design
        update_design(
            job_id,
            design_i,
            step=step_name,
            message=msg,
            progress=min(100, int((done_steps / total_steps) * 100)),
            status="running"
        )

    try:
        update_job(job_id, status="running", step="start", progress=1, message="Starting bulk pipeline...")

        results: List[Dict[str, Any]] = []
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
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    failures.append(
                        {
                            "design_index": i,
                            "error": str(e),
                            "trace": traceback.format_exc(),
                        }
                    )
                    update_job(job_id, step=f"design_{i}", message=f"Variation {i} failed: {e}")

                    if fail_fast:
                        # Cancel pending tasks
                        for f2 in future_map:
                            if not f2.done():
                                f2.cancel()
                        raise RuntimeError(f"Variation {i} failed: {e}")

        # failures report
        if failures:
            with open(os.path.join(job_dir, "failures.json"), "w", encoding="utf-8") as f:
                json.dump(failures, f, indent=2)

        # zip bundle (skip debug artifacts)
        zip_path = f"{job_dir}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(job_dir):
                for file in files:
                    if "_debug" in file:
                        continue
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, job_dir)
                    zipf.write(filepath, arcname)

        # zip step counts as +1
        with progress_lock:
            done_steps += 1
            pct = int((done_steps / total_steps) * 100)

        update_job(job_id, status="running", step="zip", progress=pct, message="Packaging ZIP...")

        update_job(
            job_id,
            status="done" if not failures else "done_with_errors",
            step="done",
            progress=100,
            message=("All designs completed!" if not failures else f"Completed with {len(failures)} failures (see failures.json)."),
        )

    except Exception as e:
        update_job(
            job_id,
            status="error",
            step="error",
            progress=100,
            message="Pipeline failed",
            error=str(e),
        )


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/generate-pod-design")
def generate_pod_design(req: DesignRequest):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join("outputs", job_id)
    os.makedirs(job_dir, exist_ok=True)

    init_job(job_id)
    update_job(job_id, status="queued", step="queued", progress=0, message="Queued")

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

    zip_path = f"outputs/{job_id}.zip"
    zip_url = f"/outputs/{job_id}.zip" if os.path.exists(zip_path) else None

    failures_path = os.path.join(job_dir, "failures.json")
    failures_url = f"/outputs/{job_id}/failures.json" if os.path.exists(failures_path) else None

    return JSONResponse(
        {
            "job_id": job_id,
            "images": images,
            "zip": zip_url,
            "failures": failures_url,
        }
    )