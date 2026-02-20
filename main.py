from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid, os, json, zipfile
from threading import Thread

from fastapi.middleware.cors import CORSMiddleware

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


app = FastAPI()
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

def _collect_preview_images(job_id: str):
    """
    Return ONLY 01_original and 05_mockup from each design folder.
    """
    job_dir = f"outputs/{job_id}"
    images = []
    if not os.path.exists(job_dir):
        return images

    for i in range(1, 500):  # safe upper bound
        design_dir = os.path.join(job_dir, f"design_{i}")
        if not os.path.isdir(design_dir):
            break

        for fname in ("01_original.png", "05_mockup_white.jpg","05_mockup_black.jpg","05_mockup_lifestyle.jpg"):
            fpath = os.path.join(design_dir, fname)
            if os.path.exists(fpath):
                images.append(f"/outputs/{job_id}/design_{i}/{fname}")

    return images

def _run_pipeline(job_id: str, prompt: str, num_designs: int):
    job_dir = f"outputs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    try:
        update_job(job_id, status="running", step="start", progress=2, message="Starting pipeline...")

        # Progress plan:
        # per design: 8 steps + final zip
        # steps: prompt_enhance, generate_image, upscale, remove_bg, print_ready, validate, mockup, seo, marketplace
        steps_per_design = 9
        total_steps = num_designs * steps_per_design + 1  # +1 for zip
        done_steps = 0

        def bump(step_name: str, msg: str):
            nonlocal done_steps
            done_steps += 1
            pct = int((done_steps / total_steps) * 100)
            update_job(job_id, status="running", step=step_name, progress=pct, message=msg)

        for i in range(1, num_designs + 1):
            design_dir = f"{job_dir}/design_{i}"
            os.makedirs(design_dir, exist_ok=True)

            original = f"{design_dir}/01_original.png"
            upscaled = f"{design_dir}/02_upscaled.png"
            transparent = f"{design_dir}/03_transparent.png"
            print_ready = f"{design_dir}/04_print_ready.png"
            mockup_white = f"{design_dir}/05_mockup_white.jpg"

            update_job(job_id, step=f"design_{i}", message=f"Working on variation {i}/{num_designs}...")

            # 1 Prompt enhance
            enhanced_prompt = enhance_prompt(prompt)
            bump("prompt_enhance", f"Variation {i}: prompt enhanced")

            # 2 Generate image
            generate_image(enhanced_prompt, original)
            bump("generate_image", f"Variation {i}: image generated")

            # 3 Upscale
            upscale_image(original, upscaled)
            bump("upscale", f"Variation {i}: upscaled")

            # 4 Remove background
            remove_background(upscaled, transparent)
            bump("remove_bg", f"Variation {i}: background removed")

            # 5 Print-ready canvas (your place_on_pod_canvas already tightens alpha + embeds DPI/ICC)
            place_on_pod_canvas(transparent, print_ready)
            bump("print_ready", f"Variation {i}: print-ready created")

            # 6 Validate OR fix, then standardize filename to 04_print_ready.png
            best_print_ready = validate_or_fix_print_ready(
                print_ready,
                alpha_threshold=20,
                band_px=6,
                max_edge_junk=25000,
                auto_fix_passes=1,
            )

            # If it produced a _fixed file, overwrite 04_print_ready.png so downstream is consistent
            if best_print_ready != print_ready:
                import shutil
                shutil.copyfile(best_print_ready, print_ready)
                best_print_ready = print_ready

            bump("validate", f"Variation {i}: print-ready validated")

            # 7 Mockup (white only)
            create_mockup(
                best_print_ready,
                mockup_white,
                mode="front",
                mockup_path=os.path.join("assets", "tshirt_white.jpg"),
                write_debug=True,
            )
            bump("mockup", f"Variation {i}: mockup created (white)")

            # 8 SEO (use final print-ready)
            seo = generate_seo_metadata(best_print_ready)
            with open(f"{design_dir}/seo.json", "w", encoding="utf-8") as f:
                json.dump(seo, f, indent=2)
            bump("seo", f"Variation {i}: SEO generated")

            # 9 Marketplace files
            create_marketplace_files(design_dir)
            bump("marketplace", f"Variation {i}: marketplace files created")

        # Zip bundle
        zip_path = f"{job_dir}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(job_dir):
                for file in files:
                    if "_warp_debug" in file or "_quad_outline" in file or "_area_debug" in file:
                        continue
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, job_dir)
                    zipf.write(filepath, arcname)

        bump("zip", "Packaging ZIP...")

        update_job(
            job_id,
            status="done",
            step="done",
            progress=100,
            message="All designs completed!",
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


@app.post("/generate-pod-design")
def generate_pod_design(req: DesignRequest):
    job_id = str(uuid.uuid4())
    os.makedirs(f"outputs/{job_id}", exist_ok=True)

    init_job(job_id)
    update_job(job_id, status="queued", step="queued", progress=0, message="Queued")

    # Run pipeline in a separate thread so the API can respond immediately
    t = Thread(target=_run_pipeline, args=(job_id, req.prompt, req.num_designs), daemon=True)
    t.start()

    # Return immediately so UI can start polling /job/{job_id}/status
    return {"job_id": job_id}

@app.get("/job/{job_id}")
def get_job_results(job_id: str):
    job_dir = f"outputs/{job_id}"
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    images = _collect_preview_images(job_id)

    zip_path = f"outputs/{job_id}.zip"
    zip_url = f"/outputs/{job_id}.zip" if os.path.exists(zip_path) else None

    return JSONResponse({"job_id": job_id, "images": images, "zip": zip_url})

@app.get("/job/{job_id}/status")
def job_status(job_id: str):
    return get_job(job_id)
