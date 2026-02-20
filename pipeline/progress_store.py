from typing import Dict, Any
from threading import Lock

_progress: Dict[str, Dict[str, Any]] = {}
_lock = Lock()


# -------------------------------------------------
# Job Initialization
# -------------------------------------------------

def init_job(job_id: str, total_designs: int = 0):
    with _lock:
        _progress[job_id] = {
            "job_id": job_id,
            "status": "queued",          # queued | running | done | error | done_with_errors
            "step": "queued",
            "progress": 0,               # overall 0..100
            "message": "Queued",
            "error": None,
            "total_designs": total_designs,
            "completed_designs": 0,
            "failed_designs": 0,
            "designs": {}                # per-design state
        }


# -------------------------------------------------
# Update overall job fields
# -------------------------------------------------

def update_job(job_id: str, **kwargs):
    with _lock:
        if job_id not in _progress:
            init_job(job_id)
        _progress[job_id].update(kwargs)


# -------------------------------------------------
# Per-design tracking
# -------------------------------------------------

def update_design(
    job_id: str,
    design_index: int,
    *,
    step: str = None,
    message: str = None,
    progress: int = None,
    status: str = None,      # running | done | error
    error: str = None,
):
    with _lock:
        if job_id not in _progress:
            init_job(job_id)

        job = _progress[job_id]

        if "designs" not in job:
            job["designs"] = {}

        if design_index not in job["designs"]:
            job["designs"][design_index] = {
                "design_index": design_index,
                "status": "running",
                "step": None,
                "message": None,
                "progress": 0,
                "error": None,
            }

        design = job["designs"][design_index]

        if step is not None:
            design["step"] = step
        if message is not None:
            design["message"] = message
        if progress is not None:
            design["progress"] = progress
        if status is not None:
            design["status"] = status
        if error is not None:
            design["error"] = error

        # Update aggregate counters
        completed = sum(1 for d in job["designs"].values() if d["status"] == "done")
        failed = sum(1 for d in job["designs"].values() if d["status"] == "error")

        job["completed_designs"] = completed
        job["failed_designs"] = failed


# -------------------------------------------------
# Get job state
# -------------------------------------------------

def get_job(job_id: str) -> Dict[str, Any]:
    with _lock:
        return _progress.get(
            job_id,
            {
                "job_id": job_id,
                "status": "unknown",
                "step": "unknown",
                "progress": 0,
                "message": "Job not found",
                "error": "not_found",
                "designs": {},
            },
        )