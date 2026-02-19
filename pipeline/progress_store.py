# progress_store.py (or inside main.py)
from typing import Dict, Any
from threading import Lock

_progress: Dict[str, Dict[str, Any]] = {}
_lock = Lock()

def init_job(job_id: str):
    with _lock:
        _progress[job_id] = {
            "job_id": job_id,
            "status": "queued",         # queued | running | done | error
            "step": "queued",
            "progress": 0,              # 0..100
            "message": "Queued",
            "error": None,
        }

def update_job(job_id: str, **kwargs):
    with _lock:
        if job_id not in _progress:
            init_job(job_id)
        _progress[job_id].update(kwargs)

def get_job(job_id: str) -> Dict[str, Any]:
    with _lock:
        return _progress.get(job_id, {
            "job_id": job_id,
            "status": "unknown",
            "step": "unknown",
            "progress": 0,
            "message": "Job not found",
            "error": "not_found",
        })
