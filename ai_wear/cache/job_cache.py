"""Job state machine and disk cache.

The worker thread does HTTP/file IO only. The main thread (via bpy.app.timers)
polls job state and updates the UI. Worker threads MUST NOT touch bpy.scene/data.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class JobState(str, Enum):
    IDLE = "IDLE"
    RENDER = "RENDER"
    AI = "AI"
    BUILD = "BUILD"
    BAKE = "BAKE"
    DONE = "DONE"
    ERROR = "ERROR"
    CANCEL = "CANCEL"


class JobStage(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    UV = "UV"
    CAPTURE = "CAPTURE"
    AI_SUBMIT = "AI_SUBMIT"
    AI_POLL = "AI_POLL"
    MASK = "MASK"
    SURFACE = "SURFACE"
    WEARTHRESHOLD = "WEARTHRESHOLD"
    SEAM = "SEAM"
    EXPORT = "EXPORT"


@dataclass
class Job:
    """A single pipeline run. Stored in a module-level registry (session only).

    Worker threads mutate fields here; the main thread reads them via timers.
    All mutations are simple attribute writes on a Python object, which is safe
    enough for a single-producer progress display. Heavy intermediate data
    lives on disk, not in this object.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: JobState = JobState.IDLE
    stage: JobStage = JobStage.PREFLIGHT
    progress: float = 0.0  # 0..1 within the current stage
    message: str = ""
    error: str = ""
    error_kind: str = ""  # API / NETWORK / UV / DISK / RENDER / UNKNOWN
    cancel: bool = False
    started: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    # Free-form metadata bag for intermediate file paths, hashes, counts, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated = time.time()

    def to_display(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "error_kind": self.error_kind,
        }


# --- In-process registry -----------------------------------------------------

_JOBS: Dict[str, Job] = {}


def create_job() -> Job:
    job = Job()
    _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)


def active_job() -> Optional[Job]:
    """Return the most recent non-terminal job, if any."""
    terminal = {JobState.DONE, JobState.ERROR, JobState.CANCEL}
    latest: Optional[Job] = None
    for j in _JOBS.values():
        if j.state not in terminal and (latest is None or j.updated > latest.updated):
            latest = j
    return latest


def request_cancel(job_id: str) -> None:
    j = _JOBS.get(job_id)
    if j is not None:
        j.cancel = True
        j.touch()


def clear_finished() -> None:
    """Drop terminal jobs to bound memory."""
    terminal = {JobState.DONE, JobState.ERROR, JobState.CANCEL}
    for k in [k for k, j in _JOBS.items() if j.state in terminal]:
        if time.time() - _JOBS[k].updated > 3600:
            _JOBS.pop(k, None)


# --- Disk cache --------------------------------------------------------------

def cache_root() -> str:
    """Cache root. Defaults to <blend_dir>/.ai_wear_cache, falls back to temp."""
    try:
        import bpy  # local import; this runs on main thread only

        blend = bpy.path.abspath("//")
        if blend and blend != "//":
            root = os.path.join(blend, ".ai_wear_cache")
        else:
            # Unsaved file -> use OS temp to avoid leaking into random cwd
            import tempfile

            root = os.path.join(tempfile.gettempdir(), "ai_wear_cache")
    except Exception:
        import tempfile

        root = os.path.join(tempfile.gettempdir(), "ai_wear_cache")
    os.makedirs(root, exist_ok=True)
    return root


def object_cache_dir(obj_uuid: str) -> str:
    d = os.path.join(cache_root(), obj_uuid)
    os.makedirs(d, exist_ok=True)
    return d


def clear_current_run(obj_uuid: str) -> str:
    """Clear replaceable artifacts for a new full Generate run.

    ``Replay Downstream`` deliberately depends on the previous run's views and
    UV snapshot, so it must *not* call this function.  A new full Generate, on
    the other hand, must not inherit a partial ``views.json`` or old clean/worn
    pairs after a failed run.  Keep ``experiments/`` as the user's immutable
    comparison archive and remove every other direct child of this object's
    cache directory.
    """
    cache_dir = object_cache_dir(obj_uuid)
    cache_abs = os.path.abspath(cache_dir)
    keep = "experiments"
    for entry in os.scandir(cache_dir):
        if entry.name == keep:
            continue
        target = os.path.abspath(entry.path)
        # ``scandir(cache_dir)`` should always satisfy this, but keep the guard
        # explicit because this is the destructive boundary of the cache API.
        if os.path.commonpath((cache_abs, target)) != cache_abs:
            continue
        # Do not follow a link when deciding whether it is a directory.  The
        # cache owns these files, but this also prevents an accidental linked
        # directory from extending deletion outside the cache.
        is_junction = getattr(os.path, "isjunction", lambda _p: False)(entry.path)
        if entry.is_symlink() or is_junction:
            os.unlink(entry.path)
        elif entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)
    return cache_dir


def stable_hash(*parts: Any) -> str:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        h.update(repr(p).encode("utf-8", errors="replace"))
        h.update(b"|")
    return h.hexdigest()


def cache_key(mesh_hash: str, uv_hash: str, cam_hash: str, res: int,
              provider: str, model: str, prompt: str, seed: int) -> str:
    return stable_hash(mesh_hash, uv_hash, cam_hash, res, provider, model, prompt, seed)


def cache_path(obj_uuid: str, key: str, filename: str) -> str:
    return os.path.join(object_cache_dir(obj_uuid), f"{key}_{filename}")


def write_json(obj_uuid: str, key: str, filename: str, data: Any) -> str:
    p = cache_path(obj_uuid, key, filename)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, default=asdict)
    return p


def read_json(obj_uuid: str, key: str, filename: str) -> Optional[Any]:
    p = cache_path(obj_uuid, key, filename)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def exists(obj_uuid: str, key: str, filename: str) -> bool:
    return os.path.exists(cache_path(obj_uuid, key, filename))
