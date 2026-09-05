"""Pipeline orchestration.

Threading model (per the plan): a worker thread does HTTP and numpy. Every
bpy-touching step (render, UV ops, image save/load, node setup, bmesh) runs on
the main thread through a MainThreadBridge. bpy.app.timers pumps the bridge and
refreshes the UI between steps, so Blender never freezes on network calls and
the worker never touches the scene graph.

start_pipeline() snapshots the context, creates a Job, spawns the worker, and
installs the timer. The worker drives job.state/stage/progress; the UI reads it.
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Any, Callable

import numpy as np

from .. import utils
from ..cache import job_cache
from ..cache.job_cache import JobState, JobStage


# --- prefs snapshot (thread-safe plain object) ------------------------------

class PrefsSnap:
    """Plain copy of AddonPreferences so the worker thread never touches bpy.

    Providers read attributes/methods off this; everything is a Python string.
    """

    def __init__(self, prefs):
        self.api_base_url = prefs.api_base_url
        self.model_id = prefs.model_id
        self.api_key = prefs.get_api_key() or ""
        self.image_endpoint_path = prefs.image_endpoint_path
        self.request_mode = prefs.request_mode
        self.raw_body_template = prefs.raw_body_template
        self.raw_image_field = prefs.raw_image_field
        self.raw_response_is_url = bool(prefs.raw_response_is_url)
        self.extra_headers = prefs.extra_headers
        self.timeout = float(prefs.timeout)
        self.comfyui_url = prefs.comfyui_url
        self.workflow_path = prefs.workflow_path
        self.clean_image_node = prefs.clean_image_node
        self.mask_image_node = prefs.mask_image_node
        self.prompt_node = prefs.prompt_node
        self.seed_node = prefs.seed_node
        self.output_node = prefs.output_node
        self.poll_interval = float(prefs.poll_interval)

    def get_api_key(self):
        return self.api_key or None

    def get_base_url(self):
        return self.api_base_url.rstrip("/")


# --- main-thread bridge -----------------------------------------------------

class _Task:
    __slots__ = ("ev", "fn", "args", "kwargs", "result", "error")

    def __init__(self, fn, args, kwargs):
        self.ev = threading.Event()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error: Any = None


class MainThreadBridge:
    """Worker → main-thread call dispatcher."""

    def __init__(self):
        self._q: "list[_Task]" = []
        self._lock = threading.Lock()

    def run(self, fn: Callable, *args, **kwargs):
        t = _Task(fn, args, kwargs)
        with self._lock:
            self._q.append(t)
        t.ev.wait()
        if t.error is not None:
            raise t.error
        return t.result

    def pump(self):
        """Main thread: run at most one pending task (a render may take a while)."""
        with self._lock:
            if not self._q:
                return
            t = self._q[0]
        try:
            t.result = t.fn(*t.args, **t.kwargs)
        except Exception as e:
            t.error = e
        finally:
            t.ev.set()
            with self._lock:
                self._q.pop(0)


# --- snapshot ---------------------------------------------------------------

def snapshot_context(context) -> dict:
    scene = context.scene
    s = scene.ai_wear
    import bpy
    prefs = bpy.context.preferences.addons["ai_wear"].preferences
    obj = context.active_object
    obj_uuid = (obj.ai_wear.mesh_hash if obj and obj.ai_wear.mesh_hash
                else (obj.name if obj else "obj"))
    preset_name = ""
    if 0 <= s.active_preset_index < len(s.presets):
        preset_name = s.presets[s.active_preset_index].name
    prefs_snap = PrefsSnap(prefs)
    # Providers consume this plain object on the worker thread.  Apply the
    # scene-level overrides to it as well as recording them in metadata;
    # otherwise the UI appeared to override Base URL / Model but requests still
    # used the global Add-on Preferences values.
    prefs_snap.api_base_url = s.effective_base_url(prefs)
    prefs_snap.model_id = s.effective_model(prefs)
    return {
        "object_name": obj.name if obj else None,
        "obj_uuid": obj_uuid,
        "uv_mode": s.uv_mode,
        "target_uv_layer": s.target_uv_layer,
        "work_resolution": s.work_resolution,
        "camera_preset": s.camera_preset,
        "camera_count": s.camera_count,
        "render_resolution": s.render_resolution,
        "view_context_mode": s.view_context_mode,
        "provider": s.effective_provider(prefs),
        "model": s.effective_model(prefs),
        "base_url": s.effective_base_url(prefs),
        "strategy": s.generation_strategy,
        "prompt": s.build_prompt(),
        "seed": s.seed,
        "lock_seed": s.lock_seed,
        "weights": {"w_ai": s.w_ai, "w_convex": s.w_convex,
                    "w_expose": s.w_expose, "w_cavity": s.w_cavity},
        "gamma": s.gamma,
        "alpha": s.alpha,
        "noise_amp": s.noise_amp,
        "noise_scale": s.noise_scale,
        "use_barrier": s.use_barrier,
        "mat_penalty": s.material_boundary_penalty,
        "coverage_target": s.coverage_target,
        "use_comfy_inpaint": s.use_comfy_inpaint,
        "inpaint_edge_width": s.inpaint_edge_width,
        "seam_fuse": s.seam_fuse,
        "seam_diffuse": s.seam_diffuse_texels,
        "use_padding": s.use_padding,
        "padding": s.padding_texels,
        "use_ai_mask": s.use_ai_mask,
        "use_geometry_prior": s.use_geometry_prior,
        "use_topology_growth": s.use_topology_growth,
        "save_experiment_snapshot": s.save_experiment_snapshot,
        "experiment_label": s.experiment_label,
        "preset_name": preset_name,
        "export_format": s.export_format,
        "wear_amount": s.wear_amount,
        "feather": s.feather,
        "prefs_obj": prefs_snap,  # plain snapshot; safe on worker thread
        "timeout": prefs.timeout,
        "poll_interval": prefs.poll_interval,
    }


# --- worker pipeline --------------------------------------------------------

def _check_cancel(job):
    if job.cancel:
        from ..ai.base import ProviderError
        raise ProviderError("Cancelled", kind="CANCEL")


def _require_api_key(provider: str, prefs_obj) -> None:
    """Raise an actionable RuntimeError if a key-requiring provider has no key.

    A missing key would otherwise surface later as "HTTP 401 Missing bearer or
    basic" — the OpenAI-compat path only attaches the Authorization header when
    the key is truthy (see _openai_compat.generate_openai_compat), so an empty
    key sends a headerless request. Extracted from _run_pipeline so the preflight
    is unit-testable without a full GUI pipeline run (the inline call site sits
    after UV/camera setup, which needs a screen area).
    """
    if provider != "COMFYUI" and not prefs_obj.get_api_key():
        raise RuntimeError(
            "API key is empty — the endpoint would reject with HTTP 401 "
            "'Missing bearer or basic' (no Authorization header is sent). Open "
            "Edit > Preferences > Add-ons > AI Wear Texture and paste your key "
            "into 'API Key'."
        )


def _uv_coverage_diag(obj, layer_name, uvfield) -> dict:
    """Diagnose the UV raster field's coverage. Main thread only (touches bpy.data).

    Returns a plain dict the worker thread can read to build an error message.
    The key fact is `valid_count`: if 0, bake_vertex_to_uv produces an all-zero
    WearThreshold (it zeroes every invalid texel), so the run must stop here instead
    of silently baking a useless all-black texture.
    """
    import numpy as np
    res = uvfield.res
    valid_n = int(uvfield.valid.sum())
    total = res * res
    diag = {
        "valid_count": valid_n, "total": total,
        "coverage_pct": 100.0 * valid_n / max(1, total),
        "layer": layer_name,
        "degenerate": int(uvfield.degenerate_count),
        "flipped_ratio": float(uvfield.flipped_ratio),
    }
    base = obj.data
    layers = [l.name for l in base.uv_layers] if base.uv_layers else []
    diag["base_layers"] = layers
    bl = base.uv_layers.get(layer_name) if base.uv_layers else None
    if bl is not None:
        nloops = len(base.loops)
        if nloops > 0:
            uv = np.empty(nloops * 2, dtype=np.float32)
            bl.data.foreach_get("uv", uv)
            uv = uv.reshape(nloops, 2)
            diag["base_uv_min"] = float(uv.min())
            diag["base_uv_max"] = float(uv.max())
            diag["base_nonzero_frac"] = float((np.abs(uv).sum(-1) > 1e-6).mean())
            # the rasterizer only scans the [0,1] tile; UVs outside it are skipped
            inside = ((uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0)
                      & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0))
            diag["base_in_tile_frac"] = float(inside.mean())
    return diag


def _uv_empty_error_msg(obj_name: str, diag: dict) -> str:
    """Actionable message for the 'UV covered 0 texels' failure (Q5/Q7 root cause).

    When uvfield.valid is all-False, bake_vertex_to_uv zeroes every texel and the
    shader's smoothstep(T-feather, T+feather, wear_amount) collapses to 1.0
    everywhere (T=0 <= wear_amount) — the whole surface reads as fully worn /
    dark. Rather than ship that, we stop and tell the user why.
    """
    nz = diag.get("base_nonzero_frac", -1.0)
    in_tile = diag.get("base_in_tile_frac", -1.0)
    if 0.0 <= nz < 0.5:
        why = ("The UV layer's coordinates are all (or nearly all) zero — the "
               "auto-unwrap never filled it. Re-run with Mode B, or if you already "
               "are, the mesh may be non-manifold / have degenerate faces that "
               "Smart UV Project skipped. Check the UV QC report in the log.")
    elif 0.0 <= in_tile < 0.05:
        why = (f"The UV coordinates lie OUTSIDE the [0,1] tile the rasterizer scans "
               f"(only {in_tile*100:.1f}% of loops are inside it). Scale or translate "
               f"the existing UV into the unit square, or switch to Mode B "
               f"(auto-unwrap packs islands into [0,1]).")
    else:
        why = ("The UV layer has coordinates but no triangle covered a texel — "
               "possibly all zero-area (degenerate) UV triangles. Re-unwrap.")
    return (
        f"UV rasterization covered 0 of {diag['total']} texels — WearThreshold would "
        f"be all-black (and the render would read as fully worn everywhere), so the "
        f"pipeline stopped. Layer '{diag['layer']}' on '{obj_name}': base UV range "
        f"[{diag.get('base_uv_min', 0.0):.3f}, {diag.get('base_uv_max', 0.0):.3f}], "
        f"nonzero {nz*100:.1f}%, in-tile {in_tile*100:.1f}%, "
        f"{diag.get('degenerate', 0)} degenerate tris. {why} "
        f"Layers on object: {diag.get('base_layers')}."
    )


def _save_view_diff_mask(path: str, mask: np.ndarray) -> str:
    """Persist a screen-space clean/worn diff mask as a viewable RGB image.

    ``utils.save_image`` treats a 2D array as a one-channel image and therefore
    writes the scalar only into R.  These per-view QA files are meant to be
    inspected directly in an image viewer, so replicate the scalar into RGB and
    keep alpha opaque.  PNG16 preserves the interpolated float mask used by the
    projection stage rather than reducing it to an 8-bit debug preview.
    """
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    rgba = np.empty((m.shape[0], m.shape[1], 4), dtype=np.float32)
    rgba[..., 0] = m
    rgba[..., 1] = m
    rgba[..., 2] = m
    rgba[..., 3] = 1.0
    return utils.save_image(path, rgba, "PNG16")


def _binary_erode4(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        if not out.any():
            break
        nxt = np.zeros_like(out)
        nxt[1:-1, 1:-1] = (out[1:-1, 1:-1]
                              & out[:-2, 1:-1] & out[2:, 1:-1]
                              & out[1:-1, :-2] & out[1:-1, 2:])
        out = nxt
    return out


def _binary_dilate4(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant")
        out = (padded[1:-1, 1:-1] | padded[:-2, 1:-1]
               | padded[2:, 1:-1] | padded[1:-1, :-2]
               | padded[1:-1, 2:])
    return out


def _build_geometry_inpaint_mask(depth: np.ndarray, coverage: np.ndarray,
                                 edge_width: int) -> np.ndarray:
    """Build a screen-space local-redraw mask from silhouette + depth edges."""
    cov = np.asarray(coverage, dtype=bool)
    width = max(1, int(edge_width))
    silhouette = cov & ~_binary_erode4(cov, width)

    z = np.asarray(depth, dtype=np.float32)
    grad = np.zeros_like(z)
    valid_x = cov[:, 1:] & cov[:, :-1]
    valid_y = cov[1:, :] & cov[:-1, :]
    dx = np.zeros_like(z)
    dy = np.zeros_like(z)
    with np.errstate(invalid="ignore"):
        delta_x = np.abs(z[:, 1:] - z[:, :-1])
        delta_y = np.abs(z[1:, :] - z[:-1, :])
    delta_x = np.nan_to_num(delta_x, nan=0.0, posinf=0.0, neginf=0.0)
    delta_y = np.nan_to_num(delta_y, nan=0.0, posinf=0.0, neginf=0.0)
    dx[:, 1:] = np.where(valid_x, delta_x, 0.0)
    dy[1:, :] = np.where(valid_y, delta_y, 0.0)
    grad = np.maximum(dx, dy)
    samples = grad[(grad > 0) & np.isfinite(grad)]
    if samples.size:
        threshold = float(np.percentile(samples, 92.0))
        depth_edges = cov & (grad >= max(threshold, 1e-7))
        depth_edges = _binary_dilate4(depth_edges, max(1, width // 3))
    else:
        depth_edges = np.zeros_like(cov)
    return (silhouette | depth_edges).astype(np.float32)


def _save_comfy_inpaint_mask(path: str, mask: np.ndarray) -> str:
    """Save an RGBA mask matching ComfyUI LoadImage's inverse-alpha convention."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    rgba = np.empty((m.shape[0], m.shape[1], 4), dtype=np.float32)
    rgba[..., :3] = m[..., None]
    rgba[..., 3] = 1.0 - m  # ComfyUI LoadImage outputs MASK = 1 - alpha.
    return utils.save_image(path, rgba, "PNG8")


def _save_uv_texture(path: str, rgba: np.ndarray, fmt: str) -> str:
    """Save a UV-domain array with Blender/image row orientation corrected.

    UV raster fields use row 0 for V=0 (the bottom of an image texture), while
    PNG scanline 0 is the top row.  Flip only UV-domain outputs here; screen
    captures and per-view diff masks are already top-row-first.
    """
    return utils.save_image(path, np.ascontiguousarray(rgba[::-1]), fmt)


def _effective_weights(snap: dict) -> dict:
    weights = dict(snap["weights"])
    if not snap.get("use_ai_mask", True):
        weights["w_ai"] = 0.0
    if not snap.get("use_geometry_prior", True):
        weights["w_convex"] = 0.0
        weights["w_expose"] = 0.0
        weights["w_cavity"] = 0.0
    return weights


def _select_view_context(mode: str, supported: bool,
                         first_anchor: str | None,
                         previous_worn: str | None) -> str | None:
    """Return the exact worn image attached to the next AI request."""
    if not supported or mode == "NONE":
        return None
    if mode == "FIRST_ANCHOR":
        return first_anchor
    if mode == "PREVIOUS_VIEW":
        return previous_worn
    return None


def _experiment_config(snap: dict, n_views: int, replayed: bool) -> dict:
    keys = (
        "uv_mode", "target_uv_layer", "work_resolution",
        "camera_preset", "camera_count", "render_resolution", "view_context_mode", "provider",
        "model", "strategy", "prompt", "seed", "lock_seed", "weights",
        "gamma", "alpha", "noise_amp", "noise_scale", "use_barrier",
        "mat_penalty", "coverage_target", "use_comfy_inpaint",
        "inpaint_edge_width", "seam_fuse", "seam_diffuse", "use_padding",
        "padding", "use_ai_mask", "use_geometry_prior",
        "use_topology_growth", "export_format", "wear_amount", "feather",
        "preset_name",
    )
    cfg = {key: snap.get(key) for key in keys}
    cfg["effective_weights"] = _effective_weights(snap)
    cfg["effective_view_count"] = int(n_views)
    cfg["replayed"] = bool(replayed)
    return cfg


def _save_experiment_bundle(cache_dir: str, snap: dict, job,
                            n_views: int, ai_field: np.ndarray,
                            wearthreshold_before: np.ndarray,
                            wearthreshold_after: np.ndarray,
                            replayed: bool) -> str:
    """Persist comparable ablation inputs/outputs without changing main outputs."""
    import json
    import shutil

    label = utils.safe_name(snap.get("experiment_label") or "experiment")
    exp_dir = os.path.join(cache_dir, "experiments", f"{label}_{job.id}")
    utils.ensure_dir(exp_dir)

    def _gray(field, name):
        rgba = np.empty((*field.shape, 4), dtype=np.float32)
        rgba[..., 0] = field
        rgba[..., 1] = field
        rgba[..., 2] = field
        rgba[..., 3] = 1.0
        _save_uv_texture(os.path.join(exp_dir, name), rgba, "PNG16")

    _gray(ai_field, "M_Wear.png")
    _gray(wearthreshold_before, "WearThreshold_before_seam_padding.png")
    _gray(wearthreshold_after, "WearThreshold_after_seam_padding.png")
    for name in ("WearThreshold.png", "AIWear_WornTex.png", "AIWear_UVSnapshot.npz"):
        src = os.path.join(cache_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(exp_dir, name))

    # Keep the experiment quantitative output deliberately small: these are
    # visual production comparisons, not a metric-optimization benchmark.
    import time
    metrics = {"elapsed_seconds": float(time.time() - job.started)}

    # Context/camera experiments need the actual per-view sequence, not only
    # the final UV textures. Snapshot all reviewable view assets and manifest.
    source_views = os.path.join(cache_dir, "views")
    target_views = os.path.join(exp_dir, "views")
    if os.path.isdir(source_views):
        utils.ensure_dir(target_views)
        prefixes = ("clean_V", "worn_V", "diff_mask_V", "inpaint_mask_V")
        for entry in os.scandir(source_views):
            if entry.is_file() and (entry.name == "views.json"
                                    or entry.name.startswith(prefixes)):
                shutil.copy2(entry.path, os.path.join(target_views, entry.name))
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_experiment_config(snap, n_views, replayed), f,
                  ensure_ascii=False, indent=2)
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return exp_dir


UV_SNAPSHOT_FILENAME = "AIWear_UVSnapshot.npz"


def _save_uv_snapshot(obj, layer_name: str, path: str) -> str:
    """Persist the exact per-loop Wear UV needed to replay cached projections.

    A Mode-B UV layer lives in the .blend, but users commonly run the pipeline
    and close/reopen an older unsaved copy. The cached views then outlive the
    generated layer. Camera matrices alone are not sufficient for replay: the
    exact UV coordinates are part of the cache contract too.
    """
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Cannot snapshot Wear UV: target is not a mesh.")
    layer = obj.data.uv_layers.get(layer_name)
    if layer is None:
        raise RuntimeError(f"Cannot snapshot missing UV layer '{layer_name}'.")
    uv = np.empty(len(layer.data) * 2, dtype=np.float32)
    layer.data.foreach_get("uv", uv)
    utils.ensure_dir(os.path.dirname(path))
    np.savez_compressed(
        path,
        schema=np.asarray([1], dtype=np.int32),
        layer=np.asarray([layer_name]),
        uv=uv,
        vertices=np.asarray([len(obj.data.vertices)], dtype=np.int64),
        loops=np.asarray([len(obj.data.loops)], dtype=np.int64),
        polygons=np.asarray([len(obj.data.polygons)], dtype=np.int64),
    )
    return path


def _restore_uv_snapshot(obj, layer_name: str, path: str) -> dict:
    """Restore a missing Wear UV from cache, with topology validation.

    New caches contain an exact compressed per-loop snapshot. For legacy
    caches, recovery is allowed only when the object has exactly one UV layer;
    that conservative fallback matches projects where ``AI_WearUV`` was an
    unsaved copy of the sole authored UV. Ambiguous multi-UV meshes fail loud
    instead of silently sampling the wrong atlas.
    """
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Replay target is not a mesh.")
    existing = obj.data.uv_layers.get(layer_name)
    if existing is not None:
        return {"restored": False, "method": "existing", "source": layer_name}

    mesh = obj.data
    uv = None
    method = None
    source_name = None
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as data:
            expected = (
                int(data["vertices"][0]),
                int(data["loops"][0]),
                int(data["polygons"][0]),
            )
            actual = (len(mesh.vertices), len(mesh.loops), len(mesh.polygons))
            if actual != expected:
                raise RuntimeError(
                    f"Replay: cached UV snapshot topology {expected} does not "
                    f"match current mesh {actual}. Re-run the full pipeline.")
            uv = np.asarray(data["uv"], dtype=np.float32).reshape(-1)
        if uv.size != len(mesh.loops) * 2:
            raise RuntimeError(
                f"Replay: cached UV snapshot has {uv.size // 2} loops, current "
                f"mesh has {len(mesh.loops)}. Re-run the full pipeline.")
        method = "cached_snapshot"
        source_name = os.path.basename(path)
    else:
        layers = list(mesh.uv_layers)
        if len(layers) != 1:
            names = [layer.name for layer in layers]
            raise RuntimeError(
                f"Replay: UV layer '{layer_name}' is missing and this legacy "
                f"cache has no UV snapshot. Existing UV layers are {names}; "
                f"automatic recovery is ambiguous. Re-run the full pipeline.")
        source = layers[0]
        uv = np.empty(len(source.data) * 2, dtype=np.float32)
        source.data.foreach_get("uv", uv)
        method = "legacy_single_uv_copy"
        source_name = source.name

    restored = mesh.uv_layers.new(name=layer_name, do_init=False)
    if restored.name != layer_name:
        mesh.uv_layers.remove(restored)
        raise RuntimeError(
            f"Replay could not recreate UV layer with exact name '{layer_name}'.")
    restored.data.foreach_set("uv", uv)
    mesh.uv_layers.active_index = list(mesh.uv_layers).index(restored)
    restored.active_render = True
    mesh.update()
    return {"restored": True, "method": method, "source": source_name}


def _step_preflight(snap, job, bridge):
    import bpy
    from ..uv import qc, unwrap_blender
    obj_name = snap["object_name"]

    def _do():
        obj = bpy.data.objects.get(obj_name) if obj_name else bpy.context.active_object
        if obj is None or obj.type != "MESH":
            raise RuntimeError("Active object must be a mesh.")
        # UV/Smart-Project ops need this object active + selected
        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        report = {"mesh": True}
        if snap["uv_mode"] == "MODE_A":
            ok, layer = unwrap_blender.setup_mode_a(obj, snap["target_uv_layer"])
            if not ok:
                raise RuntimeError(layer)
        else:
            ok, layer, uvr = unwrap_blender.setup_mode_b(
                obj, snap["target_uv_layer"] or "AI_WearUV",
                snap["work_resolution"], depsgraph=bpy.context.evaluated_depsgraph_get())
            if not ok:
                raise RuntimeError(layer)
            report["uv_qc"] = uvr
            if not uvr.get("ok"):
                raise RuntimeError(
                    "Mode B could not create a production-quality wear UV: "
                    f"{qc.failure_summary(uvr)}. "
                    "Use a valid authored UV in Mode A or adjust the mesh unwrap.")
        # Compute UV QC once, store for the panel AND print to the console.
        qr = qc.compute_uv_qc(obj, layer, low_res=256)
        report_txt = qc.format_report(qr)
        obj.ai_wear.last_qc_report = report_txt
        print(f"\n[AI Wear] UV QC ({snap['uv_mode']}, layer={layer}):")
        print(report_txt)
        # one-line summary in the Info editor (full report in System Console)
        _ui_log("INFO" if qr.get("ok") else "WARNING",
                f"[AI Wear] UV QC ok={qr.get('ok')} "
                f"overlap={qr.get('overlap_ratio', 0):.2%}")
        return {"layer": layer}
    return bridge.run(_do)


def _save_oblique_preview(obj_name, cache_dir, snap, bridge):
    """Render a fixed oblique comparison shot after a successful run.

    Best-effort: a failed comparison render must never fail the run. The wear
    overlay is already attached to the object by this point, so the fixed
    oblique camera captures the actual worn result. Named by the experiment
    preset name under ``<cache>/experiments/oblique_renders``. Generate and
    Replay intentionally update the same per-preset comparison image.

    Returns a plain metadata dict on success, or ``None`` if the best-effort
    render failed.
    """
    def _render():
        import bpy
        obj = bpy.data.objects.get(obj_name)
        if obj is None or obj.type != "MESH":
            return
        from ..render import oblique
        label = utils.safe_name(
            snap.get("preset_name") or snap.get("experiment_label") or "render")
        out_dir = os.path.join(cache_dir, "experiments", "oblique_renders")
        out_path = os.path.join(out_dir, label + "_oblique.png")
        oblique.render_oblique(bpy.context.scene, obj, out_path)
        return {"path": out_path, "camera": oblique.CAMERA_NAME}
    try:
        result = bridge.run(_render)
        if result:
            _log_text(
                f"[AI Wear] Fixed oblique render saved to '{result['path']}' "
                f"with camera '{result['camera']}'.")
        return result
    except Exception:
        # A comparison shot must not break a successful run.
        traceback.print_exc()
        return None


def _run_pipeline(job, snap, bridge):
    """The full worker pipeline. May only touch bpy via bridge.run()."""
    import bpy
    from ..uv import qc as uvqc
    from ..uv.rasterizer import build_uv_field
    from ..uv import seam_registry
    from ..render import view_sampler, passes
    from ..ai.base import get_provider, GenRequest, ProviderError, ProviderCapabilities
    from ..ai import base as ai_base
    from ..surface import projection, fusion, geometry_prior, wear_growth
    from ..shader import wear_nodegroup as shader

    ai_base.ensure_providers_registered()

    # A full Generate replaces the replay source and main outputs. Preserve
    # only explicit experiment snapshots; Replay itself never calls this path.
    cache_dir = job_cache.clear_current_run(snap["obj_uuid"])
    job.meta["cache_dir"] = cache_dir

    # 1. Preflight + UV
    job.state = JobState.RENDER; job.stage = JobStage.PREFLIGHT
    job.message = "Preflight + UV setup…"; job.progress = 0.02
    pf = _step_preflight(snap, job, bridge)
    layer_name = pf["layer"]
    obj_name = snap["object_name"]
    _check_cancel(job)

    # 2. Build UV field (main thread; returns pure numpy)
    job.stage = JobStage.UV; job.message = "Building UV raster field…"; job.progress = 0.05

    def _build_uv():
        obj = bpy.data.objects.get(obj_name)
        uvf = build_uv_field(obj, layer_name, snap["work_resolution"],
                             depsgraph=bpy.context.evaluated_depsgraph_get())
        if uvf is None:
            layers = ([l.name for l in obj.data.uv_layers]
                      if (obj.data and obj.data.uv_layers) else [])
            raise RuntimeError(
                f"No usable UV layer on '{obj_name}' named '{layer_name}'. "
                f"Existing layers: {layers}. Use Mode B (auto-unwrap) or pick a "
                f"valid UV layer in the panel.")
        diag = _uv_coverage_diag(obj, layer_name, uvf)
        if diag["valid_count"] == 0:
            raise RuntimeError(_uv_empty_error_msg(obj_name, diag))
        _log_text(f"[AI Wear] UV field coverage {diag['coverage_pct']:.1f}% "
                  f"({diag['valid_count']}/{diag['total']} texels), "
                  f"layer='{layer_name}', {diag['degenerate']} degenerate tris.")
        return uvf, diag
    uvfield, uv_diag = bridge.run(_build_uv)
    job.meta["uv_coverage_pct"] = uv_diag["coverage_pct"]
    _check_cancel(job)

    res = uvfield.res
    # flat texel arrays for accumulation
    texel_pos = uvfield.reconstruct_positions().reshape(-1, 3)
    texel_norm = uvfield.reconstruct_normals().reshape(-1, 3)
    valid_flat = uvfield.valid.reshape(-1)
    valid_idx = np.nonzero(valid_flat)[0]
    acc_mask = np.zeros(res * res, dtype=np.float32)
    acc_w = np.zeros(res * res, dtype=np.float32)
    count = np.zeros(res * res, dtype=np.float32)
    # worn-texture accumulator (RGB projected to UV, same weighting as the mask)
    acc_rgb = np.zeros((res * res, 3), dtype=np.float32)
    acc_rgb_w = np.zeros(res * res, dtype=np.float32)
    acc_support = np.zeros(res * res, dtype=np.float32)
    exposure_count = np.zeros(len(uvfield.vpos), dtype=np.float32)

    # 3. Cameras
    job.stage = JobStage.CAPTURE; job.message = "Generating views…"; job.progress = 0.08
    cam_names = bridge.run(lambda: [c.name for c in view_sampler.generate_views(
        bpy.context.scene, bpy.data.objects.get(obj_name),
        snap["camera_preset"], snap["camera_count"],
        depsgraph=bpy.context.evaluated_depsgraph_get())])
    n_views = len(cam_names)
    job.meta["effective_view_count"] = n_views
    if n_views == 0:
        raise RuntimeError("No cameras generated.")
    _check_cancel(job)

    # provider + capabilities (validate config up front)
    provider = get_provider(snap["provider"])
    cap = provider.capabilities()
    cfg_probs = provider.validate_config(snap["prefs_obj"])
    if cfg_probs:
        raise RuntimeError("Provider config: " + "; ".join(cfg_probs))

    # API-key preflight: a missing key would otherwise surface later as a cryptic
    # "HTTP 401 Missing bearer or basic" (the OpenAI-compat path only attaches
    # the Authorization header when the key is truthy). ComfyUI is local and
    # commonly needs no key.
    _require_api_key(snap["provider"], snap["prefs_obj"])

    # cache dirs (the non-experiment contents were cleared at run start)
    out_dir = os.path.join(cache_dir, "views")
    utils.ensure_dir(out_dir)
    uv_snapshot_path = os.path.join(cache_dir, UV_SNAPSHOT_FILENAME)
    bridge.run(lambda: _save_uv_snapshot(
        bpy.data.objects.get(obj_name), layer_name, uv_snapshot_path))

    # seed policy
    base_seed = snap["seed"] if (snap["seed"] and snap["lock_seed"]) else \
        int.from_bytes(os.urandom(4), "big")
    seed = base_seed

    worn_paths: list[str] = []
    first_anchor_path = None
    previous_worn_path = None
    context_mode = snap.get("view_context_mode", "FIRST_ANCHOR")
    context_supported = bool(cap.supports_multi_turn_context
                             and cap.max_reference_images > 1)
    job.meta["view_context_mode"] = context_mode
    job.meta["view_context_supported"] = context_supported
    if context_mode != "NONE" and not context_supported:
        warning = (
            f"Provider '{snap['provider']}' cannot send an additional worn-view "
            f"reference; '{context_mode}' will run as independent views.")
        job.meta["view_context_warning"] = warning
        _log_text("[AI Wear] WARNING: " + warning)
    # Per-view projection records, saved to views.json so the downstream
    # (mask → projection → fusion → WearThreshold → bake) can be replayed later
    # WITHOUT re-running AI: the replay reuses the exact camera matrices the
    # clean/worn images were rendered from (regenerating cameras would give a
    # different framing — esp. after the half-diagonal framing fix — and the
    # screen-space masks would no longer align with the surface). Q3.
    view_records: list[dict] = []

    for vi, cam_name in enumerate(cam_names):
        _check_cancel(job)
        job.message = f"View {vi+1}/{n_views}: rendering clean…"
        job.progress = _capture_view_progress(vi, n_views, 0.0)
        clean_png = os.path.join(out_dir, f"clean_V{vi}.png")
        bridge.run(lambda cn=cam_name, cp=clean_png: passes.render_clean(
            bpy.context.scene, bpy.data.objects.get(cn), cp, snap["render_resolution"], job,
            lighting="unlit", unlit_object=bpy.data.objects.get(obj_name)))

        # Projection/depth are known before AI. ComfyUI can therefore receive a
        # deterministic geometry-derived inpaint mask rather than repainting the
        # whole view and changing lighting/background/silhouette.
        cam_data = bridge.run(lambda cn=cam_name: _cam_proj_data(
            bpy.data.objects.get(cn), bpy.data.objects.get(obj_name),
            snap["render_resolution"]))
        view = cam_data["view"]
        cam_loc = cam_data["cam_loc"]
        lens = cam_data["lens"]
        sensor_w = cam_data["sensor_w"]
        rx = ry = snap["render_resolution"]
        depth_buf, screen_coverage, screen_normals = projection.rasterize_screen_depth_normals(
            uvfield.vpos, uvfield.tri_vert, view, lens, sensor_w, rx, ry)
        inpaint_mask_png = None
        if (snap["provider"] == "COMFYUI" and snap.get("use_comfy_inpaint", True)
                and cap.supports_mask):
            inpaint_mask = _build_geometry_inpaint_mask(
                depth_buf, screen_coverage, snap.get("inpaint_edge_width", 12))
            inpaint_mask_png = os.path.join(out_dir, f"inpaint_mask_V{vi}.png")
            _save_comfy_inpaint_mask(inpaint_mask_png, inpaint_mask)

        # AI generate (worker thread, HTTP only)
        job.state = JobState.AI; job.stage = JobStage.AI_SUBMIT
        job.message = f"View {vi+1}/{n_views}: AI generating…"
        job.progress = _capture_view_progress(vi, n_views, 0.20)
        refs = []
        context_path = _select_view_context(
            context_mode, context_supported,
            first_anchor_path, previous_worn_path)
        if context_path:
            refs = [context_path]
        view_prompt = snap["prompt"]
        if refs:
            view_prompt += (
                " The first input image is the current clean target. The second "
                "image is a previous worn-view style reference only: match its "
                "wear material, color, scratch scale and severity, but preserve "
                "the current target image's camera, silhouette and geometry.")
        req = GenRequest(
            clean_image_path=clean_png,
            prompt=view_prompt,
            seed=seed,
            output_size=snap["render_resolution"],
            reference_images=refs,
            reference_labels=["context_worn"] if refs else [],
            mask_path=inpaint_mask_png,
            depth_path=None,
            normal_path=None,
            workflow_path=getattr(snap["prefs_obj"], "workflow_path", None) or None,
            node_mapping={
                "clean_image_node": snap["prefs_obj"].clean_image_node,
                "mask_image_node": snap["prefs_obj"].mask_image_node,
                "prompt_node": snap["prefs_obj"].prompt_node,
                "seed_node": snap["prefs_obj"].seed_node,
                "output_node": snap["prefs_obj"].output_node,
            } if snap["provider"] == "COMFYUI" else {},
            should_cancel=lambda: job.cancel,
            on_progress=lambda p, m, v=vi, n=n_views:
                _set_ai_progress(job, p, m, v, n),
            out_dir=out_dir,
        )
        result = provider.generate(req, snap["prefs_obj"], None)
        worn_png = result.worn_image_path
        worn_paths.append(worn_png)
        # Canonical per-view worn image so replay can pair clean_V{i}.png with
        # worn_V{i}.png without guessing the AI provider's random filename.
        worn_canon = os.path.join(out_dir, f"worn_V{vi}.png")
        try:
            import shutil
            shutil.copyfile(worn_png, worn_canon)
        except Exception:
            worn_canon = worn_png  # fall back to the provider's own path
        if first_anchor_path is None:
            first_anchor_path = worn_canon
        previous_worn_path = worn_canon
        if not snap["lock_seed"]:
            seed = int.from_bytes(os.urandom(4), "big")

        # Screen mask (main thread image load) + accumulation (worker numpy)
        job.state = JobState.BUILD; job.stage = JobStage.MASK
        job.message = f"View {vi+1}/{n_views}: projecting mask…"
        job.progress = _capture_view_progress(vi, n_views, 0.88)
        mask, conf, worn_rgb = bridge.run(lambda cp=clean_png, wp=worn_png:
                                           projection.extract_screen_mask(
                                               cp, wp, snap["render_resolution"],
                                               return_worn_rgb=True))
        mask, worn_rgb, screen_safe = projection.reject_depth_edge_payload(
            mask, worn_rgb, depth_buf, screen_coverage, cam_data["radius"],
            normal_buf=screen_normals)
        # Save the exact interpolated diff mask consumed by projection next to
        # clean_Vi/worn_Vi.  It is RGB grayscale (not red-channel-only) so it is
        # immediately readable in ordinary image viewers and review tools.
        diff_mask_png = os.path.join(out_dir, f"diff_mask_V{vi}.png")
        _save_view_diff_mask(diff_mask_png, mask)

        # record for replay: the exact projection used for this view's mask
        view_records.append({
            "index": vi,
            "clean": os.path.basename(clean_png),
            "worn": os.path.basename(worn_canon),
            "mask": os.path.basename(diff_mask_png),
            "inpaint_mask": (os.path.basename(inpaint_mask_png)
                             if inpaint_mask_png else None),
            "context_mode": context_mode,
            "context_source": (os.path.basename(context_path)
                               if context_path else None),
            # Persist the exact text sent for this view. In context modes views
            # 1..N include an extra role-clarification clause, so the scene's
            # base prompt alone is not a complete audit record.
            "prompt": view_prompt,
            "view": [list(map(float, row)) for row in np.asarray(view).tolist()],
            "cam_loc": [float(x) for x in np.asarray(cam_loc).ravel().tolist()],
            "lens": float(lens), "sensor_w": float(sensor_w),
            "radius": float(cam_data["radius"]),
            "confidence": float(conf),
            "depth_edge_rejected_ratio": float(
                1.0 - screen_safe.sum() / max(screen_coverage.sum(), 1)),
        })
        # accumulate (worker numpy)
        depth_eps = _visibility_depth_epsilon(cam_data["radius"])
        projection.accumulate_view(
            acc_mask, acc_w, count, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=depth_eps)
        # Accumulate the encoded clean→worn color residual into UV too (same
        # visibility/facing weight as the scalar diff mask).
        projection.accumulate_rgb_view(
            acc_rgb, acc_rgb_w, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, worn_rgb,
            snap["gamma"], depth_eps=depth_eps)
        projection.accumulate_evidence_support(
            acc_support, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=depth_eps)

        # exposure per vertex (worker numpy)
        exposure_count += projection.vertex_visibility(
            uvfield.vpos, view, cam_loc, lens, sensor_w, rx, ry,
            depth_buf, depth_eps)

        job.stage = JobStage.SURFACE
        # early-coverage check
        cov = float((count.reshape(res, res) > 0).sum()) / float(res * res)
        job.meta["coverage"] = cov
        job.progress = _capture_view_progress(vi, n_views, 1.0)

    # cleanup cameras
    bridge.run(lambda: view_sampler.cleanup_views(
        [bpy.data.objects.get(n) for n in cam_names if bpy.data.objects.get(n)]))

    # Persist the per-view projection records so the downstream can be replayed
    # later without re-running AI (Q3). This is written AFTER the cameras are
    # gone — it carries the matrices, so replay never needs to regenerate them.
    try:
        import json
        with open(os.path.join(out_dir, "views.json"), "w", encoding="utf-8") as f:
            json.dump({"resolution": snap["render_resolution"],
                       "work_resolution": snap["work_resolution"],
                       "uv_mode": snap["uv_mode"],
                       "layer": layer_name, "object": obj_name,
                       "view_context_mode": context_mode,
                       "view_context_supported": context_supported,
                       "uv_snapshot": UV_SNAPSHOT_FILENAME,
                       "views": view_records}, f, indent=2)
    except Exception:
        pass

    # 4. Fusion
    _check_cancel(job)
    job.message = "Fusing multi-view field…"; job.progress = 0.58
    fused = fusion.finalize_ai_field(acc_mask, acc_w, count)
    ai_field = fused["ai_field"]
    job.meta["coverage_ratio"] = fused["coverage_ratio"]
    support = acc_support.reshape(res, res)
    observed = count.reshape(res, res)
    consensus = fusion.evidence_consensus(support, observed)
    multi_observed = observed >= 2.0
    nonzero_before_consensus = ai_field > 0.0
    rejected_single_view = multi_observed & (support < 2.0) & nonzero_before_consensus
    ai_field *= consensus
    job.meta["single_view_evidence_rejected_ratio"] = float(
        rejected_single_view.sum() / max(nonzero_before_consensus.sum(), 1))
    # Finalize the encoded color residual, then derive its alpha from actual AI
    # wear evidence (not camera coverage). Coverage is nearly 1 on a successful
    # run and using it as alpha would whiten the whole model at Amount=100.
    fused_rgb = fusion.finalize_rgb_field(acc_rgb, acc_rgb_w)
    rgb_valid = fused_rgb["valid"]
    overlay = fusion.prepare_worn_overlay(fused_rgb["rgb"], ai_field, rgb_valid)
    worn_uv = overlay["rgb"]                # (res,res,3), 0.5 = neutral residual
    wear_alpha = overlay["alpha"]           # (res,res), AI wear evidence
    worn_uv = 0.5 + (worn_uv - 0.5) * consensus[..., None]
    wear_alpha *= consensus
    job.meta["worn_coverage_ratio"] = fused_rgb["coverage_ratio"]

    # 5. Geometry priors
    job.message = "Computing geometry priors…"; job.progress = 0.64
    if snap.get("use_geometry_prior", True):
        convexity = bridge.run(lambda: geometry_prior.signed_convexity(
            bpy.data.objects.get(obj_name)))
        exposure = geometry_prior.normalize_exposure(exposure_count, n_views)
    else:
        convexity = np.zeros(len(uvfield.vpos), dtype=np.float32)
        exposure = np.zeros(len(uvfield.vpos), dtype=np.float32)
    adj_world = bridge.run(lambda: wear_growth.build_topology_graph(
        bpy.data.objects.get(obj_name)))
    adj, world = adj_world

    # 6. WearThreshold
    _check_cancel(job)
    job.message = "Growing WearThreshold topology field…"; job.progress = 0.70
    effective_weights = _effective_weights(snap)
    if snap.get("use_topology_growth", True):
        wt = wear_growth.build_wearthreshold_from_graph(
            uvfield, ai_field, convexity, exposure, adj, world,
            effective_weights, snap["gamma"], snap["alpha"],
            snap["noise_amp"], snap["noise_scale"], snap["use_barrier"],
            snap["mat_penalty"], base_seed,
            float(np.linalg.norm(world.max(0)-world.min(0))))
    else:
        wt = wear_growth.build_direct_wearthreshold(
            uvfield, ai_field, convexity, exposure, effective_weights,
            snap["noise_amp"], snap["noise_scale"], base_seed, world)
    wearthreshold_uv = wt["wearthreshold_uv"]
    wearthreshold_before_post = wearthreshold_uv.copy()

    # 7. Seam fusion and island padding are independent ablation switches.
    if snap["seam_fuse"]:
        job.message = "Fusing UV seams…"; job.progress = 0.80
        registry = bridge.run(lambda: seam_registry.build_seam_registry(
            bpy.data.objects.get(obj_name), layer_name))
        qa_before = seam_registry.seam_qa(wearthreshold_uv, registry, res)
        wearthreshold_uv = seam_registry.fuse_seam(
            wearthreshold_uv, registry, res, snap["seam_diffuse"],
            valid=uvfield.valid)
        worn_uv = seam_registry.fuse_seam_rgb(
            worn_uv, registry, res, snap["seam_diffuse"], valid=rgb_valid)
        wear_alpha = seam_registry.fuse_seam(
            wear_alpha, registry, res, snap["seam_diffuse"], valid=rgb_valid)
        job.meta["seam_before_p95"] = qa_before["p95"]
        bridge.run(lambda: seam_registry.visualize_seams(
            bpy.data.objects.get(obj_name), registry))
    if snap.get("use_padding", True) and snap["padding"] > 0:
        job.message = "Padding UV islands…"; job.progress = 0.84
        wearthreshold_uv, _valid2 = seam_registry.dilate(
            wearthreshold_uv, uvfield.valid, snap["padding"])
        worn_uv, _rgb_valid2 = seam_registry.dilate_rgb(
            worn_uv, rgb_valid, snap["padding"])
        wear_alpha, _wear_valid2 = seam_registry.dilate(
            wear_alpha, rgb_valid, snap["padding"])
    if snap["seam_fuse"]:
        qa_after = seam_registry.seam_qa(wearthreshold_uv, registry, res)
        job.meta["seam_after_p95"] = qa_after["p95"]

    # 8. Bake WearThreshold to image + attach shader (main thread)
    job.state = JobState.BAKE; job.message = "Baking WearThreshold texture…"; job.progress = 0.90
    # Catch-all: an all-zero WearThreshold makes the shader read every surface as
    # fully worn (T=0 <= wear_amount -> smoothstep -> 1 -> all dark). The early
    # uvfield.valid guard catches the most common cause (no UV coverage); this
    # catches the rarer ones — e.g. alpha=1.0 + noise_amp=0.0 on a mesh whose
    # Dijkstra arrival distance is uniform (single component / all verts equidistant
    # from the seed) so T_base collapses to 0. Stop here rather than ship black.
    if not wearthreshold_uv.any():
        vcov = float(uvfield.valid.mean()) if uvfield.valid.size else 0.0
        T_vert = wt.get("wearthreshold_vertex")
        t_min = float(T_vert.min()) if T_vert is not None else float("nan")
        t_max = float(T_vert.max()) if T_vert is not None else float("nan")
        raise RuntimeError(
            f"WearThreshold came out all-zero — the render would read as fully worn "
            f"everywhere, so stopping instead of baking a useless texture. UV "
            f"coverage was {vcov*100:.1f}% (UV was fine); the collapse is in the "
            f"topology-growth step. Vertex WearThreshold T range [{t_min:.4f}, {t_max:.4f}]. "
            f"Most likely: alpha=1.0 with noise_amp=0.0 on a mesh where Dijkstra "
            f"arrival distance is uniform (single disconnected component, or all "
            f"verts equidistant from the seed). Fix: lower alpha toward 0.5–0.7, "
            f"raise noise_amp above 0, and make sure the mesh is manifold/connected "
            f"(build_topology_graph found {len(adj)} verts)."
        )
    fmt = "PNG16" if snap["export_format"] == "PNG16" else ("EXR" if snap["export_format"] == "EXR" else "PNG8")
    rgba = np.zeros((res, res, 4), dtype=np.float32)
    rgba[..., 0] = wearthreshold_uv; rgba[..., 1] = wearthreshold_uv
    rgba[..., 2] = wearthreshold_uv; rgba[..., 3] = 1.0
    wearthreshold_path = os.path.join(cache_dir, "WearThreshold.png")
    bridge.run(lambda: _save_uv_texture(wearthreshold_path, rgba, fmt))
    # M_Wear.png — the directly-reprojected wear mask (the "重投影完的
    # mask"). Persisted for QA/production (the shader gate is WearThreshold, but the
    # user asked where the reprojected mask lives — here it is).
    mask_rgba = np.zeros((res, res, 4), dtype=np.float32)
    mask_rgba[..., 0] = ai_field; mask_rgba[..., 1] = ai_field
    mask_rgba[..., 2] = ai_field; mask_rgba[..., 3] = 1.0
    m_wear_path = os.path.join(cache_dir, "M_Wear.png")
    bridge.run(lambda: _save_uv_texture(m_wear_path, mask_rgba, fmt))
    # AIWear_WornTex.png — bounded encoded clean→worn color residual.
    # RGB: 0.5 + 0.5*clamped_delta, so 0.5 is neutral; A: actual AI wear
    # evidence. Camera coverage must never be used as the appearance alpha.
    # The shader reads it as Non-Color, decodes the signed residual and adds it
    # to the authored Base Color, avoiding baked-view-lighting as albedo.
    wtex = np.zeros((res, res, 4), dtype=np.float32)
    wtex[..., :3] = worn_uv
    wtex[..., 3] = wear_alpha
    worn_tex_path = os.path.join(cache_dir, "AIWear_WornTex.png")
    bridge.run(lambda: _save_uv_texture(worn_tex_path, wtex, fmt))

    job.message = "Attaching wear overlay shader…"; job.progress = 0.95

    def _attach():
        wt_img = bpy.data.images.load(wearthreshold_path)
        wt_img.name = "AIWear_WearThreshold"
        wt_img.colorspace_settings.name = "Non-Color"
        worn_img = bpy.data.images.load(worn_tex_path)
        worn_img.name = "AIWear_WornTex"
        worn_img.colorspace_settings.name = "Non-Color"
        obj = bpy.data.objects.get(obj_name)
        # inject mix(existing_base_color, worn_tex, gate*AI-evidence) into the
        # existing material's Principled BSDF (preserves the rest of the mat).
        mats = shader.attach_wear_overlay(obj, wt_img, worn_img, layer_name)
        if not mats:
            raise RuntimeError(f"Could not attach a wear preview material to '{obj_name}'.")
        for mat in mats:
            shader.set_amount(mat, snap["wear_amount"] / 100.0)
            shader.set_feather(mat, snap["feather"] / 100.0)
        return mats
    bridge.run(_attach)

    job.meta["wearthreshold_path"] = wearthreshold_path
    job.meta["worn_tex_path"] = worn_tex_path
    job.meta["m_wear_path"] = m_wear_path
    job.meta["worn_views"] = worn_paths
    job.meta["diff_masks"] = [os.path.join(out_dir, f"diff_mask_V{i}.png")
                              for i in range(n_views)]
    if snap.get("save_experiment_snapshot", False):
        exp_dir = _save_experiment_bundle(
            cache_dir, snap, job, n_views, ai_field,
            wearthreshold_before_post, wearthreshold_uv, replayed=False)
        job.meta["experiment_dir"] = exp_dir
    oblique_meta = _save_oblique_preview(obj_name, cache_dir, snap, bridge)
    if oblique_meta:
        job.meta["oblique_render_path"] = oblique_meta["path"]
        job.meta["oblique_camera_name"] = oblique_meta["camera"]
    job.state = JobState.DONE
    job.stage = JobStage.EXPORT
    job.progress = 1.0
    job.message = "Done."


def _run_replay(job, snap, bridge):
    """Replay ONLY the downstream (mask → projection → fusion → WearThreshold → bake
    → shader) from the cached per-view clean/worn images + the saved camera
    matrices (views.json). No render, no AI. This is the per-stage testing
    workflow (Q3): once a known-good AI pass is cached, you can iterate on the
    surface field / WearThreshold parameters without spending API budget.

    Mirrors _run_pipeline's downstream exactly so the result is comparable.
    """
    import json
    import bpy
    from ..uv.rasterizer import build_uv_field
    from ..uv import qc as uvqc, seam_registry, unwrap_blender
    from ..surface import projection, fusion, geometry_prior, wear_growth
    from ..shader import wear_nodegroup as shader

    obj_name = snap["object_name"]
    cache_dir = job_cache.object_cache_dir(snap["obj_uuid"])
    out_dir = os.path.join(cache_dir, "views")
    views_json = os.path.join(out_dir, "views.json")
    if not os.path.isfile(views_json):
        raise RuntimeError(
            f"No cached views.json at {views_json}. Run the full pipeline once "
            f"(it now saves per-view camera data + canonical worn_V{{i}}.png) so "
            f"the downstream can be replayed without re-running AI.")
    with open(views_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    view_records = manifest.get("views", [])
    if not view_records:
        raise RuntimeError("views.json has no views. Run the full pipeline once first.")
    layer_name = manifest.get("layer") or (snap["target_uv_layer"] or "AI_WearUV")
    uv_mode = manifest.get("uv_mode", snap.get("uv_mode", "MODE_B"))
    res = int(manifest.get("work_resolution", snap["work_resolution"]))
    uv_snapshot_path = os.path.join(
        cache_dir, manifest.get("uv_snapshot") or UV_SNAPSHOT_FILENAME)

    # validate the cached images exist before doing any work
    missing = []
    for rec in view_records:
        cp = os.path.join(out_dir, rec["clean"]); wp = os.path.join(out_dir, rec["worn"])
        if not os.path.isfile(cp): missing.append(rec["clean"])
        if not os.path.isfile(wp): missing.append(rec["worn"])
    if missing:
        raise RuntimeError(
            f"Missing cached images in {out_dir}: {missing[:6]}. Re-run the full "
            f"pipeline — it saves clean_V{{i}}.png + worn_V{{i}}.png per view.")

    job.state = JobState.BUILD; job.stage = JobStage.UV
    job.message = f"Replay: building UV field ({len(view_records)} cached views)…"; job.progress = 0.05

    def _build_uv():
        obj = bpy.data.objects.get(obj_name)
        restore = _restore_uv_snapshot(obj, layer_name, uv_snapshot_path)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        if restore["restored"]:
            depsgraph.update()
            _log_text(
                f"[AI Wear] Replay restored missing UV layer '{layer_name}' "
                f"via {restore['method']} from '{restore['source']}'.")

        # Caches made by the old Mode-B unwrap can contain a technically valid
        # but unusably sparse atlas (the reported asset occupied only 2.1% of a
        # 1K texture). Repair it before reprojection so existing known-good AI
        # views can be reused without another render/API call.
        if uv_mode == "MODE_B":
            uv_report = uvqc.compute_uv_qc(
                obj, layer_name, depsgraph=depsgraph, low_res=256)
            if not uv_report.get("ok", False):
                ok, repaired_layer, repaired_report = unwrap_blender.setup_mode_b(
                    obj, layer_name, res, depsgraph=depsgraph)
                if not ok or not repaired_report.get("ok", False):
                    raise RuntimeError(
                        "Replay could not repair the cached Mode-B UV: "
                        f"{uvqc.failure_summary(repaired_report)}.")
                layer_name_local = repaired_layer
                restore["mode_b_repaired"] = True
                restore["repair_strategy"] = repaired_report.get("mode_b_strategy")
                restore["repair_source"] = repaired_report.get("source_uv_layer")
                _log_text(
                    f"[AI Wear] Replay repaired sparse Mode-B UV '{layer_name_local}' "
                    f"via {restore['repair_strategy']} "
                    f"(utilization={repaired_report.get('utilization', 0):.1%}).")

        if restore["restored"] or restore.get("mode_b_repaired"):
            # Upgrade legacy/bad caches immediately so later replays restore
            # the exact corrected coordinates.
            _save_uv_snapshot(obj, layer_name, uv_snapshot_path)
        uvf = build_uv_field(obj, layer_name, res,
                             depsgraph=depsgraph)
        if uvf is None:
            raise RuntimeError(
                f"Replay: UV layer '{layer_name}' not found on '{obj_name}'. The "
                f"object's UV may have changed since the cached run — re-run the "
                f"full pipeline, or switch uv_mode.")
        diag = _uv_coverage_diag(obj, layer_name, uvf)
        if diag["valid_count"] == 0:
            raise RuntimeError("Replay: " + _uv_empty_error_msg(obj_name, diag))
        _log_text(f"[AI Wear] Replay UV coverage {diag['coverage_pct']:.1f}% "
                  f"({diag['valid_count']}/{diag['total']} texels).")
        return uvf, diag, restore
    uvfield, _udiag, uv_restore = bridge.run(_build_uv)
    job.meta["uv_restore"] = uv_restore
    _check_cancel(job)

    res = uvfield.res
    texel_pos = uvfield.reconstruct_positions().reshape(-1, 3)
    texel_norm = uvfield.reconstruct_normals().reshape(-1, 3)
    valid_flat = uvfield.valid.reshape(-1)
    valid_idx = np.nonzero(valid_flat)[0]
    acc_mask = np.zeros(res * res, dtype=np.float32)
    acc_w = np.zeros(res * res, dtype=np.float32)
    count = np.zeros(res * res, dtype=np.float32)
    # worn-texture accumulator (RGB projected to UV, same weighting as the mask)
    acc_rgb = np.zeros((res * res, 3), dtype=np.float32)
    acc_rgb_w = np.zeros(res * res, dtype=np.float32)
    acc_support = np.zeros(res * res, dtype=np.float32)
    exposure_count = np.zeros(len(uvfield.vpos), dtype=np.float32)
    n_views = len(view_records)
    job.meta["effective_view_count"] = n_views

    # per-view mask extraction + accumulation, using the SAVED camera matrices
    for i, rec in enumerate(view_records):
        _check_cancel(job)
        job.stage = JobStage.MASK
        job.message = f"Replay view {i+1}/{n_views}: projecting cached mask…"
        job.progress = 0.08 + 0.50 * (i / max(1, n_views))
        clean_png = os.path.join(out_dir, rec["clean"])
        worn_png = os.path.join(out_dir, rec["worn"])
        mask, _conf, worn_rgb = projection.extract_screen_mask(clean_png, worn_png, res, return_worn_rgb=True)
        diff_mask_png = os.path.join(out_dir, f"diff_mask_V{i}.png")
        rec["mask"] = os.path.basename(diff_mask_png)
        view = np.array(rec["view"], dtype=np.float32)
        cam_loc = np.array(rec["cam_loc"], dtype=np.float32)
        lens = float(rec["lens"]); sensor_w = float(rec["sensor_w"])
        radius = float(rec["radius"])
        rx = ry = res
        depth_buf, screen_coverage, screen_normals = projection.rasterize_screen_depth_normals(
            uvfield.vpos, uvfield.tri_vert, view, lens, sensor_w, rx, ry)
        mask, worn_rgb, screen_safe = projection.reject_depth_edge_payload(
            mask, worn_rgb, depth_buf, screen_coverage, radius,
            normal_buf=screen_normals)
        rec["depth_edge_rejected_ratio"] = float(
            1.0 - screen_safe.sum() / max(screen_coverage.sum(), 1))
        # The persisted mask is the exact guarded payload consumed below.
        _save_view_diff_mask(diff_mask_png, mask)
        depth_eps = _visibility_depth_epsilon(radius)
        projection.accumulate_view(
            acc_mask, acc_w, count, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=depth_eps)
        projection.accumulate_rgb_view(
            acc_rgb, acc_rgb_w, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, worn_rgb,
            snap["gamma"], depth_eps=depth_eps)
        projection.accumulate_evidence_support(
            acc_support, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=depth_eps)
        exposure_count += projection.vertex_visibility(
            uvfield.vpos, view, cam_loc, lens, sensor_w, rx, ry,
            depth_buf, depth_eps)
        job.stage = JobStage.SURFACE
        job.meta["coverage"] = float((count.reshape(res, res) > 0).sum()) / float(res * res)

    # Backfill the mask filenames into an older views.json when Replay is used
    # on caches created before per-view diff-mask persistence was added.
    manifest["views"] = view_records
    manifest["uv_mode"] = uv_mode
    manifest["uv_snapshot"] = UV_SNAPSHOT_FILENAME
    with open(views_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Fusion
    _check_cancel(job)
    job.message = "Replay: fusing multi-view field…"; job.progress = 0.60
    fused = fusion.finalize_ai_field(acc_mask, acc_w, count)
    ai_field = fused["ai_field"]
    job.meta["coverage_ratio"] = fused["coverage_ratio"]
    support = acc_support.reshape(res, res)
    observed = count.reshape(res, res)
    consensus = fusion.evidence_consensus(support, observed)
    multi_observed = observed >= 2.0
    nonzero_before_consensus = ai_field > 0.0
    rejected_single_view = multi_observed & (support < 2.0) & nonzero_before_consensus
    ai_field *= consensus
    job.meta["single_view_evidence_rejected_ratio"] = float(
        rejected_single_view.sum() / max(nonzero_before_consensus.sum(), 1))
    # Finalize the encoded color residual and build an AI-evidence alpha.  View
    # coverage is deliberately not used as the material mask.
    fused_rgb = fusion.finalize_rgb_field(acc_rgb, acc_rgb_w)
    rgb_valid = fused_rgb["valid"]
    overlay = fusion.prepare_worn_overlay(fused_rgb["rgb"], ai_field, rgb_valid)
    worn_uv = overlay["rgb"]                # (res,res,3), 0.5 = neutral residual
    wear_alpha = overlay["alpha"]           # (res,res), AI wear evidence
    worn_uv = 0.5 + (worn_uv - 0.5) * consensus[..., None]
    wear_alpha *= consensus
    job.meta["worn_coverage_ratio"] = fused_rgb["coverage_ratio"]

    # Geometry priors
    job.message = "Replay: computing geometry priors…"; job.progress = 0.66
    if snap.get("use_geometry_prior", True):
        convexity = bridge.run(lambda: geometry_prior.signed_convexity(
            bpy.data.objects.get(obj_name)))
        exposure = geometry_prior.normalize_exposure(exposure_count, n_views)
    else:
        convexity = np.zeros(len(uvfield.vpos), dtype=np.float32)
        exposure = np.zeros(len(uvfield.vpos), dtype=np.float32)
    adj, world = bridge.run(lambda: wear_growth.build_topology_graph(
        bpy.data.objects.get(obj_name)))

    # WearThreshold
    _check_cancel(job)
    job.message = "Replay: growing WearThreshold field…"; job.progress = 0.72
    base_seed = snap["seed"] if (snap["seed"] and snap["lock_seed"]) else 0
    effective_weights = _effective_weights(snap)
    if snap.get("use_topology_growth", True):
        wt = wear_growth.build_wearthreshold_from_graph(
            uvfield, ai_field, convexity, exposure, adj, world,
            effective_weights, snap["gamma"], snap["alpha"],
            snap["noise_amp"], snap["noise_scale"], snap["use_barrier"],
            snap["mat_penalty"], base_seed,
            float(np.linalg.norm(world.max(0)-world.min(0))))
    else:
        wt = wear_growth.build_direct_wearthreshold(
            uvfield, ai_field, convexity, exposure, effective_weights,
            snap["noise_amp"], snap["noise_scale"], base_seed, world)
    wearthreshold_uv = wt["wearthreshold_uv"]
    wearthreshold_before_post = wearthreshold_uv.copy()

    # Seam fusion and island padding are independent ablation switches.
    if snap["seam_fuse"]:
        job.message = "Replay: fusing seams…"; job.progress = 0.82
        registry = bridge.run(lambda: seam_registry.build_seam_registry(
            bpy.data.objects.get(obj_name), layer_name))
        qa_before = seam_registry.seam_qa(wearthreshold_uv, registry, res)
        wearthreshold_uv = seam_registry.fuse_seam(
            wearthreshold_uv, registry, res, snap["seam_diffuse"],
            valid=uvfield.valid)
        worn_uv = seam_registry.fuse_seam_rgb(
            worn_uv, registry, res, snap["seam_diffuse"], valid=rgb_valid)
        wear_alpha = seam_registry.fuse_seam(
            wear_alpha, registry, res, snap["seam_diffuse"], valid=rgb_valid)
        job.meta["seam_before_p95"] = qa_before["p95"]
        bridge.run(lambda: seam_registry.visualize_seams(
            bpy.data.objects.get(obj_name), registry))
    if snap.get("use_padding", True) and snap["padding"] > 0:
        job.message = "Replay: padding UV islands…"; job.progress = 0.86
        wearthreshold_uv, _valid2 = seam_registry.dilate(
            wearthreshold_uv, uvfield.valid, snap["padding"])
        worn_uv, _rgb_valid2 = seam_registry.dilate_rgb(
            worn_uv, rgb_valid, snap["padding"])
        wear_alpha, _wear_valid2 = seam_registry.dilate(
            wear_alpha, rgb_valid, snap["padding"])
    if snap["seam_fuse"]:
        qa_after = seam_registry.seam_qa(wearthreshold_uv, registry, res)
        job.meta["seam_after_p95"] = qa_after["p95"]

    # Bake + attach shader
    job.state = JobState.BAKE; job.message = "Replay: baking WearThreshold…"; job.progress = 0.92
    if not wearthreshold_uv.any():
        raise RuntimeError(
            "Replay: WearThreshold came out all-zero (render would read as fully worn). "
            "UV coverage was fine; the topology-growth collapsed — see the full "
            "pipeline's all-zero guard for the param fixes (lower alpha, raise "
            "noise_amp, check mesh connectivity).")
    fmt = "PNG16" if snap["export_format"] == "PNG16" else ("EXR" if snap["export_format"] == "EXR" else "PNG8")
    rgba = np.zeros((res, res, 4), dtype=np.float32)
    rgba[..., 0] = wearthreshold_uv; rgba[..., 1] = wearthreshold_uv
    rgba[..., 2] = wearthreshold_uv; rgba[..., 3] = 1.0
    wearthreshold_path = os.path.join(cache_dir, "WearThreshold.png")
    bridge.run(lambda: _save_uv_texture(wearthreshold_path, rgba, fmt))
    # M_Wear.png — the directly-reprojected wear mask (the "重投影完的
    # mask"). Persisted for QA/production (the shader gate is WearThreshold, but the
    # user asked where the reprojected mask lives — here it is).
    mask_rgba = np.zeros((res, res, 4), dtype=np.float32)
    mask_rgba[..., 0] = ai_field; mask_rgba[..., 1] = ai_field
    mask_rgba[..., 2] = ai_field; mask_rgba[..., 3] = 1.0
    m_wear_path = os.path.join(cache_dir, "M_Wear.png")
    bridge.run(lambda: _save_uv_texture(m_wear_path, mask_rgba, fmt))
    # AIWear_WornTex.png — encoded LINEAR clean→worn color residual.
    # RGB: bounded encoded residual; A: actual AI wear evidence (not coverage).
    wtex = np.zeros((res, res, 4), dtype=np.float32)
    wtex[..., :3] = worn_uv
    wtex[..., 3] = wear_alpha
    worn_tex_path = os.path.join(cache_dir, "AIWear_WornTex.png")
    bridge.run(lambda: _save_uv_texture(worn_tex_path, wtex, fmt))

    job.message = "Replay: attaching wear overlay shader…"; job.progress = 0.97

    def _attach():
        wt_img = bpy.data.images.load(wearthreshold_path)
        wt_img.name = "AIWear_WearThreshold"
        wt_img.colorspace_settings.name = "Non-Color"
        worn_img = bpy.data.images.load(worn_tex_path)
        worn_img.name = "AIWear_WornTex"
        worn_img.colorspace_settings.name = "Non-Color"
        obj = bpy.data.objects.get(obj_name)
        mats = shader.attach_wear_overlay(obj, wt_img, worn_img, layer_name)
        if not mats:
            raise RuntimeError(f"Could not attach a wear preview material to '{obj_name}'.")
        for mat in mats:
            shader.set_amount(mat, snap["wear_amount"] / 100.0)
            shader.set_feather(mat, snap["feather"] / 100.0)
        return mats
    bridge.run(_attach)

    job.meta["wearthreshold_path"] = wearthreshold_path
    job.meta["worn_tex_path"] = worn_tex_path
    job.meta["m_wear_path"] = m_wear_path
    job.meta["replayed"] = True
    if snap.get("save_experiment_snapshot", False):
        exp_dir = _save_experiment_bundle(
            cache_dir, snap, job, n_views, ai_field,
            wearthreshold_before_post, wearthreshold_uv, replayed=True)
        job.meta["experiment_dir"] = exp_dir
    job.message = "Replay: rendering fixed oblique preview…"; job.progress = 0.99
    oblique_meta = _save_oblique_preview(obj_name, cache_dir, snap, bridge)
    if oblique_meta:
        job.meta["oblique_render_path"] = oblique_meta["path"]
        job.meta["oblique_camera_name"] = oblique_meta["camera"]
    job.state = JobState.DONE
    job.stage = JobStage.EXPORT
    job.progress = 1.0
    job.message = "Done (replay)."


_CAPTURE_PROGRESS_START = 0.08
_CAPTURE_PROGRESS_END = 0.55


def _capture_view_progress(view_index: int, view_count: int, local: float) -> float:
    """Map one view's local progress into its non-overlapping global interval."""
    count = max(1, int(view_count))
    index = max(0, min(int(view_index), count - 1))
    fraction = max(0.0, min(1.0, float(local)))
    view_fraction = (index + fraction) / count
    return (_CAPTURE_PROGRESS_START
            + (_CAPTURE_PROGRESS_END - _CAPTURE_PROGRESS_START) * view_fraction)


def _set_ai_progress(job, p, msg, view_index=0, view_count=1):
    # AI providers report progress local to one request.  Put it inside the
    # current view's slice instead of repeatedly mapping every view to the full
    # capture band (which made the displayed percentage jump backwards).
    local = 0.20 + 0.65 * max(0.0, min(1.0, float(p)))
    mapped = _capture_view_progress(view_index, view_count, local)
    job.progress = max(float(job.progress), mapped)
    if msg:
        job.message = f"View {view_index + 1}/{max(1, view_count)}: {msg}"
    job.touch()


def _visibility_depth_epsilon(radius: float) -> float:
    """Scale-relative tolerance for the self-consistent software Z test."""
    return max(float(radius) * 0.02, 1e-3)


def _cam_proj_data(cam, target_obj, res):
    import bpy
    import numpy as np_
    from ..render.view_sampler import compute_framing
    # Framing (bounding-sphere radius) is of the TARGET MESH, not the camera.
    # Cameras carry no geometry, so compute_framing(camera) would raise "Object
    # does not have geometry data" on to_mesh(). The radius feeds the per-view
    # scale used by the software z-buffer visibility tolerance below.
    dg = bpy.context.evaluated_depsgraph_get()
    _center, radius = compute_framing(target_obj, dg)
    return {
        "view": np_.array(cam.matrix_world.inverted(), dtype=np.float32),
        "cam_loc": np_.array(cam.matrix_world.translation, dtype=np.float32),
        "lens": cam.data.lens,
        "sensor_w": cam.data.sensor_width,
        "radius": float(radius),
    }


# --- timer + launch ---------------------------------------------------------

LOG_TEXT_NAME = "ai_wear.log"


def ensure_log_text():
    """Return the persistent Text data block used as the in-UI log (creating it).

    Main thread only (touches bpy.data). Open a Text Editor and select
    'ai_wear.log', or click 'Open Log' in the panel.
    """
    import bpy
    t = bpy.data.texts.get(LOG_TEXT_NAME)
    if t is None:
        t = bpy.data.texts.new(LOG_TEXT_NAME)
    return t


def _log_text(line: str) -> None:
    """Append one line to the 'ai_wear.log' Text data block.

    This is the RELIABLE in-UI log channel. The Info editor is fed by
    self.report() from a timer-invoked bpy.ops.ai_wear.log call, which is both
    filtered by the Info editor's severity toggles (INFO lines are hidden by
    default) and can fail on timer context, silently falling back to print().
    A Text data block is written directly and is visible in any Text Editor
    with no filters — every progress line, AI retry message, and the full
    error + traceback land here. Main thread only (called from the timer).
    """
    try:
        t = ensure_log_text()
        t.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def _log_clear() -> None:
    """Reset the log at the start of each run so it never grows unbounded."""
    try:
        t = ensure_log_text()
        t.clear()
    except Exception:
        pass


def _ui_log(level: str, text: str) -> None:
    """Write one line to the Info editor (visible in-UI) via an internal operator.

    Falls back to the System Console (print) if the operator can't run, e.g.
    during a modal op or with no window context.
    """
    import bpy
    try:
        bpy.ops.ai_wear.log(level=level, message=text)
    except Exception:
        print(f"[AI Wear] ({level}) {text}")


def _console_log(job, force=False):
    """Print progress milestones to the System Console AND the Info editor.

    Dedupes on (state, stage, message) so a 0.25s timer doesn't spam. The Info
    editor line is what most users see (no need to open the System Console);
    the System Console still gets the full text + traceback on error.
    """
    key = (job.state.value, job.stage.value, job.message)
    if not force and getattr(_console_log, "_last", None) == key:
        return
    _console_log._last = key
    pct = max(0.0, min(1.0, job.progress)) * 100.0
    line = f"[AI Wear] {pct:5.1f}%  {job.stage.value:<10}  {job.message}"
    print(line)
    # reliable in-UI log: a Text data block visible in any Text Editor, immune
    # to Info-editor severity filters and timer/bpy.ops context failures.
    _log_text(line)
    level = ("ERROR" if job.state == JobState.ERROR
             else "WARNING" if job.state == JobState.CANCEL
             else "INFO")
    _ui_log(level, line)


def _tick(job, bridge):
    job_cache.clear_finished()
    bridge.pump()
    _console_log(job)
    # live-refresh the viewport so the Generate button progress overlay updates
    try:
        import bpy
        screen = bpy.context.screen
        if screen is not None:
            for area in screen.areas:
                area.tag_redraw()
    except Exception:
        pass
    # auto-stop when terminal
    if job.state in (JobState.DONE, JobState.ERROR, JobState.CANCEL):
        _console_log(job, force=True)
        if job.state == JobState.DONE:
            cov = job.meta.get("coverage_ratio")
            sb = job.meta.get("seam_before_p95"); sa = job.meta.get("seam_after_p95")
            parts = []
            if cov is not None:
                parts.append(f"coverage {cov*100:.1f}%")
            if sa is not None and sb is not None:
                parts.append(f"seam p95 {sb:.4f} -> {sa:.4f}")
            if parts:
                done = f"[AI Wear] Done. " + " ".join(parts)
                print(done)
                _ui_log("INFO", done)
                _log_text(done)
        elif job.state == JobState.ERROR:
            # full traceback: System Console AND the in-UI Text log (so it is
            # visible without opening the hidden System Console). Not sent to
            # the Info editor (too long/noisy for self.report).
            if job.meta.get("traceback"):
                tb = job.meta["traceback"]
                print(tb)
                _log_text(tb)
        return None
    return getattr(bridge, "_interval", 0.25)


def start_pipeline(context) -> str:
    snap = snapshot_context(context)
    _log_clear()  # fresh in-UI Text log per run ('ai_wear.log')
    job = job_cache.create_job()
    bridge = MainThreadBridge()
    bridge._interval = snap["poll_interval"]

    def worker():
        try:
            _run_pipeline(job, snap, bridge)
        except Exception as e:
            job.state = JobState.ERROR
            job.error = str(e)
            kind = getattr(e, "kind", "UNKNOWN")
            job.error_kind = "CANCEL" if kind == "CANCEL" else (
                "API" if kind in ("API",) else (
                "NETWORK" if kind in ("NETWORK",) else "UNKNOWN"))
            if job.error_kind == "CANCEL":
                job.state = JobState.CANCEL
                job.message = "Cancelled."
            else:
                job.message = f"Failed: {e}"
            job.meta["traceback"] = traceback.format_exc()[-1200:]
        finally:
            job.touch()

    t = threading.Thread(target=worker, name="ai_wear_pipeline", daemon=True)
    t.start()

    context.scene.ai_wear.active_job_id = job.id

    def timer_cb():
        return _tick(job, bridge)

    import bpy
    try:
        bpy.app.timers.register(timer_cb, first_interval=snap["poll_interval"])
    except Exception:
        # already a timer registered; the existing one will pick up the job
        pass
    return job.id


def is_running(context) -> bool:
    job = get_active_job(context)
    return job is not None and job.state not in (JobState.DONE, JobState.ERROR, JobState.CANCEL)


def get_active_job(context):
    jid = context.scene.ai_wear.active_job_id
    return job_cache.get_job(jid) if jid else None


def _apply_active_preset(context) -> str:
    """Re-apply the selected preset and return its name.

    Selection normally applies a preset via its update callback, but explicit
    re-application here guarantees Replay cannot accidentally snapshot values
    left over from manual edits or another preset.
    """
    settings = context.scene.ai_wear
    index = settings.active_preset_index
    if not (0 <= index < len(settings.presets)):
        return ""
    from .. import presets
    preset = settings.presets[index]
    preset_name = preset.name
    presets.apply_preset(settings, preset)
    return preset_name


def start_replay(context) -> str:
    """Launch a downstream-only replay (no render, no AI). See _run_replay.

    Reuses the cached clean_V{i}.png + worn_V{i}.png + views.json from the last
    full run on the active object. Lets you iterate on surface/WearThreshold params
    without spending API budget (Q3).
    """
    _log_clear()
    preset_name = _apply_active_preset(context)
    snap = snapshot_context(context)
    if preset_name:
        _log_text(f"[AI Wear] Replay applied current preset '{preset_name}'.")
    job = job_cache.create_job()
    bridge = MainThreadBridge()
    bridge._interval = snap["poll_interval"]

    def worker():
        try:
            _run_replay(job, snap, bridge)
        except Exception as e:
            job.state = JobState.ERROR
            job.error = str(e)
            kind = getattr(e, "kind", "UNKNOWN")
            job.error_kind = "CANCEL" if kind == "CANCEL" else "UNKNOWN"
            if job.error_kind == "CANCEL":
                job.state = JobState.CANCEL
                job.message = "Cancelled."
            else:
                job.message = f"Replay failed: {e}"
            job.meta["traceback"] = traceback.format_exc()[-1200:]
        finally:
            job.touch()

    t = threading.Thread(target=worker, name="ai_wear_replay", daemon=True)
    t.start()
    context.scene.ai_wear.active_job_id = job.id

    def timer_cb():
        return _tick(job, bridge)

    import bpy
    try:
        bpy.app.timers.register(timer_cb, first_interval=snap["poll_interval"])
    except Exception:
        pass
    return job.id
