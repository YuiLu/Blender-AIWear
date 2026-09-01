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
    return {
        "object_name": obj.name if obj else None,
        "obj_uuid": obj_uuid,
        "uv_mode": s.uv_mode,
        "target_uv_layer": s.target_uv_layer,
        "work_resolution": s.work_resolution,
        "texture_size": s.texture_size,
        "camera_preset": s.camera_preset,
        "camera_count": s.camera_count,
        "render_resolution": s.render_resolution,
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
        "seam_fuse": s.seam_fuse,
        "seam_diffuse": s.seam_diffuse_texels,
        "padding": s.padding_texels,
        "export_format": s.export_format,
        "wear_amount": s.wear_amount,
        "feather": s.feather,
        "prefs_obj": PrefsSnap(prefs),  # plain snapshot; safe on worker thread
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
            "into 'API Key'. If 'Key Env Var' is set, that env var must exist in "
            "Blender's own process environment (a Blender launched from the "
            "Start Menu does NOT inherit your terminal's env vars — set a "
            "system-level env var, or just paste the key into the field)."
        )


def _uv_coverage_diag(obj, layer_name, uvfield) -> dict:
    """Diagnose the UV raster field's coverage. Main thread only (touches bpy.data).

    Returns a plain dict the worker thread can read to build an error message.
    The key fact is `valid_count`: if 0, bake_vertex_to_uv produces an all-zero
    WearTime (it zeroes every invalid texel), so the run must stop here instead
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
        f"UV rasterization covered 0 of {diag['total']} texels — WearTime would "
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


def _save_uv_texture(path: str, rgba: np.ndarray, fmt: str) -> str:
    """Save a UV-domain array with Blender/image row orientation corrected.

    UV raster fields use row 0 for V=0 (the bottom of an image texture), while
    PNG scanline 0 is the top row.  Flip only UV-domain outputs here; screen
    captures and per-view diff masks are already top-row-first.
    """
    return utils.save_image(path, np.ascontiguousarray(rgba[::-1]), fmt)


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
                snap["texture_size"], depsgraph=bpy.context.evaluated_depsgraph_get())
            report["uv_qc"] = uvr
            if not uvr.get("ok"):
                job.message = f"Mode B UV QC warns: overlap={uvr.get('overlap_ratio',0):.2%}"
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
    exposure_count = np.zeros(len(uvfield.vpos), dtype=np.float32)

    # 3. Cameras
    job.stage = JobStage.CAPTURE; job.message = "Generating views…"; job.progress = 0.08
    cam_names = bridge.run(lambda: [c.name for c in view_sampler.generate_views(
        bpy.context.scene, bpy.data.objects.get(obj_name),
        snap["camera_preset"], snap["camera_count"],
        depsgraph=bpy.context.evaluated_depsgraph_get())])
    n_views = len(cam_names)
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

    # cache dirs
    cache_dir = job_cache.object_cache_dir(snap["obj_uuid"])
    out_dir = os.path.join(cache_dir, "views")
    utils.ensure_dir(out_dir)

    # seed policy
    base_seed = snap["seed"] if (snap["seed"] and snap["lock_seed"]) else \
        int.from_bytes(os.urandom(4), "big")
    seed = base_seed

    worn_paths: list[str] = []
    anchor_path = None
    # Per-view projection records, saved to views.json so the downstream
    # (mask → projection → fusion → WearTime → bake) can be replayed later
    # WITHOUT re-running AI: the replay reuses the exact camera matrices the
    # clean/worn images were rendered from (regenerating cameras would give a
    # different framing — esp. after the half-diagonal framing fix — and the
    # screen-space masks would no longer align with the surface). Q3.
    view_records: list[dict] = []

    for vi, cam_name in enumerate(cam_names):
        _check_cancel(job)
        job.message = f"View {vi+1}/{n_views}: rendering clean…"
        job.progress = 0.08 + 0.45 * (vi / max(1, n_views))
        clean_png = os.path.join(out_dir, f"clean_V{vi}.png")
        bridge.run(lambda cn=cam_name, cp=clean_png: passes.render_clean(
            bpy.context.scene, bpy.data.objects.get(cn), cp, snap["render_resolution"], job))

        # AI generate (worker thread, HTTP only)
        job.state = JobState.AI; job.stage = JobStage.AI_SUBMIT
        job.message = f"View {vi+1}/{n_views}: AI generating…"
        refs = []
        if anchor_path and cap.max_reference_images >= 2:
            refs = [anchor_path]
        req = GenRequest(
            clean_image_path=clean_png,
            prompt=snap["prompt"],
            seed=seed,
            output_size=snap["render_resolution"],
            reference_images=refs,
            mask_path=None,
            depth_path=None,
            normal_path=None,
            workflow_path=getattr(snap["prefs_obj"], "workflow_path", None) or None,
            node_mapping={
                "clean_image_node": snap["prefs_obj"].clean_image_node,
                "prompt_node": snap["prefs_obj"].prompt_node,
                "seed_node": snap["prefs_obj"].seed_node,
                "output_node": snap["prefs_obj"].output_node,
            } if snap["provider"] == "COMFYUI" else {},
            should_cancel=lambda: job.cancel,
            on_progress=lambda p, m: _set_ai_progress(job, p, m),
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
        if anchor_path is None:
            anchor_path = worn_png
        if not snap["lock_seed"]:
            seed = int.from_bytes(os.urandom(4), "big")

        # Screen mask (main thread image load) + accumulation (worker numpy)
        job.state = JobState.BUILD; job.stage = JobStage.MASK
        job.message = f"View {vi+1}/{n_views}: projecting mask…"
        mask, conf, worn_rgb = bridge.run(lambda cp=clean_png, wp=worn_png:
                                           projection.extract_screen_mask(
                                               cp, wp, snap["render_resolution"],
                                               return_worn_rgb=True))
        # Save the exact interpolated diff mask consumed by projection next to
        # clean_Vi/worn_Vi.  It is RGB grayscale (not red-channel-only) so it is
        # immediately readable in ordinary image viewers and review tools.
        diff_mask_png = os.path.join(out_dir, f"diff_mask_V{vi}.png")
        _save_view_diff_mask(diff_mask_png, mask)

        # camera projection data (main thread)
        cam_data = bridge.run(lambda cn=cam_name: _cam_proj_data(
            bpy.data.objects.get(cn), bpy.data.objects.get(obj_name),
            snap["render_resolution"]))
        view = cam_data["view"]; cam_loc = cam_data["cam_loc"]
        lens = cam_data["lens"]; sensor_w = cam_data["sensor_w"]
        # record for replay: the exact projection used for this view's mask
        view_records.append({
            "index": vi,
            "clean": os.path.basename(clean_png),
            "worn": os.path.basename(worn_canon),
            "mask": os.path.basename(diff_mask_png),
            "view": [list(map(float, row)) for row in np.asarray(view).tolist()],
            "cam_loc": [float(x) for x in np.asarray(cam_loc).ravel().tolist()],
            "lens": float(lens), "sensor_w": float(sensor_w),
            "radius": float(cam_data["radius"]),
            "confidence": float(conf),
        })
        rx = ry = snap["render_resolution"]

        # software z-buffer (worker numpy)
        depth_buf, _cov = projection.rasterize_screen_depth(
            uvfield.vpos, uvfield.tri_vert, view, lens, sensor_w, rx, ry)

        # accumulate (worker numpy)
        projection.accumulate_view(
            acc_mask, acc_w, count, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=max(cam_data["radius"] * 0.02, 1e-3))
        # Accumulate the encoded clean→worn color residual into UV too (same
        # visibility/facing weight as the scalar diff mask).
        projection.accumulate_rgb_view(
            acc_rgb, acc_rgb_w, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, worn_rgb,
            snap["gamma"], depth_eps=max(cam_data["radius"] * 0.02, 1e-3))

        # exposure per vertex (worker numpy)
        exposure_count += projection.vertex_visibility(
            uvfield.vpos, view, cam_loc, lens, sensor_w, rx, ry,
            depth_buf, max(cam_data["radius"] * 0.02, 1e-3))

        job.stage = JobStage.SURFACE
        # early-coverage check
        cov = float((count.reshape(res, res) > 0).sum()) / float(res * res)
        job.meta["coverage"] = cov

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
                       "layer": layer_name, "object": obj_name,
                       "views": view_records}, f, indent=2)
    except Exception:
        pass

    # 4. Fusion
    _check_cancel(job)
    job.message = "Fusing multi-view field…"; job.progress = 0.58
    fused = fusion.finalize_ai_field(acc_mask, acc_w, count)
    ai_field = fused["ai_field"]
    job.meta["coverage_ratio"] = fused["coverage_ratio"]
    # Finalize the encoded color residual, then derive its alpha from actual AI
    # wear evidence (not camera coverage). Coverage is nearly 1 on a successful
    # run and using it as alpha would whiten the whole model at Amount=100.
    fused_rgb = fusion.finalize_rgb_field(acc_rgb, acc_rgb_w)
    rgb_valid = fused_rgb["valid"]
    overlay = fusion.prepare_worn_overlay(fused_rgb["rgb"], ai_field, rgb_valid)
    worn_uv = overlay["rgb"]                # (res,res,3), 0.5 = neutral residual
    wear_alpha = overlay["alpha"]           # (res,res), AI wear evidence
    job.meta["worn_coverage_ratio"] = fused_rgb["coverage_ratio"]

    # 5. Geometry priors
    job.message = "Computing geometry priors…"; job.progress = 0.64
    convexity = bridge.run(lambda: geometry_prior.signed_convexity(
        bpy.data.objects.get(obj_name)))
    exposure = geometry_prior.normalize_exposure(exposure_count, n_views)
    adj_world = bridge.run(lambda: wear_growth.build_topology_graph(
        bpy.data.objects.get(obj_name)))
    adj, world = adj_world

    # 6. WearTime
    _check_cancel(job)
    job.message = "Growing WearTime topology field…"; job.progress = 0.70
    wt = wear_growth.build_weartime_from_graph(
        uvfield, ai_field, convexity, exposure, adj, world,
        snap["weights"], snap["gamma"], snap["alpha"],
        snap["noise_amp"], snap["noise_scale"], snap["use_barrier"],
        snap["mat_penalty"], base_seed, float(np.linalg.norm(world.max(0)-world.min(0))))
    weartime_uv = wt["weartime_uv"]

    # 7. Seam fusion + dilation (worker numpy after registry build on main)
    if snap["seam_fuse"]:
        job.message = "Fusing seams + padding…"; job.progress = 0.82
        registry = bridge.run(lambda: seam_registry.build_seam_registry(
            bpy.data.objects.get(obj_name), layer_name))
        qa_before = seam_registry.seam_qa(weartime_uv, registry, res)
        weartime_uv = seam_registry.fuse_seam(weartime_uv, registry, res, snap["seam_diffuse"])
        weartime_uv, _valid2 = seam_registry.dilate(weartime_uv, uvfield.valid, snap["padding"])
        # Fuse the encoded color residual too (same seam registry, per-channel).
        worn_uv = seam_registry.fuse_seam_rgb(worn_uv, registry, res, snap["seam_diffuse"])
        worn_uv, _rgb_valid2 = seam_registry.dilate_rgb(
            worn_uv, rgb_valid, snap["padding"])
        wear_alpha = seam_registry.fuse_seam(
            wear_alpha, registry, res, snap["seam_diffuse"])
        wear_alpha, _wear_valid2 = seam_registry.dilate(
            wear_alpha, rgb_valid, snap["padding"])
        qa_after = seam_registry.seam_qa(weartime_uv, registry, res)
        job.meta["seam_before_p95"] = qa_before["p95"]
        job.meta["seam_after_p95"] = qa_after["p95"]
        bridge.run(lambda: seam_registry.visualize_seams(
            bpy.data.objects.get(obj_name), registry))

    # 8. Bake WearTime to image + attach shader (main thread)
    job.state = JobState.BAKE; job.message = "Baking WearTime texture…"; job.progress = 0.90
    # Catch-all: an all-zero WearTime makes the shader read every surface as
    # fully worn (T=0 <= wear_amount -> smoothstep -> 1 -> all dark). The early
    # uvfield.valid guard catches the most common cause (no UV coverage); this
    # catches the rarer ones — e.g. alpha=1.0 + noise_amp=0.0 on a mesh whose
    # Dijkstra arrival distance is uniform (single component / all verts equidistant
    # from the seed) so T_base collapses to 0. Stop here rather than ship black.
    if not weartime_uv.any():
        vcov = float(uvfield.valid.mean()) if uvfield.valid.size else 0.0
        T_vert = wt.get("weartime_vertex")
        t_min = float(T_vert.min()) if T_vert is not None else float("nan")
        t_max = float(T_vert.max()) if T_vert is not None else float("nan")
        raise RuntimeError(
            f"WearTime came out all-zero — the render would read as fully worn "
            f"everywhere, so stopping instead of baking a useless texture. UV "
            f"coverage was {vcov*100:.1f}% (UV was fine); the collapse is in the "
            f"topology-growth step. Vertex WearTime T range [{t_min:.4f}, {t_max:.4f}]. "
            f"Most likely: alpha=1.0 with noise_amp=0.0 on a mesh where Dijkstra "
            f"arrival distance is uniform (single disconnected component, or all "
            f"verts equidistant from the seed). Fix: lower alpha toward 0.5–0.7, "
            f"raise noise_amp above 0, and make sure the mesh is manifold/connected "
            f"(build_topology_graph found {len(adj)} verts)."
        )
    fmt = "PNG16" if snap["export_format"] == "PNG16" else ("EXR" if snap["export_format"] == "EXR" else "PNG8")
    rgba = np.zeros((res, res, 4), dtype=np.float32)
    rgba[..., 0] = weartime_uv; rgba[..., 1] = weartime_uv
    rgba[..., 2] = weartime_uv; rgba[..., 3] = 1.0
    weartime_path = os.path.join(cache_dir, "WearTime.png")
    bridge.run(lambda: _save_uv_texture(weartime_path, rgba, fmt))
    # AIWear_Mask.png — the directly-reprojected wear mask (the "重投影完的
    # mask"). Persisted for QA/production (the shader gate is WearTime, but the
    # user asked where the reprojected mask lives — here it is).
    mask_rgba = np.zeros((res, res, 4), dtype=np.float32)
    mask_rgba[..., 0] = ai_field; mask_rgba[..., 1] = ai_field
    mask_rgba[..., 2] = ai_field; mask_rgba[..., 3] = 1.0
    mask_path = os.path.join(cache_dir, "AIWear_Mask.png")
    bridge.run(lambda: _save_uv_texture(mask_path, mask_rgba, fmt))
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
        wt_img = bpy.data.images.load(weartime_path)
        wt_img.name = "AIWear_WearTime"
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

    job.meta["weartime_path"] = weartime_path
    job.meta["worn_tex_path"] = worn_tex_path
    job.meta["worn_mask_path"] = mask_path
    job.meta["worn_views"] = worn_paths
    job.meta["diff_masks"] = [os.path.join(out_dir, f"diff_mask_V{i}.png")
                              for i in range(n_views)]
    job.state = JobState.DONE
    job.stage = JobStage.EXPORT
    job.progress = 1.0
    job.message = "Done."


def _run_replay(job, snap, bridge):
    """Replay ONLY the downstream (mask → projection → fusion → WearTime → bake
    → shader) from the cached per-view clean/worn images + the saved camera
    matrices (views.json). No render, no AI. This is the per-stage testing
    workflow (Q3): once a known-good AI pass is cached, you can iterate on the
    surface field / WearTime parameters without spending API budget.

    Mirrors _run_pipeline's downstream exactly so the result is comparable.
    """
    import json
    import bpy
    from ..uv.rasterizer import build_uv_field
    from ..uv import seam_registry
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
    res = int(manifest.get("work_resolution", snap["work_resolution"]))

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
        uvf = build_uv_field(obj, layer_name, res,
                             depsgraph=bpy.context.evaluated_depsgraph_get())
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
        return uvf, diag
    uvfield, _udiag = bridge.run(_build_uv)
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
    exposure_count = np.zeros(len(uvfield.vpos), dtype=np.float32)
    n_views = len(view_records)

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
        _save_view_diff_mask(diff_mask_png, mask)
        rec["mask"] = os.path.basename(diff_mask_png)
        view = np.array(rec["view"], dtype=np.float32)
        cam_loc = np.array(rec["cam_loc"], dtype=np.float32)
        lens = float(rec["lens"]); sensor_w = float(rec["sensor_w"])
        radius = float(rec["radius"])
        rx = ry = res
        depth_buf, _cov = projection.rasterize_screen_depth(
            uvfield.vpos, uvfield.tri_vert, view, lens, sensor_w, rx, ry)
        projection.accumulate_view(
            acc_mask, acc_w, count, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, mask,
            snap["gamma"], depth_eps=max(radius * 0.02, 1e-3))
        projection.accumulate_rgb_view(
            acc_rgb, acc_rgb_w, texel_pos, texel_norm, valid_idx,
            view, cam_loc, lens, sensor_w, rx, ry, depth_buf, worn_rgb,
            snap["gamma"], depth_eps=max(radius * 0.02, 1e-3))
        exposure_count += projection.vertex_visibility(
            uvfield.vpos, view, cam_loc, lens, sensor_w, rx, ry,
            depth_buf, max(radius * 0.02, 1e-3))
        job.stage = JobStage.SURFACE
        job.meta["coverage"] = float((count.reshape(res, res) > 0).sum()) / float(res * res)

    # Backfill the mask filenames into an older views.json when Replay is used
    # on caches created before per-view diff-mask persistence was added.
    manifest["views"] = view_records
    with open(views_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Fusion
    _check_cancel(job)
    job.message = "Replay: fusing multi-view field…"; job.progress = 0.60
    fused = fusion.finalize_ai_field(acc_mask, acc_w, count)
    ai_field = fused["ai_field"]
    job.meta["coverage_ratio"] = fused["coverage_ratio"]
    # Finalize the encoded color residual and build an AI-evidence alpha.  View
    # coverage is deliberately not used as the material mask.
    fused_rgb = fusion.finalize_rgb_field(acc_rgb, acc_rgb_w)
    rgb_valid = fused_rgb["valid"]
    overlay = fusion.prepare_worn_overlay(fused_rgb["rgb"], ai_field, rgb_valid)
    worn_uv = overlay["rgb"]                # (res,res,3), 0.5 = neutral residual
    wear_alpha = overlay["alpha"]           # (res,res), AI wear evidence
    job.meta["worn_coverage_ratio"] = fused_rgb["coverage_ratio"]

    # Geometry priors
    job.message = "Replay: computing geometry priors…"; job.progress = 0.66
    convexity = bridge.run(lambda: geometry_prior.signed_convexity(
        bpy.data.objects.get(obj_name)))
    exposure = geometry_prior.normalize_exposure(exposure_count, n_views)
    adj, world = bridge.run(lambda: wear_growth.build_topology_graph(
        bpy.data.objects.get(obj_name)))

    # WearTime
    _check_cancel(job)
    job.message = "Replay: growing WearTime field…"; job.progress = 0.72
    base_seed = snap["seed"] if (snap["seed"] and snap["lock_seed"]) else 0
    wt = wear_growth.build_weartime_from_graph(
        uvfield, ai_field, convexity, exposure, adj, world,
        snap["weights"], snap["gamma"], snap["alpha"],
        snap["noise_amp"], snap["noise_scale"], snap["use_barrier"],
        snap["mat_penalty"], base_seed, float(np.linalg.norm(world.max(0)-world.min(0))))
    weartime_uv = wt["weartime_uv"]

    # Seam fusion + dilation
    if snap["seam_fuse"]:
        job.message = "Replay: fusing seams + padding…"; job.progress = 0.84
        registry = bridge.run(lambda: seam_registry.build_seam_registry(
            bpy.data.objects.get(obj_name), layer_name))
        qa_before = seam_registry.seam_qa(weartime_uv, registry, res)
        weartime_uv = seam_registry.fuse_seam(weartime_uv, registry, res, snap["seam_diffuse"])
        weartime_uv, _valid2 = seam_registry.dilate(weartime_uv, uvfield.valid, snap["padding"])
        # Fuse the encoded color residual too (same seam registry, per-channel).
        worn_uv = seam_registry.fuse_seam_rgb(worn_uv, registry, res, snap["seam_diffuse"])
        worn_uv, _rgb_valid2 = seam_registry.dilate_rgb(
            worn_uv, rgb_valid, snap["padding"])
        wear_alpha = seam_registry.fuse_seam(
            wear_alpha, registry, res, snap["seam_diffuse"])
        wear_alpha, _wear_valid2 = seam_registry.dilate(
            wear_alpha, rgb_valid, snap["padding"])
        qa_after = seam_registry.seam_qa(weartime_uv, registry, res)
        job.meta["seam_before_p95"] = qa_before["p95"]
        job.meta["seam_after_p95"] = qa_after["p95"]
        bridge.run(lambda: seam_registry.visualize_seams(
            bpy.data.objects.get(obj_name), registry))

    # Bake + attach shader
    job.state = JobState.BAKE; job.message = "Replay: baking WearTime…"; job.progress = 0.92
    if not weartime_uv.any():
        raise RuntimeError(
            "Replay: WearTime came out all-zero (render would read as fully worn). "
            "UV coverage was fine; the topology-growth collapsed — see the full "
            "pipeline's all-zero guard for the param fixes (lower alpha, raise "
            "noise_amp, check mesh connectivity).")
    fmt = "PNG16" if snap["export_format"] == "PNG16" else ("EXR" if snap["export_format"] == "EXR" else "PNG8")
    rgba = np.zeros((res, res, 4), dtype=np.float32)
    rgba[..., 0] = weartime_uv; rgba[..., 1] = weartime_uv
    rgba[..., 2] = weartime_uv; rgba[..., 3] = 1.0
    weartime_path = os.path.join(cache_dir, "WearTime.png")
    bridge.run(lambda: _save_uv_texture(weartime_path, rgba, fmt))
    # AIWear_Mask.png — the directly-reprojected wear mask (the "重投影完的
    # mask"). Persisted for QA/production (the shader gate is WearTime, but the
    # user asked where the reprojected mask lives — here it is).
    mask_rgba = np.zeros((res, res, 4), dtype=np.float32)
    mask_rgba[..., 0] = ai_field; mask_rgba[..., 1] = ai_field
    mask_rgba[..., 2] = ai_field; mask_rgba[..., 3] = 1.0
    mask_path = os.path.join(cache_dir, "AIWear_Mask.png")
    bridge.run(lambda: _save_uv_texture(mask_path, mask_rgba, fmt))
    # AIWear_WornTex.png — encoded LINEAR clean→worn color residual.
    # RGB: bounded encoded residual; A: actual AI wear evidence (not coverage).
    wtex = np.zeros((res, res, 4), dtype=np.float32)
    wtex[..., :3] = worn_uv
    wtex[..., 3] = wear_alpha
    worn_tex_path = os.path.join(cache_dir, "AIWear_WornTex.png")
    bridge.run(lambda: _save_uv_texture(worn_tex_path, wtex, fmt))

    job.message = "Replay: attaching wear overlay shader…"; job.progress = 0.97

    def _attach():
        wt_img = bpy.data.images.load(weartime_path)
        wt_img.name = "AIWear_WearTime"
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

    job.meta["weartime_path"] = weartime_path
    job.meta["worn_tex_path"] = worn_tex_path
    job.meta["worn_mask_path"] = mask_path
    job.meta["replayed"] = True
    job.state = JobState.DONE
    job.stage = JobStage.EXPORT
    job.progress = 1.0
    job.message = "Done (replay)."


def _set_ai_progress(job, p, msg):
    job.progress = 0.08 + 0.45 * p  # map provider 0..1 into the capture band
    if msg:
        job.message = msg
    job.touch()


def _cam_proj_data(cam, target_obj, res):
    import bpy
    import numpy as np_
    from ..render.view_sampler import compute_framing
    # Framing (bounding-sphere radius) is of the TARGET MESH, not the camera.
    # Cameras carry no geometry, so compute_framing(camera) would raise "Object
    # does not have geometry data" on to_mesh(). The radius feeds the per-view
    # depth epsilon used by the software z-buffer visibility test below.
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
    # live-refresh the viewport so the progress panel updates
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


def start_replay(context) -> str:
    """Launch a downstream-only replay (no render, no AI). See _run_replay.

    Reuses the cached clean_V{i}.png + worn_V{i}.png + views.json from the last
    full run on the active object. Lets you iterate on surface/WearTime params
    without spending API budget (Q3).
    """
    snap = snapshot_context(context)
    _log_clear()
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
