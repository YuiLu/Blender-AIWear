"""Run a fresh four-arm UV seam ablation on the gaming-console cache.

This script executes the add-on's real Replay Downstream path in Blender.  It
temporarily evaluates the cached six views at 256px to make texture-footprint
failures visible, restores the canonical cache afterwards, and writes only the
focused results (no duplicated clean/worn views) to a new supplemental folder.
It also renders all UV seams and the highest-risk seams directly on the model.

Run from the repository root while keeping the asset outside the repository:
  blender -b "<path-to-gaming_console_2k.blend>" \
    --python tools/run_seam_supplement.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_wear

try:
    ai_wear.unregister()
except Exception:
    pass
ai_wear.register()

from ai_wear import utils
from ai_wear.cache import job_cache
from ai_wear.operators import pipeline
from ai_wear.render import passes
from ai_wear.uv import seam_registry


class ImmediateBridge:
    @staticmethod
    def run(fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _load_uv_rgba(path: Path) -> np.ndarray:
    # Image files are top-row-first; UV arrays use row 0 = V0/bottom.
    return np.ascontiguousarray(utils.load_image_rgba(str(path))[::-1])


def _qa_rgb(field: np.ndarray, registry, res: int) -> dict:
    values = []
    for channel in range(field.shape[2]):
        for pair in registry:
            t = np.linspace(0.05, 0.95, 16)
            ua = pair.uv_a0[None] * (1 - t[:, None]) + pair.uv_a1[None] * t[:, None]
            ub = pair.uv_b0[None] * (1 - t[:, None]) + pair.uv_b1[None] * t[:, None]
            a = seam_registry._bilinear_sample(field[..., channel], ua, res)
            b = seam_registry._bilinear_sample(field[..., channel], ub, res)
            values.extend(np.abs(a - b).tolist())
    data = np.asarray(values, dtype=np.float32)
    return {
        "count": len(registry),
        "mean": float(data.mean()) if data.size else 0.0,
        "p95": float(np.percentile(data, 95)) if data.size else 0.0,
        "worst": float(data.max()) if data.size else 0.0,
    }


def _pair_risks(wearthreshold: np.ndarray, worn: np.ndarray, registry, res: int):
    risks = []
    t = np.linspace(0.05, 0.95, 24)
    fields = [wearthreshold, worn[..., 3], worn[..., 0], worn[..., 1], worn[..., 2]]
    for pair in registry:
        ua = pair.uv_a0[None] * (1 - t[:, None]) + pair.uv_a1[None] * t[:, None]
        ub = pair.uv_b0[None] * (1 - t[:, None]) + pair.uv_b1[None] * t[:, None]
        risk = 0.0
        for field in fields:
            a = seam_registry._bilinear_sample(field, ua, res)
            b = seam_registry._bilinear_sample(field, ub, res)
            risk = max(risk, float(np.percentile(np.abs(a - b), 95)))
        risks.append(risk)
    return np.asarray(risks, dtype=np.float32)


def _hero_camera(scene, obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((corner - center).length for corner in corners)
    data = bpy.data.cameras.new("AIWear_SeamSupplement_CameraData")
    data.lens = 58.0
    camera = bpy.data.objects.new("AIWear_SeamSupplement_Camera", data)
    scene.collection.objects.link(camera)
    direction = Vector((1.15, -1.35, 0.82)).normalized()
    camera.location = center + direction * radius * 3.15
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    return camera, radius


def _emission_material(name: str, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = 8.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _seam_curve(obj, pairs, name: str, color, bevel_depth: float):
    curve = bpy.data.curves.new(name + "Data", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_resolution = 0
    curve.bevel_depth = bevel_depth
    world = obj.matrix_world
    vertices = obj.data.vertices
    for pair in pairs:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        a = world @ vertices[pair.v0].co
        b = world @ vertices[pair.v1].co
        spline.points[0].co = (*a, 1.0)
        spline.points[1].co = (*b, 1.0)
    overlay = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(overlay)
    overlay.data.materials.append(_emission_material(name + "Material", color))
    return overlay


def _save_uv_overlay(path: Path, base: np.ndarray, pairs, res: int):
    overlay = np.empty((res, res, 4), dtype=np.float32)
    overlay[..., :3] = base[..., None]
    overlay[..., 3] = 1.0
    for pair in pairs:
        t = np.linspace(0.0, 1.0, 256)
        for start, end in ((pair.uv_a0, pair.uv_a1), (pair.uv_b0, pair.uv_b1)):
            uv = start[None] * (1.0 - t[:, None]) + end[None] * t[:, None]
            xs = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
            ys = np.clip((uv[:, 1] * res).astype(int), 0, res - 1)
            for x, y in zip(xs, ys):
                y0, y1 = max(0, y - 1), min(res, y + 2)
                x0, x1 = max(0, x - 1), min(res, x + 2)
                overlay[y0:y1, x0:x1, :3] = (1.0, 0.0, 0.0)
    pipeline._save_uv_texture(str(path), overlay, "PNG8")


obj = bpy.data.objects.get("gaming_console")
assert obj is not None and obj.type == "MESH", "gaming_console mesh not found"
for selected in bpy.context.selected_objects:
    selected.select_set(False)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

scene = bpy.context.scene
settings = scene.ai_wear
settings.target_uv_layer = "AI_WearUV"
settings.work_resolution = 256
settings.export_format = "PNG16"
settings.wear_amount = 100.0
settings.feather = 0.5
settings.use_ai_mask = True
settings.use_geometry_prior = True
settings.use_topology_growth = True
settings.seam_diffuse_texels = 8
settings.padding_texels = 16
settings.save_experiment_snapshot = False

cache_dir = Path(job_cache.object_cache_dir("gaming_console"))
views_dir = cache_dir / "views"
manifest_path = views_dir / "views.json"
assert manifest_path.is_file(), manifest_path

stamp = time.strftime("%Y%m%d_%H%M%S")
result_dir = cache_dir / ("supplemental_seam_retest_" + stamp)
result_dir.mkdir(parents=True, exist_ok=False)

canonical = [manifest_path, cache_dir / "AIWear_UVSnapshot.npz",
             cache_dir / "WearThreshold.png",
             cache_dir / "M_Wear.png", cache_dir / "AIWear_WornTex.png"]
canonical += sorted(views_dir.glob("diff_mask_V*.png"))

summary = {
    "created_at": stamp,
    "blender_version": bpy.app.version_string,
    "blend": bpy.data.filepath,
    "object": obj.name,
    "replay_resolution": 256,
    "render_resolution": 1024,
    "image_generation_calls": 0,
    "arms": {},
}

with tempfile.TemporaryDirectory(prefix="aiwear_seam_backup_") as backup_name:
    backup_dir = Path(backup_name)
    existed = {}
    for original in canonical:
        existed[str(original)] = original.is_file()
        if original.is_file():
            shutil.copy2(original, backup_dir / (str(len(existed)) + "_" + original.name))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_manifest = manifest_path.read_bytes()
    manifest["work_resolution"] = 256
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    camera = None
    overlay_objects = []
    raw_wearthreshold = None
    raw_worn = None
    registry = None
    try:
        camera, radius = _hero_camera(scene, obj)
        arms = (
            ("raw", False, False),
            ("fusion_only", True, False),
            ("padding_only", False, True),
            ("fusion_and_padding", True, True),
        )
        for label, use_fusion, use_padding in arms:
            settings.seam_fuse = use_fusion
            settings.use_padding = use_padding
            snap = pipeline.snapshot_context(bpy.context)
            snap["obj_uuid"] = "gaming_console"
            snap["work_resolution"] = 256
            snap["wear_amount"] = 100.0
            snap["feather"] = 0.5
            job = job_cache.create_job()
            started = time.perf_counter()
            pipeline._run_replay(job, snap, ImmediateBridge())
            elapsed = time.perf_counter() - started
            assert job.state == job_cache.JobState.DONE, (label, job.state, job.error)

            arm_dir = result_dir / label
            arm_dir.mkdir()
            for filename in ("WearThreshold.png", "M_Wear.png", "AIWear_WornTex.png"):
                shutil.copy2(cache_dir / filename, arm_dir / filename)
            render_path = arm_dir / "render_model.png"
            passes.render_clean(scene, camera, str(render_path), 1024)

            wearthreshold = _load_uv_rgba(cache_dir / "WearThreshold.png")[..., 0]
            worn = _load_uv_rgba(cache_dir / "AIWear_WornTex.png")
            if registry is None:
                registry = seam_registry.build_seam_registry(obj, "AI_WearUV")
                summary["seam_edge_count"] = len(registry)
            summary["arms"][label] = {
                "seam_fusion": use_fusion,
                "padding": use_padding,
                "padding_texels": 16 if use_padding else 0,
                "elapsed_seconds": round(elapsed, 3),
                "wearthreshold": seam_registry.seam_qa(wearthreshold, registry, 256),
                "worn_alpha": seam_registry.seam_qa(worn[..., 3], registry, 256),
                "worn_rgb": _qa_rgb(worn[..., :3], registry, 256),
                "render": str(render_path),
            }
            if label == "raw":
                raw_wearthreshold = wearthreshold.copy()
                raw_worn = worn.copy()

        assert registry is not None and raw_wearthreshold is not None and raw_worn is not None
        risks = _pair_risks(raw_wearthreshold, raw_worn, registry, 256)
        threshold = float(np.percentile(risks, 90))
        risk_pairs = [pair for pair, risk in zip(registry, risks) if risk >= threshold]
        summary["risk_overlay"] = {
            "definition": "top 10% seam edges by max p95 mismatch across raw WearThreshold, WornTex alpha and RGB",
            "threshold": threshold,
            "edge_count": len(risk_pairs),
        }

        _save_uv_overlay(result_dir / "seam_edges_uv_overlay.png",
                         raw_wearthreshold, registry, 256)
        all_overlay = _seam_curve(obj, registry, "AIWear_AllUVSeams",
                                  (1.0, 0.02, 0.02), radius * 0.0022)
        overlay_objects.append(all_overlay)
        passes.render_clean(scene, camera,
                            str(result_dir / "seam_edges_3d_all.png"), 1024)
        all_overlay.hide_render = True
        risk_overlay = _seam_curve(obj, risk_pairs, "AIWear_HighRiskUVSeams",
                                   (1.0, 0.75, 0.0), radius * 0.0035)
        overlay_objects.append(risk_overlay)
        passes.render_clean(scene, camera,
                            str(result_dir / "seam_edges_3d_high_risk.png"), 1024)
    finally:
        # Restore every canonical cache file exactly.  The supplement remains
        # self-contained without duplicating the six clean/worn view images.
        manifest_path.write_bytes(original_manifest)
        backup_files = list(backup_dir.iterdir())
        for original in canonical:
            match = next((p for p in backup_files if p.name.endswith("_" + original.name)), None)
            if existed.get(str(original), False) and match is not None:
                shutil.copy2(match, original)
            elif original.exists() and not existed.get(str(original), False):
                original.unlink()
        for overlay in overlay_objects:
            if overlay and overlay.name in bpy.data.objects:
                bpy.data.objects.remove(overlay, do_unlink=True)
        if camera and camera.name in bpy.data.objects:
            camera_data = camera.data
            bpy.data.objects.remove(camera, do_unlink=True)
            if camera_data:
                bpy.data.cameras.remove(camera_data)

summary_path = result_dir / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("AIWEAR_SEAM_SUPPLEMENT_OK")
print("AIWEAR_SEAM_SUPPLEMENT_DIR=" + str(result_dir))
print(json.dumps(summary, ensure_ascii=False, indent=2))
