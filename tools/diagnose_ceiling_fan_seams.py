"""Diagnose the visible UV seam on the lower ceiling-fan base.

Runs Replay Downstream against the existing cached views; never calls an image
provider and never saves the .blend.  Canonical cache files are restored byte
for byte after the experiment.  Results are written below the object's cache.

Run from the repository root:
  blender -b "blender/ceiling_fan_2k.blend/ceiling_fan_2k.blend" \
    --python tools/diagnose_ceiling_fan_seams.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import bpy
import bmesh
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

from ai_wear import shader, utils
from ai_wear.cache import job_cache
from ai_wear.operators import pipeline
from ai_wear.render import passes
from ai_wear.uv import rasterizer, seam_registry


OBJECT_NAME = "ceiling_fan"
LAYER_NAME = "AI_WearUV"
RENDER_RESOLUTION = (2560, 1280)


class ImmediateBridge:
    @staticmethod
    def run(fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _load_uv_rgba(path: Path) -> np.ndarray:
    # Image files are top-row-first; the UV raster arrays use row 0 = V0.
    return np.ascontiguousarray(utils.load_image_rgba(str(path))[::-1])


def _sample_center(field: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """Bilinear sample using texel-centre coordinates (u*W - 0.5)."""
    h, w = field.shape[:2]
    x = np.clip(uvs[:, 0] * w - 0.5, 0.0, w - 1.0)
    y = np.clip(uvs[:, 1] * h - 0.5, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    if field.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    return (a * (1.0 - fx) * (1.0 - fy)
            + b * fx * (1.0 - fy)
            + c * (1.0 - fx) * fy
            + d * fx * fy)


def _pair_midpoint_world(obj, pair) -> Vector:
    a = obj.matrix_world @ obj.data.vertices[pair.v0].co
    b = obj.matrix_world @ obj.data.vertices[pair.v1].co
    return 0.5 * (a + b)


def _pair_face_centres(obj, registry):
    """Return inward UV directions for both sides of every registered seam."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.get(LAYER_NAME)
    result = {}
    for pair in registry:
        edge = bm.edges[pair.edge_index]
        sides = []
        for face in edge.link_faces:
            loops = {loop.vert.index: loop for loop in face.loops}
            uv0 = np.asarray(loops[pair.v0][uv_layer].uv, dtype=np.float64)
            uv1 = np.asarray(loops[pair.v1][uv_layer].uv, dtype=np.float64)
            centre = np.mean(
                [np.asarray(loop[uv_layer].uv, dtype=np.float64)
                 for loop in face.loops], axis=0)
            da = float(np.linalg.norm(uv0 - pair.uv_a0)
                       + np.linalg.norm(uv1 - pair.uv_a1))
            db = float(np.linalg.norm(uv0 - pair.uv_b0)
                       + np.linalg.norm(uv1 - pair.uv_b1))
            sides.append(("a" if da <= db else "b", centre))
        centres = {name: centre for name, centre in sides}
        if "a" in centres and "b" in centres:
            result[pair.edge_index] = (centres["a"], centres["b"])
    bm.free()
    return result


def _inward_normal(start, end, centre):
    tangent = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(tangent))
    if length < 1e-12:
        return np.zeros(2, dtype=np.float64)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64) / length
    midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
    if float(np.dot(normal, np.asarray(centre) - midpoint)) < 0.0:
        normal *= -1.0
    return normal


def _as_scalar_difference(a, b):
    diff = np.abs(a - b)
    if diff.ndim == 2:
        diff = np.max(diff, axis=1)
    return diff


def _field_metrics(field, pairs, centres, inward_texels=4.0):
    cross = []
    halo = []
    per_pair = []
    resolution = field.shape[0]
    for pair in pairs:
        t = np.linspace(0.02, 0.98, 32)
        ua = pair.uv_a0[None] * (1.0 - t[:, None]) + pair.uv_a1[None] * t[:, None]
        ub = pair.uv_b0[None] * (1.0 - t[:, None]) + pair.uv_b1[None] * t[:, None]
        va = _sample_center(field, ua)
        vb = _sample_center(field, ub)
        pair_cross = _as_scalar_difference(va, vb)
        cross.extend(pair_cross.tolist())

        pair_halo = np.zeros_like(pair_cross)
        if pair.edge_index in centres:
            ca, cb = centres[pair.edge_index]
            na = _inward_normal(pair.uv_a0, pair.uv_a1, ca)
            nb = _inward_normal(pair.uv_b0, pair.uv_b1, cb)
            ia = ua + na[None] * (inward_texels / resolution)
            ib = ub + nb[None] * (inward_texels / resolution)
            via = _sample_center(field, ia)
            vib = _sample_center(field, ib)
            pair_halo = 0.5 * (
                _as_scalar_difference(va, via)
                + _as_scalar_difference(vb, vib))
            halo.extend(pair_halo.tolist())
        per_pair.append({
            "edge_index": pair.edge_index,
            "cross_p95": float(np.percentile(pair_cross, 95)),
            "halo_p95": float(np.percentile(pair_halo, 95)),
        })

    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()) if values.size else 0.0,
            "p95": float(np.percentile(values, 95)) if values.size else 0.0,
            "worst": float(values.max()) if values.size else 0.0,
        }

    return {"cross": stats(cross), "halo_4px": stats(halo), "per_pair": per_pair}


def _smoothstep(edge0, edge1, value):
    width = np.maximum(edge1 - edge0, 1e-6)
    x = np.clip((value - edge0) / width, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _appearance_effect(threshold, worn, amount01, feather01):
    gate = _smoothstep(threshold - feather01, threshold + feather01, amount01)
    delta = 2.0 * (worn[..., :3] - 0.5)
    return delta * (gate * worn[..., 3])[..., None]


def _legacy_sample(field, uvs, res):
    """The pre-0.3.5 half-texel-shifted sampler, retained for exact A/B."""
    x = np.clip(uvs[:, 0] * res, 0, res - 1.001)
    y = np.clip(uvs[:, 1] * res, 0, res - 1.001)
    x0 = np.floor(x).astype(np.int64); x1 = x0 + 1
    y0 = np.floor(y).astype(np.int64); y1 = y0 + 1
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    a = field[y0, x0]; b = field[y0, x1]
    c = field[y1, x0]; d = field[y1, x1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
            + c * (1 - fx) * fy + d * fx * fy)


def _legacy_stamp(field, x, y, value, radius):
    res = field.shape[0]
    x0 = max(0, x - radius); x1 = min(res, x + radius + 1)
    y0 = max(0, y - radius); y1 = min(res, y + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    weight = np.clip(1.0 - distance / max(1.0, float(radius)), 0.0, 1.0)
    region = field[y0:y1, x0:x1]
    blended = region * (1.0 - weight) + value * weight
    field[y0:y1, x0:x1] = np.maximum(region, blended)


def _legacy_fuse(field, registry, res, diffuse_texels=8, tol=None, valid=None):
    """Exact legacy max/stamp implementation used before plugin 0.3.5."""
    out = field.copy()
    tolerance = 0.02 if tol is None else float(tol)
    radius = max(1, int(diffuse_texels) // 4)
    t = np.linspace(0.05, 0.95, max(8, res // 16))
    for pair in registry:
        ua = pair.uv_a0[None] * (1 - t[:, None]) + pair.uv_a1[None] * t[:, None]
        ub = pair.uv_b0[None] * (1 - t[:, None]) + pair.uv_b1[None] * t[:, None]
        va = _legacy_sample(out, ua, res)
        vb = _legacy_sample(out, ub, res)
        merged = np.maximum(va, vb)
        for index in np.nonzero(np.abs(va - vb) > tolerance)[0]:
            xa = int(np.clip(ua[index, 0] * res, 0, res - 1))
            ya = int(np.clip(ua[index, 1] * res, 0, res - 1))
            xb = int(np.clip(ub[index, 0] * res, 0, res - 1))
            yb = int(np.clip(ub[index, 1] * res, 0, res - 1))
            _legacy_stamp(out, xa, ya, float(merged[index]), radius)
            _legacy_stamp(out, xb, yb, float(merged[index]), radius)
    return out


def _legacy_fuse_rgb(field_rgb, registry, res, diffuse_texels=8, valid=None):
    out = np.empty_like(field_rgb)
    for channel in range(field_rgb.shape[2]):
        out[..., channel] = _legacy_fuse(
            field_rgb[..., channel], registry, res, diffuse_texels)
    return out


def _symmetric_fuse(field: np.ndarray, registry, res: int,
                    diffuse_texels: int = 8, tol=None, valid=None) -> np.ndarray:
    """Order-independent experimental seam blend used as an A/B candidate."""
    original = np.asarray(field, dtype=np.float32)
    if not registry:
        return original.copy()
    tolerance = 0.02 if tol is None else float(tol)
    radius = max(1, int(diffuse_texels) // 4)
    accum = np.zeros_like(original, dtype=np.float32)
    weights = np.zeros_like(original, dtype=np.float32)
    yy_cache = {}

    def stamp(uv, value):
        x = int(np.clip(math.floor(float(uv[0]) * res), 0, res - 1))
        y = int(np.clip(math.floor(float(uv[1]) * res), 0, res - 1))
        x0 = max(0, x - radius); x1 = min(res, x + radius + 1)
        y0 = max(0, y - radius); y1 = min(res, y + radius + 1)
        key = (x0, x1, y0, y1, x, y)
        if key not in yy_cache:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            weight = np.clip(
                1.0 - np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
                / max(1.0, float(radius)), 0.0, 1.0).astype(np.float32)
            yy_cache[key] = weight
        weight = yy_cache[key]
        accum[y0:y1, x0:x1] += float(value) * weight
        weights[y0:y1, x0:x1] += weight

    for pair in registry:
        edge_texels = max(
            float(np.linalg.norm(pair.uv_a1 - pair.uv_a0)),
            float(np.linalg.norm(pair.uv_b1 - pair.uv_b0))) * res
        sample_count = max(2, int(math.ceil(edge_texels * 2.0)) + 1)
        t = np.linspace(0.0, 1.0, sample_count)
        ua = pair.uv_a0[None] * (1.0 - t[:, None]) + pair.uv_a1[None] * t[:, None]
        ub = pair.uv_b0[None] * (1.0 - t[:, None]) + pair.uv_b1[None] * t[:, None]
        va = _sample_center(original, ua)
        vb = _sample_center(original, ub)
        merged = 0.5 * (va + vb)
        active = np.abs(va - vb) > tolerance
        for index in np.nonzero(active)[0]:
            stamp(ua[index], merged[index])
            stamp(ub[index], merged[index])

    active = weights > 1e-8
    target = original.copy()
    target[active] = accum[active] / weights[active]
    strength = np.clip(weights, 0.0, 1.0)
    return original * (1.0 - strength) + target * strength


def _symmetric_fuse_rgb(field_rgb, registry, res, diffuse_texels=8, valid=None):
    out = np.empty_like(field_rgb)
    for channel in range(field_rgb.shape[2]):
        out[..., channel] = _symmetric_fuse(
            field_rgb[..., channel], registry, res, diffuse_texels)
    return out


_PAIR_CENTRES = {}


def _profile_fuse(field: np.ndarray, registry, res: int,
                  diffuse_texels: int = 8, tol=None, valid=None) -> np.ndarray:
    """Match corresponding inward strips, not only the one-pixel seam line."""
    original = np.asarray(field, dtype=np.float32)
    if not registry:
        return original.copy()
    tolerance = 0.02 if tol is None else float(tol)
    width = max(1, int(diffuse_texels))
    accum = np.zeros_like(original, dtype=np.float32)
    weights = np.zeros_like(original, dtype=np.float32)

    def accumulate(uvs, values, strength):
        xs = np.clip(np.floor(uvs[:, 0] * res).astype(np.int64), 0, res - 1)
        ys = np.clip(np.floor(uvs[:, 1] * res).astype(np.int64), 0, res - 1)
        np.add.at(accum, (ys, xs), values.astype(np.float32) * strength)
        np.add.at(weights, (ys, xs), strength)

    for pair in registry:
        centres = _PAIR_CENTRES.get(pair.edge_index)
        if centres is None:
            continue
        ca, cb = centres
        na = _inward_normal(pair.uv_a0, pair.uv_a1, ca)
        nb = _inward_normal(pair.uv_b0, pair.uv_b1, cb)
        edge_texels = max(
            float(np.linalg.norm(pair.uv_a1 - pair.uv_a0)),
            float(np.linalg.norm(pair.uv_b1 - pair.uv_b0))) * res
        sample_count = max(2, int(math.ceil(edge_texels * 2.0)) + 1)
        t = np.linspace(0.0, 1.0, sample_count)
        edge_a = pair.uv_a0[None] * (1.0 - t[:, None]) + pair.uv_a1[None] * t[:, None]
        edge_b = pair.uv_b0[None] * (1.0 - t[:, None]) + pair.uv_b1[None] * t[:, None]
        for distance in range(width + 1):
            ua = edge_a + na[None] * (distance / res)
            ub = edge_b + nb[None] * (distance / res)
            va = _sample_center(original, ua)
            vb = _sample_center(original, ub)
            active = np.abs(va - vb) > tolerance
            if not active.any():
                continue
            merged = 0.5 * (va + vb)
            # Full equality at the edge, smoothly returning to the original
            # signal at the outer edge of the strip.
            strength = float(1.0 - distance / (width + 1.0))
            accumulate(ua[active], merged[active], strength)
            accumulate(ub[active], merged[active], strength)

    active = weights > 1e-8
    target = original.copy()
    target[active] = accum[active] / weights[active]
    strength = np.clip(weights, 0.0, 1.0)
    return original * (1.0 - strength) + target * strength


def _profile_fuse_rgb(field_rgb, registry, res, diffuse_texels=8, valid=None):
    out = np.empty_like(field_rgb)
    for channel in range(field_rgb.shape[2]):
        out[..., channel] = _profile_fuse(
            field_rgb[..., channel], registry, res, diffuse_texels)
    return out


def _make_closeup_camera(scene, obj):
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    z_min = min(c.z for c in world_corners)
    z_max = max(c.z for c in world_corners)
    centre = sum(world_corners, Vector()) / len(world_corners)
    # A 2:1 frame around only the lower base.  The earlier square shot placed
    # the seam in the upper half and wasted roughly one third on empty floor.
    target = Vector((centre.x, centre.y, z_min + 0.155 * (z_max - z_min)))
    camera_data = bpy.data.cameras.new("AIWear_FanBaseCloseupData")
    camera_data.lens = 68.0
    camera = bpy.data.objects.new("AIWear_FanBaseCloseup", camera_data)
    scene.collection.objects.link(camera)
    direction = Vector((0.30, -1.0, 0.07)).normalized()
    camera.location = target + direction * 0.65
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    return camera, z_min, z_max


def _emission_material(name, color):
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


def _edge_curve(obj, pairs, name, color, width):
    curve = bpy.data.curves.new(name + "Data", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_resolution = 0
    curve.bevel_depth = width
    for pair in pairs:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        a = obj.matrix_world @ obj.data.vertices[pair.v0].co
        b = obj.matrix_world @ obj.data.vertices[pair.v1].co
        spline.points[0].co = (*a, 1.0)
        spline.points[1].co = (*b, 1.0)
    overlay = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(overlay)
    overlay.data.materials.append(_emission_material(name + "Material", color))
    return overlay


def _boundary_inventory(obj, z_limit):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = []
    bottom = []
    quantized = {}
    tolerance = max(obj.dimensions) * 1e-5
    for edge in bm.edges:
        if len(edge.link_faces) != 1:
            continue
        a = obj.matrix_world @ edge.verts[0].co
        b = obj.matrix_world @ edge.verts[1].co
        record = {"edge_index": edge.index,
                  "v0": edge.verts[0].index, "v1": edge.verts[1].index}
        boundary.append(record)
        if 0.5 * (a.z + b.z) <= z_limit:
            bottom.append(record)
        qa = tuple(int(round(float(v) / tolerance)) for v in a)
        qb = tuple(int(round(float(v) / tolerance)) for v in b)
        key = tuple(sorted((qa, qb)))
        quantized.setdefault(key, []).append(record)
    coincident_groups = [items for items in quantized.values() if len(items) > 1]
    bm.free()
    return {
        "boundary_edge_count": len(boundary),
        "bottom_boundary_edge_count": len(bottom),
        "coincident_boundary_groups": len(coincident_groups),
        "coincident_boundary_edges": sum(len(group) for group in coincident_groups),
    }


obj = bpy.data.objects.get(OBJECT_NAME)
assert obj is not None and obj.type == "MESH", f"{OBJECT_NAME} mesh not found"
for selected in bpy.context.selected_objects:
    selected.select_set(False)
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

scene = bpy.context.scene
settings = scene.ai_wear
settings.target_uv_layer = LAYER_NAME
settings.work_resolution = 1024
settings.export_format = "PNG16"
settings.save_experiment_snapshot = False

cache_dir = Path(job_cache.object_cache_dir(OBJECT_NAME))
views_dir = cache_dir / "views"
manifest_path = views_dir / "views.json"
assert manifest_path.is_file(), manifest_path
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
resolution = int(manifest.get("work_resolution", 1024))
assert resolution == 1024, resolution

stamp = time.strftime("%Y%m%d_%H%M%S")
review_only = "--review" in sys.argv
folder_prefix = "fan_base_seam_review_2k_" if review_only else "fan_base_seam_diagnosis_"
result_dir = cache_dir / (folder_prefix + stamp)
result_dir.mkdir(parents=True, exist_ok=False)

canonical = [manifest_path, cache_dir / "AIWear_UVSnapshot.npz",
             cache_dir / "WearThreshold.png", cache_dir / "M_Wear.png",
             cache_dir / "AIWear_WornTex.png"]
canonical += sorted(views_dir.glob("diff_mask_V*.png"))
canonical.append(cache_dir / "experiments" / "oblique_renders" /
                 "cams_08_oblique.png")

camera, z_min, z_max = _make_closeup_camera(scene, obj)
base_limit = z_min + 0.30 * (z_max - z_min)
registry = seam_registry.build_seam_registry(obj, LAYER_NAME)
base_registry = [pair for pair in registry
                 if _pair_midpoint_world(obj, pair).z <= base_limit]
centres = _pair_face_centres(obj, registry)
_PAIR_CENTRES = centres
uvfield = rasterizer.build_uv_field(
    obj, LAYER_NAME, resolution,
    depsgraph=bpy.context.evaluated_depsgraph_get())
assert uvfield is not None and uvfield.valid.any()

summary = {
    "created_at": stamp,
    "blender_version": bpy.app.version_string,
    "blend": bpy.data.filepath,
    "object": OBJECT_NAME,
    "resolution": resolution,
    "wear_amount": float(settings.wear_amount),
    "feather": float(settings.feather),
    "view_count": len(manifest.get("views", [])),
    "view_context_mode": manifest.get("view_context_mode"),
    "view_context_supported": manifest.get("view_context_supported"),
    "context_sources": [view.get("context_source")
                        for view in manifest.get("views", [])],
    "registered_seams": len(registry),
    "base_registered_seams": len(base_registry),
    "base_z_limit": base_limit,
    "boundary_inventory": _boundary_inventory(obj, base_limit),
    "arms": {},
}

original_fuse = seam_registry.fuse_seam
original_fuse_rgb = seam_registry.fuse_seam_rgb
original_settings = {
    name: getattr(settings, name) for name in
    ("seam_fuse", "seam_diffuse_texels", "use_padding", "padding_texels",
     "gamma")
}

production_only = "--production" in sys.argv
tuning_only = "--tuning" in sys.argv
latest_only = "--latest" in sys.argv
if review_only:
    arms = (
        ("01_raw", False, 8, False, 0, "current", 2.0),
        ("02_legacy_f8_no_padding", True, 8, False, 0, "legacy", 2.0),
        ("03_legacy_f8_padding16", True, 8, True, 16, "legacy", 2.0),
        ("04_production_f8_padding2", True, 8, True, 2, "current", 2.0),
    )
elif production_only:
    arms = (
        ("production_f8_p2", True, 8, True, 2, "current", 2.0),
    )
elif latest_only:
    # Focused rerun for the current Qwen/First-view-Anchor six-view cache.  This
    # directly compares the user's seam-off/padding-2 image with progressively
    # wider production fusion bands while keeping every other input identical.
    arms = (
        ("off_p2", False, 8, True, 2, "current", 2.0),
        ("f1_p2", True, 1, True, 2, "current", 2.0),
        ("f2_p2", True, 2, True, 2, "current", 2.0),
        ("f4_p2", True, 4, True, 2, "current", 2.0),
        ("f8_p2", True, 8, True, 2, "current", 2.0),
        ("f8_no_padding", True, 8, False, 0, "current", 2.0),
    )
elif tuning_only:
    arms = (
        ("current_f4_no_padding", True, 4, False, 0, "current", 2.0),
        ("current_f8_p2", True, 8, True, 2, "current", 2.0),
        ("current_f8_p4", True, 8, True, 4, "current", 2.0),
        ("current_f8_p2_gamma1", True, 8, True, 2, "current", 1.0),
        ("profile_f4_p2", True, 4, True, 2, "profile", 2.0),
        ("profile_f8_p2", True, 8, True, 2, "profile", 2.0),
        ("profile_f8_no_padding", True, 8, False, 0, "profile", 2.0),
    )
else:
    arms = (
        ("raw", False, 8, False, 0, "current", 2.0),
        ("current_f8_no_padding", True, 8, False, 0, "current", 2.0),
        ("current_f8_p16", True, 8, True, 16, "current", 2.0),
        ("current_f16_p8", True, 16, True, 8, "current", 2.0),
        ("symmetric_f8_p8", True, 8, True, 8, "symmetric", 2.0),
    )

with tempfile.TemporaryDirectory(prefix="aiwear_fan_seam_backup_") as backup_name:
    backup_dir = Path(backup_name)
    backup_records = []
    for index, original in enumerate(canonical):
        record = {"path": original, "existed": original.is_file(), "backup": None}
        if original.is_file():
            backup = backup_dir / f"{index:03d}_{original.name}"
            shutil.copy2(original, backup)
            record["backup"] = backup
        backup_records.append(record)

    overlay_objects = []
    try:
        for label, fuse, diffuse, padding, padding_texels, algorithm, gamma in arms:
            if algorithm == "symmetric":
                seam_registry.fuse_seam = _symmetric_fuse
                seam_registry.fuse_seam_rgb = _symmetric_fuse_rgb
            elif algorithm == "profile":
                seam_registry.fuse_seam = _profile_fuse
                seam_registry.fuse_seam_rgb = _profile_fuse_rgb
            elif algorithm == "legacy":
                seam_registry.fuse_seam = _legacy_fuse
                seam_registry.fuse_seam_rgb = _legacy_fuse_rgb
            else:
                seam_registry.fuse_seam = original_fuse
                seam_registry.fuse_seam_rgb = original_fuse_rgb

            settings.seam_fuse = fuse
            settings.seam_diffuse_texels = diffuse
            settings.use_padding = padding
            settings.padding_texels = padding_texels
            settings.gamma = gamma
            snap = pipeline.snapshot_context(bpy.context)
            snap["obj_uuid"] = OBJECT_NAME
            snap["object_name"] = OBJECT_NAME
            snap["target_uv_layer"] = LAYER_NAME
            snap["work_resolution"] = resolution
            snap["save_experiment_snapshot"] = False

            job = job_cache.create_job()
            started = time.perf_counter()
            pipeline._run_replay(job, snap, ImmediateBridge())
            elapsed = time.perf_counter() - started
            assert job.state == job_cache.JobState.DONE, (
                label, job.state, job.message, job.error)

            arm_dir = result_dir / label
            arm_dir.mkdir()
            for filename in ("WearThreshold.png", "M_Wear.png", "AIWear_WornTex.png"):
                shutil.copy2(cache_dir / filename, arm_dir / filename)
            render_path = arm_dir / "base_closeup.png"
            passes.render_clean(scene, camera, str(render_path),
                                RENDER_RESOLUTION, lighting="neutral")
            if review_only:
                shutil.copy2(render_path, result_dir / f"{label}.png")

            threshold = _load_uv_rgba(cache_dir / "WearThreshold.png")[..., 0]
            m_wear = _load_uv_rgba(cache_dir / "M_Wear.png")[..., 0]
            worn = _load_uv_rgba(cache_dir / "AIWear_WornTex.png")
            amount = float(settings.wear_amount) / 100.0
            feather = float(settings.feather) / 100.0
            effect = _appearance_effect(threshold, worn, amount, feather)
            summary["arms"][label] = {
                "algorithm": algorithm,
                "seam_fusion": fuse,
                "seam_diffuse_texels": diffuse,
                "padding": padding,
                "padding_texels": padding_texels,
                "gamma": gamma,
                "elapsed_seconds": round(elapsed, 3),
                "job_seam_before_p95": job.meta.get("seam_before_p95"),
                "job_seam_after_p95": job.meta.get("seam_after_p95"),
                "all": {
                    "M_Wear": _field_metrics(m_wear, registry, centres),
                    "WearThreshold": _field_metrics(threshold, registry, centres),
                    "WornAlpha": _field_metrics(worn[..., 3], registry, centres),
                    "WornRGB": _field_metrics(worn[..., :3], registry, centres),
                    "AppearanceEffect": _field_metrics(effect, registry, centres),
                },
                "base": {
                    "M_Wear": _field_metrics(m_wear, base_registry, centres),
                    "WearThreshold": _field_metrics(threshold, base_registry, centres),
                    "WornAlpha": _field_metrics(worn[..., 3], base_registry, centres),
                    "WornRGB": _field_metrics(worn[..., :3], base_registry, centres),
                    "AppearanceEffect": _field_metrics(effect, base_registry, centres),
                },
                "render": str(render_path),
            }

        # Render the exact registered seams over the final A/B material.  Red
        # means the current registry can process the edge; this distinguishes
        # algorithm failure from an unregistered open/split boundary.
        base_overlay = _edge_curve(
            obj, base_registry, "AIWear_FanBaseRegisteredSeams",
            (1.0, 0.01, 0.01), max(obj.dimensions) * 0.0018)
        overlay_objects.append(base_overlay)
        overlay_name = ("00_registered_seams.png" if review_only
                        else "base_registered_seams_red.png")
        overlay_path = result_dir / overlay_name
        passes.render_clean(scene, camera, str(overlay_path),
                            RENDER_RESOLUTION, lighting="neutral")
        summary["base_registered_overlay"] = str(overlay_path)
    finally:
        seam_registry.fuse_seam = original_fuse
        seam_registry.fuse_seam_rgb = original_fuse_rgb
        for name, value in original_settings.items():
            setattr(settings, name, value)
        for record in backup_records:
            original = record["path"]
            if record["existed"]:
                shutil.copy2(record["backup"], original)
            elif original.exists():
                original.unlink()
        for overlay in overlay_objects:
            if overlay and overlay.name in bpy.data.objects:
                bpy.data.objects.remove(overlay, do_unlink=True)
        if camera and camera.name in bpy.data.objects:
            data = camera.data
            bpy.data.objects.remove(camera, do_unlink=True)
            if data and data.users == 0:
                bpy.data.cameras.remove(data)

# Remove detailed per-edge payload from most arms to keep JSON reviewable, but
# retain the raw and default pair lists for locating the exact offending edge.
for label, arm in summary["arms"].items():
    if label not in {"raw", "current_f8_p16"}:
        for scope in ("all", "base"):
            for field_metrics in arm[scope].values():
                field_metrics.pop("per_pair", None)

summary_path = result_dir / "summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
if review_only:
    rows = []
    for label, arm in summary["arms"].items():
        base = arm["base"]["AppearanceEffect"]
        rows.append(
            f"{label}: cross p95={base['cross']['p95']:.3f}, "
            f"halo4 p95={base['halo_4px']['p95']:.3f}")
    (result_dir / "README.txt").write_text(
        "ceiling_fan base UV seam comparison\n"
        "Render: 2560 x 1280; same camera and cached 8 views.\n\n"
        "00_registered_seams.png: registered seams in red\n"
        "01_raw.png: fusion off, padding off\n"
        "02_legacy_f8_no_padding.png: legacy fusion 8, padding off\n"
        "03_legacy_f8_padding16.png: legacy fusion 8, padding 16\n"
        "04_production_f8_padding2.png: production fusion 8, padding 2\n\n"
        + "\n".join(rows) + "\n",
        encoding="utf-8")
print("AIWEAR_FAN_SEAM_DIAGNOSIS_OK")
print("AIWEAR_FAN_SEAM_DIAGNOSIS_DIR=" + str(result_dir))
print("AIWEAR_FAN_SEAM_SUMMARY=" + json.dumps({
    "registered_seams": summary["registered_seams"],
    "base_registered_seams": summary["base_registered_seams"],
    "boundary_inventory": summary["boundary_inventory"],
    "context_supported": summary["view_context_supported"],
    "arms": {
        label: {
            "base_effect_cross_p95": arm["base"]["AppearanceEffect"]["cross"]["p95"],
            "base_effect_halo_p95": arm["base"]["AppearanceEffect"]["halo_4px"]["p95"],
            "base_alpha_cross_p95": arm["base"]["WornAlpha"]["cross"]["p95"],
            "base_rgb_cross_p95": arm["base"]["WornRGB"]["cross"]["p95"],
        } for label, arm in summary["arms"].items()
    },
}, ensure_ascii=False))
