"""Create a non-destructive review copy with all fusible UV seams in red.

The source .blend is never overwritten. Run after opening a source file:

  blender -b source.blend --python tools/create_seam_overlay_blend.py -- OUT_DIR

Outputs ``<source>_seam_overlay.blend``, ``<source>_seam_overlay.png`` and a
small JSON manifest in OUT_DIR.  Red curves represent exactly the manifold UV
discontinuities returned by ``build_seam_registry`` and therefore the edges the
production Seam Fusion pass can process; mesh boundary edges are excluded.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_wear.uv import seam_registry


def _arguments():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not args:
        raise RuntimeError("Missing output directory after --")
    return Path(args[0]).resolve()


def _material():
    material = bpy.data.materials.get("AIWear_UVSeam_Red")
    if material is None:
        material = bpy.data.materials.new("AIWear_UVSeam_Red")
    material.diffuse_color = (1.0, 0.005, 0.005, 1.0)
    material.use_nodes = True
    principled = next(
        (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is not None:
        principled.inputs["Base Color"].default_value = (1.0, 0.002, 0.002, 1.0)
        principled.inputs["Roughness"].default_value = 0.28
    return material


def _world_bounds(objects):
    corners = [obj.matrix_world @ Vector(corner)
               for obj in objects for corner in obj.bound_box]
    low = Vector((min(v.x for v in corners), min(v.y for v in corners),
                  min(v.z for v in corners)))
    high = Vector((max(v.x for v in corners), max(v.y for v in corners),
                   max(v.z for v in corners)))
    return low, high


def _overlay_curve(obj, registry, collection, material, width):
    curve = bpy.data.curves.new(f"AIWear_UVSeams_{obj.name}_Data", "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_resolution = 0
    curve.bevel_depth = width
    for pair in registry:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        a = obj.matrix_world @ obj.data.vertices[pair.v0].co
        b = obj.matrix_world @ obj.data.vertices[pair.v1].co
        spline.points[0].co = (*a, 1.0)
        spline.points[1].co = (*b, 1.0)
    overlay = bpy.data.objects.new(f"AIWear_UVSeams_{obj.name}", curve)
    collection.objects.link(overlay)
    curve.materials.append(material)
    overlay.color = (1.0, 0.0, 0.0, 1.0)
    overlay.show_in_front = True
    overlay["ai_wear_overlay_for"] = obj.name
    overlay["ai_wear_seam_count"] = len(registry)
    return overlay


def _camera(scene, low, high):
    centre = 0.5 * (low + high)
    radius = max(0.5 * (high - low).length, 1e-3)
    data = bpy.data.cameras.new("AIWear_SeamOverview_CameraData")
    data.lens = 55.0
    camera = bpy.data.objects.new("AIWear_SeamOverview_Camera", data)
    scene.collection.objects.link(camera)
    direction = Vector((1.35, -1.75, 1.05)).normalized()
    fov_x = 2.0 * math.atan(data.sensor_width / (2.0 * data.lens))
    # At 16:9 the vertical field of view is narrower than the horizontal one.
    # Fit the bounding sphere to the limiting axis so tall assets (the chair)
    # are not cropped in the overview render.
    aspect = 1920.0 / 1080.0
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect)
    fit_fov = min(fov_x, fov_y)
    distance = radius / max(math.sin(fit_fov * 0.5), 1e-3) * 1.08
    camera.location = centre + direction * distance
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    data.clip_start = max(radius * 0.001, 0.001)
    data.clip_end = max(radius * 20.0, 100.0)
    scene.camera = camera
    return camera


output_dir = _arguments()
output_dir.mkdir(parents=True, exist_ok=True)
source = Path(bpy.data.filepath)
if not source.name:
    raise RuntimeError("The current Blender file has not been saved")

meshes = [obj for obj in bpy.context.scene.objects
          if obj.type == "MESH" and len(obj.data.polygons) and obj.data.uv_layers]
if not meshes:
    raise RuntimeError("No mesh with a UV layer found")

collection = bpy.data.collections.new("AIWear_UV_Seam_Overlay")
bpy.context.scene.collection.children.link(collection)
material = _material()
low, high = _world_bounds(meshes)
width = max((high - low).length * 0.00065, 0.0001)

records = []
for obj in meshes:
    layer_name = obj.data.uv_layers.active.name
    registry = seam_registry.build_seam_registry(obj, layer_name)
    seam_indices = {pair.edge_index for pair in registry}
    for edge in obj.data.edges:
        selected = edge.index in seam_indices
        edge.select = selected
        if selected:
            edge.use_seam = True
    obj["ai_wear_seam_count"] = len(registry)
    obj["ai_wear_seam_uv_layer"] = layer_name
    if registry:
        _overlay_curve(obj, registry, collection, material, width)
    records.append({
        "object": obj.name,
        "uv_layer": layer_name,
        "fusible_manifold_uv_seams": len(registry),
    })

scene = bpy.context.scene
_camera(scene, low, high)
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_shadows = True
scene.display.shading.show_cavity = True
if hasattr(scene.display.shading, "show_outline"):
    scene.display.shading.show_outline = True
scene.display.shading.background_type = "VIEWPORT"
scene.display.shading.background_color = (0.035, 0.035, 0.035)

for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        # Blender 5.1 exposes WIREFRAME/SOLID/RENDERED here (no MATERIAL enum).
        # SOLID + MATERIAL color shows the red curve without loading textures.
        space.shading.type = "SOLID"
        space.shading.color_type = "MATERIAL"
        if hasattr(space.shading, "show_outline"):
            space.shading.show_outline = True
        space.overlay.show_overlays = True
        if space.region_3d is not None:
            space.region_3d.view_perspective = "CAMERA"

stem = source.stem + "_seam_overlay"
blend_path = output_dir / f"{stem}.blend"
render_path = output_dir / f"{stem}.png"
manifest_path = output_dir / f"{stem}.json"

scene.render.filepath = str(render_path)
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), check_existing=False)

manifest = {
    "source": str(source),
    "overlay_blend": str(blend_path),
    "overview_render": str(render_path),
    "meaning": "red = manifold UV discontinuity processed by Seam Fusion",
    "excluded": "open/non-manifold boundary edges",
    "objects": records,
    "total_fusible_manifold_uv_seams": sum(r["fusible_manifold_uv_seams"]
                                             for r in records),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                         encoding="utf-8")
print("AIWEAR_SEAM_OVERLAY_JSON=" + json.dumps(manifest, ensure_ascii=False))
