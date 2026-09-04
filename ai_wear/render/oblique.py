"""Fixed oblique hero camera + render, for consistent comparison shots.

Single source of truth for the "oblique side view" used by the pipeline's
end-of-run comparison snapshot and by ``tools/render_oblique.py``. The camera
is deterministic from the object's world bounding box, so the same model always
yields the same framing (and EEVEE's 16-sample TAA keeps it pixel-stable across
runs). Keeping the direction/lens/distance here — not duplicated in tools/ —
prevents the two callers from drifting apart.
"""

from __future__ import annotations

import os

from mathutils import Vector

from . import passes

CAMERA_NAME = "AIWear_ObliqueCam"
# Same direction as the seam-supplement hero camera for visual consistency.
DIRECTION = Vector((1.15, -1.35, 0.82)).normalized()
LENS = 58.0
DISTANCE_FACTOR = 3.15
RENDER_RESOLUTION = 1024


def make_oblique_camera(scene, obj, name=CAMERA_NAME):
    """Create-or-replace a fixed oblique hero camera framed on obj's bbox.

    Returns ``(camera, radius)``. Idempotent: removes any existing camera of the
    same name first, so repeated runs reproduce the exact same framing.
    """
    import bpy
    existing = bpy.data.objects.get(name)
    if existing is not None:
        old_data = existing.data
        bpy.data.objects.remove(existing, do_unlink=True)
        if old_data and old_data.users == 0:
            bpy.data.cameras.remove(old_data)

    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    radius = max((c - center).length for c in corners)
    cam_data = bpy.data.cameras.new(name + "Data")
    cam_data.lens = LENS
    cam = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + DIRECTION * radius * DISTANCE_FACTOR
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    return cam, radius


def render_oblique(scene, obj, out_png, resolution=RENDER_RESOLUTION):
    """Render ``obj`` from the fixed oblique camera to ``out_png`` (EEVEE, scene-lit).

    Renders whatever the object currently looks like (base material + any wear
    overlay already attached), lit by the scene's own world/HDRI + lights so it
    matches the viewport. If this asset has no World, ``passes.render_clean``
    supplies a neutral built-in environment instead of producing a black shot.
    Scene render settings and camera are restored afterwards.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cam, _ = make_oblique_camera(scene, obj)
    # Scene lighting (the user's HDRI + lights), not the neutral diagnostic
    # world+sun, so the comparison frame matches the viewport.
    return passes.render_clean(scene, cam, str(out_png), resolution, lighting="scene")
