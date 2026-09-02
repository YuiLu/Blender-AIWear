"""Automatic coverage cameras + framing.

Compute the model's world bbox, frame it with a perspective camera, and place a
preset number of cameras around it for multi-view capture. Each camera is a
real object named AIWearCam_* so it can be reused as the scene camera; the
caller cleans them up afterwards.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from mathutils import Matrix, Vector

CAM_PREFIX = "AIWearCam_"


def compute_framing(obj, depsgraph) -> Tuple[Vector, float]:
    """Return (world-space center, bounding radius) for the evaluated mesh."""
    import numpy as np
    eval_obj = obj.evaluated_get(depsgraph)
    emesh = eval_obj.to_mesh()
    try:
        nv = len(emesh.vertices)
        if nv == 0:
            from mathutils import Vector as V
            return V((0, 0, 0)), 1.0
        co = np.empty(nv * 3, dtype=np.float32)
        emesh.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        m = np.array(obj.matrix_world, dtype=np.float32)
        world = co @ m[:3, :3].T + m[:3, 3]
        mn = world.min(axis=0); mx = world.max(axis=0)
        center = (mn + mx) / 2.0
        # Bounding-sphere radius (half the bbox diagonal), NOT the max half-extent.
        # The max half-extent only fits axis-aligned views; for an oblique/equator
        # view the projected bbox diagonal can be up to r*sqrt(3), which exceeds
        # the frame half-extent (1.18*r) and clips the object out of frame. The
        # half-diagonal is the conservative radius that fits every orientation.
        radius = float(np.linalg.norm((mx - mn) / 2.0))
        return Vector((float(center[0]), float(center[1]), float(center[2]))), max(radius, 1e-4)
    finally:
        eval_obj.to_mesh_clear()


def _make_camera(name: str, location: Vector, target: Vector) -> "bpy.types.Object":
    import bpy
    cam_data = bpy.data.cameras.new(name=name + "_data")
    cam_data.lens = 50.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "AUTO"
    cam_data.clip_start = 0.005
    cam_data.clip_end = 1e6
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    direction = (target - location)
    direction.normalize()
    quat = direction.to_track_quat("-Z", "Y")
    cam_obj.matrix_world = Matrix.Translation(location) @ quat.to_matrix().to_4x4()
    return cam_obj


def _equator_dirs(n: int) -> List[Vector]:
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        out.append(Vector((math.cos(a), math.sin(a), 0.0)))
    return out


def _fibonacci_dirs(n: int) -> List[Vector]:
    """Approximately uniform sphere directions for exact-count experiments."""
    n = max(1, int(n))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * ((i + 0.5) / n)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        phi = i * golden
        out.append(Vector((radius * math.cos(phi), radius * math.sin(phi), z)))
    return out


def generate_views(scene, obj, preset: str, count: int,
                   depsgraph=None) -> List["bpy.types.Object"]:
    import bpy
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    center, radius = compute_framing(obj, dg)
    # Distance so the object fills ~85% of frame with a 50mm / 36mm sensor.
    fov_x = 2 * math.atan(36.0 / (2 * 50.0))
    dist = (radius / math.tan(fov_x / 2.0)) * 1.18

    dirs: List[Vector] = []
    if preset == "AUTO_6":
        dirs = _equator_dirs(4)
        dirs.append(Vector((0, 0, 1)))      # top
        dirs.append(Vector((0, 0, -1)))     # bottom
    elif preset == "AUTO_8":
        dirs = _equator_dirs(6)
        dirs.append(Vector((0, 0, 1)))
        dirs.append(Vector((0, 0, -1)))
    elif preset == "TURNTABLE_4":
        dirs = _equator_dirs(4)
    elif preset == "AUTO_COUNT":
        dirs = _fibonacci_dirs(count)
    elif preset == "CUSTOM":
        # Use any existing camera tagged by the user (name starts with CAM_PREFIX)
        cams = [o for o in scene.objects if o.type == "CAMERA"
                and o.name.startswith(CAM_PREFIX)]
        return cams[:max(count, 1)] if cams else []
    else:
        dirs = _equator_dirs(max(3, count))

    # Tilt equatorial views slightly to expose top edges
    cams: List["bpy.types.Object"] = []
    for i, d in enumerate(dirs):
        loc = center + d * dist
        cams.append(_make_camera(f"{CAM_PREFIX}V{i}", loc, center))
    return cams


def cleanup_views(cameras: List["bpy.types.Object"]) -> None:
    import bpy
    for c in cameras:
        try:
            bpy.data.objects.remove(c, do_unlink=True)
        except Exception:
            pass
