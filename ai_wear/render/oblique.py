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
BACKGROUND_COLOR = (0.055, 0.055, 0.055, 1.0)


def _make_viewport_world(scene):
    """Clone the scene World and replace only camera rays with dark grey.

    Environment lighting and glossy reflections still use the user's current
    World/HDRI, matching Rendered Viewport shading. The camera sees a stable
    neutral background instead of the HDRI photograph. The clone is temporary,
    so no World nodes in the user's scene are modified.
    """
    import bpy

    source_world = scene.world
    world = (source_world.copy() if source_world is not None
             else bpy.data.worlds.new("AIWear_ObliqueWorld"))
    world.name = "AIWear_ObliqueWorld"
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    output = next((node for node in nodes
                   if node.bl_idname == "ShaderNodeOutputWorld"
                   and node.is_active_output), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputWorld")
    surface = output.inputs["Surface"]
    lighting_source = surface.links[0].from_socket if surface.links else None
    for link in list(surface.links):
        links.remove(link)
    if lighting_source is None:
        lighting = nodes.new("ShaderNodeBackground")
        lighting.name = "AIWear_ViewportLighting"
        lighting.inputs["Color"].default_value = tuple(world.color) + (1.0,)
        lighting_source = lighting.outputs["Background"]

    camera_background = nodes.new("ShaderNodeBackground")
    camera_background.name = "AIWear_CameraBackground"
    camera_background.inputs["Color"].default_value = BACKGROUND_COLOR
    camera_background.inputs["Strength"].default_value = 1.0
    light_path = nodes.new("ShaderNodeLightPath")
    light_path.name = "AIWear_LightPath"
    mix = nodes.new("ShaderNodeMixShader")
    mix.name = "AIWear_WorldMix"

    links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(lighting_source, mix.inputs[1])
    links.new(camera_background.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], surface)
    return world


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
    """Render the fixed oblique camera like Rendered Viewport on dark grey.

    Renders whatever the object currently looks like (base material + any wear
    overlay already attached). The scene World/HDRI and lights continue to drive
    shading and reflections, but the camera sees a neutral grey background.
    The user's World, render settings, and active camera are restored afterwards.
    """
    import bpy

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cam, _ = make_oblique_camera(scene, obj)
    original_world = scene.world
    world = None
    try:
        world = _make_viewport_world(scene)
        scene.world = world
        return passes.render_clean(
            scene, cam, str(out_png), resolution, lighting="scene")
    finally:
        scene.world = original_world
        if world is not None:
            try:
                bpy.data.worlds.remove(world)
            except Exception:
                pass
