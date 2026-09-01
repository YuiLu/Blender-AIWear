"""Clean-view render passes.

Renders a stable Clean RGB image per camera (neutral background + simple
lighting) for the AI provider and as the reference for mask extraction. Depth
and surface normals are NOT rendered here — they come from the mesh via the
rasterizer + projection, which keeps the visibility test self-consistent and
avoids fragile multilayer-EXR loading.

Everything the render touches (engine, resolution, world, temp lights, output
format) is snapshotted and restored so the user's scene is untouched.
"""

from __future__ import annotations

import os
from typing import Optional


class _RenderState:
    def __init__(self, scene):
        self.scene = scene
        r = scene.render
        self.engine = r.engine
        self.res_x = r.resolution_x
        self.res_y = r.resolution_y
        self.res_pct = r.resolution_percentage
        self.film_transparent = r.film_transparent
        self.filepath = r.filepath
        self.file_format = r.image_settings.file_format
        self.color_depth = r.image_settings.color_depth
        self.color_mode = r.image_settings.color_mode
        self.world = scene.world
        self.cam = scene.camera
        self.eevee_samples = getattr(scene.eevee, "taa_render_samples", 16) if hasattr(scene, "eevee") else None
        self._temp_lights = []
        self._temp_world = None

    def restore(self):
        import bpy
        r = self.scene.render
        r.engine = self.engine
        r.resolution_x = self.res_x
        r.resolution_y = self.res_y
        r.resolution_percentage = self.res_pct
        r.film_transparent = self.film_transparent
        r.filepath = self.filepath
        r.image_settings.file_format = self.file_format
        r.image_settings.color_depth = self.color_depth
        r.image_settings.color_mode = self.color_mode
        if self.eevee_samples is not None and hasattr(self.scene, "eevee"):
            self.scene.eevee.taa_render_samples = self.eevee_samples
        self.scene.world = self.world
        if self._temp_world and self._temp_world.users == 0:
            try:
                bpy.data.worlds.remove(self._temp_world)
            except Exception:
                pass
        for L in self._temp_lights:
            try:
                bpy.data.objects.remove(L, do_unlink=True)
            except Exception:
                pass
        self.scene.camera = self.cam


def _eevee_id() -> str:
    """Return the EEVEE engine id valid in the running Blender.

    3.6 uses ``BLENDER_EEVEE``; 4.2–4.x renamed it to ``BLENDER_EEVEE_NEXT``;
    5.x reverted the branding so the id is ``BLENDER_EEVEE`` again. Hard-coding
    on ``bpy.app.version`` is therefore wrong on 5.x. Probe the live render
    engine enum instead so a single code path works across all versions.
    """
    import bpy
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        valid = {e.identifier for e in prop.enum_items}
    except Exception:
        return "BLENDER_EEVEE"
    for c in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if c in valid:
            return c
    return "BLENDER_EEVEE"


def _set_engine(scene) -> None:
    """Set the render engine to EEVEE, tolerant of the 4.x rename / 5.x revert."""
    import bpy
    tried = []
    for eng in (_eevee_id(), "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        if eng in tried:
            continue
        tried.append(eng)
        try:
            scene.render.engine = eng
            return
        except (TypeError, ValueError):
            continue
    # leave whatever the scene already uses


def configure(scene, resolution: int, state: _RenderState) -> None:
    import bpy
    r = scene.render
    _set_engine(scene)
    r.resolution_x = resolution
    r.resolution_y = resolution
    r.resolution_percentage = 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_depth = "8"
    r.image_settings.color_mode = "RGBA"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 16
    # Temp neutral world
    w = bpy.data.worlds.new("AIWear_World")
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background") or w.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    state._temp_world = w
    scene.world = w
    # A single soft sun for shading (reads material form to the AI)
    light_data = bpy.data.lights.new(name="AIWear_Sun_data", type="SUN")
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("AIWear_Sun", light_data)
    light_obj.matrix_world = (
        __import__("mathutils").Matrix.Translation((3, -3, 5))
        @ __import__("mathutils").Vector((1, -1, 1)).normalized().to_track_quat("-Z", "Y").to_matrix().to_4x4()
    )
    scene.collection.objects.link(light_obj)
    state._temp_lights.append(light_obj)


def render_clean(scene, camera, out_png_path: str, resolution: int = 1024,
                 job=None) -> str:
    """Render the clean view for `camera` to a PNG. Restores scene afterwards."""
    import bpy
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    state = _RenderState(scene)
    try:
        configure(scene, resolution, state)
        scene.camera = camera
        scene.render.filepath = out_png_path
        if job is not None:
            job.touch()
        bpy.ops.render.render(write_still=True)
    finally:
        state.restore()
    if not os.path.exists(out_png_path):
        raise RuntimeError(f"Render did not produce {out_png_path}")
    return out_png_path


def render_depth(scene, camera, out_exr_path: str, resolution: int = 1024,
                 job=None) -> Optional[str]:
    """Optional Blender depth pass to OpenEXR (kept for advanced users).

    The main pipeline uses the software z-buffer in surface.projection for
    consistency; this is here as a fallback / debugging aid.
    """
    import bpy
    os.makedirs(os.path.dirname(out_exr_path), exist_ok=True)
    state = _RenderState(scene)
    try:
        configure(scene, resolution, state)
        scene.camera = camera
        scene.view_layers[0].use_pass_z = True
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = out_exr_path
        bpy.ops.render.render(write_still=True)
    finally:
        state.restore()
    return out_exr_path if os.path.exists(out_exr_path) else None
