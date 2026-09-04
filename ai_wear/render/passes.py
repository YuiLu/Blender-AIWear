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
        self._unlit_material_state = []

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
        _restore_unlit_materials(self._unlit_material_state)
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


def _socket_state(sock):
    """Capture an input's value and incoming links for exact restoration."""
    try:
        value = tuple(sock.default_value)
    except TypeError:
        value = sock.default_value
    return value, [link.from_socket for link in sock.links]


def _restore_unlit_materials(states) -> None:
    """Restore Principled inputs temporarily changed for an unlit capture."""
    for sock, value, sources in reversed(states):
        try:
            for link in list(sock.links):
                sock.node.id_data.links.remove(link)
            sock.default_value = value
            for source in sources:
                sock.node.id_data.links.new(source, sock)
        except Exception:
            # Render-state restoration is best effort, like the other saved
            # scene settings.  Do not mask the original rendering exception.
            pass


def _set_socket_value(sock, value, states) -> None:
    if sock is None:
        return
    states.append((sock, *_socket_state(sock)))
    for link in list(sock.links):
        sock.node.id_data.links.remove(link)
    try:
        sock.default_value = value
    except Exception:
        pass


def _make_object_materials_unlit(obj, state: _RenderState) -> None:
    """Temporarily render Principled materials as base-color emission.

    A white World with no lamps is still *lit* in Eevee: metallic or glossy
    Principled surfaces reflect that World and leave highlight gradients in the
    image sent to image-edit models.  This conversion preserves each material's
    Base Color source (including the AI overlay) but feeds it into emission,
    while blacking out the reflective BSDF contribution.
    """
    if obj is None:
        return
    try:
        from ..shader.wear_nodegroup import _find_surface_principled
    except Exception:
        return
    seen = set()
    for slot in getattr(obj, "material_slots", ()):
        mat = slot.material
        if mat is None or not mat.use_nodes or mat.node_tree is None:
            continue
        ptr = mat.as_pointer()
        if ptr in seen:
            continue
        seen.add(ptr)
        bsdf = _find_surface_principled(mat)
        if bsdf is None:
            continue
        base = bsdf.inputs.get("Base Color")
        emission = bsdf.inputs.get("Emission Color")
        if emission is None:
            emission = bsdf.inputs.get("Emission")
        strength = bsdf.inputs.get("Emission Strength")
        if base is None or emission is None:
            continue
        # Capture the source before clearing Base Color.  An output socket may
        # feed both inputs, so textures/procedural colors remain exact.
        source = base.links[0].from_socket if base.links else None
        try:
            base_value = tuple(base.default_value)
        except TypeError:
            base_value = base.default_value
        _set_socket_value(base, (0.0, 0.0, 0.0, 1.0), state._unlit_material_state)
        _set_socket_value(emission, base_value, state._unlit_material_state)
        if source is not None:
            try:
                mat.node_tree.links.new(source, emission)
            except Exception:
                pass
        _set_socket_value(bsdf.inputs.get("Metallic"), 0.0, state._unlit_material_state)
        _set_socket_value(bsdf.inputs.get("Roughness"), 1.0, state._unlit_material_state)
        specular = bsdf.inputs.get("Specular IOR Level")
        if specular is None:
            specular = bsdf.inputs.get("Specular")
        _set_socket_value(specular, 0.0, state._unlit_material_state)
        _set_socket_value(strength, 1.0, state._unlit_material_state)


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


def configure(scene, resolution: int, state: _RenderState,
              lighting: str = "neutral", unlit_object=None) -> None:
    """Set engine/resolution/format and (optionally) swap in temp lighting.

    ``lighting`` selects what lights the frame:

    - ``"neutral"`` — flat 0.5 world + one soft sun. Diagnostic; gives the
      material a bit of form but adds directional specular highlights.
    - ``"unlit"`` — temporarily converts the target's Principled Base Color to
      emission, removing all diffuse and specular lighting from the AI input.
    - ``"scene"`` — leave the scene's own world + lights untouched (e.g. the
      user's HDRI). If the scene has no World, use a neutral built-in World and
      sun so comparison renders cannot turn black.
    """
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
    if lighting == "scene" and scene.world is not None:
        return
    if lighting == "scene":
        # A .blend without a World otherwise renders as a black environment in
        # Eevee. Reuse the neutral diagnostic setup as a deterministic fallback.
        lighting = "fallback"
    # Temp world: neutral grey for diagnostics, flat white for the unlit pass.
    w = bpy.data.worlds.new("AIWear_World")
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background") or w.node_tree.nodes.new("ShaderNodeBackground")
    if lighting == "unlit":
        bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    else:
        bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    state._temp_world = w
    scene.world = w
    if lighting == "unlit":
        _make_object_materials_unlit(unlit_object, state)
        return
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
                 job=None, lighting: str = "neutral", unlit_object=None) -> str:
    """Render the clean view for `camera` to a PNG. Restores scene afterwards."""
    import bpy
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    state = _RenderState(scene)
    try:
        configure(scene, resolution, state, lighting, unlit_object=unlit_object)
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
