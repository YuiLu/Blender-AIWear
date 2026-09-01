"""Mode A / Mode B + Blender auto-unwrap.

Mode A: pick an existing UV layer as the Target Wear UV, do NOT modify it.
Mode B: keep every original UV; add a fresh AI_WearUV layer and auto-unwrap it
with Smart UV Project + Pack Islands, then run the same UV QC. If QC fails the
first time, retry once with a wider angle limit and larger margin.

Scene state (active object / mode / selection / UV layer / the area we may have
borrowed) is saved and restored, so the user's workspace is untouched afterwards.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from . import qc
from .rasterizer import _find_uv_layer


# --- scene state -------------------------------------------------------------

class _State:
    def __init__(self):
        import bpy
        self.obj = bpy.context.active_object
        self.mode = self.obj.mode if self.obj else "OBJECT"
        self.selected = [o for o in bpy.context.selected_objects]
        self.active_layer_index = None
        self.area = None
        self.saved_area_type = None
        self.saved_mesh_select_mode = None

    def save_uv(self, obj):
        import bpy
        if obj and obj.type == "MESH" and obj.data.uv_layers:
            self.active_layer_index = obj.data.uv_layers.active_index
        # remember the user's mesh select mode so we can restore it after the
        # temporary face-select-mode used for UV operations
        try:
            self.saved_mesh_select_mode = tuple(bpy.context.tool_settings.mesh_select_mode)
        except Exception:
            self.saved_mesh_select_mode = None

    def restore(self):
        import bpy
        if self.obj and self.obj.mode != self.mode:
            try:
                bpy.ops.object.mode_set(mode=self.mode)
            except Exception:
                pass
        if self.obj and self.obj.type == "MESH" and self.active_layer_index is not None:
            try:
                self.obj.data.uv_layers.active_index = self.active_layer_index
            except Exception:
                pass
        # restore selection
        try:
            for o in bpy.context.selected_objects:
                o.select_set(False)
            for o in self.selected:
                o.select_set(True)
            if self.obj:
                bpy.context.view_layer.objects.active = self.obj
        except Exception:
            pass
        if self.area is not None and self.saved_area_type is not None:
            try:
                self.area.type = self.saved_area_type
            except Exception:
                pass
        if self.saved_mesh_select_mode is not None:
            try:
                bpy.context.tool_settings.mesh_select_mode = self.saved_mesh_select_mode
            except Exception:
                pass


def _ensure_uv_area() -> Tuple[object, Optional[object], Optional[str]]:
    """Return (area, region, saved_area_type). Borrow a 3D view if needed."""
    import bpy
    screen = bpy.context.screen
    if screen is None:
        return None, None, None
    for area in screen.areas:
        if area.type == "IMAGE_EDITOR":
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            return area, region, None
    for area in screen.areas:
        if area.type == "VIEW_3D":
            saved = area.type
            try:
                area.type = "IMAGE_EDITOR"
            except Exception:
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            return area, region, saved
    return None, None, None


def _run_uv_ops(obj, angle_rad: float, island_margin: float,
                pack_margin: float, state: _State) -> None:
    import bpy
    bpy.ops.object.mode_set(mode="EDIT")
    # Face select mode + select all faces so UV operators act on the whole mesh.
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    bpy.ops.mesh.select_all(action="SELECT")
    area, region, saved = _ensure_uv_area()
    state.area = area
    state.saved_area_type = saved
    if area is None:
        raise RuntimeError("No screen area available for UV operations. Run from the GUI.")
    # NOTE: do not pass "mode" — it is a derived context member, and "EDIT" is not
    # a valid value (the real mesh-edit mode is "EDIT_MESH"). An invalid override
    # key can confuse operator poll checks.
    kwargs = {"window": bpy.context.window, "screen": bpy.context.screen,
              "area": area, "edit_object": obj,
              "selected_editable_objects": [obj], "object": obj}
    if region is not None:
        kwargs["region"] = region
    try:
        with bpy.context.temp_override(**kwargs):
            # Smart UV Project produces the actual unwrap. It works on the whole
            # mesh regardless of UV selection.
            bpy.ops.uv.smart_project(angle_limit=angle_rad,
                                     island_margin=island_margin,
                                     scale_to_bounds=False)
            # Pack Islands only tightens the layout; it needs UVs selected, and
            # smart_project may leave the selection cleared. Re-select all so
            # pack has something to operate on.
            try:
                bpy.ops.uv.select_all(action="SELECT")
            except Exception:
                pass
            try:
                bpy.ops.uv.pack_islands(rotate=True, margin=pack_margin)
            except Exception as e:
                # pack_islands poll can fail from a timer/worker context even
                # with the override (a freshly-converted image-editor area's
                # space may not be fully ready, or no UVs report as selected).
                # Smart UV Project already gave a usable, mostly non-overlapping
                # unwrap, so packing is optional — skip it and let the UV QC
                # step flag any real overlaps. This keeps the pipeline running
                # instead of crashing on a cosmetic step.
                print(f"[AI Wear] pack_islands skipped ({e}); using Smart UV Project result")
    finally:
        if saved is not None and area is not None:
            try:
                area.type = saved
            except Exception:
                pass


# --- public API --------------------------------------------------------------

def setup_mode_a(obj, layer_name: Optional[str]) -> Tuple[bool, str]:
    """Select an existing UV layer as the Target Wear UV without modifying it."""
    import bpy
    if obj.type != "MESH":
        return False, "Object is not a mesh."
    layer = _find_uv_layer(obj.data, layer_name)
    if layer is None:
        if layer_name:
            return False, f"UV layer '{layer_name}' not found. Switch to Mode B or pick another."
        return False, "Object has no UV layer."
    # Set as active so downstream reads use it, but DO NOT alter its coordinates.
    idx = obj.data.uv_layers[:].index(layer)
    obj.data.uv_layers.active_index = idx
    return True, layer.name


def setup_mode_b(obj, layer_name: str, texture_size: int,
                 angle_deg: float = 66.0,
                 padding_texels: int = 16,
                 depsgraph=None) -> Tuple[bool, str, dict]:
    """Create AI_WearUV, preserve all existing UVs, auto-unwrap, then QC."""
    import bpy
    if obj.type != "MESH":
        return False, "Object is not a mesh.", {}
    state = _State()
    state.save_uv(obj)
    try:
        # Create the new layer (or reuse if it already exists)
        layer = None
        for l in obj.data.uv_layers:
            if l.name == layer_name:
                layer = l
                break
        if layer is None:
            layer = obj.data.uv_layers.new(name=layer_name)
        # Capture the layer's real name as a plain string NOW, before any UV op.
        # _run_uv_ops toggles EDIT mode and runs smart_project/pack_islands, which
        # invalidates this RNA object reference; accessing layer.name afterwards
        # reads freed memory and raises UnicodeDecodeError (garbage bytes that
        # differ run-to-run, e.g. 0xb0 / 0xd0). We pass the captured string to QC
        # and re-derive the layer object after each UV op below.
        real_name = layer.name
        obj.data.uv_layers.active_index = obj.data.uv_layers[:].index(layer)

        island_margin = max(0.002, padding_texels / max(1, texture_size))
        pack_margin = max(0.001, padding_texels / max(1, texture_size * 2))
        angle = math.radians(angle_deg)

        _run_uv_ops(obj, angle, island_margin, pack_margin, state)
        bpy.ops.object.mode_set(mode="OBJECT")
        layer = _find_uv_layer(obj.data, real_name) or obj.data.uv_layers.active

        report = qc.compute_uv_qc(obj, real_name, depsgraph=depsgraph, low_res=256)
        if not report.get("ok", False) and report.get("overlap_ratio", 0) > 0.02:
            # One retry with a wider angle + larger margin
            angle2 = math.radians(min(89.0, angle_deg + 15.0))
            margin2 = island_margin * 2.0
            pack2 = pack_margin * 2.0
            _run_uv_ops(obj, angle2, margin2, pack2, state)
            bpy.ops.object.mode_set(mode="OBJECT")
            layer = _find_uv_layer(obj.data, real_name) or obj.data.uv_layers.active
            report = qc.compute_uv_qc(obj, real_name, depsgraph=depsgraph, low_res=256)
        return True, real_name, report
    finally:
        state.restore()
