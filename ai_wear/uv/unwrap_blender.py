"""Mode A / Mode B + Blender auto-unwrap.

Mode A: pick an existing UV layer as the Target Wear UV, do NOT modify it.
Mode B: keep every original UV; reuse a production-ready authored layout when
possible, otherwise add an AI_WearUV layer and auto-unwrap it with Smart UV
Project + Pack Islands. If QC fails, retry with a wider angle and smaller
packing margin so surface texel density is not sacrificed.

Scene state (active object / mode / selection / UV layer / the area we may have
borrowed) is saved and restored, so the user's workspace is untouched afterwards.
"""

from __future__ import annotations

from array import array
import math
from typing import Optional, Tuple

from . import qc
from .rasterizer import _find_uv_layer


# Blender's Smart UV ``island_margin`` uses a scaled packing factor, not a
# literal texel distance. Feeding padding_texels / resolution into it produced
# a 0.015625 margin at 1K; on a mesh with thousands of islands that shrank the
# actual UV surface coverage to ~2%. Padding is a later texture-domain step and
# must not be mapped 1:1 to this packing control.
MIN_SCALED_MARGIN = 0.00025
MAX_SCALED_MARGIN = 0.002
COPY_MAX_OVERLAP = 0.0001


def _scaled_uv_margin(work_resolution: int) -> float:
    """A conservative Smart-UV margin that preserves usable texel density."""
    return min(MAX_SCALED_MARGIN,
               max(MIN_SCALED_MARGIN, 0.5 / max(1, int(work_resolution))))


def _reusable_uv_report(report: dict) -> bool:
    """Whether an authored UV can safely seed the independent wear layer."""
    return bool(
        report.get("ok", False)
        and report.get("overlap_ratio", 1.0) <= COPY_MAX_OVERLAP
        and report.get("utilization", 0.0) >= qc.MIN_UTILIZATION
    )


def _copy_uv_layer(source, target) -> None:
    """Copy per-loop coordinates without making the source the render UV."""
    if len(source.data) != len(target.data):
        raise RuntimeError("UV layer loop counts do not match.")
    values = array("f", [0.0]) * (len(source.data) * 2)
    source.data.foreach_get("uv", values)
    target.data.foreach_set("uv", values)


def _tag_report(report: dict, strategy: str, source: Optional[str] = None) -> dict:
    report["mode_b_strategy"] = strategy
    if source:
        report["source_uv_layer"] = source
    return report


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
        self.mesh_flags = None

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
        # Hidden edit-mode elements are still renderable geometry. Smart UV's
        # select-all deliberately skips them, which left half of dining_chair
        # collapsed at (0,0). Temporarily reveal them during unwrap, but retain
        # every hide/selection bit so the artist's edit state is unchanged.
        if obj and obj.type == "MESH":
            mesh = obj.data
            self.mesh_flags = {
                "vert_hide": [item.hide for item in mesh.vertices],
                "vert_select": [item.select for item in mesh.vertices],
                "edge_hide": [item.hide for item in mesh.edges],
                "edge_select": [item.select for item in mesh.edges],
                "face_hide": [item.hide for item in mesh.polygons],
                "face_select": [item.select for item in mesh.polygons],
            }

    def restore(self):
        import bpy
        target_mode = self.mode
        if self.obj and self.obj.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        if self.obj and self.obj.type == "MESH" and self.mesh_flags:
            mesh = self.obj.data
            groups = (
                (mesh.vertices, "vert_hide", "hide"),
                (mesh.vertices, "vert_select", "select"),
                (mesh.edges, "edge_hide", "hide"),
                (mesh.edges, "edge_select", "select"),
                (mesh.polygons, "face_hide", "hide"),
                (mesh.polygons, "face_select", "select"),
            )
            for elements, key, attribute in groups:
                values = self.mesh_flags[key]
                if len(elements) == len(values):
                    for element, value in zip(elements, values):
                        setattr(element, attribute, value)
            mesh.update()
        if self.obj and target_mode != "OBJECT" and self.obj.mode != target_mode:
            try:
                bpy.ops.object.mode_set(mode=target_mode)
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
    # Edit-mode hidden faces are omitted by select_all and would retain the
    # newly-created layer's default (0,0) UV. Reveal them only for this unwrap;
    # _State.restore reinstates the exact original hide/selection flags.
    bpy.ops.mesh.reveal(select=False)
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
                                     scale_to_bounds=True)
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


def setup_mode_b(obj, layer_name: str, work_resolution: int,
                 angle_deg: float = 66.0,
                 padding_texels: int = 2,
                 depsgraph=None) -> Tuple[bool, str, dict]:
    """Create AI_WearUV, preserve all existing UVs, auto-unwrap, then QC."""
    import bpy
    if obj.type != "MESH":
        return False, "Object is not a mesh.", {}
    state = _State()
    state.save_uv(obj)
    try:
        dg = depsgraph or bpy.context.evaluated_depsgraph_get()

        # Preserve an artist-authored wear layer when it already has enough
        # usable texel area. Re-running Smart UV on every Generate used to
        # destroy a good layout even though nothing about the mesh had changed.
        layer = _find_uv_layer(obj.data, layer_name)
        if layer is not None:
            existing_report = qc.compute_uv_qc(
                obj, layer_name, depsgraph=dg, low_res=256)
            if _reusable_uv_report(existing_report):
                obj.data.uv_layers.active_index = list(obj.data.uv_layers).index(layer)
                return True, layer_name, _tag_report(
                    existing_report, "kept_existing", layer_name)

        # If the asset already has a production-quality, non-overlapping UV,
        # clone it into a separate wear layer. This keeps the base material UV
        # untouched while avoiding needless fragmentation from Smart Project.
        candidates = []
        for source in obj.data.uv_layers:
            if source.name == layer_name:
                continue
            source_report = qc.compute_uv_qc(
                obj, source.name, depsgraph=dg, low_res=256)
            if _reusable_uv_report(source_report):
                candidates.append((source_report["utilization"], source, source_report))
        if candidates:
            _util, source, _source_report = max(candidates, key=lambda item: item[0])
            if layer is None:
                layer = obj.data.uv_layers.new(name=layer_name, do_init=False)
            _copy_uv_layer(source, layer)
            obj.data.uv_layers.active_index = list(obj.data.uv_layers).index(layer)
            obj.data.update()
            # The downstream rasterizer reads the evaluated mesh. Force the
            # depsgraph to see the copied loop UVs immediately; otherwise a
            # Generate started in the same UI event can still receive the old
            # evaluated AI_WearUV for one cycle.
            try:
                dg.update()
            except Exception:
                pass
            report = qc.compute_uv_qc(obj, layer.name, depsgraph=dg, low_res=256)
            return True, layer.name, _tag_report(
                report, "copied_existing", source.name)

        # No suitable source exists: generate a fresh, unique wear atlas.
        if layer is None:
            layer = obj.data.uv_layers.new(name=layer_name, do_init=False)
        # Capture the layer's real name as a plain string NOW, before any UV op.
        # _run_uv_ops toggles EDIT mode and runs smart_project/pack_islands, which
        # invalidates this RNA object reference; accessing layer.name afterwards
        # reads freed memory and raises UnicodeDecodeError (garbage bytes that
        # differ run-to-run, e.g. 0xb0 / 0xd0). We pass the captured string to QC
        # and re-derive the layer object after each UV op below.
        real_name = layer.name
        obj.data.uv_layers.active_index = obj.data.uv_layers[:].index(layer)

        island_margin = _scaled_uv_margin(work_resolution)
        pack_margin = island_margin
        angle = math.radians(angle_deg)

        _run_uv_ops(obj, angle, island_margin, pack_margin, state)
        bpy.ops.object.mode_set(mode="OBJECT")
        layer = _find_uv_layer(obj.data, real_name) or obj.data.uv_layers.active

        report = qc.compute_uv_qc(obj, real_name, depsgraph=dg, low_res=256)
        if not report.get("ok", False):
            # One retry with a wider angle and a smaller packing margin. A low
            # utilization failure needs more texel area, not the old doubled
            # margin which made the shrinkage worse.
            angle2 = math.radians(min(89.0, angle_deg + 15.0))
            margin2 = max(MIN_SCALED_MARGIN, island_margin * 0.5)
            pack2 = margin2
            _run_uv_ops(obj, angle2, margin2, pack2, state)
            bpy.ops.object.mode_set(mode="OBJECT")
            layer = _find_uv_layer(obj.data, real_name) or obj.data.uv_layers.active
            report = qc.compute_uv_qc(obj, real_name, depsgraph=dg, low_res=256)
        return True, real_name, _tag_report(report, "smart_project")
    finally:
        state.restore()
