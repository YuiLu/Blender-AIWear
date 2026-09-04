"""Exercise Mode-B unwrap in a real Blender window without saving the .blend.

Background Blender has no screen/UV-editor context, so it cannot faithfully
exercise the same Smart UV operators used by the interactive pipeline.  This
script schedules the check after the UI is ready, prints one machine-readable
result line, then exits Blender.  It never calls an AI provider.

Usage:
  blender "asset.blend" --python tools/verify_mode_b.py -- [object-name]
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_wear
from ai_wear.uv import qc, unwrap_blender


def _json_uv_stats(mesh, layer_name):
    mesh.calc_loop_triangles()
    stats = qc._uv_triangle_stats(mesh, layer_name)
    return {key: value for key, value in stats.items()
            if key not in {"tri_uv", "signed_area"}}


def _edit_state_counts(obj):
    return {
        "hidden_vertices": sum(item.hide for item in obj.data.vertices),
        "hidden_edges": sum(item.hide for item in obj.data.edges),
        "hidden_faces": sum(item.hide for item in obj.data.polygons),
        "selected_vertices": sum(item.select for item in obj.data.vertices),
        "selected_edges": sum(item.select for item in obj.data.edges),
        "selected_faces": sum(item.select for item in obj.data.polygons),
    }


def _argument_object_name():
    if "--" not in sys.argv:
        return None
    args = sys.argv[sys.argv.index("--") + 1:]
    return args[0] if args else None


def _main():
    try:
        try:
            ai_wear.unregister()
        except Exception:
            pass
        ai_wear.register()
        requested = _argument_object_name()
        obj = bpy.data.objects.get(requested) if requested else None
        if obj is None:
            obj = bpy.context.view_layer.objects.active
        if obj is None or obj.type != "MESH":
            obj = next((item for item in bpy.context.scene.objects
                        if item.type == "MESH"), None)
        if obj is None:
            raise RuntimeError("No mesh object found")
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        edit_state_before = _edit_state_counts(obj)
        ok, layer, report = unwrap_blender.setup_mode_b(
            obj, "AI_WearUV", 1024,
            depsgraph=bpy.context.evaluated_depsgraph_get())
        base_stats = _json_uv_stats(obj.data, layer)
        eval_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        eval_mesh = eval_obj.to_mesh()
        try:
            evaluated_stats = _json_uv_stats(eval_mesh, layer)
        finally:
            eval_obj.to_mesh_clear()
        print("AIWEAR_MODE_B_JSON=" + json.dumps({
            "ok": ok,
            "layer": layer,
            "report": report,
            "base_stats": base_stats,
            "evaluated_stats": evaluated_stats,
            "edit_state_before": edit_state_before,
            "edit_state_after": _edit_state_counts(obj),
            "modifiers": [
                {"name": modifier.name, "type": modifier.type,
                 "show_viewport": modifier.show_viewport}
                for modifier in obj.modifiers
            ],
        }, ensure_ascii=False), flush=True)
    except Exception:
        traceback.print_exc()
        print("AIWEAR_MODE_B_FAILED", flush=True)
    finally:
        bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(_main, first_interval=0.5)
