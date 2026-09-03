"""Operator classes: run pipeline, cancel, preflight, export, presets."""

from __future__ import annotations

import os

import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, FloatProperty
from bpy_extras.io_utils import ExportHelper

from . import pipeline
from .. import presets
from ..cache import job_cache
from ..cache.job_cache import JobState


class AIWEAR_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "ai_wear.run_pipeline"
    bl_label = "Generate Wear Texture"
    bl_description = "Run the full UV → render → AI → surface field → WearTime pipeline"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if pipeline.is_running(context):
            self.report({"WARNING"}, "A pipeline is already running. Cancel it first.")
            return {"CANCELLED"}
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}
        try:
            pipeline.start_pipeline(context)
            self.report({"INFO"}, "Pipeline started. See the panel for progress.")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to start: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class AIWEAR_OT_replay_downstream(bpy.types.Operator):
    bl_idname = "ai_wear.replay_downstream"
    bl_label = "Replay Downstream (no AI)"
    bl_description = ("Re-run only mask → projection → fusion → WearTime → bake "
                      "from the cached clean/worn images of the last full run on "
                      "this object. No render, no AI call — for iterating on the "
                      "surface field without spending API budget")
    bl_options = {"REGISTER"}

    def execute(self, context):
        if pipeline.is_running(context):
            self.report({"WARNING"}, "A pipeline is already running. Cancel it first.")
            return {"CANCELLED"}
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object first.")
            return {"CANCELLED"}
        try:
            pipeline.start_replay(context)
            self.report({"INFO"}, "Downstream replay started (no AI). See the panel.")
        except Exception as e:
            self.report({"ERROR"}, f"Failed to start replay: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class AIWEAR_OT_cancel(bpy.types.Operator):
    bl_idname = "ai_wear.cancel"
    bl_label = "Cancel"
    bl_description = "Cancel the running pipeline"
    bl_options = {"REGISTER"}

    def execute(self, context):
        job = pipeline.get_active_job(context)
        if job is None:
            self.report({"WARNING"}, "No active job.")
            return {"CANCELLED"}
        job_cache.request_cancel(job.id)
        self.report({"INFO"}, "Cancellation requested.")
        return {"FINISHED"}


class AIWEAR_OT_preflight(bpy.types.Operator):
    bl_idname = "ai_wear.preflight"
    bl_label = "Run Preflight / UV QC"
    bl_description = "Check the mesh and UV without running AI"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..uv import qc
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}
        s = context.scene.ai_wear
        layer = s.target_uv_layer if s.uv_mode == "MODE_A" else (s.target_uv_layer or "AI_WearUV")
        report = qc.compute_uv_qc(obj, layer if s.uv_mode == "MODE_A" else None, low_res=256)
        text = qc.format_report(report)
        obj.ai_wear.last_qc_report = text
        obj.ai_wear.qc_ok = bool(report.get("ok"))
        self.report({"INFO"} if report.get("ok") else {"WARNING"},
                    f"UV QC ok={report.get('ok')} overlap={report.get('overlap_ratio',0):.2%}")
        # print full report to the info console
        print(text)
        return {"FINISHED"}


class AIWEAR_OT_export_weartime(bpy.types.Operator, ExportHelper):
    bl_idname = "ai_wear.export_weartime"
    bl_label = "Export WearTime"
    bl_description = "Export the 16-bit WearTime map (Substance/Blender ready)"
    bl_options = {"REGISTER"}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png;*.exr", options={"HIDDEN"})

    def execute(self, context):
        src = _current_weartime_path(context)
        if not src or not os.path.exists(src):
            self.report({"ERROR"}, "No WearTime texture found. Run the pipeline first.")
            return {"CANCELLED"}
        fmt = context.scene.ai_wear.export_format
        from .. import utils
        import numpy as np
        arr = utils.load_image_rgba(src)
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.float32)
        v = arr[..., 0]
        rgba[..., 0] = v; rgba[..., 1] = v; rgba[..., 2] = v; rgba[..., 3] = 1.0
        out_fmt = "PNG16" if fmt == "PNG16" else ("EXR" if fmt == "EXR" else "PNG8")
        utils.save_image(self.filepath, rgba, out_fmt)
        self.report({"INFO"}, f"Exported WearTime → {self.filepath}")
        return {"FINISHED"}


class AIWEAR_OT_export_mask(bpy.types.Operator, ExportHelper):
    bl_idname = "ai_wear.export_mask"
    bl_label = "Export Current Mask"
    bl_description = "Bake the current Wear Amount mask to 8/16-bit PNG"
    bl_options = {"REGISTER"}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})

    def execute(self, context):
        src = _current_weartime_path(context)
        if not src or not os.path.exists(src):
            self.report({"ERROR"}, "No WearTime texture. Run the pipeline first.")
            return {"CANCELLED"}
        from .. import utils
        import numpy as np
        s = context.scene.ai_wear
        T = utils.load_image_rgba(src)[..., 0]
        amount = s.wear_amount / 100.0
        feather = (s.feather / 100.0) or 1e-4
        # smoothstep(T - feather, T + feather, amount)
        d = (amount - T) / (2.0 * feather)
        mask = np.clip(d * 0.5 + 0.5, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)
        utils.save_array_png8(mask, self.filepath)
        self.report({"INFO"}, f"Exported mask ({s.wear_amount:.0f}) → {self.filepath}")
        return {"FINISHED"}


class AIWEAR_OT_export_batch(bpy.types.Operator, ExportHelper):
    """Export 30/60/100 masks to demonstrate monotonicity (30 ⊆ 60 ⊆ 100)."""
    bl_idname = "ai_wear.export_batch"
    bl_label = "Export 30/60/100 Masks"
    bl_options = {"REGISTER"}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png", options={"HIDDEN"})

    def execute(self, context):
        src = _current_weartime_path(context)
        if not src or not os.path.exists(src):
            self.report({"ERROR"}, "No WearTime texture. Run the pipeline first.")
            return {"CANCELLED"}
        from .. import utils
        import numpy as np
        T = utils.load_image_rgba(src)[..., 0]
        s = context.scene.ai_wear
        feather = (s.feather / 100.0) or 1e-4
        out_dir = os.path.dirname(self.filepath) or "."
        for amt in (30, 60, 100):
            a = amt / 100.0
            d = (a - T) / (2.0 * feather)
            mask = np.clip(d * 0.5 + 0.5, 0.0, 1.0)
            mask = mask * mask * (3.0 - 2.0 * mask)
            path = os.path.join(out_dir, f"wearmask_{amt}.png")
            utils.save_array_png8(mask, path)
        self.report({"INFO"}, f"Exported 30/60/100 masks → {out_dir}")
        return {"FINISHED"}


def _current_weartime_path(context) -> str:
    job = pipeline.get_active_job(context)
    if job and job.meta.get("weartime_path"):
        return job.meta["weartime_path"]
    img = bpy.data.images.get("AIWear_WearTime")
    if img and img.filepath:
        return bpy.path.abspath(img.filepath)
    return ""


# --- presets ----------------------------------------------------------------

class AIWEAR_OT_preset_save(bpy.types.Operator):
    bl_idname = "ai_wear.preset_save"
    bl_label = "Save Preset"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Preset Name", default="My Preset")

    def execute(self, context):
        s = context.scene.ai_wear
        p = s.presets.add()
        p.name = self.name
        p.material = s.prompt_material
        p.wear_type = s.prompt_wear_type
        p.max_state = s.max_wear_state
        p.prompt_extra = s.prompt_extra
        p.camera_preset = s.camera_preset
        p.camera_count = s.camera_count
        p.view_context_mode = s.view_context_mode
        p.w_ai = s.w_ai; p.w_convex = s.w_convex
        p.w_expose = s.w_expose; p.w_cavity = s.w_cavity
        p.alpha = s.alpha; p.noise_amp = s.noise_amp; p.noise_scale = s.noise_scale
        p.use_ai_mask = s.use_ai_mask
        p.use_geometry_prior = s.use_geometry_prior
        p.use_topology_growth = s.use_topology_growth
        p.seam_fuse = s.seam_fuse
        p.use_padding = s.use_padding
        p.wear_amount = s.wear_amount
        p.feather = s.feather
        s.active_preset_index = len(s.presets) - 1
        self.report({"INFO"}, f"Saved preset '{self.name}'")
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class AIWEAR_OT_preset_load(bpy.types.Operator):
    bl_idname = "ai_wear.preset_load"
    bl_label = "Load Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.ai_wear
        if not s.presets:
            self.report({"WARNING"}, "No presets.")
            return {"CANCELLED"}
        i = min(s.active_preset_index, len(s.presets) - 1)
        p = s.presets[i]
        presets.apply_preset(s, p)
        self.report({"INFO"}, f"Loaded preset '{p.name}'")
        return {"FINISHED"}


class AIWEAR_OT_preset_delete(bpy.types.Operator):
    bl_idname = "ai_wear.preset_delete"
    bl_label = "Delete Preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.ai_wear
        if not s.presets:
            return {"CANCELLED"}
        i = min(s.active_preset_index, len(s.presets) - 1)
        s.presets.remove(i)
        s.active_preset_index = max(0, i - 1)
        return {"FINISHED"}


class AIWEAR_OT_restore_presets(bpy.types.Operator):
    bl_idname = "ai_wear.restore_presets"
    bl_label = "Restore Experiment Presets"
    bl_description = "Clear and re-add the built-in EXPERIMENTS.md preset arms"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        presets.restore_experiment_presets(context.scene.ai_wear)
        self.report({"INFO"}, "Restored experiment presets.")
        return {"FINISHED"}


class AIWEAR_OT_log(bpy.types.Operator):
    """Internal: write one line to the Info editor so progress/errors are visible
    without opening the System Console. Called from the pipeline timer."""
    bl_idname = "ai_wear.log"
    bl_label = "AI Wear Log"
    bl_options = {"INTERNAL"}

    message: StringProperty()  # type: ignore[assignment]
    level: EnumProperty(  # type: ignore[assignment]
        items=[("INFO", "Info", ""), ("WARNING", "Warning", ""), ("ERROR", "Error", "")],
        default="INFO")

    def execute(self, context):
        msg = self.message or ""
        # self.report on an INTERNAL operator still lands in the Info editor log
        # and briefly in the status bar, which is what we want.
        self.report({self.level}, msg[:300])
        return {"FINISHED"}


class AIWEAR_OT_open_log(bpy.types.Operator):
    """Open the 'ai_wear.log' Text block in a Text Editor so every progress
    line, AI retry message, and the full error+traceback is visible without
    opening the hidden System Console"""
    bl_idname = "ai_wear.open_log"
    bl_label = "Open Log"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        import bpy
        from .pipeline import ensure_log_text
        t = ensure_log_text()
        # 1) prefer an existing Text Editor area
        target = None
        for area in context.screen.areas:
            if area.type == 'TEXT_EDITOR':
                target = area
                break
        # 2) otherwise convert the largest non-3D area into a Text Editor
        if target is None:
            best = None
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    continue
                if best is None or (area.width * area.height) > (best.width * best.height):
                    best = area
            if best is not None:
                best.type = 'TEXT_EDITOR'
                target = best
        if target is None:
            self.report({'WARNING'}, "Open a Text Editor and select 'ai_wear.log'")
            return {'CANCELLED'}
        target.spaces[0].text = t
        # pin near the bottom so the latest error is in view
        try:
            target.spaces[0].top = max(0, len(t.lines) - 20)
        except Exception:
            pass
        return {'FINISHED'}


CLASSES = (
    AIWEAR_OT_run_pipeline,
    AIWEAR_OT_replay_downstream,
    AIWEAR_OT_cancel,
    AIWEAR_OT_log,
    AIWEAR_OT_open_log,
    AIWEAR_OT_preflight,
    AIWEAR_OT_export_weartime,
    AIWEAR_OT_export_mask,
    AIWEAR_OT_export_batch,
    AIWEAR_OT_preset_save,
    AIWEAR_OT_preset_load,
    AIWEAR_OT_preset_delete,
    AIWEAR_OT_restore_presets,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
