"""Main UI panels — the AI Wear sidebar (N > AI Wear).

Sections: Run, UV/Preflight, Capture, WearThreshold, Seam, Export, Presets. Each is a
sub-panel so users can collapse what they don't need. Provider/model/endpoint and
the AI prompt live in Add-on Preferences (Edit > Preferences > Add-ons), not here;
UV QC + progress milestones print to the System Console.
"""

from __future__ import annotations

import bpy


class AIWEAR_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "AI Wear Texture"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        s = scene.ai_wear
        obj = context.active_object
        col = layout.column()
        if obj is None or obj.type != "MESH":
            col.label(text="Select a mesh object.", icon="ERROR")
        col.prop(s, "wear_amount", slider=True)
        col.prop(s, "feather", slider=True)
        row = col.row(align=True)
        row.scale_y = 1.4
        row.operator("ai_wear.run_pipeline", icon="IMAGE_RGB")
        from ..operators import pipeline
        if pipeline.is_running(context):
            row.operator("ai_wear.cancel", text="", icon="CANCEL")
        # Per-stage testing (Q3): replay only the downstream from the last run's
        # cached clean/worn images + camera matrices — no render, no AI. Lets you
        # iterate on surface/WearThreshold params without spending API budget.
        replay = col.row(align=True)
        replay.enabled = not pipeline.is_running(context)
        replay.operator("ai_wear.replay_downstream", icon="FILE_REFRESH")


class AIWEAR_PT_uv(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "UV Mode & Preflight"
    bl_parent_id = "AIWEAR_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "uv_mode")
        layout.prop(s, "target_uv_layer")
        layout.prop(s, "work_resolution")
        layout.operator("ai_wear.preflight", icon="MOD_UVPROJECT")
        # UV QC report is printed to the System Console only (Window > Toggle
        # System Console) — kept out of the panel to reduce clutter.


class AIWEAR_PT_capture(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Capture"
    bl_parent_id = "AIWEAR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "camera_preset")
        if s.camera_preset in {"AUTO_COUNT", "CUSTOM"}:
            layout.prop(s, "camera_count")
        layout.prop(s, "render_resolution")
        layout.prop(s, "view_context_mode")
        layout.prop(s, "coverage_target")
        layout.prop(s, "use_comfy_inpaint")
        if s.use_comfy_inpaint:
            layout.prop(s, "inpaint_edge_width")


class AIWEAR_PT_prompt(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Prompt"
    bl_parent_id = "AIWEAR_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "prompt_material")
        layout.prop(s, "prompt_wear_type")
        layout.prop(s, "max_wear_state")
        layout.prop(s, "prompt_extra")
        layout.prop(s, "lock_seed")
        layout.prop(s, "seed")


class AIWEAR_PT_wearthreshold(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "WearThreshold Parameters"
    bl_parent_id = "AIWEAR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        box = layout.box()
        box.label(text="Propensity weights", icon="MOD_WAVE")
        row = box.row(align=True); row.prop(s, "w_ai"); row.prop(s, "w_convex")
        row = box.row(align=True); row.prop(s, "w_expose"); row.prop(s, "w_cavity")
        box.prop(s, "gamma")
        box = layout.box()
        box.label(text="Growth / noise", icon="MOD_SIMPLIFY")
        box.prop(s, "alpha")
        box.prop(s, "noise_amp")
        box.prop(s, "noise_scale")
        box.prop(s, "use_barrier")
        box.prop(s, "material_boundary_penalty")


class AIWEAR_PT_seam(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "UV Seam"
    bl_parent_id = "AIWEAR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "seam_fuse")
        if s.seam_fuse:
            layout.prop(s, "seam_diffuse_texels")
        layout.prop(s, "use_padding")
        if s.use_padding:
            layout.prop(s, "padding_texels")


class AIWEAR_PT_experiments(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Experiments / Ablation"
    bl_parent_id = "AIWEAR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "use_ai_mask")
        layout.prop(s, "use_geometry_prior")
        layout.prop(s, "use_topology_growth")
        layout.prop(s, "save_experiment_snapshot")
        if s.save_experiment_snapshot:
            layout.prop(s, "experiment_label")


class AIWEAR_PT_export(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Export"
    bl_parent_id = "AIWEAR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        layout.prop(s, "export_format")
        layout.operator("ai_wear.export_wearthreshold", icon="IMAGE_DATA")
        layout.operator("ai_wear.export_mask", icon="MOD_MASK")
        layout.operator("ai_wear.export_batch", icon="RENDERLAYERS")


class AIWEAR_PT_presets(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Wear"
    bl_label = "Presets"
    bl_parent_id = "AIWEAR_PT_main"

    def draw(self, context):
        layout = self.layout
        s = context.scene.ai_wear
        row = layout.row()
        row.template_list("UI_UL_list", "ai_wear_presets", s, "presets",
                          s, "active_preset_index", rows=3)
        col = row.column(align=True)
        col.operator("ai_wear.preset_save", icon="ADD", text="")
        col.operator("ai_wear.preset_load", icon="FILE_TICK", text="")
        col.operator("ai_wear.preset_delete", icon="REMOVE", text="")
        layout.operator("ai_wear.restore_presets", icon="FILE_REFRESH")


CLASSES = (
    AIWEAR_PT_main,
    AIWEAR_PT_presets,
    AIWEAR_PT_uv,
    AIWEAR_PT_capture,
    AIWEAR_PT_prompt,
    AIWEAR_PT_wearthreshold,
    AIWEAR_PT_seam,
    AIWEAR_PT_experiments,
    AIWEAR_PT_export,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    from .progress import AIWEAR_PT_progress
    bpy.utils.register_class(AIWEAR_PT_progress)


def unregister():
    from .progress import AIWEAR_PT_progress
    bpy.utils.unregister_class(AIWEAR_PT_progress)
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
