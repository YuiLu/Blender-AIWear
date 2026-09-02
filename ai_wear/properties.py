"""Scene / Object settings.

All per-project tunables live on the Scene (so they save with the .blend). The
only thing that must NOT save with the .blend is the API key, which lives in
AddonPreferences instead. Scene-level provider/model/base_url fields override
the AddonPreferences defaults when non-empty.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
    CollectionProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup


UV_MODE_ITEMS = (
    ("MODE_A", "Mode A — Use Existing UV",
     "Map wear onto the chosen existing UV layer. Overlapping UVs share wear (expected limitation)"),
    ("MODE_B", "Mode B — New AI_WearUV",
     "Keep all original UVs; add a fresh AI_WearUV layer and auto-unwrap it"),
)

CAMERA_PRESET_ITEMS = (
    ("AUTO_6", "Auto 6 views", "4 equatorial cameras plus one top and one bottom camera"),
    ("AUTO_8", "Auto 8 views", "8 symmetric oblique cameras along cube-corner directions"),
    ("TURNTABLE_4", "Turntable 4", "4 equatorial views for a fast preview"),
    ("AUTO_COUNT", "Counted Auto", "Generate exactly Cam Count approximately even Fibonacci-sphere views"),
    ("CUSTOM", "Custom", "Use the cameras currently marked in the scene"),
)

STRATEGY_ITEMS = (
    ("IMAGE_EDIT", "Image Edit",
     "Single-image edit: clean view + prompt (anchor optional)"),
    ("CONTROL_ASSISTED", "Control-assisted",
     "Include depth/normal/edge as extra reference where the provider supports it"),
    ("CUSTOM_WORKFLOW", "Custom Workflow",
     "Provider-specific custom workflow (e.g. ComfyUI node graph)"),
)

VIEW_CONTEXT_ITEMS = (
    ("NONE", "Independent", "Generate every view from its clean render only"),
    ("FIRST_ANCHOR", "First-view Anchor", "Use the first generated worn view as context for every later view"),
    ("PREVIOUS_VIEW", "Previous View", "Use the immediately preceding worn view as context for the next view"),
)

EXPORT_ITEMS = (
    ("PNG16", "PNG 16-bit", "16-bit PNG (Non-Color) — Substance/Blender ready"),
    ("EXR", "OpenEXR 32-bit float", "Full float precision"),
    ("PNG8", "PNG 8-bit", "8-bit mask export only"),
)


def _make_wear_prompt(self, _context):
    """Build the full wear prompt from material + wear type + max state + extra.

    The prompt strength scales with max_wear_state so heavier wear yields a
    stronger description, satisfying the 'prompt strength follows wear param' ask.
    """
    mat = self.prompt_material or "the material"
    wt = self.prompt_wear_type or "wear"
    state = self.max_wear_state or "moderate"
    extra = self.prompt_extra or ""
    base = (f"Edit this render of a 3D model to add realistic, physically plausible "
            f"{state} {wt} on the {mat} surface. "
            f"Concentrate the {wt} on edges, convex ridges and high-contact areas; "
            f"keep recesses and hidden areas clean. "
            f"Preserve the exact silhouette, camera, geometry and lighting. "
            f"Do not change the shape, pose or background.")
    if extra:
        base += " " + extra.strip()
    return base


class AIWearPreset(PropertyGroup):
    """A saved wear parameter preset (save/load加分项)."""

    name: StringProperty(name="Preset Name", default="New Preset")
    material: StringProperty(name="Material", default="painted metal")
    wear_type: StringProperty(name="Wear Type", default="chipping and scratches")
    max_state: StringProperty(name="Max Wear State", default="heavy")
    w_ai: FloatProperty(name="w AI", default=0.6, min=0, max=2)
    w_convex: FloatProperty(name="w Convex", default=0.3, min=0, max=2)
    w_expose: FloatProperty(name="w Expose", default=0.2, min=0, max=2)
    w_cavity: FloatProperty(name="w Cavity", default=0.2, min=0, max=2)
    alpha: FloatProperty(name="alpha", default=0.7, min=0, max=1)
    noise_amp: FloatProperty(name="noise amp", default=0.12, min=0, max=1)
    noise_scale: FloatProperty(name="noise scale", default=8.0, min=0.1, max=64)


class AIWearSceneSettings(PropertyGroup):
    # --- UV ---------------------------------------------------------------
    uv_mode: EnumProperty(items=UV_MODE_ITEMS, name="UV Mode", default="MODE_B")
    target_uv_layer: StringProperty(
        name="Target UV Layer",
        description="Mode A: pick an existing UV layer. Mode B: name of the new AI_WearUV layer",
        default="AI_WearUV",
    )
    work_resolution: IntProperty(name="Work Res", default=1024, min=256, max=4096,
                                  description="Internal raster resolution for surface field")
    texture_size: IntProperty(name="Texture Size", default=2048, min=512, max=8192,
                               description="Output WearTime/mask texture size")

    # --- Capture ----------------------------------------------------------
    camera_preset: EnumProperty(items=CAMERA_PRESET_ITEMS, name="Cameras", default="AUTO_6")
    camera_count: IntProperty(name="Cam Count", default=6, min=1, max=16)
    render_resolution: IntProperty(name="Render Res", default=1024, min=512, max=4096)
    view_context_mode: EnumProperty(
        items=VIEW_CONTEXT_ITEMS, name="View Context", default="FIRST_ANCHOR",
        description="Sequential multi-view context; used only by providers that support reference images")

    # --- AI provider overrides --------------------------------------------
    provider_override: EnumProperty(
        name="Provider",
        description="NONE = use the AddonPreferences default",
        items=(("NONE", "— Default —", "Use AddonPreferences provider"),
               ("OPENAI", "OpenAI", ""), ("GEMINI", "Gemini", ""),
               ("COMFYUI", "ComfyUI", ""), ("CUSTOM", "Custom HTTP", "")),
        default="NONE",
    )
    model_override: StringProperty(name="Model", default="")
    base_url_override: StringProperty(name="Base URL", default="",
                                      description="Override AddonPreferences base URL for this project")
    generation_strategy: EnumProperty(items=STRATEGY_ITEMS, name="Strategy", default="IMAGE_EDIT")

    # --- Prompt -----------------------------------------------------------
    prompt_material: StringProperty(name="Material", default="painted metal")
    prompt_wear_type: StringProperty(name="Wear Type", default="chipping and scratches")
    max_wear_state: StringProperty(name="Max Wear State", default="heavy",
                                   description="The single maximal wear state the AI generates once")
    prompt_extra: StringProperty(name="Extra Prompt", default="",
                                 description="Appended verbatim to the generated prompt")
    seed: IntProperty(name="Seed", default=0, min=0,
                      description="0 = random per run")
    lock_seed: BoolProperty(name="Lock Seed", default=True,
                            description="Reuse the same seed across views for determinism")

    # --- WearTime parameters ---------------------------------------------
    # Update callbacks drive the shader preview in real time (no AI re-run).
    def _wa_cb(s, c):
        from . import shader as _sh
        _sh.on_wear_amount(s, c)

    def _feather_cb(s, c):
        from . import shader as _sh
        _sh.on_feather(s, c)

    wear_amount: FloatProperty(name="Wear Amount", default=60.0, min=0.0, max=100.0,
                                description="Real-time threshold 0..100, no AI re-run",
                                update=_wa_cb)
    feather: FloatProperty(name="Feather", default=4.0, min=0.0, max=50.0,
                           description="Smoothstep half-width in 0..100 units",
                           update=_feather_cb)
    w_ai: FloatProperty(name="w AI", default=0.6, min=0, max=2)
    w_convex: FloatProperty(name="w Convex", default=0.3, min=0, max=2)
    w_expose: FloatProperty(name="w Expose", default=0.2, min=0, max=2)
    w_cavity: FloatProperty(name="w Cavity", default=0.2, min=0, max=2)
    gamma: FloatProperty(name="gamma", default=2.0, min=0.1, max=6.0,
                         description="Exponent on facing weight")
    alpha: FloatProperty(name="alpha", default=0.7, min=0, max=1,
                         description="Blend of Dijkstra arrival vs propensity")
    noise_amp: FloatProperty(name="noise amp", default=0.12, min=0, max=1)
    noise_scale: FloatProperty(name="noise scale", default=8.0, min=0.1, max=64)
    material_boundary_penalty: FloatProperty(name="Mat Boundary", default=4.0, min=1.0, max=20.0)
    use_barrier: BoolProperty(name="Material Boundary Barrier", default=True)

    # --- Coverage / fusion -----------------------------------------------
    coverage_target: FloatProperty(name="Coverage Target", default=0.95, min=0.5, max=1.0,
                                   description="Stop adding views once this surface fraction is covered")

    # --- ComfyUI inpaint --------------------------------------------------
    use_comfy_inpaint: BoolProperty(
        name="Geometry Inpaint Mask", default=True,
        description="For ComfyUI, generate an object-silhouette/depth-edge mask and restrict regeneration to likely wear zones")
    inpaint_edge_width: IntProperty(
        name="Inpaint Edge Width", default=12, min=1, max=64,
        description="Screen-space width in pixels of the generated wear/inpaint bands")

    # --- Seam / bake ------------------------------------------------------
    seam_fuse: BoolProperty(name="Seam Fusion", default=True)
    seam_diffuse_texels: IntProperty(name="Seam Diffuse", default=8, min=0, max=64)
    use_padding: BoolProperty(
        name="Island Padding", default=True,
        description="Dilate valid UV islands after projection; independent of seam fusion for ablation")
    padding_texels: IntProperty(name="Padding", default=16, min=0, max=64,
                                 description="Island dilation to prevent bilinear/mipmap bleed")

    # --- Experiments / ablation ------------------------------------------
    use_ai_evidence: BoolProperty(
        name="AI Evidence", default=True,
        description="Include the reprojected AI difference field in wear propensity")
    use_geometry_prior: BoolProperty(
        name="Geometry Prior", default=True,
        description="Include convexity, exposure and cavity terms in wear propensity")
    use_topology_growth: BoolProperty(
        name="Topology Growth", default=True,
        description="Use multi-source Dijkstra growth; disabled uses direct propensity and still completes")
    save_experiment_snapshot: BoolProperty(
        name="Save Experiment Snapshot", default=False,
        description="Save config, metrics and pre/post seam textures into a uniquely named experiment folder")
    experiment_label: StringProperty(
        name="Experiment Label", default="baseline",
        description="Short label used in the experiment output directory")

    # --- Export -----------------------------------------------------------
    export_format: EnumProperty(items=EXPORT_ITEMS, name="Export", default="PNG16")

    # --- State ------------------------------------------------------------
    active_job_id: StringProperty(name="Active Job", default="")

    # --- Presets ----------------------------------------------------------
    presets: CollectionProperty(type=AIWearPreset)
    active_preset_index: IntProperty(default=0)

    def build_prompt(self) -> str:
        return _make_wear_prompt(self, None)

    def effective_provider(self, prefs) -> str:
        return self.provider_override if self.provider_override != "NONE" else prefs.provider

    def effective_model(self, prefs) -> str:
        return self.model_override or prefs.model_id

    def effective_base_url(self, prefs) -> str:
        return (self.base_url_override.rstrip("/") or prefs.get_base_url())


class AIWearObjectSettings(PropertyGroup):
    mesh_hash: StringProperty(name="Mesh Hash", default="")
    uv_hash: StringProperty(name="UV Hash", default="")
    processing_copy_name: StringProperty(name="Processing Copy", default="")
    last_qc_report: StringProperty(name="Last QC Report", default="")
    qc_ok: BoolProperty(name="QC OK", default=False)


CLASSES = (AIWearPreset, AIWearSceneSettings, AIWearObjectSettings)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.ai_wear = PointerProperty(type=AIWearSceneSettings)
    bpy.types.Object.ai_wear = PointerProperty(type=AIWearObjectSettings)


def unregister():
    del bpy.types.Scene.ai_wear
    del bpy.types.Object.ai_wear
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
