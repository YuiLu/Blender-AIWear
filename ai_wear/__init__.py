"""AI Wear Texture — Blender addon.

Pipeline: Mesh Preflight / UV QC → Mode A/B → multi-view clean render →
AI provider (OpenAI / Gemini / ComfyUI / fully-configurable Custom HTTP) →
screen-mask extraction → 3D Surface Field multi-view fusion → topology-growth
WearTime → UV seam fusion + dilation → 16-bit WearTime + real-time shader
preview + export.

The external image-generation API endpoint is configured in
Edit > Preferences > Add-ons > AI Wear Texture (Base URL, Model, Key, and for
Custom HTTP: request path, request mode, body template, image JSON path).

See README.md for install / config / workflow mapping.
"""

bl_info = {
    "name": "AI Wear Texture",
    "author": "AI Wear Texture — implementation per plan",
    "version": (0, 3, 2),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > AI Wear",
    "description": "AI-driven model wear-texture generation: UV QC, surface field, "
                   "topology-growth WearTime, seam fusion, multi-API providers",
    "category": "Texture",
    "doc_url": "",
    "tracker_url": "",
}

import bpy

from . import preferences
from . import properties
from . import presets
from . import utils  # noqa: F401  (numpy helpers; no classes)
from .operators import runner
from .operators import pipeline  # noqa: F401  (no classes)
from . import ui


def _register_providers():
    from .ai import base as ai_base
    ai_base.ensure_providers_registered()


def _seed_all_scenes():
    # During startup addon-enable, bpy.data is a restricted proxy with no
    # .scenes — skip and let the load_post handler seed once the file is loaded.
    try:
        scenes = bpy.data.scenes
    except Exception:
        return
    for scene in scenes:
        presets.seed_experiment_presets(scene.ai_wear)


def _seed_presets_on_load(_dummy):
    _seed_all_scenes()


def _seed_retry_timer():
    """One-shot safety net: seed every scene once bpy.data is writable.

    During startup addon-enable ``bpy.data.scenes`` is a restricted proxy, so the
    register-time seed bails; and for a scene opened without a .blend load (the
    default startup scene) load_post never fires. This timer retries every 0.2s
    until the data is accessible, then seeds and stops — so the Presets panel is
    never left empty.
    """
    try:
        scenes = bpy.data.scenes
    except Exception:
        return 0.2
    for scene in scenes:
        presets.seed_experiment_presets(scene.ai_wear)
    return None


def register():
    # Order matters: preferences (AddonPreferences) → properties (PropertyGroups
    # + PointerProperty on Scene/Object) → operators → UI panels.
    preferences.register()
    properties.register()
    runner.register()
    ui.panels.register()
    _register_providers()
    # Seed the experiment presets so the Presets panel isn't blank: at register
    # time, on every subsequent .blend load, and via a retry timer that covers
    # the startup window where bpy.data is still restricted.
    _seed_all_scenes()
    bpy.app.handlers.load_post.append(_seed_presets_on_load)
    if not bpy.app.timers.is_registered(_seed_retry_timer):
        bpy.app.timers.register(_seed_retry_timer, first_interval=0.2)


def unregister():
    if _seed_presets_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_seed_presets_on_load)
    if bpy.app.timers.is_registered(_seed_retry_timer):
        bpy.app.timers.unregister(_seed_retry_timer)
    ui.panels.unregister()
    runner.unregister()
    properties.unregister()
    preferences.unregister()


if __name__ == "__main__":
    register()
