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
    "version": (0, 3, 0),
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
from . import utils  # noqa: F401  (numpy helpers; no classes)
from .operators import runner
from .operators import pipeline  # noqa: F401  (no classes)
from . import ui


def _register_providers():
    from .ai import base as ai_base
    ai_base.ensure_providers_registered()


def register():
    # Order matters: preferences (AddonPreferences) → properties (PropertyGroups
    # + PointerProperty on Scene/Object) → operators → UI panels.
    preferences.register()
    properties.register()
    runner.register()
    ui.panels.register()
    _register_providers()


def unregister():
    ui.panels.unregister()
    runner.unregister()
    properties.unregister()
    preferences.unregister()


if __name__ == "__main__":
    register()
