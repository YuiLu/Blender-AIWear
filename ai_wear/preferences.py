"""Addon preferences: provider / API key / URL / model / workflow / cache.

The external image-generation API endpoint is fully configurable here. Keys use
the PASSWORD subtype so they are not printed in tooltips/logs and are not
written into the .blend. Environment variables take precedence over stored keys.
"""

from __future__ import annotations

import os
from typing import Optional

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    StringProperty,
    PointerProperty,
)
from bpy.types import AddonPreferences

# Provider identifiers. The enum is static so registration is stable; the actual
# base URL / model / key are data, so a new endpoint never needs a code change.
PROVIDER_OPENAI = "OPENAI"
PROVIDER_GEMINI = "GEMINI"
PROVIDER_COMFYUI = "COMFYUI"
PROVIDER_CUSTOM = "CUSTOM"


def _env_or_stored(stored: str, env_name: str) -> Optional[str]:
    """Environment variable wins over stored value; never log the result."""
    if env_name:
        v = os.environ.get(env_name)
        if v:
            return v
    return stored or None


class AIWearPreferences(AddonPreferences):
    bl_idname = "ai_wear"

    # --- Active global default provider -------------------------------------
    provider: EnumProperty(
        name="Provider",
        description="Default AI image provider. The scene panel can override per project",
        items=(
            (PROVIDER_OPENAI, "OpenAI (GPT-Image)", "OpenAI image-edit compatible endpoint"),
            (PROVIDER_GEMINI, "Gemini", "Google Gemini image generation/edit"),
            (PROVIDER_COMFYUI, "ComfyUI", "Local ComfyUI workflow server"),
            (PROVIDER_CUSTOM, "Custom HTTP", "Fully configurable OpenAI-compatible or raw HTTP endpoint"),
        ),
        default=PROVIDER_CUSTOM,
    )

    # --- OpenAI / Gemini / Custom shared fields ----------------------------
    api_base_url: StringProperty(
        name="API Base URL",
        description="Base URL of the image-generation endpoint, e.g. https://api.openai.com/v1. "
                    "For Custom HTTP this is the request root; the full path is appended",
        default="https://api.openai.com/v1",
        subtype="NONE",
    )
    api_key: StringProperty(
        name="API Key",
        description="API key. Stored locally only; never written into the .blend. "
                    "Set the env var below to avoid storing it at all",
        default="",
        subtype="PASSWORD",
    )
    api_key_env: StringProperty(
        name="Key Env Var",
        description="If set, the key is read from this environment variable and the stored value is ignored",
        default="",
        subtype="NONE",
    )
    model_id: StringProperty(
        name="Model ID",
        description="Provider model identifier, e.g. gpt-image-2 or gemini-2.5-flash-image",
        default="gpt-image-2",
    )

    # --- Request / response knobs (Custom HTTP & OpenAI-compatible) -------
    image_endpoint_path: StringProperty(
        name="Image Edit Path",
        description="Path appended to API Base URL for image edit/generation, e.g. /images/edits or /images/generations",
        default="/images/edits",
    )
    request_mode: EnumProperty(
        name="Request Mode",
        description="How to talk to the endpoint",
        items=(
            ("OPENAI_COMPAT", "OpenAI-compatible", "Use OpenAI images edit/generation JSON form"),
            ("RAW_JSON", "Raw JSON template", "Render a configurable JSON body template; parse image by JSON path"),
        ),
        default="OPENAI_COMPAT",
    )
    raw_body_template: StringProperty(
        name="Raw Body Template",
        description="JSON body template. Use {{prompt}}, {{seed}}, {{image_b64}} placeholders. "
                    "Only used when Request Mode = Raw JSON",
        default='{"prompt": "{{prompt}}", "image": "{{image_b64}}", "seed": {{seed}}}',
        subtype="NONE",
    )
    raw_image_field: StringProperty(
        name="Image JSON Path",
        description="Dot-path into the JSON response to the image (base64) or image URL, e.g. data.0.b64_json or data.0.url",
        default="data.0.b64_json",
    )
    raw_response_is_url: BoolProperty(
        name="Response Is URL",
        description="If the image field is a URL, fetch it afterwards; otherwise treat as base64",
        default=False,
    )
    extra_headers: StringProperty(
        name="Extra Headers",
        description='Optional extra HTTP headers as JSON object string, e.g. {"X-Client": "ai-wear"}',
        default="",
        subtype="NONE",
    )

    # --- ComfyUI -----------------------------------------------------------
    comfyui_url: StringProperty(
        name="ComfyUI URL",
        description="ComfyUI server root, e.g. http://127.0.0.1:8188",
        default="http://127.0.0.1:8188",
    )
    workflow_path: StringProperty(
        name="Workflow JSON",
        description="Path to a ComfyUI workflow (API-format JSON) with input/output node mapping",
        default="",
        subtype="FILE_PATH",
    )
    clean_image_node: StringProperty(
        name="Clean Image Node",
        description="ComfyUI node id that receives the clean view image",
        default="",
    )
    prompt_node: StringProperty(
        name="Prompt Node",
        description="ComfyUI node id that receives the text prompt",
        default="",
    )
    seed_node: StringProperty(
        name="Seed Node",
        description="ComfyUI node id that receives the seed",
        default="",
    )
    output_node: StringProperty(
        name="Output Node",
        description="ComfyUI SaveImage / PreviewImage node id to fetch results from",
        default="",
    )

    # --- Transport / limits ------------------------------------------------
    timeout: FloatProperty(
        name="Timeout (s)",
        description="Per-request network timeout",
        default=120.0,
        min=10.0,
        max=3600.0,
    )
    max_concurrency: IntProperty(
        name="Max Concurrency",
        description="Maximum simultaneous in-flight API requests (limits cost / VRAM)",
        default=1,
        min=1,
        max=8,
    )
    poll_interval: FloatProperty(
        name="Poll Interval (s)",
        description="How often the main thread polls job state to refresh UI",
        default=0.25,
        min=0.05,
        max=5.0,
    )

    # --- Cache -------------------------------------------------------------
    cache_path_override: StringProperty(
        name="Cache Path",
        description="Override the disk cache root. Leave empty to use <blend>/.ai_wear_cache",
        default="",
        subtype="DIR_PATH",
    )

    def get_api_key(self) -> Optional[str]:
        return _env_or_stored(self.api_key, self.api_key_env)

    def get_base_url(self) -> str:
        return self.api_base_url.rstrip("/")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "provider")

        box = layout.box()
        box.label(text="Endpoint & Auth", icon="URL")
        box.prop(self, "api_base_url")
        row = box.row(align=True)
        row.prop(self, "api_key")
        row.prop(self, "api_key_env", text="Env")
        box.prop(self, "model_id")

        if self.provider == PROVIDER_CUSTOM:
            box2 = layout.box()
            box2.label(text="Custom HTTP Request", icon="SCRIPT")
            box2.prop(self, "image_endpoint_path")
            box2.prop(self, "request_mode")
            if self.request_mode == "RAW_JSON":
                box2.prop(self, "raw_body_template")
                box2.prop(self, "raw_image_field")
                box2.prop(self, "raw_response_is_url")
            box2.prop(self, "extra_headers")

        if self.provider == PROVIDER_COMFYUI:
            cbox = layout.box()
            cbox.label(text="ComfyUI Workflow", icon="NODE")
            cbox.prop(self, "comfyui_url")
            cbox.prop(self, "workflow_path")
            cbox.prop(self, "clean_image_node")
            cbox.prop(self, "prompt_node")
            cbox.prop(self, "seed_node")
            cbox.prop(self, "output_node")

        tbox = layout.box()
        tbox.label(text="Transport & Cache", icon="TIME")
        tbox.prop(self, "timeout")
        tbox.prop(self, "max_concurrency")
        tbox.prop(self, "poll_interval")
        tbox.prop(self, "cache_path_override")


CLASSES = (AIWearPreferences,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


def get_prefs() -> AIWearPreferences:
    return bpy.context.preferences.addons["ai_wear"].preferences
