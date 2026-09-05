"""Provider abstraction.

The addon never hardcodes a specific model's logic in an Operator. Operators
build a GenRequest and hand it to a provider; the provider only does HTTP,
polling and file IO on a worker thread. The main thread (bpy.app.timers) reads
job state to update the UI.

A new endpoint never requires a code change: OPENAI_COMPAT mode talks to any
OpenAI-style images endpoint; RAW_JSON mode renders a body template and parses
the response image by JSON path. Both are driven entirely by AddonPreferences.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import http_util


class ProviderError(Exception):
    def __init__(self, message: str, kind: str = "API"):
        super().__init__(message)
        self.kind = kind


@dataclass
class ProviderCapabilities:
    max_reference_images: int = 1
    supports_image_edit: bool = True
    supports_mask: bool = False
    supports_depth_or_control: bool = False
    supports_multi_turn_context: bool = False
    supports_seed: bool = False
    supports_model_snapshot: bool = False


@dataclass
class GenRequest:
    clean_image_path: str
    prompt: str
    seed: int = 0
    output_size: int = 1024
    # Optional references (anchor worn view / projected guide / depth/normal)
    reference_images: List[str] = field(default_factory=list)
    reference_labels: List[str] = field(default_factory=list)  # informational
    mask_path: Optional[str] = None
    depth_path: Optional[str] = None
    normal_path: Optional[str] = None
    # ComfyUI workflow override (path to API-format JSON)
    workflow_path: Optional[str] = None
    node_mapping: Dict[str, str] = field(default_factory=dict)
    # Threading controls provided by the operator
    should_cancel: Callable[[], bool] = field(default=lambda: False)
    on_progress: Callable[[float, str], None] = field(default=lambda p, m: None)
    # Where to write the result
    out_dir: str = ""


@dataclass
class GenResult:
    worn_image_path: str
    used_seed: int = 0
    raw_response: str = ""


class AIProvider:
    name: str = "BASE"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def validate_config(self, prefs) -> List[str]:
        """Return a list of human-readable config problems (empty == ok)."""
        return []

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        raise NotImplementedError


# --- Registry / factory ------------------------------------------------------

_REGISTRY: Dict[str, Callable[[], AIProvider]] = {}


def register_provider(name: str, factory: Callable[[], AIProvider]) -> None:
    _REGISTRY[name] = factory


def get_provider(name: str) -> AIProvider:
    if name not in _REGISTRY:
        raise ProviderError(f"Unknown provider '{name}'", kind="CONFIG")
    return _REGISTRY[name]()


def provider_names() -> List[str]:
    return list(_REGISTRY.keys())


def _register_builtin():
    # Imported here to avoid import cycles at package load.
    from .openai_provider import OpenAIProvider
    from .gemini_provider import GeminiProvider
    from .comfyui_provider import ComfyUIProvider
    from .qwen_image_provider import QwenImageProvider
    from .custom_http_provider import CustomHTTPProvider

    register_provider("OPENAI", OpenAIProvider)
    register_provider("GEMINI", GeminiProvider)
    register_provider("QWEN", QwenImageProvider)
    register_provider("COMFYUI", ComfyUIProvider)
    register_provider("CUSTOM", CustomHTTPProvider)


def ensure_providers_registered():
    if not _REGISTRY:
        _register_builtin()


def auth_header(prefs) -> Dict[str, str]:
    key = prefs.get_api_key() if hasattr(prefs, "get_api_key") else None
    return {"Authorization": f"Bearer {key}"} if key else {}
