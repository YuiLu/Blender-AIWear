"""OpenAI provider (gpt-image style)."""

from __future__ import annotations

from .base import AIProvider, GenRequest, GenResult, ProviderCapabilities, ProviderError
from ._openai_compat import generate_openai_compat


class OpenAIProvider(AIProvider):
    name = "OPENAI"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_reference_images=1,
            supports_image_edit=True,
            supports_mask=True,
            supports_seed=False,  # stock OpenAI image edit doesn't honor seed
        )

    def validate_config(self, prefs) -> list:
        probs = []
        if not prefs.api_base_url:
            probs.append("API Base URL is empty")
        if not prefs.get_api_key():
            probs.append("API Key is empty (or set an env var)")
        if not prefs.model_id:
            probs.append("Model ID is empty")
        return probs

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        path = prefs.image_endpoint_path or "/images/edits"
        return generate_openai_compat(req, prefs, prefs.get_base_url(), path)
