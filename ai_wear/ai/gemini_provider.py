"""Google Gemini image generation/edit provider.

Endpoint: {base}/models/{model}:generateContent
Auth: x-goog-api-key header (or ?key= query).
For wear we send the clean render + prompt as inline_data + text parts and ask
for an IMAGE response. Gemini supports multiple inline_data parts, so anchor /
guide images are attached when available.
"""

from __future__ import annotations

import base64
import json
import os
from typing import List

from . import http_util
from .base import AIProvider, GenRequest, GenResult, ProviderCapabilities, ProviderError
from .. import utils


class GeminiProvider(AIProvider):
    name = "GEMINI"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_reference_images=3,
            supports_image_edit=True,
            supports_mask=False,
            supports_depth_or_control=True,
            supports_multi_turn_context=True,
            supports_seed=True,
        )

    def validate_config(self, prefs) -> list:
        probs = []
        if not prefs.api_base_url:
            probs.append("API Base URL is empty")
        if not prefs.get_api_key():
            probs.append("API Key is empty (or set an env var)")
        if not prefs.model_id:
            probs.append("Model ID is empty (e.g. gemini-2.5-flash-image)")
        return probs

    def _build_body(self, req: GenRequest) -> dict:
        parts: List[dict] = [{"text": req.prompt}]
        # Clean view first
        b64, mime = utils.file_to_base64(req.clean_image_path)
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        # Optional reference images (anchor / projected guide / depth)
        for ref in req.reference_images:
            if os.path.exists(ref):
                b, m = utils.file_to_base64(ref)
                parts.append({"inline_data": {"mime_type": m, "data": b}})
        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 0.4,
            },
        }

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        headers = {"Content-Type": "application/json"}
        key = prefs.get_api_key()
        if key:
            headers["x-goog-api-key"] = key
        url = f"{prefs.get_base_url().rstrip('/')}/models/{prefs.model_id}:generateContent"
        body = self._build_body(req)

        req.on_progress(0.3, "Requesting Gemini edit…")
        code, resp = http_util.post_json(url, body, headers, prefs.timeout)
        if code != 200:
            raise ProviderError(f"Gemini failed ({code}): {resp.decode('utf-8','replace')[:500]}", kind="API")
        try:
            data = json.loads(resp)
        except Exception:
            raise ProviderError(f"Non-JSON Gemini response: {resp.decode('utf-8','replace')[:500]}", kind="API")

        out_path = os.path.join(req.out_dir, f"worn_{os.urandom(4).hex()}.png")
        try:
            cands = data.get("candidates", [])
            parts = cands[0]["content"]["parts"] if cands and "content" in cands[0] else []
        except Exception:
            parts = []
        image_b64 = None
        for p in parts:
            inline = p.get("inline_data") or p.get("inlineData")
            if inline and inline.get("data"):
                image_b64 = inline["data"]
                break
        if not image_b64:
            # Fallback: some responses nest images differently
            raise ProviderError(f"No image in Gemini response: {str(data)[:500]}", kind="API")

        req.on_progress(0.8, "Saving worn view…")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return GenResult(worn_image_path=out_path, used_seed=req.seed,
                         raw_response=json.dumps(data)[:800])
