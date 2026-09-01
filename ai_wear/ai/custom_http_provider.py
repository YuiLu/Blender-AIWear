"""Fully configurable HTTP provider.

This is the provider the user emphasized: the external image-generation model
endpoint is configured entirely inside the plugin (AddonPreferences). Two modes:

  * OPENAI_COMPAT — POST {base}{path} as OpenAI-style multipart images/edits.
    Works with any OpenAI-compatible vendor; just set Base URL + Path + Model.

  * RAW_JSON — render a JSON body template (with {{prompt}}, {{seed}},
    {{image_b64}}, {{output_size}} placeholders) and parse the response image by
    a dot-path (e.g. data.0.b64_json or data.0.url). Lets you wire up almost any
    image-edit HTTP API without writing code.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from . import http_util
from .base import AIProvider, GenRequest, GenResult, ProviderCapabilities, ProviderError
from ._openai_compat import generate_openai_compat
from .. import utils


def _render_template(tpl: str, ctx: dict) -> str:
    out = tpl
    for k, v in ctx.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def _resolve_dotpath(data: Any, path: str) -> Any:
    cur = data
    for token in path.split("."):
        token = token.strip()
        if token == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(token)]
            except (ValueError, IndexError):
                raise ProviderError(f"List index '{token}' out of range in path {path}", kind="API")
        elif isinstance(cur, dict):
            if token in cur:
                cur = cur[token]
            else:
                raise ProviderError(f"Key '{token}' not found in path {path}", kind="API")
        else:
            raise ProviderError(f"Cannot descend '{token}' on {type(cur).__name__}", kind="API")
    return cur


class CustomHTTPProvider(AIProvider):
    name = "CUSTOM"

    def capabilities(self) -> ProviderCapabilities:
        if getattr(self, "_caps_override", None):
            return self._caps_override
        return ProviderCapabilities(
            max_reference_images=1,
            supports_image_edit=True,
            supports_mask=True,
            supports_seed=True,
            supports_depth_or_control=False,
        )

    def validate_config(self, prefs) -> list:
        probs = []
        if not prefs.api_base_url:
            probs.append("API Base URL is empty")
        if prefs.request_mode == "RAW_JSON":
            if not prefs.raw_image_field:
                probs.append("Raw JSON mode needs an Image JSON Path")
        else:
            if not prefs.image_endpoint_path:
                probs.append("OpenAI-compat mode needs an Image Edit Path")
        return probs

    def _generate_raw_json(self, req: GenRequest, prefs) -> GenResult:
        b64, _ = utils.file_to_base64(req.clean_image_path)
        ctx = {
            "prompt": req.prompt,
            "seed": req.seed,
            "image_b64": b64,
            "output_size": req.output_size,
            "model": prefs.model_id,
        }
        body_str = _render_template(prefs.raw_body_template, ctx)
        try:
            body = json.loads(body_str)
        except Exception as e:
            raise ProviderError(f"Raw body template is not valid JSON: {e}", kind="CONFIG")

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = prefs.get_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if prefs.extra_headers:
            try:
                extra = json.loads(prefs.extra_headers)
                headers.update({str(k): str(v) for k, v in extra.items()})
            except Exception:
                pass

        url = prefs.get_base_url().rstrip("/") + prefs.image_endpoint_path
        req.on_progress(0.4, "Posting to custom endpoint…")
        code, resp = http_util.post_json(url, body, headers, prefs.timeout)
        if code != 200:
            raise ProviderError(f"Custom endpoint failed ({code}): {resp.decode('utf-8','replace')[:500]}", kind="API")
        try:
            data = json.loads(resp)
        except Exception:
            raise ProviderError(f"Non-JSON response: {resp.decode('utf-8','replace')[:500]}", kind="API")

        val = _resolve_dotpath(data, prefs.raw_image_field)
        if not isinstance(val, str) or not val:
            raise ProviderError(f"Image field '{prefs.raw_image_field}' is not a string", kind="API")
        out_path = os.path.join(req.out_dir, f"worn_{os.urandom(4).hex()}.png")
        req.on_progress(0.8, "Saving worn view…")
        if prefs.raw_response_is_url or val.startswith("http"):
            http_util.download(val, out_path, headers, prefs.timeout)
        else:
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(val))
        return GenResult(worn_image_path=out_path, used_seed=req.seed,
                         raw_response=json.dumps(data)[:800])

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        if prefs.request_mode == "RAW_JSON":
            return self._generate_raw_json(req, prefs)
        # OpenAI-compatible
        path = prefs.image_endpoint_path or "/images/edits"
        return generate_openai_compat(req, prefs, prefs.get_base_url(), path,
                                      model=prefs.model_id)
