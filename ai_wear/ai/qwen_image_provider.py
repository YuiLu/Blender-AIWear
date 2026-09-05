"""Alibaba Cloud Model Studio provider for Qwen Image 3.0.

Qwen Image does not expose image edit through the OpenAI-compatible
``/compatible-mode/v1/images/edits`` route.  It uses the native synchronous
multimodal-generation API and accepts the clean render plus optional worn-view
references as data-URI image parts.
"""

from __future__ import annotations

import base64
import json
import os
from urllib.parse import urlsplit

from . import http_util
from .base import AIProvider, GenRequest, GenResult, ProviderCapabilities, ProviderError
from .. import utils


NATIVE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
_COPY_PUNCTUATION = " 　,，。;；、"


def _clean_base_url(value: str) -> str:
    """Clean common copy/paste punctuation without altering the hostname."""
    return (value or "").strip().rstrip(_COPY_PUNCTUATION).rstrip("/")


def native_endpoint(value: str) -> str:
    """Convert a workspace root (or old compatible-mode URL) to Qwen native API.

    Accepting the compatible-mode suffix makes migration from a previously
    configured Custom HTTP provider painless; the request itself is always sent
    using the documented native protocol.
    """
    base = _clean_base_url(value)
    if base.endswith(NATIVE_PATH):
        return base
    for suffix in ("/compatible-mode/v1", "/compatible-mode", "/api/v1"):
        if base.endswith(suffix):
            base = base[:-len(suffix)].rstrip("/")
            break
    return base + NATIVE_PATH


def _data_uri(path: str) -> str:
    data, mime = utils.file_to_base64(path)
    return f"data:{mime};base64,{data}"


def _build_body(req: GenRequest, model: str) -> dict:
    # The native API accepts 1–3 input images total.  The current clean render is
    # always first; up to two context images follow, then exactly one text part.
    content = [{"image": _data_uri(req.clean_image_path)}]
    for ref in req.reference_images[:2]:
        if os.path.isfile(ref):
            content.append({"image": _data_uri(ref)})
    content.append({"text": req.prompt})

    # Qwen Image 3.0 supports output up to 2048 square.  The downstream mask
    # extractor resamples back to render_resolution when these differ.
    output_size = max(512, min(int(req.output_size), 2048))
    parameters = {
        "prompt_extend": False,
        "enable_thinking": True,
        "n": 1,
        "size": f"{output_size}*{output_size}",
        "watermark": False,
    }
    if req.seed and req.seed > 0:
        parameters["seed"] = min(int(req.seed), 2147483647)
    return {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }


def _extract_image(data: dict) -> str:
    try:
        content = data["output"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = []
    for item in content if isinstance(content, list) else []:
        value = item.get("image") if isinstance(item, dict) else None
        if isinstance(value, str) and value:
            return value
    message = data.get("message") if isinstance(data, dict) else None
    raise ProviderError(
        f"No image in Qwen response{': ' + str(message) if message else ''}: "
        f"{str(data)[:500]}", kind="API")


class QwenImageProvider(AIProvider):
    name = "QWEN"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_reference_images=3,
            supports_image_edit=True,
            supports_mask=False,
            supports_depth_or_control=False,
            supports_multi_turn_context=True,
            supports_seed=True,
        )

    def validate_config(self, prefs) -> list:
        problems = []
        raw_url = getattr(prefs, "api_base_url", "")
        if not raw_url:
            problems.append("API Base URL is empty")
        else:
            endpoint = native_endpoint(raw_url)
            parsed = urlsplit(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                problems.append("API Base URL must be an absolute http(s) URL")
            try:
                endpoint.encode("ascii")
            except UnicodeEncodeError:
                problems.append(
                    "API Base URL contains non-ASCII characters; remove copied Chinese punctuation")
        if not prefs.get_api_key():
            problems.append("API Key is empty")
        if not getattr(prefs, "model_id", ""):
            problems.append("Model ID is empty (e.g. qwen-image-3.0)")
        return problems

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        if req.should_cancel():
            raise ProviderError("Cancelled", kind="CANCEL")

        url = native_endpoint(prefs.get_base_url())
        body = _build_body(req, prefs.model_id)
        headers = {"Authorization": f"Bearer {prefs.get_api_key()}"}

        req.on_progress(0.3, "Requesting Qwen Image edit…")
        code, response = http_util.post_json(url, body, headers, prefs.timeout)
        if code != 200:
            raise ProviderError(
                f"Qwen Image failed ({code}): "
                f"{response.decode('utf-8', 'replace')[:500]}", kind="API")
        try:
            data = json.loads(response)
        except Exception:
            raise ProviderError(
                f"Non-JSON Qwen response: {response.decode('utf-8', 'replace')[:500]}",
                kind="API")

        if req.should_cancel():
            raise ProviderError("Cancelled", kind="CANCEL")
        image_value = _extract_image(data)
        os.makedirs(req.out_dir, exist_ok=True)
        out_path = os.path.join(req.out_dir, f"worn_{os.urandom(4).hex()}.png")
        req.on_progress(0.8, "Saving Qwen worn view…")
        if image_value.startswith("data:image"):
            try:
                encoded = image_value.split(",", 1)[1]
                with open(out_path, "wb") as handle:
                    handle.write(base64.b64decode(encoded))
            except Exception as exc:
                raise ProviderError(f"Invalid Qwen image data URI: {exc}", kind="API")
        elif image_value.startswith(("http://", "https://")):
            # The returned object-storage URL is signed.  Do not forward the
            # Model Studio bearer token to that separate host.
            http_util.download(image_value, out_path, None, prefs.timeout)
        else:
            raise ProviderError("Qwen response image is neither a URL nor data URI", kind="API")

        return GenResult(
            worn_image_path=out_path,
            used_seed=req.seed,
            raw_response=json.dumps(data, ensure_ascii=False)[:800],
        )
