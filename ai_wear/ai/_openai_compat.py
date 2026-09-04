"""Shared OpenAI-compatible image-edit logic.

Used by both OpenAIProvider and the OPENAI_COMPAT mode of CustomHTTPProvider.
Any endpoint that speaks OpenAI's /images/edits multipart form works here,
so a new vendor is just a Base URL + Model + Key in preferences.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Optional, Tuple

from . import http_util
from .base import GenRequest, GenResult, ProviderError
from .. import utils


MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_RETRY_AFTER_SECONDS = 10.0


def _retry_after_seconds(error: Exception) -> float:
    """Read Azure's human-readable retry delay, with a safe default."""
    match = re.search(r"retry\s+after\s+(\d+(?:\.\d+)?)\s*seconds?",
                      str(error), flags=re.IGNORECASE)
    if match:
        return max(0.5, min(float(match.group(1)), 120.0))
    return DEFAULT_RETRY_AFTER_SECONDS


def _wait_for_retry(req: GenRequest, seconds: float, attempt: int) -> None:
    req.on_progress(0.3,
                    f"Rate limited; retrying in {seconds:g}s "
                    f"({attempt}/{MAX_RATE_LIMIT_RETRIES})…")
    deadline = time.monotonic() + seconds
    # Short sleeps keep Cancel responsive while a worker waits for Azure.
    while True:
        if req.should_cancel():
            raise ProviderError("Cancelled", kind="CANCEL")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def _post_edit_with_retry(url, fields, files, headers, timeout, req):
    """Submit a multipart edit, retrying only the explicitly transient 429."""
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        if req.should_cancel():
            raise ProviderError("Cancelled", kind="CANCEL")
        try:
            return http_util.post_multipart(url, fields, files, headers, timeout)
        except http_util.NetworkError as e:
            if getattr(e, "status", 0) != 429 or attempt >= MAX_RATE_LIMIT_RETRIES:
                raise
            _wait_for_retry(req, _retry_after_seconds(e), attempt + 1)
    raise AssertionError("unreachable")


def _parse_extra_headers(prefs) -> dict:
    if not getattr(prefs, "extra_headers", ""):
        return {}
    try:
        obj = json.loads(prefs.extra_headers)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def _extract_image_field(item: dict) -> Tuple[str, str]:
    """Return (kind, value): kind in {'b64','url'}."""
    for k in ("b64_json", "base64", "b64", "data"):
        v = item.get(k)
        if isinstance(v, str) and v:
            if v.startswith("http"):
                return ("url", v)
            return ("b64", v)
    v = item.get("url") or item.get("image")
    if isinstance(v, str) and v.startswith("http"):
        return ("url", v)
    if isinstance(v, str) and v:
        return ("b64", v)
    raise ProviderError("No image field in response item", kind="API")


def _extract_from_response(data) -> Tuple[str, str]:
    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data") or data.get("images") or data.get("results")
    if not items or not isinstance(items, list):
        raise ProviderError(f"Unexpected response shape: {str(data)[:300]}", kind="API")
    return _extract_image_field(items[0])


def generate_openai_compat(req: GenRequest, prefs, base_url: str, path: str,
                           model: Optional[str] = None) -> GenResult:
    headers = {}
    key = prefs.get_api_key() if hasattr(prefs, "get_api_key") else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    headers.update(_parse_extra_headers(prefs))

    url = base_url.rstrip("/") + path
    with open(req.clean_image_path, "rb") as f:
        img_bytes = f.read()
    fname = os.path.basename(req.clean_image_path) or "clean.png"

    fields = {
        "prompt": req.prompt,
        "model": model or prefs.model_id,
        "n": "1",
        "size": f"{req.output_size}x{req.output_size}",
    }
    # 'seed' is non-standard for OpenAI's /images/edits. Some clones use it for
    # reproducibility; strict OpenAI-compatible endpoints (e.g. gpt-image-*)
    # reject it with 400 unknown_parameter. We send it when requested and retry
    # once without if rejected (see below).
    send_seed = bool(req.seed and req.seed > 0)
    if send_seed:
        fields["seed"] = str(req.seed)
    files = {"image": (fname, img_bytes, "image/png")}
    # Optional mask
    if req.mask_path and os.path.exists(req.mask_path):
        with open(req.mask_path, "rb") as f:
            files["mask"] = (os.path.basename(req.mask_path), f.read(), "image/png")

    req.on_progress(0.3, "Uploading clean view…")
    try:
        code, body = _post_edit_with_retry(url, fields, files, headers, prefs.timeout, req)
    except http_util.NetworkError as e:
        # Strict OpenAI-compat endpoints (e.g. gpt-image-*) reject the non-standard
        # 'seed' param with 400 unknown_parameter. http_util raises on 4xx, so we
        # catch it here and retry once without seed. The 400 is returned at request
        # validation (before the model runs), so the extra round-trip is cheap.
        # Reproducibility isn't possible on such endpoints anyway — their
        # images/edits API has no seed parameter.
        if send_seed and getattr(e, "status", 0) == 400 and "unknown_parameter" in str(e) and "seed" in str(e):
            fields.pop("seed", None)
            req.on_progress(0.3, "Endpoint rejected 'seed'; retrying without…")
            code, body = _post_edit_with_retry(url, fields, files, headers, prefs.timeout, req)
        else:
            raise
    if code != 200:
        raise ProviderError(f"Image edit failed ({code}): {body.decode('utf-8','replace')[:500]}", kind="API")
    try:
        data = json.loads(body)
    except Exception:
        raise ProviderError(f"Non-JSON response: {body.decode('utf-8','replace')[:500]}", kind="API")

    kind, val = _extract_from_response(data)
    out_path = os.path.join(req.out_dir, f"worn_{os.urandom(4).hex()}.png")
    req.on_progress(0.8, "Downloading worn view…")
    if kind == "url":
        http_util.download(val, out_path, headers, prefs.timeout)
    else:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(val))
    return GenResult(worn_image_path=out_path, used_seed=req.seed,
                     raw_response=json.dumps(data)[:800])
