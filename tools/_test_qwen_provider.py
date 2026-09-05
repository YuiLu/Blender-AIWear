"""Offline contract tests for the Qwen Image native provider."""

import json
import os
import tempfile

from ai_wear.ai import http_util
from ai_wear.ai.base import GenRequest
from ai_wear.ai.qwen_image_provider import QwenImageProvider, native_endpoint


USER_URL = (
    "https://llm-lqutzmjw0v5wojik.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1，"
)
EXPECTED = (
    "https://llm-lqutzmjw0v5wojik.cn-beijing.maas.aliyuncs.com/"
    "api/v1/services/aigc/multimodal-generation/generation"
)


class Prefs:
    api_base_url = USER_URL
    model_id = "qwen-image-3.0"
    timeout = 10.0

    def get_api_key(self):
        return "test-key"

    def get_base_url(self):
        return self.api_base_url.rstrip("/")


assert native_endpoint(USER_URL) == EXPECTED

try:
    http_util._request("https://example.com/，", b"", {}, 1.0)
except http_util.NetworkError as exc:
    assert exc.kind == "CONFIG" and "non-ASCII" in str(exc)
else:
    raise AssertionError("non-ASCII URL was not rejected before network access")

with tempfile.TemporaryDirectory() as tmp:
    clean = os.path.join(tmp, "clean.png")
    anchor = os.path.join(tmp, "anchor.png")
    with open(clean, "wb") as handle:
        handle.write(b"clean")
    with open(anchor, "wb") as handle:
        handle.write(b"anchor")

    captured = {}
    old_post_json = http_util.post_json
    old_download = http_util.download

    def fake_post_json(url, body, headers, timeout):
        captured.update(url=url, body=body, headers=headers, timeout=timeout)
        response = {
            "output": {"choices": [{"message": {"content": [
                {"image": "https://signed.example/result.png"}
            ]}}]}
        }
        return 200, json.dumps(response).encode("utf-8")

    def fake_download(url, dest_path, headers, timeout):
        captured.update(download_url=url, download_headers=headers)
        with open(dest_path, "wb") as handle:
            handle.write(b"result")
        return dest_path

    http_util.post_json = fake_post_json
    http_util.download = fake_download
    try:
        req = GenRequest(
            clean_image_path=clean,
            prompt='wear "only"',
            seed=123,
            output_size=4096,
            reference_images=[anchor],
            out_dir=tmp,
        )
        result = QwenImageProvider().generate(req, Prefs(), None)
    finally:
        http_util.post_json = old_post_json
        http_util.download = old_download

    assert captured["url"] == EXPECTED
    assert captured["headers"] == {"Authorization": "Bearer test-key"}
    assert captured["download_headers"] is None
    body = captured["body"]
    assert body["model"] == "qwen-image-3.0"
    assert body["parameters"]["size"] == "2048*2048"
    assert body["parameters"]["seed"] == 123
    content = body["input"]["messages"][0]["content"]
    assert len([part for part in content if "image" in part]) == 2
    assert content[-1] == {"text": 'wear "only"'}
    assert os.path.isfile(result.worn_image_path)

print("Qwen provider offline contract: PASS")
