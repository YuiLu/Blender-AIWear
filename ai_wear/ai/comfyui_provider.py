"""ComfyUI provider.

Loads an API-format workflow JSON, wires the configured input nodes (clean
image / prompt / seed), queues it via /prompt, polls /history until the run
finishes, then downloads the configured output node's image via /view.

Changing the workflow never requires a code change — only the node-id mapping
in AddonPreferences. Optional depth/normal/control images can be wired to any
extra LoadImage node id listed in node_mapping (key=label, value=node_id).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional, Tuple

from . import http_util
from .base import AIProvider, GenRequest, GenResult, ProviderCapabilities, ProviderError


class ComfyUIProvider(AIProvider):
    name = "COMFYUI"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_reference_images=8,
            supports_image_edit=True,
            supports_mask=True,
            supports_depth_or_control=True,
            supports_seed=True,
            supports_model_snapshot=True,
        )

    def validate_config(self, prefs) -> list:
        probs = []
        if not prefs.comfyui_url:
            probs.append("ComfyUI URL is empty")
        if not prefs.workflow_path or not os.path.exists(bpy_abspath(prefs.workflow_path)):
            probs.append("Workflow JSON path is invalid or missing")
        return probs

    # --- helpers ----------------------------------------------------------

    def _root(self, prefs) -> str:
        return prefs.comfyui_url.rstrip("/")

    def _upload_image(self, prefs, image_path: str) -> str:
        root = self._root(prefs)
        with open(image_path, "rb") as f:
            data = f.read()
        fname = f"aiwear_{uuid.uuid4().hex[:8]}.png"
        code, body = http_util.post_multipart(
            f"{root}/upload/image",
            fields={"overwrite": "true", "type": "input"},
            files={"image": (fname, data, "image/png")},
            headers=None, timeout=prefs.timeout,
        )
        if code != 200:
            raise ProviderError(f"ComfyUI upload failed ({code})", kind="API")
        info = json.loads(body)
        name = info.get("name", fname)
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    @staticmethod
    def _load_workflow(prefs) -> dict:
        path = bpy_abspath(prefs.workflow_path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _wire_inputs(self, wf: dict, req: GenRequest, prefs) -> Tuple[str, Optional[str]]:
        """Wire clean image / prompt / seed nodes. Returns (output_node_id or None)."""
        clean_node = prefs.clean_image_node
        prompt_node = prefs.prompt_node
        seed_node = prefs.seed_node
        out_node = prefs.output_node

        # Auto-detect by class_type when mapping is missing
        if not (clean_node and prompt_node and seed_node):
            for nid, n in wf.items():
                if not isinstance(n, dict):
                    continue
                ct = n.get("class_type", "")
                if not clean_node and ct == "LoadImage":
                    clean_node = nid
                if not prompt_node and ct in ("CLIPTextEncode", "BERTPositivePromptModelList"):
                    prompt_node = nid
                if not seed_node and ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                    seed_node = nid

        # Upload clean view and set LoadImage
        clean_filename = self._upload_image(prefs, req.clean_image_path)
        if clean_node and clean_node in wf:
            wf[clean_node].setdefault("inputs", {})["image"] = clean_filename

        if prompt_node and prompt_node in wf:
            wf[prompt_node].setdefault("inputs", {})["text"] = req.prompt

        if seed_node and seed_node in wf:
            wf[seed_node].setdefault("inputs", {})["seed"] = int(req.seed) if req.seed > 0 else \
                int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        # Wire extra references (depth/normal/anchor/guide) per node_mapping
        extra = dict(req.node_mapping or {})
        extra.update({l: n for l, n in ({"depth": "", "normal": ""}.items())})  # no-op keep
        label_path = {
            "depth": req.depth_path,
            "normal": req.normal_path,
        }
        for i, ref in enumerate(req.reference_images):
            label_path[f"ref{i}"] = ref
        for label, node_id in (req.node_mapping or {}).items():
            p = label_path.get(label)
            if p and node_id and node_id in wf and os.path.exists(p):
                fn = self._upload_image(prefs, p)
                wf[node_id].setdefault("inputs", {})["image"] = fn

        return out_node

    def _queue(self, prefs, wf: dict) -> str:
        client_id = uuid.uuid4().hex
        body = {"prompt": wf, "client_id": client_id}
        code, resp = http_util.post_json(
            f"{self._root(prefs)}/prompt", body, headers=None, timeout=prefs.timeout)
        if code != 200:
            raise ProviderError(f"ComfyUI /prompt failed ({code}): {resp.decode('utf-8','replace')[:500]}", kind="API")
        data = json.loads(resp)
        pid = data.get("prompt_id")
        if not pid:
            raise ProviderError(f"ComfyUI did not return prompt_id: {data}", kind="API")
        return pid

    def _poll(self, prefs, prompt_id: str, req: GenRequest) -> dict:
        root = self._root(prefs)
        t0 = time.time()
        last_status = ""
        while True:
            if req.should_cancel():
                raise ProviderError("Cancelled", kind="CANCEL")
            if time.time() - t0 > prefs.timeout * 4:
                raise ProviderError("ComfyUI job timed out", kind="TIMEOUT")
            code, body = http_util.get_bytes(f"{root}/history/{prompt_id}", timeout=prefs.timeout)
            if code == 200 and body:
                data = json.loads(body)
                if prompt_id in data:
                    return data[prompt_id]
            req.on_progress(0.5, "ComfyUI generating…")
            time.sleep(prefs.poll_interval if hasattr(prefs, "poll_interval") else 1.0)

    def _download_output(self, prefs, history_entry: dict, out_node: Optional[str],
                         req: GenRequest) -> str:
        root = self._root(prefs)
        outputs = history_entry.get("outputs", {}) or {}
        node = outputs.get(out_node) if out_node else None
        if not node:
            # fallback: first node that has 'images'
            for nid, nout in outputs.items():
                if isinstance(nout, dict) and nout.get("images"):
                    node = nout
                    break
        if not node or not node.get("images"):
            raise ProviderError("ComfyUI run produced no image outputs", kind="API")
        img_info = node["images"][0]
        fname = img_info.get("filename")
        sub = img_info.get("subfolder", "")
        typ = img_info.get("type", "output")
        params = {"filename": fname, "subfolder": sub, "type": typ}
        from urllib.parse import urlencode
        url = f"{root}/view?{urlencode(params)}"
        out_path = os.path.join(req.out_dir, f"worn_{uuid.uuid4().hex[:8]}.png")
        http_util.download(url, out_path, timeout=prefs.timeout)
        return out_path

    def generate(self, req: GenRequest, prefs, scene) -> GenResult:
        if not prefs.workflow_path or not os.path.exists(bpy_abspath(prefs.workflow_path)):
            raise ProviderError("ComfyUI workflow path not set", kind="CONFIG")
        wf = self._load_workflow(prefs)
        out_node = self._wire_inputs(wf, req, prefs)
        req.on_progress(0.2, "Queuing ComfyUI job…")
        prompt_id = self._queue(prefs, wf)
        req.on_progress(0.4, "Polling ComfyUI history…")
        history = self._poll(prefs, prompt_id, req)
        req.on_progress(0.85, "Downloading output…")
        out_path = self._download_output(prefs, history, out_node, req)
        return GenResult(worn_image_path=out_path, used_seed=req.seed,
                         raw_response="comfyui:" + prompt_id)


def bpy_abspath(path: str) -> str:
    """Resolve a Blender file path (//relative or absolute) for filesystem use."""
    try:
        import bpy
        return bpy.path.abspath(path)
    except Exception:
        return path
