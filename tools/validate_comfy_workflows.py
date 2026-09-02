"""Static validation for the bundled ComfyUI UI/API workflow pair."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_PATH = ROOT / "examples" / "comfyui" / "aiwear_inpaint_api.json"
UI_PATH = ROOT / "examples" / "comfyui" / "aiwear_inpaint_workflow.json"
BUNDLED_API_PATH = ROOT / "ai_wear" / "workflows" / "aiwear_inpaint_api.json"


def main() -> None:
    api = json.loads(API_PATH.read_text(encoding="utf-8"))
    ui = json.loads(UI_PATH.read_text(encoding="utf-8"))
    bundled_api = json.loads(BUNDLED_API_PATH.read_text(encoding="utf-8"))
    assert bundled_api == api, "bundled and example API workflows differ"
    api_ids = set(api)
    ui_nodes = {str(node["id"]): node for node in ui["nodes"]}
    assert api_ids == set(ui_nodes), "UI/API node ids differ"

    core_nodes = {
        "CheckpointLoaderSimple", "LoadImage", "CLIPTextEncode",
        "VAEEncodeForInpaint", "KSampler", "VAEDecode",
        "ImageCompositeMasked", "SaveImage",
    }
    for node_id, node in api.items():
        assert node["class_type"] in core_nodes, f"custom node at {node_id}"
        assert ui_nodes[node_id]["type"] == node["class_type"]
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2:
                assert str(value[0]) in api_ids, f"missing API source {value[0]}"

    links = {int(link[0]): link for link in ui["links"]}
    assert len(links) == len(ui["links"]), "duplicate UI link id"
    for link_id, (_, src, _slot, dst, _input, _kind) in links.items():
        assert str(src) in ui_nodes, f"link {link_id} missing source"
        assert str(dst) in ui_nodes, f"link {link_id} missing destination"
    for node in ui["nodes"]:
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert link_id in links and links[link_id][1] == node["id"]
        for input_socket in node.get("inputs", []):
            link_id = input_socket.get("link")
            if link_id is not None:
                assert link_id in links and links[link_id][3] == node["id"]

    expected_mapping = {"clean": "2", "mask": "3", "prompt": "4",
                        "seed": "7", "output": "10"}
    assert api[expected_mapping["clean"]]["class_type"] == "LoadImage"
    assert api[expected_mapping["mask"]]["class_type"] == "LoadImage"
    assert api[expected_mapping["prompt"]]["class_type"] == "CLIPTextEncode"
    assert api[expected_mapping["seed"]]["class_type"] == "KSampler"
    assert api[expected_mapping["output"]]["class_type"] == "SaveImage"
    print(f"OK: {len(api)} core nodes, {len(links)} links, UI/API pair aligned")


if __name__ == "__main__":
    main()
