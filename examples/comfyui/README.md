# AIWear ComfyUI / Liblib 工作流

- `aiwear_inpaint_workflow.json`：UI 格式，拖入 ComfyUI 或 Liblib 工作流画布。
- `aiwear_inpaint_api.json`：API 格式，供 Blender 插件通过 `/prompt` 调用。

只使用 ComfyUI 原生节点：`CheckpointLoaderSimple`、`LoadImage`、
`CLIPTextEncode`、`VAEEncodeForInpaint`、`KSampler`、`VAEDecode`、
`ImageCompositeMasked`、`SaveImage`，不依赖自定义节点。

导入 UI 工作流后，只需把节点 1 换成平台已有的 inpainting checkpoint。
Blender Preferences 中填写：

| 字段 | 节点 ID |
| --- | ---: |
| Clean Image Node | 2 |
| Inpaint Mask Node | 3 |
| Prompt Node | 4 |
| Seed Node | 7 |
| Output Node | 10 |

mask 由 Blender 根据物体轮廓和深度突变自动生成；工作流最后再次按 mask
合成，保证未选区域严格沿用 clean view。关闭 `Geometry Inpaint Mask` 后，
请改用不要求 mask 的自定义 API 工作流。
