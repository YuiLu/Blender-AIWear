# Blender AI Wear

把 2D 生图模型给出的磨损语义，稳定重投影到 Blender 网格表面，生成可单调调节的
`WearTime` 与材质叠加层。AI 只负责“磨损长什么样”，相机、深度、几何与 UV 决定
“磨损落在哪里”。

## 安装与运行

要求 Blender 3.6+，无需额外 pip 依赖。

1. 将 `ai_wear/` 放入 Blender 用户 `scripts/addons/`，或把仓库打成顶层为
   `ai_wear/` 的 zip 后安装。
2. 在 `Edit > Preferences > Add-ons > AI Wear Texture` 配置 Provider、模型和密钥。
3. 选中一个 Mesh，在 `N > AI Wear` 选择 UV、相机与参数，先运行 Preflight，
   再点 `Generate Wear Texture`。
4. 完成后拖动 `Wear Amount` 和 `Feather` 实时预览；这两个参数不重新请求 AI。
5. 调试下游参数时使用 `Replay Downstream`，复用缓存视图，不消耗生图额度。

Provider 支持 OpenAI、Gemini、ComfyUI 和可配置的 OpenAI-compatible / Raw JSON
HTTP 端点。API key 只保存在 Add-on Preferences，也可用环境变量覆盖。

## 算法

```text
UV/QC → 多视角 clean render → AI edit/inpaint → clean/worn diff
     → 相机 + Z-buffer 重投影 → 多视角表面融合 → 几何先验
     → 拓扑 WearTime → seam fusion / padding → 贴图与 Shader
```

### 1. 表面参数化

- Mode A 使用指定现有 UV；重叠岛共享磨损。
- Mode B 新建 `AI_WearUV`，不修改原材质 UV。
- UV 光栅化为每个 texel 保存三角形 id 与重心坐标，因此可直接恢复 3D 位置和法线，
  不做逐 texel ray cast。
- 完整运行保存 per-loop UV 快照；Replay 在拓扑一致时可精确恢复缺失的 Wear UV。

### 2. 视图差分与重投影

每个固定相机生成 `clean_Vi`，AI 在相同构图上生成 `worn_Vi`。颜色、梯度和结构差异
经鲁棒归一化得到屏幕 mask `M_i`。已知相机矩阵与软件 Z-buffer 后，UV texel 的表面点
只有在当前视图可见时才接收该像素观测：

```text
w_i = visible_i · max(0, dot(n, view_i))^gamma · confidence_i
F_ai = robust_weighted_mean(M_i, w_i)
```

`views/diff_mask_V*.png` 保存实际参与重投影的 16-bit RGB 灰度 mask；
`AIWear_Mask.png` 是融合后的 UV 域 AI 证据。

### 3. 几何先验与 WearTime

AI 证据先从 UV 转到顶点，再与凸度、曝光度、凹陷惩罚组合：

```text
P(v) = clamp(w_ai·F_ai + w_convex·C+ + w_expose·E - w_cavity·C-)
edge_cost(i,j) = length(i,j) / (eps + mean(P_i,P_j)^gamma)
T(v) = clamp(alpha·normalize(Dijkstra(v)) + (1-alpha)·(1-P(v)) + noise3D)
```

高 `P` 的局部极大值是磨损种子；多源 Dijkstra 沿真实网格拓扑传播。噪声采样物体
空间而非 UV，因此不会在 UV seam 处自行断裂。`Wear Amount` 只在 Shader 中阈值化：

```text
gate = smoothstep(T - feather, T + feather, amount)
final_mask = gate · AI_evidence
```

因此 Amount 单调增长，100% 也只显示有 AI 磨损证据的区域。

### 4. 外观叠加

插件不把 AI 渲染图直接当 albedo。它重投影并编码 `clean → worn` 的有界颜色残差，
写入 `AIWear_WornTex.png`：RGB 为残差，Alpha 为 AI evidence。节点组通过显式 Wear UV
解码残差并叠加到原 Principled Base Color，同时保留原法线、粗糙度、金属度和其他材质连接。
共享材质会先复制为对象私有材质，避免影响其他对象。

### 5. Seam 与 padding

同一 3D 边在两个 UV 岛上的 texel 构成 seam pair；融合阶段在 pair 间平均并做有限扩散。
padding 单独对岛外 texel 膨胀，抑制双线性采样和 mipmap 漏色。二者是独立开关，便于
做无处理 / 仅融合 / 仅 padding / 两者都开四组实验。

## ComfyUI / Liblib

仓库包含一对同源的纯原生节点工作流：

- [`examples/comfyui/aiwear_inpaint_workflow.json`](examples/comfyui/aiwear_inpaint_workflow.json)：
  UI 格式，导入 ComfyUI 或 Liblib 画布。
- [`examples/comfyui/aiwear_inpaint_api.json`](examples/comfyui/aiwear_inpaint_api.json)：
  API 格式，Blender 插件通过 `/prompt` 执行。

工作流使用 `VAEEncodeForInpaint` 做局部重绘，并在输出前再次按 mask 与 clean view 合成，
未选区域保持不变。Blender 会从物体轮廓与深度突变生成每视角
`inpaint_mask_V*.png`。节点映射固定为：Clean `2`、Mask `3`、Prompt `4`、Seed `7`、
Output `10`。平台上只需在节点 1 选择已有的 inpainting checkpoint。

插件会在排队前验证 API 图中所有节点引用与 Preferences 映射。UI JSON 不能误选为插件
API JSON；两者用途见 [`examples/comfyui/README.md`](examples/comfyui/README.md)。

## 实验与消融

`N > AI Wear > Experiments / Ablation` 可独立关闭：

- AI Evidence
- Geometry Prior
- Topology Growth
- Seam Fusion
- Island Padding
- Geometry Inpaint Mask

相机选 `Counted Auto` 后，`Cam Count` 精确控制为 1–16 个 Fibonacci-sphere 视角。
开启 `Save Experiment Snapshot` 后，每次运行或 Replay 会写入：

```text
.ai_wear_cache/<object>/experiments/<label>_<job_id>/
├─ config.json
├─ metrics.json
├─ AIWear_Mask.png
├─ WearTime_before_seam_padding.png
├─ WearTime_after_seam_padding.png
├─ WearTime.png
└─ AIWear_WornTex.png
```

推荐实验矩阵、指标与变量控制见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 输出与代码结构

主要输出位于 `<blend>/.ai_wear_cache/<object>/`：

- `views/clean_V*`, `worn_V*`, `diff_mask_V*`, `inpaint_mask_V*`, `views.json`
- `AIWear_Mask.png`：重投影融合后的 AI 磨损证据
- `WearTime.png`：可单调阈值化的磨损时间场
- `AIWear_WornTex.png`：颜色残差 + evidence alpha
- `AIWear_UVSnapshot.npz`：Replay 所需 Wear UV

核心模块：

```text
ai_wear/render      相机与 clean pass
ai_wear/ai          Provider 与 ComfyUI 图执行
ai_wear/uv          UV QC、光栅化、seam registry
ai_wear/surface     mask 投影、多视角融合、几何先验、拓扑生长
ai_wear/shader      对象私有材质与实时 WearTime 节点组
ai_wear/operators   后台任务、缓存、Replay、实验快照
```

模块拆解与验收映射见 `Blender_AI_Wear_Texture_Plugin_Implementation_Plan.md`。
