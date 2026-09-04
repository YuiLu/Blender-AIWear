# Blender AI 模型磨损纹理生成插件

技术实现文档 / Implementation Plan

范围：UV 质检与自动展开 · 3D Surface Field · 拓扑生长式 WearThreshold · UV Seam Fusion · 多 API 接入

V0.2 — 面向 MVP 落地（补充多视角一致性与 Demo Mesh）

# 0. 实现边界

本实现以现有需求文档中的 Mode A / Mode B、UV 接缝、磨损强度控制、多模型 / 自定义 ComfyUI 接入为需求基线。重点不是复刻参考路线，而是把“AI 生成的表面变化”稳定、可重复地落回 3D 模型表面和目标 UV。

> **明确不做** AI 高清放大、背景移除、完整 AI PBR、训练 / 微调 / LoRA、Blender 内加载 PyTorch、强依赖 RizomUV、第一版 dense optical flow。

MVP 默认处理单个活动 Mesh；多对象批处理、xatlas、RizomUV Bridge、完整 PBR 生成、复杂局部重投影作为 Future Work。

# 1. Problem Reframing

单张未知参考图反推到模型 UV，本质是欠定问题：相机参数未知、遮挡不可见、同类几何重复、透视与表面展开不同构，无法仅凭 2D 像素保证其唯一对应某个 3D 表面点。直接“把参考图摊到 UV”只能在少数受控案例中成立。

当前方案把输入改造成受控问题：由 Blender 自己生成固定相机的 Clean View，并同时输出 Depth / Normal / Target UV 等几何辅助数据；AI 只负责在相同构图下编辑表面磨损。这样每个 AI 像素都可以通过已知相机和模型几何重新定位到目标表面。

因此整个系统的核心不是“AI 生图”，而是：

```text
AI：决定磨损分布与视觉语义
Blender Geometry：决定像素到底属于模型表面的哪里
3D Surface Field：承载跨视角、跨 UV 岛的一致表面数据
WearThreshold：把一次 AI 结果转换成连续 0~100 的可控磨损过程
UV：最后的存储 / 导出载体，而不是匹配算法本身
```

# 2. 管线概述

```text
[Mesh Preflight / UV QC]
        ↓
[Mode A: 使用现有 UV]  或  [Mode B: 创建 AI_WearUV + Blender 自动展开]
        ↓
[自动多视角 Camera]
        ↓
[Render Clean RGB + Alpha + Depth + Normal]
        ↓
[AI Provider: OpenAI / Gemini / ComfyUI / Custom]
        ↓
[Anchor Worn View → Partial Surface Field]
        ↓
[后续视角：Clean View + Anchor / Projected Wear Guide → Consistent Worn Views]
        ↓
[Clean ↔ Worn 差异提取 → Screen-space Wear Mask]
        ↓
[3D Surface Field Reconstruction]
        ↓
[Multi-view Fusion + Geometry Prior]
        ↓
[Topology-aware WearThreshold / Wear Activation Field]
        ↓
[Seam Registry + Seam Fusion + Padding]
        ↓
[Blender Shader: Wear Amount 0~100 实时阈值 / Dissolve]
        ↓
[WearThreshold / WearMask 输出；ORM 等派生贴图后续扩展]
```

| 阶段 | 输入 | 输出 | 核心职责 |
| --- | --- | --- | --- |
| Preflight | 活动 Mesh | 检查报告 | 验证拓扑、UV、Modifier、尺度、材质与可处理性 |
| UV | UV0 / 无 UV | Target Wear UV | Mode A 保持现有；Mode B 新建 AI_WearUV |
| Capture | 模型 + Target UV | Clean View + G-buffer | 固定相机与几何对应关系 |
| AI | Clean View + Prompt | Worn View | 只改表面状态，不负责 UV |
| Mask | Clean/Worn | 2D Wear Mask | 提取 AI 引入的磨损变化 |
| Surface Field | Mask + Camera + Mesh | Surface Wear Score | 将多视角信号融合到模型表面 |
| WearThreshold | AI Field + Geometry | 0~1 Activation Time | 用拓扑传播构造连续磨损过程 |
| Seam / Bake | Surface Field + UV | 无缝纹理 | 跨 UV 岛约束与 padding |
| Preview / Export | WearThreshold | Shader / PNG16 | 0~100 实时预览与导出 |

# 3. Feature 与实现思路

## 3.1 UV Mode A / Mode B + 模型 UV 质检

Mode A 的定义是“直接使用用户指定的现有 UV 层作为 Target Wear UV”；Mode B 的定义是“保留现有 UV，不覆盖，在对象上新增 AI_WearUV，并使用 Blender 内置 UV 算法自动展开”。

Mode A 不自动修 UV。质检发现 overlap / degenerate / 0~1 外溢等问题时，给出明确风险和切换 Mode B 的建议；用户强制继续时，结果允许出现共享磨损，这是该模式的预期限制。Mode B 的目标是生成唯一、可 bake 的磨损 UV。

| QC 项 | 实现 | 判定 / 输出 |
| --- | --- | --- |
| UV 是否存在 | 读取 mesh.uv_layers / active layer | Mode A 必须存在；Mode B 可从无 UV 开始 |
| 退化 UV 三角形 | 三角化后计算 signed UV area | area≈0 记为 invalid |
| UV overlap | 低分辨率 UV raster collision map；统计同 texel 多三角形覆盖率 | 输出 overlap ratio；Mode A 警告，Mode B 失败则重新展开 |
| 镜像 / 翻转 | signed UV area < 0 的三角形比例 | 标记为信息项；scalar mask 可用，但用于诊断 |
| 0~1 / UDIM | 统计 UV bbox 与 tile | MVP 默认 WearUV 目标为单 0~1 tile |
| UV 利用率 | raster valid texel / atlas texel | 低利用率提示重新 pack |
| 极小岛 / 极端密度差 | 岛面积与 3D 面积比值 | 输出异常岛列表 |
| Modifier 风险 | 比较 base / evaluated mesh 与拓扑变化 modifier | Mode B 如无法安全回写则使用 Processing Copy 或阻止执行 |

Mode B 自动展开建议流程：

```text
1. 新建 UV Layer: AI_WearUV
2. 保存当前 active object / mode / selection / UV layer 状态
3. 选择全部目标面
4. 默认执行 Blender Smart UV Project（硬表面优先）
5. Pack Islands，按目标贴图尺寸换算 island margin
6. 运行同一套 UV QC
7. 若 overlap / degenerate 未达标，调整 angle_limit / margin 做一次自动重试
8. 恢复用户场景状态；AI_WearUV 仅作为磨损贴图坐标
```

xatlas 仅保留统一的 UVBackend 接口，未来可加入，不进入当前依赖。

## 3.2 3D Surface Field

Surface Field 是模型表面的标量数据层，用来承接多视角 AI Mask，并为后续拓扑生长、接缝融合和 Bake 提供统一数据。建议采用“mesh-domain 低频场 + UV-backed 高分辨率场”的混合实现：拓扑和生长在 mesh domain 完成，高频 AI 细节与最终输出在 UV texture domain 完成。

核心数据：

```text
SurfaceField
- target_uv_layer
- tri_id / barycentric lookup（按工作分辨率 rasterize）
- position P / normal N（可分块计算，不必常驻全部内存）
- ai_score        : AI 多视角融合后的磨损概率
- geom_score      : 凸度 / 可见度等几何先验
- wear_threshold       : 0~1，何时开始出现磨损
- confidence      : 多视角覆盖与 AI 对齐置信度
- valid_mask      : 有效表面 texel
```

关键点是避免 Python 对每个 texel 做 ray_cast。Target UV 确定后，先把每个 UV triangle rasterize 到工作分辨率，保存 triangle id + barycentric；由重心坐标直接恢复 3D Position / Normal。之后针对每个 Camera 用矩阵投影回屏幕，并与 Blender Depth Pass 做可见性比较，再采样 Screen-space Wear Mask。整个步骤可用 NumPy 分块向量化。

```text
for each target UV texel x:
    tri, bary = uv_lookup[x]
    P = barycentric_interpolate(vertex_position[tri])
    N = barycentric_interpolate(vertex_normal[tri])

    for each camera i:
        screen_xy, z = project(P, camera_i)
        visible = abs(z - depth_i(screen_xy)) < depth_eps
        facing  = max(0, dot(N, view_dir_i))
        m       = sample(mask_i, screen_xy)

        w = visible * facing^gamma * alignment_confidence_i
        sum_mask[x]   += w * m
        sum_weight[x] += w

AI_Field[x] = robust_fusion(sum / weight)
```

多视角融合默认用加权均值并配合 outlier clipping；如果某个视角的 silhouette / edge alignment 评分低于阈值，整张 view 降权或自动重试。AI 不同磨损程度不会重复生成；多视角只是为同一个“最大合理磨损状态”补齐模型表面覆盖。

### 3.2.1 多视角 Worn Views 一致性

多视角不能只靠提示词解决。**相同 Prompt、相同 Seed 只能降低随机性，不能保证不同相机下同一个 3D 位置产生相同磨损。** 对当前工具来说，Worn View 只是生成 Surface Field 的观测，因此更稳妥的做法是把一致性约束放回 3D，而不是要求图像模型自己完成严格的多视角重建。

默认采用 **Anchor-conditioned Sequential Generation**：

```text
Step 0  固定一次全局 Wear Spec
        - material / wear type / max wear state
        - model snapshot / prompt template / seed policy

Step 1  选择信息量最高的主视角 V0
        Clean(V0) → AI → Worn(V0)
        ↓
        提取 Mask(V0) → Partial Surface Field F0

Step 2  生成下一视角 Vi 前
        将当前 Surface Field Fi-1 投影到 Vi
        → Projected Wear Guide Gi

Step 3  AI 编辑 Vi
        输入优先级：
        1. Clean(Vi)：必须保持几何、相机、轮廓
        2. Anchor Worn View：统一材质老化语言与磨损形态
        3. Projected Wear Guide / Depth / Edge：约束已观察区域
        ↓
        Worn(Vi)

Step 4  Mask(Vi) → Surface Field Fusion
        已有高置信度区域以 3D Field 为主
        新视角主要补充 previously-unseen surface

Step 5  重复直到 coverage 达标
```

核心原则是：**已知区域不允许每个视角重新“发明”一套磨损；AI 的自由度主要留给当前相机中新出现、尚未被 Surface Field 覆盖的区域。** 如果模型支持 mask / control image，可以把已覆盖区域作为强约束；如果只支持普通 image edit，则把 `Projected Wear Guide` 做成额外参考图，并在融合阶段拒绝与现有 3D Field 冲突过大的新观测。

一次性把所有视角输入不是默认方案。原因是很多图片 API 的输出仍以单图为中心，即使支持多参考图，也不代表能稳定返回严格一一对应的 N 个正交视图。Google 当前的 Gemini 图片模型支持多参考图，Gemini 3.1 Flash Image 可在一个 workflow 中使用多张对象参考图；Google 文档同时提示模型不一定严格遵循请求的图片输出数量。Qwen-Image-Edit-2509 在 ComfyUI 中原生支持 1-3 张输入图，并加强了 multi-image editing 与一致性。这些能力适合用来传递 Anchor / Style / Projected Guide，而不是把“生成整套严格一致正交图”本身当作唯一保证。

可保留一个可选的 **Contact Sheet Batch Mode**：把 4 个 Clean Views 排成固定 2×2 contact sheet，一次 edit 整张图，再裁切回 4 个视角。优点是四个视角共享一次生成上下文；缺点是单视角有效分辨率下降、panel 之间可能串扰、输出几何精度通常不如逐视角编辑。因此它更适合作为快速预览或 provider 能力测试，不作为最终 bake 默认路径。

Provider 能力建议抽象为：

```text
ProviderCapabilities
- max_reference_images
- supports_image_edit
- supports_mask
- supports_depth_or_control
- supports_multi_turn_context
- supports_seed
- supports_model_snapshot
```

调度器根据能力自动决定输入组合。例如 Qwen 2509 的 1-3 图预算可以使用 `CleanTarget + AnchorWorn + ProjectedGuide`；支持更多参考图的 Provider 可以额外加入相邻视角。只支持单图编辑时则退化为 `CleanTarget + Prompt`，但最终仍由 3D Surface Field 做冲突过滤。

验收指标不以“几张 Worn View 看起来像”为准，而以 3D 一致性衡量：同一 Surface Point 在两个可见视角反投影得到的 mask 差值、重叠可见区域的 p95 discrepancy、以及最终 Surface Field 的 coverage/confidence。这样 Provider 更换后仍可用相同指标验证。

## 3.3 WearThreshold：用拓扑自然生长的 Shader / Dissolve 数据

不为 30% / 60% / 100% 分别请求 AI。AI 只给一次磨损分布参考，插件把它转成固定的 WearThreshold（又称 Wear Activation Map）。每个表面点 T∈[0,1] 表示“磨损强度达到多少时该位置开始被激活”。用户的 Wear Amount 只在 Shader 中对 T 做连续阈值。

```text
T(P) = 0.15  → 很早就出现（高凸边 / 高频接触）
T(P) = 0.55  → 中度磨损时出现
T(P) = 0.90  → 只有严重磨损时出现

mask(P, s) = smoothstep(T(P) - feather, T(P) + feather, s)
s = Wear Amount / 100
```

当前推荐的规则化生成是“Propensity → Seed → Topology Propagation → 3D Noise Breakup”。它不需要训练，也不依赖 UV 形状。

1. 生成基础 propensity：AI_Field 与几何先验组合。几何先验 MVP 至少包含 signed convexity；可见视角次数可以作为 exposure 的廉价近似。

1. 从高 propensity 区域选 seed。用局部极值 + 最小拓扑距离做抑制，避免种子过密。

1. 在 Mesh Vertex / Edge Graph 上做 multi-source Dijkstra。边的 traversal cost 由 3D 边长、凸凹性、AI propensity、材质边界共同决定，使磨损更容易沿高凸边和相邻表面扩张。

1. 把 geodesic arrival distance 归一化为基础 WearThreshold，再叠加低频 object-space 3D noise。噪声用 3D Position 采样，不用 UV noise，因此跨 UV seam 连续。

1. 对 WearThreshold 做少量拓扑 Laplacian / edge-aware smoothing，避免离散顶点产生阶梯；材质边界或用户 Barrier 不跨越。

1. Bake 到 AI_WearUV，得到 16-bit WearThreshold。Shader 只根据 Wear Amount + Feather 计算最终 mask。

可直接采用如下第一版公式：

```text
Propensity P = clamp(
      w_ai     * AI_Field
    + w_convex * Convexity
    + w_expose * VisibilityExposure
    - w_cavity * ConcavityPenalty
)

EdgeCost(i,j) =
    Length(i,j)
    * MaterialBoundaryPenalty
    / (epsilon + ((P_i + P_j) / 2))^gamma

D = MultiSourceDijkstra(seed_set, EdgeCost)

T_base = normalize(D)
T = clamp(
      alpha * T_base
    + (1-alpha) * (1-P)
    + noise_amp * Noise3D(Position * noise_scale),
    0, 1
)
```

> **实现原则** Wear Amount 只控制阈值，不重新请求 AI、不重新生成 topology field。改变 0~100 的结果必须满足单调性：30% 的磨损区域应当是 60% 的子集，60% 应当是 100% 的子集。

## 3.4 UV Seam：自动标记用于质检，直接融合用于最终结果

两件事都做，但职责不同：Seam Registry / 可视化用于诊断；Surface / Seam Fusion 用于真正消除接缝。不要把“让美术手动修 seam”当成核心解决方案。

Seam Registry 的检测方式：遍历拓扑 edge 的两侧 face loops；如果同一 3D edge 在两侧对应的 UV 端点不同，则它是 UV discontinuity。不要直接覆盖用户原来的 edge.use_seam；Mode A 建议保存为自定义 edge attribute / 临时选择集，Mode B 可记录自动展开后的 seam。

| 步骤 | 做法 | 用途 |
| --- | --- | --- |
| Detect | 同一拓扑 edge 两侧 UV 坐标不连续 → SeamPair | 建立真实 3D 邻接关系 |
| QA | 沿 SeamPair 采样两侧纹理，计算 mean / p95 abs diff | 自动找出高风险接缝 |
| Visualize | 高差异 seam 边在 viewport 选择 / 高亮 | 方便美术定位问题 |
| Fuse | 按同一 3D 参数 t 对两侧 UV sample 配对，robust average / weighted blend | 强制 scalar field 连续 |
| Diffuse | 从 seam 向岛内若干 texel 衰减融合 | 避免只有一条像素被硬改 |
| Padding | 岛外 dilation | 解决双线性采样 / mipmap bleed |

WearThreshold 低频部分在 mesh topology 上生成，本身已经跨 seam 连续；Seam Fusion 主要修复 AI_Field 的高频投影差异和 raster / sampling 误差。对 scalar mask / WearThreshold 可以融合；未来如果输出 tangent normal，不应直接跨 seam 平均 RGB，而应从连续 height / object-space detail 重新 bake。

## 3.5 多 API 平台的 Blender 接入

AI 作为独立 Provider，不把任何具体模型逻辑写死在 Operator 里。Blender 插件负责输入图、提示词、任务状态、结果文件与缓存；Provider 只负责请求 / 轮询 / 下载。

```text
AIProvider
    get_models()
    validate_config()
    submit(job) -> job_id
    poll(job_id) -> status
    fetch(job_id) -> GeneratedView

OpenAIProvider
GeminiProvider
ComfyUIProvider
CustomHTTPProvider (可选 P1)
```

插件设置（AddonPreferences）建议提供：Provider、API Base URL、API Key、Model ID、默认 Generation Strategy、超时、最大并发、ComfyUI Workflow JSON。Key 字段使用 PASSWORD UI，不写入 .blend、不打印到日志；优先允许从环境变量读取。

ComfyUI 不做节点写死：保存 workflow JSON + Input Mapping + Output Mapping。至少支持 clean image、prompt、seed、可选 depth/normal/control image 的节点映射，以及输出图片节点。

网络请求必须和 bpy 操作隔离：后台线程只做 HTTP / 文件 IO，主线程通过 bpy.app.timers 轮询并更新 UI；worker thread 不能访问 Blender scene/data。所有任务均可 Cancel，失败要保留 provider 原始错误和可重试信息。

| 设置项 | 位置 | 说明 |
| --- | --- | --- |
| Provider / Model | Addon Preferences + Scene Quick Settings | 全局默认 + 当前项目可覆盖 |
| API Key | Addon Preferences | Password UI；环境变量优先；不进 .blend |
| ComfyUI URL | Addon Preferences | 如 http://127.0.0.1:8188 |
| Workflow | Addon Preferences | JSON 路径 + 节点输入输出映射 |
| Generation Strategy | Scene | Image Edit / Control-assisted / Custom Workflow |
| Prompt Preset | Scene | 材质 / 风格 / 最大合理磨损状态 |
| Seed | Scene | 可锁定以便重复结果 |
| Concurrency | Addon Preferences | 限制 API 成本与本地显存占用 |

## 3.6 Blender Shader 与输出

核心输出不是多套 AI 贴图，而是一张稳定的 WearThreshold。最终即时预览使用一个固定 Node Group：读取 WearThreshold（Non-Color）→ 与 Wear Amount 比较 → Smoothstep / ColorRamp Feather → 得到 WearMask。

```text
WearThreshold (0..1, PNG16 / EXR)
        ↓
WearAmount (0..1) - WearThreshold
        ↓
Smoothstep(-Feather, +Feather)
        ↓
WearMask
        ↓
Existing Material / Exposed Material / Scratch Detail
```

MVP 导出：WearThreshold 16-bit + 当前 WearMask。ORM、Height、材质层合成接口保留，但不在本阶段做完整 AI PBR。若需要 Substance，只需把 WearThreshold / WearMask 作为 generator 或 layer mask 输入。

# 4. 代码结构与关键接口

```text
ai_wear/
├─ __init__.py
├─ preferences.py          # Provider / Key / URL / Workflow
├─ properties.py           # Scene / Object settings
├─ operators/
│  ├─ preflight.py
│  ├─ generate_views.py
│  ├─ generate_ai.py
│  ├─ build_surface.py
│  ├─ build_wearthreshold.py
│  ├─ seam_fix.py
│  └─ export.py
├─ uv/
│  ├─ qc.py
│  ├─ unwrap_blender.py
│  ├─ rasterizer.py
│  └─ seam_registry.py
├─ surface/
│  ├─ projection.py
│  ├─ fusion.py
│  ├─ geometry_prior.py
│  └─ wear_growth.py
├─ ai/
│  ├─ base.py
│  ├─ openai_provider.py
│  ├─ gemini_provider.py
│  └─ comfyui_provider.py
├─ render/
│  ├─ view_sampler.py
│  └─ passes.py
├─ shader/
│  └─ wear_nodegroup.py
├─ cache/
│  └─ job_cache.py
└─ ui/
   ├─ panels.py
   └─ progress.py
```

建议的处理缓存目录：`//.ai_wear_cache/<object_uuid>/`。缓存 key 至少包含 mesh/UV hash、camera matrices、render resolution、provider/model、prompt、seed。调整 Wear Amount 不触发 AI、不触发 Surface Field 重算。

# 5. Detailed TODO List

优先级定义：P0 = MVP 必须；P1 = 交付前增强；P2 = Future Work。以下顺序按依赖关系排列。

| ID | 优先级 | 模块 | TODO | Definition of Done |
| --- | --- | --- | --- | --- |
| M0-01 | P0 | 插件骨架 | 注册 Addon、Properties、Panel、Preferences | 启停插件无报错；面板可见 |
| M0-02 | P0 | 设置系统 | Provider / API Key / URL / Model / Workflow / Cache 路径 | 设置持久化且 Key 不进 .blend |
| M0-03 | P0 | 任务状态 | Job state machine: Idle/Render/AI/Build/Bake/Done/Error/Cancel | UI 可显示进度、失败原因、Cancel |
| M0-04 | P0 | 缓存 | 对象 UUID、hash、磁盘缓存目录 | 重复执行可命中 Clean/AI 中间结果 |
| M1-01 | P0 | Mesh Preflight | 活动对象、Mesh 类型、空面、non-manifold、法线、材质、modifier 风险 | 生成结构化检查报告 |
| M1-02 | P0 | UV QC 基础 | UV layer / zero-area / 0~1 / flipped 检查 | 问题可定位到 face / island |
| M1-03 | P0 | UV overlap QC | 低分辨率 raster collision map | 输出 overlap ratio + collision heatmap 数据 |
| M1-04 | P0 | Mode A | 选择现有 UV 作为 Target Wear UV，不修改 UV | clean UV 可直接进入后续流程 |
| M1-05 | P0 | Mode A 风险策略 | overlap 时 warning + 明确结果为 shared wear；允许切 B | 无静默错误 |
| M1-06 | P0 | Mode B UV Layer | 创建 AI_WearUV，保留所有原 UV | 不破坏 BaseColor 等现有 UV |
| M1-07 | P0 | Blender Auto Unwrap | Smart UV Project + Pack + margin | 无 overlap / 无 degenerate 或自动重试 |
| M1-08 | P1 | UV QC 报告 UI | 利用率、岛数量、密度异常、风险等级 | 一键查看 Mode A/B 质检结果 |
| M1-09 | P2 | UVBackend 接口 | 为 xatlas / Rizom 预留 backend protocol | 不影响当前 Blender backend |
| M2-01 | P0 | View Sampler | 自动 6~8 个覆盖视角 + framing | 模型主要表面都有覆盖 |
| M2-02 | P0 | Clean Render | 固定背景 / 光照 / 分辨率 / 相机 | 视图可重复，模型轮廓稳定 |
| M2-03 | P0 | Geometry Pass | Depth / Normal；必要时 Position / TargetUV AOV | 浮点 EXR 精度可用 |
| M2-04 | P0 | AIProvider Base | 统一 submit/poll/fetch/error 接口 | operator 不依赖具体 API |
| M2-05 | P0 | OpenAI Provider | Image edit 请求、模型 ID、超时、下载 | 单 view 完整往返 |
| M2-06 | P0 | Gemini Provider | Image edit 请求、模型 ID、超时、下载 | 单 view 完整往返 |
| M2-07 | P0 | ComfyUI Provider | prompt queue / history / view result | 本地工作流完整往返 |
| M2-08 | P0 | Custom Workflow Mapping | 输入节点 / 输出节点可配置 | 更换 workflow 不改插件代码 |
| M2-09 | P0 | 异步任务 | thread 仅网络；bpy.app.timers 更新 UI | Blender 主界面不因网络请求冻结 |
| M2-10 | P0 | Prompt Template | 固定相机/几何/文字标记，只改表面状态 | 明显结构漂移率可接受 |
| M2-11 | P0 | View Alignment QC | clean/worn edge / silhouette 一致性评分 | 低质量 view 自动降权/重试 |
| M2-12 | P0 | Multi-view Anchor | 选主视角，生成 Anchor Worn View 并立即回投 Partial Surface Field | 后续视角存在统一的 3D 磨损锚点 |
| M2-13 | P0 | Projected Wear Guide | 将当前 Surface Field 投影到待生成 Camera | 已覆盖区域可作为下一次 AI 的一致性条件 |
| M2-14 | P0 | Provider Capability | 记录 reference image / mask / control / seed / multi-turn 能力 | 调度器能自动选择输入策略 |
| M2-15 | P0 | Cross-view Conflict QC | 比较新观测与已有 Surface Field 的重叠区域 | 冲突 view 可降权、局部拒绝或重试 |
| M2-16 | P1 | Contact Sheet Mode | 2×2 Clean Views 单次 edit + 裁切 | 作为快速预览，不影响默认逐视角路径 |
| M3-01 | P0 | UV Rasterizer | Target UV triangle → texel tri_id + barycentric | 工作分辨率下无 hole / ID 错配 |
| M3-02 | P0 | Surface P/N | 由 barycentric 分块恢复 position / normal | 与 Blender 几何误差在容差内 |
| M3-03 | P0 | Screen Mask Extractor | Clean/Worn 差异 → 0..1 mask + confidence | 同视角磨损区域可稳定提取 |
| M3-04 | P0 | Projection | P → camera screen；Depth 做 visibility | 遮挡面不被错误采样 |
| M3-05 | P0 | View Weight | facing / visibility / alignment confidence | 侧视、失败视图自动降权 |
| M3-06 | P0 | Multi-view Fusion | weighted fusion + outlier clipping | 重叠视角结果一致且无单 view 主导 |
| M3-07 | P1 | Coverage Map | 记录每 texel 有效 view count / confidence | 可视化未覆盖区域 |
| M3-08 | P1 | Synthetic Test | 不用 AI 的程序 mask 验证 projection | 几何投影误差可量化 |
| M4-01 | P0 | Convexity | BMesh signed face angle → vertex/edge convex score | 凸边分数稳定，凹边可区分 |
| M4-02 | P0 | Exposure Proxy | 多视角 visibility count 归一化 | 外露面分数高于隐藏凹槽 |
| M4-03 | P0 | Propensity Field | AI + convexity + exposure 组合 | 权重参数可序列化 |
| M4-04 | P0 | Seed Selection | 局部极值 + geodesic min distance | seed 不扎堆，覆盖主要高磨损区 |
| M4-05 | P0 | Topology Graph | vertex/edge adjacency + material boundary penalty | 可用于 multi-source shortest path |
| M4-06 | P0 | Wear Growth | Multi-source Dijkstra 生成 arrival distance | 磨损从 seed 连续向邻接区域扩张 |
| M4-07 | P0 | 3D Noise Breakup | object-space correlated noise 扰动 T | 跨 UV seam 不跳变 |
| M4-08 | P0 | WearThreshold Bake | mesh field → AI_WearUV 16-bit | 0~1 连续、无明显量化 banding |
| M4-09 | P0 | Wear Shader | WearThreshold + Amount + Feather → mask | 0~100 实时预览，无 AI 重算 |
| M4-10 | P0 | Monotonic Test | 比较 30/60/100 mask 包含关系 | 30⊆60⊆100 在容差内成立 |
| M4-11 | P1 | Barrier / Artist Override | 材质边界、手绘保护区影响 traversal | 指定区域不跨越或延迟磨损 |
| M5-01 | P0 | Seam Registry | 拓扑 edge 两侧 UV discontinuity 检测 | 生成 SeamPair 列表 |
| M5-02 | P0 | Seam QA Metric | paired sample mean / p95 abs diff | 能自动报告最差 seam |
| M5-03 | P0 | Seam Fusion | 同 3D t 位置两侧 robust blend | scalar field seam 差异显著下降 |
| M5-04 | P0 | Seam Diffusion | 向岛内 k texel 衰减融合 | 无明显硬线 |
| M5-05 | P0 | Dilation | 岛外 padding | mipmap / bilinear 不露黑边 |
| M5-06 | P1 | Viewport QA | 选择 / 高亮高误差 seam | 可一键定位问题边 |
| M5-07 | P1 | Before/After Report | 输出 seam 指标和对比图 | 满足演示与验收材料 |
| M6-01 | P0 | Export WearThreshold | PNG16 / EXR，Non-Color | Substance / Blender 可正确读取 |
| M6-02 | P0 | Export Current Mask | 按当前 Amount 固化 8/16-bit Mask | 30/60/100 可批量导出 |
| M6-03 | P0 | Error Handling | API/网络/UV/磁盘/渲染失败分类 | 错误信息可操作、可重试 |
| M6-04 | P0 | Case 1 | 喷漆金属硬表面 | 展示边缘磨损、Mode A/B、seam 修复 |
| M6-05 | P0 | Case 2 | 塑料/橡胶 | 验证拓扑生长不是固定金属模板 |
| M6-06 | P0 | Case 3 | 木材/石材或第二类硬表面 | 验证多材质参数与泛化 |
| M6-07 | P0 | Performance Profile | 记录 UV build / AI / fusion / bake 时间 | 给出 1K/2K/4K 指标 |
| M6-08 | P0 | README | 安装、API 配置、Workflow mapping、常见错误 | 新用户可独立跑通 |
| M6-09 | P1 | Preset | Wear 参数 / Prompt / Provider preset 保存加载 | 可复现同一风格 |
| M6-10 | P2 | xatlas Backend | 独立 UV backend | 无需改 Surface Field |
| M6-11 | P2 | PBR Derivation | Mask/Height/ORM 派生 | 不改变核心 Surface/WearThreshold 数据模型 |

# 6. 推荐开发里程碑与验收

| 里程碑 | 完成条件 | 可独立验证方式 |
| --- | --- | --- |
| Milestone 1 — UV 闭环 | M0 + M1 | 拿 3 个模型跑 Mode A/B；输出 UV QC 报告与 AI_WearUV |
| Milestone 2 — 2D→Surface | M2 + M3 | 先用人工 synthetic mask，不依赖 AI，验证投影准确性 |
| Milestone 3 — WearThreshold | M4 | 同一 Surface Field 下 0~100 实时变化；30⊆60⊆100 |
| Milestone 4 — Seam | M5 | 统计 seam p95 差值前后；输出高风险边可视化 |
| Milestone 5 — AI Providers | OpenAI/Gemini/ComfyUI 至少两条链路稳定 | 同一 Clean View 可切 Provider，不改后续算法 |
| Milestone 6 — 交付 | M6 P0 | 3 个真实案例、性能记录、README、错误处理 |

# 7. MVP 验收口径

- Mode A：已有 clean UV 可直接映射；UV overlap 时必须显式告警，不能静默产出错误结果。

- Mode B：原 UV 保持不变，新建 AI_WearUV；自动展开后 overlap / degenerate 达到可 bake 状态。

- Surface Field：遮挡不会把正面 mask 投到背面；多视角同一位置结果可融合。

- WearThreshold：只生成一次 AI 磨损参考；Wear Amount 0~100 通过 Shader 连续控制，且结果单调增长。

- Topology Growth：磨损优先从 AI 高概率区域 / 凸边 seed 向 3D 邻接面扩张，而不是在 UV 图里做 2D dilation。

- UV Seam：自动建立 seam registry；能量化 seam 差异；fusion 后误差明显下降；最后再做 padding。

- API：至少两种 provider + ComfyUI custom workflow；Key 不写入 .blend；网络请求不冻结 Blender。

- 输出：WearThreshold 16-bit + 当前 WearMask；可以直接进入 Blender / Substance 的现有材质流程。

> **最关键的实现顺序** 先把 UV Rasterizer + Synthetic Mask → Surface Field 做准，再接 AI；先把 WearThreshold / Topology Growth 做成确定性算法，再做视觉调参。这样即使 API 或模型效果波动，也不会影响核心 3D 管线的可验证性。

# 附：需求基线

本实现文档基于用户提供的《AI模型磨损纹理生成工具 - 需求》及效果示意图，保留其中 Mode A / Mode B、UV 接缝、边缘磨损强度、多大模型切换、自定义 ComfyUI 工作流等交付目标；具体实现路线按本文件重新组织。

# 8. Demo Mesh 建议

测试资产优先选择 **几何细节足够、边缘/凹槽明显、材质边界丰富、可直接拿到 Blender/FBX/glTF，并且授权清晰** 的模型。对于本工具，原资产已经有磨损贴图不是问题：Demo 前将其 Base Color / Roughness 替换为干净材质即可，原贴图反而可以作为人工视觉参考或结果对比。第一批建议尽量采用 Poly Haven，因为其模型库整体定位就是高质量 3D 资产，并且全部为 CC0，可直接用于 Demo 和商业演示。

| 优先级 | Mesh | 规模 / 特征 | 为什么适合测试 | 免费与来源 |
| --- | --- | --- | --- | --- |
| A | [Metal Jerrycan](https://polyhaven.com/a/metal_jerrycan) | 20K tris，8K 资产；喷漆金属、压筋、把手、凹凸和大轮廓边 | **最适合作为主 Demo**。可以明显验证凸边起磨、沿拓扑扩张、正反面多视角一致性，以及 Mode A/B 的 mask bake | Poly Haven，CC0 |
| A | [Power Box 01](https://polyhaven.com/a/power_box_01) | 21K tris；金属/塑料、开关、电线、复杂硬表面 | 很适合测试复杂遮挡、多材质边界、凹槽不应过度磨损，以及 Camera coverage | Poly Haven，CC0 |
| A | [Camera 01](https://polyhaven.com/a/Camera_01) | 27K tris；金属/皮革/玻璃，镜头和机身细节多 | 适合验证**材质边界 + 小尺度细节 + seam**。可以规定只让金属边框磨损，皮革区域降低 propensity | Poly Haven，CC0 |
| A | [Lantern 01](https://polyhaven.com/a/Lantern_01) | 34K tris；黄铜/玻璃，细框架与大量曲面 | 对多视角覆盖要求高，能暴露薄结构、凹凸边、UV seam 和遮挡判断的问题 | Poly Haven，CC0 |
| B | [Modular Pipes Plastic 01](https://polyhaven.com/a/modular_pipes_plastic_01) | 50K tris；塑料/橡胶/金属，多段管件、阀门、夹具 | 用来证明算法不是“金属掉漆模板”。适合测试 material-specific weights、拓扑传播和多对象/多部件边界 | Poly Haven，CC0 |
| B | [Wheelchair 01](https://polyhaven.com/a/wheelchair_01) | 40K tris；金属管、塑料、软垫、轮子与细杆 | 压力测试 Camera 自动覆盖、细长结构、曲面投影和 visibility；比简单箱体更容易暴露 Surface Field bug | Poly Haven，CC0 |
| B | [Industrial Coffee Table](https://polyhaven.com/a/industrial_coffee_table) | 41K tris；钢 + 木 | 适合做“同一资产不同材质不同磨损规律”的对照案例，且整体结构不至于太复杂 | Poly Haven，CC0 |

推荐实际 Demo 组合不是全用最复杂模型，而是分层：`Metal Jerrycan` 做主流程与 30/60/100 WearThreshold 展示，`Camera 01` 或 `Power Box 01` 做 seam / 多材质 / 遮挡案例，`Modular Pipes Plastic 01` 做非金属材质规则案例，最后再用 `Wheelchair 01` 做 stress test。

如果希望直接从 Blender 内搜资产，也可以用 Blendkit（原 BlenderKit）的免费库，例如 **Industrial green metal box**、**Industrial red/blue metal chest**、**Cantilever Toolbox Rigged and Animated** 等，当前工具分类页明确列有 Free 资产。它们的优势是直接通过 Blender 插件拖入场景；但作为正式 Demo 基准，我仍优先 Poly Haven，因为每个资产页面的三角面数、贴图规格和 CC0 授权信息更清晰，方便写验收报告。

## 9. 外部资料（用于多视角能力与 Demo 资产核验）

- Google Gemini Image Generation: https://ai.google.dev/gemini-api/docs/image-generation
- ComfyUI — Qwen-Image-Edit 2509 Native Support: https://blog.comfy.org/p/wan22-animate-and-qwen-image-edit-2509
- OpenAI GPT-Image-2: https://developers.openai.com/api/docs/models/gpt-image-2
- Poly Haven: https://polyhaven.com/
- Poly Haven License: https://polyhaven.com/license
- Blendkit / BlenderKit free tools category: https://www.blendkit.com/?query=category%3Atool
