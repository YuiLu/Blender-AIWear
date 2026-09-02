# AI Wear Texture — Blender 插件

> 按照本仓库 `Blender_AI_Wear_Texture_Plugin_Implementation_Plan.md`（V0.2）完整实现的 Blender 插件。
> 外部生图模型 API 端点完全可在插件内配置（Addon Preferences）。

把“AI 生成的表面磨损变化”稳定、可重复、可单调控制地落回 3D 模型表面与目标 UV 的工具：

```
Mesh Preflight / UV QC  →  Mode A/B UV  →  多视角 Clean Render  →  AI Provider
   →  Screen Mask  →  3D Surface Field 多视角融合  →  拓扑生长 WearTime
   →  UV Seam Fusion + Padding  →  16-bit WearTime + 实时 Shader 预览  →  导出
```

核心设计（与实现文档一致）：

- **AI 只决定磨损分布与语义；Blender 几何决定像素落在模型表面的哪里。** 插件自己生成固定相机/光照的 Clean View，AI 只在相同构图下编辑表面磨损，因此每个 AI 像素都能通过已知相机 + 模型几何重新定位到目标表面点。
- **WearTime 只生成一次。** 用户拖动 Wear Amount 0~100 只是在 Shader 里对 WearTime 做连续阈值（smoothstep），不重新请求 AI、不重算 Surface Field，且结果单调增长（30⊆60⊆100）。
- **API 端点可配置。** 不把任何具体模型逻辑写死在 Operator 里；换端点只是改 Addon Preferences，不需要改代码。

---

## 目录

1. [安装](#1-安装)
2. [依赖](#2-依赖)
3. [插件配置（Addon Preferences）](#3-插件配置addon-preferences)
4. [使用流程](#4-使用流程)
5. [ComfyUI Workflow Mapping](#5-comfyui-workflow-mapping)
6. [常见错误与排查](#6-常见错误与排查)
7. [架构概览与扩展点](#7-架构概览与扩展点)
8. [预设 / 缓存 / 导出](#8-预设--缓存--导出)

---

## 1. 安装

要求 **Blender 3.6+ / 4.x**（脚本在 4.2+ 的 EEVEE-Next 上自动选择 `BLENDER_EEVEE_NEXT`）。

1. 把整个 `ai_wear/` 目录保留在仓库中。
2. Blender → `Edit > Preferences > Add-ons > Install…`
3. 选择本仓库根目录下的安装入口。若仓库未打包成 zip，可直接把 `ai_wear/` 文件夹放进 Blender 的用户脚本 addons 目录，然后在 Add-ons 列表里勾选 **AI Wear Texture**。
4. 勾选后，`View3D` 侧栏（按 `N` 键）出现 **AI Wear** 标签页。

> 打包成 zip 安装时，请确保 zip 内顶层就是 `ai_wear/`（含 `__init__.py` 的 `bl_info`）。

---

## 2. 依赖

**无需任何 pip 安装。**

| 依赖 | 来源 | 用途 |
| --- | --- | --- |
| `numpy` | Blender 自带 | UV 光栅化、3D Surface Field、多视角融合、WearTime 生长、Z-buffer |
| `urllib`（标准库） | Python 内置 | 全部 HTTP（OpenAI / Gemini / ComfyUI / Custom），无 `requests` 依赖 |
| `bmesh` / `bpy` / `mathutils` | Blender 内置 | 拓扑、UV 操作、渲染、Shader |

> 插件不在 Blender 内加载 PyTorch，也不做训练 / LoRA / 超分 / 完整 PBR —— 这些在实现边界里明确“不做”。
>
> 因此没有 `requirements.txt`：唯一运行期依赖就是 Blender 自带的 `numpy`。如需在 Blender 外做静态检查，`requirements.txt` 形如：
> ```
> # 无第三方运行依赖；numpy 由 Blender 自带
> # 仅开发期静态检查可选：
> # bpy-stubs
> ```

---

## 3. 插件配置（Addon Preferences）

**这是用户强调的重点：外部生图模型 API 端点完全在插件内配置。**

`Edit > Preferences > Add-ons > AI Wear Texture` 展开后：

### 3.1 通用字段（OpenAI / Gemini / Custom 共用）

| 字段 | 说明 |
| --- | --- |
| **Provider** | 全局默认供应商：OpenAI / Gemini / ComfyUI / Custom HTTP。Scene 面板可按项目覆盖。 |
| **API Base URL** | 端点根地址，如 `https://api.openai.com/v1`、`https://generativelanguage.googleapis.com/v1beta`。 |
| **API Key** | `PASSWORD` 子类型：不写入 `.blend`、不打印到日志。 |
| **Key Env Var** | 环境变量名。**填了就用环境变量，忽略上面存的值**（推荐做法：完全不存 key）。 |
| **Model ID** | 模型标识，如 `gpt-image-2`、`gemini-2.5-flash-image`。 |
| Timeout / Max Concurrency / Poll Interval | 网络超时、最大并发、UI 轮询间隔。 |
| Cache Path | 磁盘缓存根；留空用 `<blend>/.ai_wear_cache`。 |

Key 安全策略（实现文档 M0-02 要求）：

- `PASSWORD` UI，不进 `.blend`、不进日志；
- 环境变量优先于存储值；
- Scene 面板的“Base URL / Model 覆盖”是普通字符串（可保存进项目），但 key 永远只在 Preferences。

### 3.2 OpenAI（gpt-image 系）

```
Provider      = OpenAI (GPT-Image)
API Base URL  = https://api.openai.com/v1
Image Edit Path = /images/edits        （默认值）
Model ID      = gpt-image-2
API Key       = sk-...        （或 Key Env Var = OPENAI_API_KEY）
```

走 OpenAI 标准 multipart `/images/edits`：上传 Clean View + prompt + model + size，可选 mask。响应解析 `data[0].b64_json` 或 `data[0].url`。任何兼容 OpenAI images 接口的厂商只需改 Base URL + Model + Key。

### 3.3 Gemini

```
Provider      = Gemini
API Base URL  = https://generativelanguage.googleapis.com/v1beta
Model ID      = gemini-2.5-flash-image
API Key       = AIza...       （或 Key Env Var = GEMINI_API_KEY）
```

请求 `{base}/models/{model}:generateContent`，用 `x-goog-api-key` 头鉴权；Clean View 与参考图作为 `inline_data` parts，`generationConfig.responseModalities=["TEXT","IMAGE"]`。响应从 `candidates[0].content.parts[].inline_data.data` 取图。Gemini 支持 `max_reference_images=3`，因此会把前一视角的 Anchor Worn View 作为参考图传入（见多视角一致性）。

### 3.4 ComfyUI（本地工作流）

见 [§5 ComfyUI Workflow Mapping](#5-comfyui-workflow-mapping)。

### 3.5 Custom HTTP（完全可配置端点）

这是“换任何端点都不改代码”的关键供应商。两种模式：

#### 模式 A：`OpenAI-compatible`

任何讲 OpenAI multipart images 协议的端点，只填三项：

```
Provider          = Custom HTTP
Request Mode      = OpenAI-compatible
API Base URL      = https://your-vendor.example.com/v1
Image Edit Path   = /images/edits     （或 /images/generations）
Model ID          = your-model-id
API Key           = ...
```

#### 模式 B：`Raw JSON template`

用于几乎任意“收图返图”的 HTTP 图片编辑 API。你给一个 JSON body 模板，插件用占位符渲染后 POST，再按 JSON dot-path 解析返回的图。

模板占位符：

| 占位符 | 替换为 |
| --- | --- |
| `{{prompt}}` | 当前 wear prompt（字符串） |
| `{{seed}}` | 当前 seed（整数） |
| `{{image_b64}}` | Clean View 的 base64（无前缀） |
| `{{output_size}}` | 输出边长（整数，如 1024） |
| `{{model}}` | Model ID |

配置项：

| 字段 | 说明 |
| --- | --- |
| **Raw Body Template** | JSON 模板字符串（见下）。 |
| **Image JSON Path** | 响应里图片字段的 dot-path，如 `data.0.b64_json`、`data.0.url`、`result.image`。 |
| **Response Is URL** | 勾选：图片字段是 URL，插件会再下载；否则当 base64 解码。 |
| **Extra Headers** | 可选 JSON 对象字符串，如 `{"X-Client":"ai-wear"}`。 |

示例 1 —— 返回 base64：

```
Request Mode        = Raw JSON template
API Base URL        = https://api.example.com
Image Edit Path     = /v1/wear-edit
Raw Body Template   = {"prompt":"{{prompt}}","image":"{{image_b64}}","seed":{{seed}},"size":{{output_size}}}
Image JSON Path     = data.0.b64_json
Response Is URL     = ☐
```

示例 2 —— 返回 URL：

```
Request Mode        = Raw JSON template
API Base URL        = https://api.example.com
Image Edit Path     = /v1/wear-edit
Raw Body Template   = {"prompt":"{{prompt}}","image":"{{image_b64}}","seed":{{seed}}}
Image JSON Path     = result.url
Response Is URL     = ☑
```

请求头默认带 `Content-Type: application/json` 与（有 key 时）`Authorization: Bearer <key>`；`Extra Headers` 会合并进去。

---

## 4. 使用流程

1. **选中一个 Mesh**（MVP 处理单个活动 Mesh）。
2. 在 `N > AI Wear` 面板：
   - **UV Mode & Preflight**
     - Mode A：用现有 UV 层（overlap 会显式告警，结果允许共享磨损）。
     - Mode B：新建 `AI_WearUV` 层 + Smart UV Project + Pack，不破坏原 UV。点 `Run Preflight / UV QC` 看报告。
   - **Capture**：选相机预设（Auto 6 / Auto 8 / Turntable 4 / Custom）、渲染分辨率、覆盖目标。
   - **AI Provider**：可在 Scene 里按项目覆盖 Provider / Model / Base URL；设 Prompt（材质 / 磨损类型 / 最大磨损状态 / 附加词）、Seed、Lock Seed。
   - **WearTime Parameters**：propensity 权重（w_ai / w_convex / w_expose / w_cavity）、gamma、生长 alpha、3D noise、材质边界 barrier。
   - **UV Seam**：Seam Fusion、Diffuse texels、Padding texels。
3. 点 **Generate Wear Texture**。管线在后台线程跑 HTTP/numpy，主线程经 `bpy.app.timers` 轮询刷新 UI（Blender 不卡死）。**Progress** 面板显示 State/Stage/进度/消息/错误/Seam QA/Coverage。
4. 完成后，插件为活动对象的全部材质槽创建对象私有预览副本，自动注入 `AIWearMask` 节点组，并用显式 `UV Map` 节点读取目标 Wear UV（不会修改共享给其他对象的原材质）。拖动 **Wear Amount** 0~100 与 **Feather** 实时预览，**不触发 AI、不重算 Surface Field**。
   每个离散视角的差值遮罩同时保存为 `views/diff_mask_V0.png`、`diff_mask_V1.png`…；文件为 16-bit RGB 灰度图，可直接查看，文件名也写入 `views.json` 的 `mask` 字段。
   `AIWear_WornTex.png` 的 RGB 保存受限的 clean→worn 残差，Alpha 保存经过对比度整形的真实磨损证据；它不是相机覆盖图。因此 100% 只显示 AI 检出的完整磨损范围，不会把整件模型变白。
   完整管线还会在缓存根目录保存 `AIWear_UVSnapshot.npz`（精确的 per-loop Wear UV）。如果 `.blend` 没有保存 Mode-B 创建的 `AI_WearUV`，Replay 会先校验当前网格拓扑，再从快照自动恢复；旧缓存只有在模型恰好只有一个 UV 层时才执行一次保守复制，并立即升级生成快照。多 UV 情况不会猜测，以免静默映射到错误图集。
5. **Export**：导出 WearTime（PNG16 / EXR / PNG8）、当前 Mask（按当前 Amount 固化）、或一键导出 30/60/100（验证单调性）。

> 随时可点 **Cancel** 中断；失败时 Progress 面板会显示分类错误（API / NETWORK / CONFIG / CANCEL）与原始信息。

---

## 5. ComfyUI Workflow Mapping

ComfyUI 不写死任何节点。你提供一份 **API 格式** 的 workflow JSON（在 ComfyUI 界面里 `Save (API Format)` 得到），然后在 Preferences 里映射节点 id：

| Preference 字段 | 对应 ComfyUI 节点 |
| --- | --- |
| `Clean Image Node` | `LoadImage` 节点 id —— 接收 Clean View |
| `Prompt Node` | `CLIPTextEncode`（正向）节点 id —— 接收 prompt |
| `Seed Node` | `KSampler` / `KSamplerAdvanced` / `SamplerCustom` 节点 id —— 接收 seed |
| `Output Node` | `SaveImage` / `PreviewImage` 节点 id —— 取结果图 |

工作流示例见 [`examples/comfyui_workflow_example.json`](examples/comfyui_workflow_example.json)（简化 API 格式，含 LoadImage / CLIPTextEncode / KSampler / SaveImage 四个节点，可直接映射）。

**自动检测**：如果某些映射留空，插件会按 `class_type` 自动猜测第一个 `LoadImage` / `CLIPTextEncode` / `KSampler`；填了就用你填的。

**流程**：上传 Clean View（`/upload/image`）→ 注入 prompt / seed / 图 → `/prompt` 排队 → 轮询 `/history/{prompt_id}` 直到完成 → `/view` 下载 `Output Node` 的图。整个过程不碰 bpy，只在 worker 线程跑 HTTP。

> Depth / Normal / Anchor / Projected Guide 等额外参考图，可通过 `node_mapping`（key=label, value=node_id）接到任意 `LoadImage` 节点；当前版本由管线按 provider 能力决定是否生成这些辅助图。

---

## 6. 常见错误与排查

| 现象 | 原因 / 解决 |
| --- | --- |
| `Provider config: API Base URL is empty` | Preferences 里 Base URL 没填。 |
| `Provider config: API Key is empty` | OpenAI/Gemini 需要 key；或填 `Key Env Var` 从环境变量读。 |
| `Raw body template is not valid JSON` | Custom HTTP 的 Raw JSON 模板渲染后不是合法 JSON。检查占位符是否破坏了引号/逗号（`{{seed}}`/`{{output_size}}` 是裸数字，其余需在引号内）。 |
| `Key 'xxx' not found in path ...` | Custom HTTP 的 `Image JSON Path` 与实际响应结构不符。用浏览器/curl 看一次原始响应，再调 dot-path（如 `data.0.url`）。 |
| `Image edit failed (401/403)` | key 无效或无权限。 |
| `Image edit failed (4xx)` + 长文本 | 多半是端点路径或 model id 不对；错误信息里有端点返回的前 500 字。 |
| `ComfyUI /prompt failed` / `did not return prompt_id` | `ComfyUI URL` 不对，或 workflow JSON 不是 API 格式。 |
| `ComfyUI run produced no image outputs` | `Output Node` id 填错，或该节点没有 `images` 输出。 |
| `ComfyUI job timed out` | 生成太久；调大 Timeout，或检查显存/OOM。 |
| `No UV on the object after setup` | Mode A 选了不存在的 UV 层；改 Mode B 或选对层。 |
| Mode B 报 `overlap` 未达标 | 自动重试一次后仍不达标；检查模型是否有极端 island，或手动在 UV Editor 里 Pack。 |
| 拖 Wear Amount 没反应 | 确认生成或 Replay 已完成，并在 Material Preview/Rendered 模式查看；材质中应有 `AIWear_UVMap`、`AIWear_MaskGroup` 和 `AIWear_OverlayMix`。 |
| Blender 卡住 / 无响应 | 不应发生（网络全在 worker 线程）。若卡住多半是渲染步骤；检查 `Render Resolution` 与 `Work Res` 是否过大。 |
| `Non-JSON response: ...` | 端点返回了 HTML（通常是 404/反代错误页）。核对 Base URL + Path。 |

错误分类（Progress 面板与 Job 一致）：

- `API` —— 端点返回非 2xx 或响应结构不符；
- `NETWORK` —— 连接失败 / 超时 / DNS；
- `CONFIG` —— Preferences 配置缺失；
- `CANCEL` —— 用户取消。

---

## 7. 架构概览与扩展点

### 线程模型（实现文档 M2-09）

```
worker 线程：只做 HTTP + numpy，绝不碰 bpy scene/data
   │
   ▼  bridge.run(fn)  入队
MainThreadBridge ── bpy.app.timers ──▶ 主线程执行 fn（渲染 / UV ops / 图像 IO / Shader）
   │
   ▼  job.state/stage/progress
UI（Progress 面板）读 JobRegistry，timer 周期 tag_redraw
```

- 所有 bpy 触碰（render、Smart UV Project、image save/load、bmesh、node 设置）都经 `bridge.run()` 在主线程跑。
- worker 只拿 `PrefsSnap`（纯 Python 对象快照）做 HTTP，不会读 bpy RNA。
- 任何阶段均可 Cancel；失败保留 provider 原始错误与 traceback 尾部。

### 代码结构

```
ai_wear/
├─ __init__.py              # bl_info + register/unregister
├─ preferences.py           # Provider/Key/URL/Model/Workflow/Cache（可配置端点）
├─ properties.py            # Scene/Object settings + prompt builder + slider 回调
├─ utils.py                 # numpy 图像 IO（PNG8/16/EXR）、base64、resize
├─ cache/job_cache.py       # Job 状态机 + 磁盘缓存
├─ operators/
│  ├─ pipeline.py           # 编排核心：PrefsSnap/MainThreadBridge/snapshot/_run_pipeline/_tick
│  └─ runner.py             # Operators: run/cancel/preflight/export_*/preset_*
├─ uv/
│  ├─ qc.py                 # UV QC（overlap/degenerate/flipped/utilization）
│  ├─ unwrap_blender.py     # Mode A / Mode B（Smart UV Project + Pack + 重试）
│  ├─ rasterizer.py         # UVField：tri_id + barycentric → P/N（无需 ray_cast）
│  └─ seam_registry.py      # SeamPair 检测 / QA / 融合 / 扩散 / dilation
├─ render/
│  ├─ view_sampler.py       # 自动多视角相机 + framing
│  └─ passes.py             # Eevee Clean/Depth，中性世界 + 太阳
├─ ai/
│  ├─ base.py               # AIProvider/GenRequest/GenResult/ProviderCapabilities/注册表
│  ├─ http_util.py          # urllib 版 post_json/post_multipart/download/get_bytes
│  ├─ _openai_compat.py     # OpenAI multipart 共享逻辑
│  ├─ openai_provider.py    # OpenAI
│  ├─ gemini_provider.py    # Gemini（:generateContent + inline_data）
│  ├─ comfyui_provider.py   # upload/prompt/history/view + 节点自动检测
│  └─ custom_http_provider.py # OPENAI_COMPAT / RAW_JSON（端点完全可配置）
├─ surface/
│  ├─ projection.py         # 相机投影 + 软件 Z-buffer + Screen Mask 提取 + 累积
│  ├─ fusion.py             # 多视角加权均值 + outlier clipping + coverage
│  ├─ geometry_prior.py     # signed convexity / exposure / propensity
│  └─ wear_growth.py        # 顶点→UV、拓扑图、seed、multi-source Dijkstra、3D value noise
├─ shader/wear_nodegroup.py # 显式 Wear UV + 对象私有材质副本 + 颜色残差叠加 + 实时回调
└─ ui/
   ├─ panels.py             # 主面板 + 子面板（UV/Capture/AI/WearTime/Seam/Export/Presets）
   └─ progress.py           # Progress 面板
```

### 关键实现取舍

- **软件 Z-buffer（numpy 屏幕光栅化）** 而非读 Blender Depth Pass：保证投影/可见性约定自洽，避免脆弱的多层 EXR 读取。
- **手动透视投影**（`camera.matrix_world.inverted()` + lens/sensor）匹配 Blender 方形渲染约定；方形 + sensor_fit AUTO 下与 Blender 渲染精确一致。
- **UV 光栅化存 tri_id + barycentric**，由重心坐标直接恢复 3D P/N，**不做 per-texel ray_cast**（NumPy 向量化）。
- **多视角一致性放回 3D**：Anchor-conditioned 顺序生成，前一视角 Worn View 作为后续视角参考图（provider 能力允许时），最终由 3D Surface Field 融合 + outlier clipping 拒绝冲突观测。
- **WearTime 在 mesh topology 上生长**（multi-source Dijkstra + object-space 3D noise），因此跨 UV seam 连续；Seam Fusion 只修高频投影差异。
- **Wear Amount 只阈值**：`mask = smoothstep(T - feather, T + feather, amount) × AI wear evidence`，0~100 单调，无 AI 重算。AI evidence 由重投影磨损遮罩生成，而不是相机覆盖率；因此 100% 表示“显示完整的 AI 磨损范围”，不会把整个模型漂白。

### 扩展点

- **新供应商**：实现 `AIProvider` 子类（`capabilities / validate_config / generate`），在 `ai/base.py::_register_builtin` 注册；或在 Preferences 里用 Custom HTTP 的 RAW_JSON 模式，零代码接入。
- **新 UV backend**：`uv/unwrap_blender.py` 可替换为 xatlas / RizomUV（实现文档 M1-09 预留接口，不影响 Surface Field）。
- **派生贴图**（ORM/Height）：基于已生成的 WearTime / WearMask 派生，不改核心数据模型（M6-11，P2）。
- **Contact Sheet 批量模式**：2×2 Clean Views 一次 edit 再裁切（M2-16，P1，预览/能力测试用）。

---

## 8. 预设 / 缓存 / 导出

### 预设（M6-09）

`N > AI Wear > Presets` 面板可保存/加载/删除 wear 参数预设（材质、磨损类型、最大磨损状态、propensity 权重、alpha、noise）。预设存 Scene，随 `.blend` 保存，便于复现同一风格。

### 缓存

默认 `<blend>/.ai_wear_cache/<object_uuid>/`，下含 `views/`（每视角 clean/worn PNG）与 `WearTime.png`。`object_uuid` 由 mesh/UV hash 决定，相同几何 + UV + 相机 + 渲染分辨率 + provider/model/prompt/seed 可命中中间结果。调整 Wear Amount 不触发 AI、不重算 Surface Field。

> 在 Preferences > Cache Path 可覆盖缓存根目录。

### 导出

| 操作 | 输出 |
| --- | --- |
| `Export WearTime` | 16-bit PNG（Non-Color）/ EXR 32-bit / PNG 8-bit，可直接进 Substance/Blender 材质。 |
| `Export Current Mask` | 按当前 Wear Amount + Feather 固化的 8-bit mask。 |
| `Export 30/60/100 Masks` | 一次导出 `wearmask_30.png` / `60` / `100`，验证 30⊆60⊆100 单调性。 |

生产材质只需把 `AIWearMask` 节点组的 **Mask** 输出接到自己的 wear 层 / mix shader / Substance layer mask 即可。

---

## 参考

- 实现文档：`Blender_AI_Wear_Texture_Plugin_Implementation_Plan.md`（V0.2）
- 需求基线：`AI模型磨损纹理生成工具需求.md`
- ComfyUI 工作流示例：`examples/comfyui_workflow_example.json`
- 外部资料：OpenAI GPT-Image-2、Google Gemini Image Generation、ComfyUI（见实现文档 §9）
