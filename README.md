# Blender AI Wear

AI Wear 是一个 Blender 插件：它把**图像生成模型画出来的磨损**重新贴回到 3D 模型的表面上，
而不是给你一张带光照的整图。你选中一个模型、点一下按钮，它就会自动拍照、调用 AI、把结果
算成能在模型上直接使用的磨损贴图。

跑完后你会得到四个东西：

| 文件 | 是什么 |
| --- | --- |
| `AIWear_Mask.png` | **AI 磨损 mask** —— AI 认为哪些表面位置出现了磨损 |
| `WearTime.png` | **磨损时刻图** —— 每个表面位置「从什么时候开始磨损」的 0~1 数值（拖滑块就用它） |
| `AIWear_WornTex.png` | **磨损颜色残差** —— AI 生成的局部颜色变化，而不是整张渲染图 |
| Shader 预览 | 自动接进原材质的预览，拖滑块实时看磨损程度 |

一句话分工：**AI 决定磨损长什么样；Blender 的相机、深度、法线、拓扑和 UV 决定磨损出现在哪、怎么扩散。**

```text
Mesh / UV
  ↓
多视角 Clean Render ──→ AI Edit ──→ Clean/Worn 差分 Mask
  ↓ 相机矩阵 + Z-buffer                    ↓
表面重投影 ←───────────────────────────────┘
  ↓
多视角融合成 AI mask + 几何先验
  ↓
网格拓扑传播得到 WearTime
  ↓
UV Seam 融合 + Padding
  ↓
WearTime / WornTex / Shader
```

---

## 快速开始

### 安装

要求 Blender 3.6+（推荐 5.x）。插件只用 Blender 自带的 `bpy`、`bmesh`、`mathutils`、
`numpy` 和 Python 标准库，不需要额外 pip 装包。

1. 把 `ai_wear/` 目录放进 Blender 用户目录的 `scripts/addons/`，或安装打包好的 zip。
2. `Edit > Preferences > Add-ons` 搜索 "AI Wear Texture"，勾选启用。

### 配置 Provider

在同一个 Add-on 偏好设置里，填 `Endpoint & Auth` 这一栏：

- `Provider`：OpenAI / Gemini / ComfyUI / Custom HTTP，四选一。
- `API Base URL`、`API Key`、`Model ID`。Key 是密码框，只存在本地偏好里，**不会写进 `.blend`**。
- 选 Custom HTTP 时，再填请求路径、请求模式、body 模板、图片字段路径等。

### 跑一次

1. 选中一个 Mesh 物体。
2. 打开侧栏 `N > AI Wear`。
3. 选 `UV Mode`：A = 用已有 UV；B = 自动新建 `AI_WearUV` 并展开（推荐先用 B）。
4. 点 `Run Preflight` 检查 UV 是否 OK。
5. 点 `Generate Wear Texture`。管线会自动拍照 → 调 AI → 重投影 → 融合 → 生成 WearTime → 接进材质。
6. 完成后直接拖 `Wear Amount` / `Feather`，实时看磨损程度（**不会再次调用 AI**）。

### 只想调下游参数时

如果只改几何先验、拓扑生长、seam、padding 或生长参数，用 **`Replay Downstream`**：它复用
上次缓存的 clean/worn 图和相机矩阵，不重新渲染、不调 AI，省 API 预算。改相机数量、
`View Context` 或 Prompt 则必须重跑完整管线。

---

## 使用指南（面板每个分区做什么）

### UV Mode & Preflight

- **`UV Mode`**：`Mode A` 用已有 UV（重叠 UV 会共享磨损）；`Mode B` 新建一个独立的
  `AI_WearUV` 并自动展开，不影响你原来贴图用的 UV。
- **`Work Res`**：内部光栅化分辨率，默认 1024（越大越细，也越慢）。
- **`Texture Size`**：输出贴图分辨率，默认 2048。
- **`Run Preflight`**（运行前的预检）：检查 UV 是否存在、是否落在 [0,1]、退化三角形、翻转、重叠。Mode B
  会保存精确的 per-loop UV 快照；Replay 时如果发现 UV 丢失，会按顶点/loop/polygon 数量
  校验后原样恢复，不会瞎猜。

### Capture（采集）

- **`Cameras`**：`Auto 6`（4 环绕 + 顶 + 底）、`Auto 8`（8 个正方体顶点斜向）、
  `Turntable 4`（4 水平）、`Counted Auto`（精确 `Cam Count` 个 Fibonacci 球面均匀视角）、
  `Custom`（用场景里名为 `AIWearCam_*` 的相机）。
- **`Cam Count`**：`Counted Auto` 时的相机数量。
- **`Render Res`**：每张 clean 渲染的分辨率，默认 1024。
- **`View Context`**：多视角是否顺序条件化 —— `Independent`（每张只看自己的 clean）、
  `First-view Anchor`（后续都参考第一张 worn）、`Previous View`（第 i 张参考第 i-1 张）。

### Prompt

- **`Material` / `Wear Type` / `Max Wear State`**：拼成主 prompt；`Max Wear State` 越重，
  描述越强。
- **`Extra Prompt`**：原样追加到自动 prompt 末尾（实验 7 用它做「有/无」对比）。
- **`Seed` / `Lock Seed`**：锁定种子保证可复现。

### WearTime Parameters（生长与噪声）

- **`w AI / w Convex / w Expose / w Cavity`**：四个倾向权重，具体含义见下文「工作原理·几何先验」。
- **`gamma` / `alpha` / `noise amp` / `noise scale`**：控制磨损生长形态（见「拓扑传播」）。
- **`Material Boundary Barrier` / `Mat Boundary`**：让磨损不容易跨过材质边界串色。

### UV Seam

- **`Seam Fusion` / `Seam Diffuse`**：把同一 3D 边两侧的 UV 数值拉平，消除接缝。
- **`Island Padding` / `Padding`**：向 UV 岛外膨胀填充，防止 mipmap / 双线性采样漏黑边。

### Experiments / Ablation（实验开关）

- **`AI Mask`**：关掉 = `w AI` 置 0，只看几何先验。
- **`Geometry Prior`**：关掉 = 凸起/暴露/凹陷三项几何权重全置 0。
- **`Topology Growth`**：关掉 = 不跑 Dijkstra，直接由 `1-P` 生成 WearTime。
- **`Save Experiment Snapshot` / `Experiment Label`**：把这次运行的配置、耗时、前后对比贴图
  存进独立目录，方便做对照实验。

### Export

导出 16-bit WearTime（PNG/EXR）、当前 mask（8-bit PNG）、或 30/60/100 三档 mask
（用于演示「磨损量单调递增」）。

---

## 工作原理（从网格到磨损贴图）

### 1. 从 UV 反查 3D（UV 栅格化）

要把屏幕上的磨损搬回模型表面，得知道「输出贴图的每个 texel 对应模型上哪一点、朝向哪」。每个点
的世界坐标和法线**本来就在网格上**，渲染器也能输出 Position / Normal pass，UV 编辑器展示的也
确实是「3D ↔ UV」的对应关系——所以这里没有任何「新算出来的概念」，只有一次普通的**栅格化**。

真正卡住的是两件事：**方向反了**，而且**要能向量化**。

- 渲染器的 Position / Normal pass 是「**屏幕像素 → 3D**」（某台相机看到的位置），而重投影需要
  的是「**UV texel → 3D**」——这个输出贴图像素对应模型上哪一点，才能去各相机画面里采样。这是
  反方向，渲染 pass 给不了。
- UV 编辑器里的映射是给**人看**的交互展示，不是一张程序能按 texel 直接索引的数值表。

所以做法是把 UV 三角形一次性光栅化到一个 `Work Res × Work Res` 的网格，每个 texel 只记两样
东西——落在哪个三角形、重心坐标是多少：

```text
texel(u,v) → (triangle_id, b0, b1, b2)
```

世界坐标和法线**不单独存**，用的时候按重心坐标现算：

```text
P = b0·V0 + b1·V1 + b2·V2
N = normalize(b0·N0 + b1·N1 + b2·N2)
```

这是向量化的 gather + 线性组合，不是对 `1024×1024 ≈ 100 万`个 texel 逐个 `ray_cast`（每调用
一次都走一遍 Python↔C，慢到没法用）。唯一会「进内存」的中间量是 P 和 N 两张图：因为 6~8 台
相机都要把每个 texel 投到自己画面里，把 P/N 先算好存住（类似 GBuffer 的 position/normal
buffer），比每台相机都重算一遍便宜。所以它既不是一个新的「场」，也不是一张「查找表」，就是
**栅格化 + 按需重建**，一个实现细节而已。

### 2. 多视角拍照，让 AI 在干净的图上画磨损

插件在世界空间包围盒上算中心和半径，用 50mm 透视相机自动 framing，从多个角度渲染出
`clean_V0.png, clean_V1.png, ...`（构图、材质、背景、光照都固定）。然后把这些干净图发给 AI，
让它「在原图相同的位置加磨损」，得到对应的 `worn_V0.png, worn_V1.png, ...`。

注意 `Auto 8` 和 `Counted Auto + Cam Count=8` **不是**同一组相机：前者是严格中心对称的
`normalize(±1,±1,±1)`（适合盒状道具），后者是黄金角/等面积的 Fibonacci 球面采样，8 个点
不会正好落在正方体顶点上。

### 3. 提取磨损 mask（曝光匹配 + 差分归一化）

AI 经常让整张图略微变亮、变暗或偏色。如果直接 `|worn - clean|`，这种全局色调变化会被误认
成「全模型都在磨损」。所以先做**逐通道曝光匹配**，把 worn 拉回 clean 的亮度：

```text
scale_c = mean(clean_c) / mean(worn_c)
worn_matched_c = clamp(worn_c · scale_c)
```

然后同时看亮度差和最大通道差，取两者较大的那个，再按 P95 归一化到 0~1：

```text
d = max(|luma(worn_matched) - luma(clean)|,  max_c |worn_matched_c - clean_c|)
M = clamp(d / max(P95(d > 0.02), 0.05))
```

这样得到**屏幕空间**的磨损 mask `M_i`，回答的是「AI 在这张图的哪些像素做了局部修改」。
注意两点：第一，这一步产出的就是标准 mask，**永远都会执行**——面板上的 `AI Mask` 开关
不是关掉这个过程，而是后面决定「这张 mask 要不要参与倾向计算」（见第 6 节）。第二，`M_i`
回答「哪里被改了」，而不是「这里有没有被相机看见」；两者必须分开，否则相机覆盖率接近 100%
时整个模型都会变白。

### 4. 把屏幕 mask 重投影回表面

对栅格化出的每个表面点 `P`：

1. 用相机逆世界矩阵把它变换到相机空间；
2. 用 lens / sensor width 做透视投影，得到屏幕坐标 `(x,y)` 和深度 `z`；
3. 用软件 Z-buffer 判断它是否被遮挡（`z <= Z_i(x,y) + eps`）；
4. 被遮挡的点，这个视角不能给它写 mask。

可见还不够——掠射角（几乎侧对相机的面）在屏幕上的像素被压扁，重投影误差大。所以再乘一个
朝向权重：

```text
view_dir = normalize(camera_position - P)
facing   = clamp(dot(N, view_dir), 0, 1)^gamma
w        = visible · facing
```

`gamma` 越大越信任正对相机的视角。mask 和颜色残差用**同一套**坐标、可见性和权重，保证它们
不会各自落到不同 texel。

### 5. 多视角融合成 AI mask

每个 UV texel 把所有可见相机看到的磨损加权平均：

```text
F_ai = clamp(sum(w · M) / sum(w))
```

再做一个 8 邻域局部均值抑制：某个 texel 和邻域差得离谱时拉回中点，压掉单视角的极端亮点。
最终结果 `F_ai` 就是**AI 磨损 mask**，存成 `AIWear_Mask.png`。

**大白话**：一个 texel 被 6 个相机看到，其中 4 个都说「这里有磨损」、2 个说没有，那它就是
「偏有磨损」。这张图是后面一切生长的「种子」。

### 6. 几何先验（凸起 / 暴露 / 凹陷）

AI 只看得到「画面里哪里变了」，不知道资产的真实拓扑，也容易把平面上的光照变化当成磨损。
几何先验把磨损倾向拉回物理上合理的位置。它由三项组成：

- **Signed Convexity（凸起/凹陷）**：对每条连接两个面的边，取两面法线夹角和朝向符号。
  凸棱、外角 → 正（容易碰撞磨损）；凹槽、内角 → 负（受保护）。
- **Exposure（暴露度）**：某顶点被 `n` 个视角看到、总视角 `N`，`E = n/N`。外露表面高，封闭内腔低。
- **Cavity（凹陷）**：就是 `max(-convexity, 0)`，用来「扣分」保护凹槽。

然后把 AI mask 和几何先验**按权重相加**，得到最终的磨损倾向 `P`：

```text
P = clamp( w_ai·F_ai + w_convex·C+ + w_expose·E - w_cavity·C-, 0, 1 )
```

**这里最容易混淆的一点，说清楚**：几何先验**不会修改** `F_ai`（AI mask）。它和 AI mask 是
`P` 这条公式里的两个**平行输入**，各自按权重贡献。`AI Mask` 开关只是把 `w_ai` 置 0
（等价于把 `w AI` 滑块拉到 0），`Geometry Prior` 开关只是把后三项权重置 0——它们是对称的
消融开关，用来回答「去掉 AI 信号 / 去掉几何，分别能得到什么」。

| 参数 | 调大之后 |
| --- | --- |
| `w AI` | 更服从 AI 画出来的划痕/磕碰分布 |
| `w Convex` | 更集中在凸棱和硬边 |
| `w Expose` | 更偏向外露表面，少在隐藏面磨损 |
| `w Cavity` | 更强地保护凹槽和内角 |

### 7. 拓扑传播：多源 Dijkstra 生成 WearTime

`P` 只表示「哪里适合磨损」，还不是一个能随 `Wear Amount` 单调展开的「先后顺序」。WearTime
把它变成 `T ∈ [0,1]`：值越小越早磨损。

**挑种子**：候选顶点要 `P >= 0.18` 且不低于所有一环邻居（局部极大值），再按 `P` 从高到低
做空间抑制（新种子必须离已保留种子超过包围盒半径的 10%），避免同一条棱上堆几十个几乎相同
的起点。

**传播**：把网格顶点当图节点、真实 mesh 边当边，在**这个 3D 网格图上**跑多源 Dijkstra——
所有种子以距离 0 同时入队，每个顶点得到「从最近的高倾向种子走到这里」的最小累计代价：

```text
P_mean    = max((P_i + P_j)/2, 1e-3)
cost(i,j) = edge_length / (1e-4 + P_mean^gamma)
```

高倾向区域代价低、磨损优先沿凸边和 AI mask 走；低倾向区域代价高。如果边两侧材质不同且开了
`Material Boundary Barrier`，代价再乘 `Mat Boundary`。最后：

```text
T_base = D / max(D)
T = clamp( alpha·T_base + (1-alpha)·(1-P) + noise_amp·(2·noise-1), 0, 1 )
```

再做两轮一环平滑。

**大白话**：这就是「磨损从真实 mask 区沿着表面拓扑往外长」。因为 Dijkstra 是在 3D 网格图上
跑的，所以哪怕 UV 岛被切开、贴图里不连续，同一块 mesh 上的磨损在 3D 上依然是连续的。
关掉 `Topology Growth` 时跳过这一步，直接 `1-P` 当 WearTime（可运行的消融分支）。

| 参数 | 小值 | 大值 |
| --- | --- | --- |
| `alpha` | 贴着局部 `P`，碎、直接 | 强调从种子沿拓扑扩张，连续生长带 |
| `noise amp` | 边界平滑、规则 | 边界破碎、斑驳 |
| `noise scale` | 大块、低频 | 小块、细碎 |

### 8. WearTime 回到 UV + seam 融合

顶点 WearTime 再按第 1 节的三角形/重心坐标插值回 UV：

```text
T(u,v) = b0·T(V0) + b1·T(V1) + b2·T(V2)
```

所以贴图只负责「存储 + 给 shader 采样」，拓扑连续性不依赖两个 UV 岛在贴图里是否相邻。

不过，虽然 WearTime 主体已经在 3D 上连续，不同 UV 岛在贴图里的 texel 中心、插值和多视角
高频颜色残差仍可能不一致，双线性采样/mipmap 会把这点差异放大成接缝。所以：

- **Seam Fusion**：遍历每条恰好连接两个面、且两端 UV 坐标不同的拓扑边（即 UV seam），把
  同一 3D 位置两侧的数值采样后取平均再写回两侧，消除接缝。
- **Island Padding**：从 UV 岛有效 texel 向空白区做多轮四邻域膨胀，防止岛边漏色。

两者已解耦，可分别开关：`Seam Fusion` 管「让同一条 3D 边两侧数值相等」，`Padding` 管
「扩展岛边颜色」，不是一回事。

### 9. Shader：把磨损叠回原材质

插件**不会**把 AI 的 worn 渲染图直接当 Base Color——那会把视图光照烘进材质，再被 Blender
灯光照第二次。它保存的是曝光匹配后的**颜色差**（残差）：

```text
delta        = worn_matched - clean
encoded_rgb  = 0.5 + 0.5·delta      # 写入贴图（0.5 = 无变化）
```

写入前把 delta 限制在 `[-0.06, +0.28]`，抑制 AI 的全局重打光。WornTex 的 alpha 来自
`F_ai` 的取值窗口 `[0.06, 0.55]`（不是相机 coverage）。

Shader 里，一个可复用的节点组 `AIWearMask` 算出磨损门：

```text
gate = smoothstep(T - feather, T + feather, wear_amount)
mask = gate · worn_tex.alpha
worn_color = original_base_color + 2·(worn_tex.rgb - 0.5)
final = mix(original_base_color, worn_color, mask)
```

然后把 `final` 注入原材质 Principled BSDF 的 Base Color。所以：

- `Wear Amount` 30/60/100 只是同一张 WearTime 的阈值，天然单调递增；
- 100% 只打开所有「有 AI mask」的区域，不会把相机覆盖区全部变白；
- 原材质的 Normal / Roughness / Metallic 等连接保留，只有 Base Color 被改；
- 共享材质会先复制成对象私有预览材质，不影响用同一材质源的其他物体；
- 换模型也安全（每次独立拷贝注入）；换 shader 时，有 Principled BSDF 就正常注入，没有
  则退化成中灰底 + 磨损的独立材质（会丢掉原自定义 shader 的外观）。

---

## 实验

定性对照实验的计划（相机数量、视角上下文、几何/拓扑开关、seam、磨损量、额外提示词）见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。每个实验臂在 `Presets` 面板里都有对应的内置预设
（`cams_01`、`geometry_off`、`amount_30`、`extra_on` 等），Load 一下就能切过去。

---

## 故障排查

- **`API key is empty — ... HTTP 401`**：Provider 不是 ComfyUI 但没有 Key。去
  `Edit > Preferences > Add-ons > AI Wear Texture` 的 `API Key` 里填上。
- **`UV rasterization covered 0 texels`**：UV 是空的或全在 [0,1] 之外。改用 Mode B 重新
  展开；若已用 Mode B，检查网格是否非流形/有退化面（Smart UV Project 会跳过它们）。
- **`A pipeline is already running`**：上一次还没结束，先点 `Cancel`。
- **`No WearTime texture found`**（导出时）：先跑一次 `Generate Wear Texture`。
- **改了下游参数但没变化**：确认用的是 `Replay Downstream`（不是只拖滑块——滑块只改
  `Wear Amount` / `Feather`，其余参数要重跑/Replay 才生效）。

---

## 目录结构

```text
ai_wear/render       自动相机与 clean render
ai_wear/ai           Provider、ComfyUI graph 执行
ai_wear/uv           UV QC、UVField、seam registry
ai_wear/surface      差分、重投影、融合、几何先验、Dijkstra
ai_wear/shader       WearTime gate 与原材质叠加
ai_wear/operators    管线、缓存、Replay、实验快照
ai_wear/presets.py   内置实验预设
```

默认缓存目录是 `<blend>/.ai_wear_cache/<object>/`。网络和 NumPy 在 worker 线程跑，渲染/UV/
材质等 Blender 数据操作通过 `MainThreadBridge` 回到主线程，由 `bpy.app.timers` 驱动，网络
等待不会锁死 UI。

模块拆解与验收映射见 `Blender_AI_Wear_Texture_Plugin_Implementation_Plan.md`。
