# Blender AI Wear

Blender AI Wear 把图像生成模型给出的磨损语义重新定位到网格表面，输出：

- `M_Wear.png`：AI 在哪些表面区域观察到了磨损；
- `WearThreshold.png`：每个表面位置从什么时候开始出现磨损；
- `AIWear_WornTex.png`：AI 生成的局部颜色变化，而不是整张带光照渲染图；
- 自动接入原材质的 Shader 预览。

核心分工是：**AI 决定磨损外观，Blender 的相机、深度、法线、拓扑与 UV 决定磨损位置和生长方式。**

```text
Mesh / UV
  ↓
多视角 Clean Render ──→ AI Edit ──→ Clean/Worn 差分 Mask
  ↓ 相机矩阵 + Z-buffer                    ↓
表面重投影 ←───────────────────────────────┘
  ↓
多视角融合 AI mask + 几何先验
  ↓
网格拓扑传播得到 WearThreshold
  ↓
UV Seam Fusion + Padding
  ↓
WearThreshold / WornTex / Shader
```

## 1. 安装与基本使用

要求 Blender 3.6+；运行期只使用 Blender 自带的 `bpy`、`bmesh`、`mathutils`、`numpy`
和 Python 标准库，不需要额外安装 pip 包。

1. 把 `ai_wear/` 放入 Blender 用户 `scripts/addons/`，或者安装本仓库生成的 zip。
2. 在 `Edit > Preferences > Add-ons > AI Wear Texture` 中配置 Provider。
3. 选中一个 Mesh，打开 `N > AI Wear`。
4. 选择 UV Mode、相机、Prompt 和 WearThreshold 参数。
5. 先运行 Preflight，再运行 `Generate Wear Texture`。
6. 完成后直接拖动 `Wear Amount` / `Feather`；不会重新请求 AI。
7. 只修改几何先验、拓扑、seam 或生长参数时，用 `Replay Downstream` 复用上次 AI 图。

Provider 支持 OpenAI、Gemini、Qwen Image 3.0、ComfyUI，以及 OpenAI-compatible / Raw JSON 自定义端点。
API key 保存在 Add-on Preferences，不写入 `.blend`；也可以通过环境变量提供。

使用阿里云百炼的 `qwen-image-3.0` / `qwen-image-3.0-pro` 时，应选择
`Qwen Image (DashScope)`，Model 填对应模型名，Base URL 填工作空间根地址，例如
`https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com`。Qwen Image 不走
`/compatible-mode/v1/images/edits`；插件会调用原生同步多模态端点，并能把 clean 与 Anchor/Previous
参考图一并发送。为了兼容旧配置，Base URL 尾部的 `/compatible-mode/v1` 和复制时带入的全角逗号会
自动移除。Base URL、API key 和模型必须属于同一区域。

## 2. UV 与网格的对应关系

### 2.1 两种 UV 模式

- **Mode A**：使用已有 UV。适合生产资产已经有可靠贴图 UV 的情况；重叠 UV 会共享同一磨损。
- **Mode B**：创建独立的 `AI_WearUV` 并自动展开，不改变原材质纹理使用的 UV。

Preflight 检查 UV 是否存在、是否落在 `[0,1]`、退化三角形、翻转和重叠情况。Mode B
创建后会保存精确的 per-loop UV 快照 `AIWear_UVSnapshot.npz`。Replay 如果发现 Wear UV
丢失，会先验证顶点、loop、polygon 数量，再恢复完全相同的 UV；不会猜测其他 UV 层。

Mode B 会临时 Reveal 处于 Edit Mode 隐藏状态的顶点、边和面，再对完整渲染网格执行 Smart UV，
结束后恢复原来的隐藏和选择状态。否则 Blender 的 `Select All` 会跳过隐藏面，让新 UV 层中对应
区域全部坍缩到 `(0,0)`。不超过 0.1% 的零面积 UV 三角形会作为可定位警告并由光栅器跳过，超过
该比例才会阻断管线；报错会列出全部失败项，而不是只显示利用率和重叠率。

### 2.2 用 UV 光栅化取得每个 texel 的 3D 几何信息

DCC 中已经有 3D mesh，也已经有每个 face loop 的 UV。这里做的是常规纹理烘焙准备：把目标
UV 的三角形光栅化到 `work_resolution × work_resolution` 的像素网格，并为每个有效 texel
保存它落入的三角形和重心坐标：

```text
triangle_id(u,v)
barycentric(u,v) = (b0, b1, b2)
```

假设该 texel 位于三角形 `(V0,V1,V2)`，直接用 mesh 顶点数据插值还原世界位置和法线：

```text
P(u,v) = b0·V0 + b1·V1 + b2·V2
N(u,v) = normalize(b0·N0 + b1·N1 + b2·N2)
```

这不需要引入新的数据模型，也不是从图片猜 3D；只是利用 mesh 和 UV 原本就有的对应关系，
把几何信息按输出贴图像素组织起来。后续可用数组运算批量投影，不必对每个 texel 调用
Blender `ray_cast`。代码中的 `UVField` 只是这组光栅化数组的历史类型名。

## 3. 多视角采集与 AI 生成

### 3.1 相机

插件在 evaluated mesh 的世界空间包围盒上计算中心与半对角线半径，用 50mm 透视相机
自动 framing。预设包括：

- Auto 6：四个环绕视角 + 顶部 + 底部；
- Auto 8：沿正方体 8 个顶点方向放置四个上方、四个下方的斜向视角；
- Turntable 4：四个水平视角；
- Counted Auto：按 Fibonacci sphere 均匀生成精确的 `Cam Count`；
- Custom：使用场景中名为 `AIWearCam_*` 的相机。

`Auto 8` 和 `Counted Auto, Cam Count=8` 不是同一组坐标。前者是严格中心对称的
`normalize(±1, ±1, ±1)`，适合盒状道具；后者使用黄金角和等面积纬度近似均匀铺球，8 个
Fibonacci 点不会恰好落在正方体顶点，但同样都是斜向视角，没有单独的纯顶部或纯底部图。

每个相机生成构图、材质、背景和光照固定的 `clean_Vi.png`。AI 的任务不是重新设计物体，
而是在相同图像坐标中加入磨损。

### 3.2 视角上下文

`View Context` 控制多张图是否顺序条件化：

| 模式 | 第 i 张图的输入 |
| --- | --- |
| Independent | 当前 `clean_Vi`，没有其他 worn 图 |
| First-view Anchor | 当前 clean + 第一张 `worn_V0` |
| Previous View | 当前 clean + 紧邻的上一张 `worn_V(i-1)` |

第一张始终只有 clean。`First-view Anchor` 倾向锁定一种统一磨损风格；`Previous View`
传递的是局部连续上下文，但也可能逐帧累积偏差；`Independent` 没有误差传播，但不同视角
可能各自生成不同颜色和磨损尺度。

存在 context 时，插件会额外明确告诉模型：第一张图是必须保持构图的当前 clean target，
第二张只是磨损材质、颜色、划痕尺度和严重程度的风格参考，不能复制它的相机与几何。

该输入只在 Provider 声明支持参考图时生效，目前内置实现中是 Gemini 与 Qwen Image。ComfyUI 的可移植
原生 inpaint 图不带 IP-Adapter，因此不伪装成支持跨图 context。每个视角实际使用的
`context_source` 和最终发送的完整 `prompt` 都会写入 `views.json`，实验时可以检查，而不是
只相信 UI 开关或根据面板字段反推。

旋转对称或高度重复的模型并非不能使用多视角，但它缺少稳定的方位身份：从不同方位看到的
轮廓近乎可互换时，独立生成更容易把划痕旋转、镜像或重新布局。此类资产应优先使用真正生效的
First-view Anchor，并避免仅靠增加独立视角数量解决覆盖；带靠背、扶手等方向锚点的椅子更适合
检验多视角一致性。

## 4. 从 AI 图提取磨损 mask

AI 经常让整张图略微变亮、变暗或偏色。如果直接 `abs(worn-clean)`，这种全局色调变化会
被误认为全模型磨损。插件先对 worn 做逐通道均值匹配：

```text
scale_c = mean(clean_c) / mean(worn_c)
worn_matched_c = clamp(worn_c · scale_c)
```

然后同时计算亮度差与最大通道差：

```text
d_luma = |luma(worn_matched) - luma(clean)|
d_rgb  = max_c |worn_matched_c - clean_c|
d      = max(d_luma, d_rgb)
M_i    = clamp(d / max(P95(d > 0.02), 0.05))
```

这一步得到屏幕空间 mask `M_i`。它回答的是“AI 在这张图的哪些像素做了局部修改”，而不是
“这个像素是否被相机看见”。两者必须分开，否则相机覆盖率接近 100% 时整个模型都会变白。

实际参与投影的浮点 mask 会以 16-bit RGB 灰度图保存为：

```text
views/diff_mask_V0.png
views/diff_mask_V1.png
...
```

## 5. 屏幕 Mask 如何落回模型表面

### 5.1 软件 Z-buffer

插件把网格三角形投影到每个相机的屏幕，光栅化出该像素最靠近相机的深度 `Z_i(x,y)`。
对于 UV 光栅化得到的网格位置 `P(u,v)`：

1. 用相机逆世界矩阵把 `P` 变换到相机空间；
2. 用 lens / sensor width 做透视投影，得到屏幕坐标 `(x,y)` 和深度 `z`；
3. 检查点在画面内且 `z <= Z_i(x,y) + depth_epsilon`；
4. 不满足时说明该表面点被遮挡，当前视角不能给它写 mask。

### 5.2 朝向权重

可见不等于可靠。掠射角的像素被压缩，重投影误差大，因此使用：

```text
view_dir = normalize(camera_position - P)
facing_i = clamp(dot(N, view_dir), 0, 1)^gamma
w_i      = visible_i · facing_i
```

`gamma` 越大，越信任正对相机的视角，越排斥侧视角。mask 和颜色残差使用完全相同的
坐标、可见性与权重，因此二者不会各自落到不同 texel。

## 6. 多视角融合

每个 UV texel 累加来自所有可见相机的观测：

```text
sum_mask += w_i · M_i(x,y)
sum_weight += w_i
F_ai = clamp(sum_mask / sum_weight)
```

融合后还会计算 8 邻域局部均值。如果某 texel 与邻域差值大于 `0.5`，就把它拉回自身与
邻域均值的中点，抑制单视角产生的极端亮点：

```text
F_ai ← 0.5 · (F_ai + neighbor_mean)
```

最终 `F_ai` 保存为 `M_Wear.png`，也是后续几何先验中 `w_ai` 对应的输入。

## 7. 几何先验是什么、怎么计算、有什么作用

AI mask 只描述可见图像变化，不知道资产真实拓扑，也容易把平面光照变化当磨损。几何先验
用于把磨损倾向拉回物理上更合理的位置。

### 7.1 Signed Convexity

对每条恰好连接两个面的拓扑边，取两个面法线 `n0,n1`、面中心 `c0,c1` 和边中点 `m`：

```text
magnitude = (1 - clamp(dot(n0,n1), -1, 1)) / 2
sign      = sign(dot(n0,m-c0) + dot(n1,m-c1))
C_edge    = sign · magnitude
C_vertex  = incident edge C_edge 的平均
```

- `C > 0`：凸起、棱边、容易碰撞的位置；
- `C < 0`：凹槽、内角、相对受保护的位置；
- `|C|`：由相邻面夹角决定，平面附近接近 0。

`w_convex` 提高凸边的磨损倾向；`w_cavity` 从倾向中扣除凹陷区域，防止所有高对比边都被磨损。

### 7.2 Exposure

每个自动相机都用同一 Z-buffer 判断顶点是否可见。某顶点被 `n` 个视角看到、总视角为 `N`：

```text
E(v) = clamp(n / N, 0, 1)
```

它是可接近性近似：长期暴露、容易被观察到的外表面权重高，封闭内腔低。它不是严格的接触
或受力模拟，但能减少磨损向隐藏区域无条件扩散。

### 7.3 Propensity

AI mask 先通过 UV 重心权重累积到顶点，得到 `F_ai(v)`。最终磨损倾向为：

```text
C+ = max(C, 0)
C- = max(-C, 0)

P(v) = clamp(
    w_ai     · F_ai(v)
  + w_convex · C+(v)
  + w_expose · E(v)
  - w_cavity · C-(v),
  0, 1)
```

各权重的作用：

| 参数 | 增大后的结果 |
| --- | --- |
| `w_ai` | 更服从 AI 图中的 scratches/chipping 分布 |
| `w_convex` | 更集中于凸棱和硬边 |
| `w_expose` | 更偏向外露表面，减少隐藏面的磨损 |
| `w_cavity` | 更强地保护凹槽和内角 |

关闭 `Geometry Prior` 时，后三项被置零，但管线仍正常运行；关闭 `AI Mask` 时只把
`w_ai` 置零，用于观察纯几何规则能产生什么结果。

## 8. 拓扑传播如何生成 WearThreshold

`P(v)` 只表示“哪里适合磨损”，还不是可随 Amount 单调展开的先后顺序。WearThreshold 把它转换为
`T(v)∈[0,1]`：值越小越早出现磨损。

### 8.1 选择生长种子

候选顶点需要：

- `P(v) >= 0.18`；
- `P(v)` 不低于所有一环邻居，是局部极大值。

候选按 `P` 从高到低排序，再做空间抑制：新种子必须与已保留种子相距超过模型包围半径的
`10%`。这样不会在同一条棱上密集产生几十个几乎相同的起点。没有局部极大值时退化为
`P` 最高的一小组顶点。

### 8.2 在网格图上计算传播代价

网格顶点是图节点，真实 mesh edge 是图边。边 `(i,j)` 的代价为：

```text
P_mean = max((P_i + P_j)/2, 1e-3)
cost(i,j) = edge_length / (1e-4 + P_mean^gamma)
```

高倾向区域 `P_mean` 大，传播代价低；低倾向区域代价高。于是磨损优先沿凸边、AI mask 和外露
区域走，而不是在 UV 图上直线扩散。

如果边两侧 polygon 的 `material_index` 不同，并开启 `Material Boundary Barrier`：

```text
cost(i,j) *= material_boundary_penalty
```

这让涂漆塑料与金属、外壳与按钮之间不容易互相串色，但不是绝对禁止跨越。

### 8.3 多源 Dijkstra

所有种子以距离 0 同时进入优先队列，运行 multi-source Dijkstra，得到每个顶点从“最容易
磨损种子”到达的最小累计代价 `D(v)`：

```text
T_base(v) = D(v) / max(D)
```

这就是拓扑传播：它只沿真实网格边移动。这里跨 UV seam 连续的直接原因不是“先回到 3D”这句
话，而是同一个 mesh 顶点只保存一份 `T(v)`；UV seam 只复制 UV 坐标，并没有复制这个顶点值。

关闭 `Topology Growth` 时不会运行 Dijkstra，而是直接使用 `1-P` 作为 WearThreshold 主体；
这是可运行的消融分支，不会停在半条管线上。

### 8.4 最终生长形态

开启拓扑传播时：

```text
T(v) = clamp(
    alpha · T_base(v)
  + (1-alpha) · (1-P(v))
  + noise_amp · (2·noise3D(position·noise_scale)-1),
  0, 1)
```

最后做两轮一环平滑：

```text
T_new(v) = 0.5·T(v) + 0.5·mean(T(neighbors))
```

参数如何控制形态：

| 参数 | 小值 | 大值 |
| --- | --- | --- |
| `alpha` | 贴着局部 `P`，碎、直接、像阈值 AI/几何图 | 强调从种子沿拓扑扩张，形成连续生长带 |
| `noise_amp` | 边界平滑、规则 | 边界破碎、斑驳；过大会产生孤立噪点 |
| `noise_scale` | 大块、低频变化 | 小块、细碎变化 |
| `gamma` | 传播较容易穿过中低 P 区域；视角权重也较宽松 | 更沿高 P 通道生长；投影也更偏向正视角 |
| `material_boundary_penalty` | 容易跨材质传播 | 材质边界更像阻挡 |
| `Feather` | Shader 阈值边缘硬 | 只改变最终显示过渡，不重算 T |

注意当前 `gamma` 同时用于投影视角朝向和 Dijkstra 代价指数，这是当前实现的共享“选择性”
控制：增大后既更排斥掠射投影，也更强迫传播沿高 propensity 区域前进。

## 9. WearThreshold 如何回到 UV

顶点 WearThreshold 再按 UV 光栅化保存的三角形与重心坐标插值：

```text
T(u,v) = b0·T(V0) + b1·T(V1) + b2·T(V2)
```

对于一条由两侧面共享的 mesh 边，两侧 UV 副本都从相同的两个端点值插值，所以理想的边界值
相同；两个 UV 岛在贴图中是否相邻不影响这一点。贴图仍会受到有限分辨率、过滤和岛外像素影响，
因此“边界值相同”不等于最终采样一定看不到缝。

## 10. Seam Fusion 与 Padding

### 10.1 先明确：重投影本身并不消除 UV 接缝

把多视角结果按已知相机和 mesh 几何重投影，是常规的 3D→2D texture bake 路径。它解决的是
“屏幕像素应落到模型哪里”，并不自动保证 UV seam 两侧相等。是否出现接缝，取决于同一条
mesh 邻接边的两个 UV 副本是否得到一致数值，以及纹理过滤时是否读到了岛外内容。

当前管线里要区分三类情况：

1. **数据不一致**：独立 AI 视图可能在同一位置画出不同磨损；两侧面也可能由不同相机覆盖，
   使用不同的可见性、法线朝向权重或高频图像像素。`M_Wear` 和 `AIWear_WornTex` 是按
   UV texel 直接累计的高分辨率结果，因此仍可能在 seam 两侧不同。
2. **采样污染**：即使连续的每顶点 WearThreshold 在边界两侧理论值相同，有限分辨率下两个 UV 岛
   的 texel 中心并不落在完全相同的 3D 位置；双线性、mipmap 和各向异性过滤还会读到岛外的
   黑色/中性色或相邻岛。这由 padding、岛间距和输出分辨率决定。
3. **不是可配对的 seam**：重叠/镜像 UV 会让不同表面争用同一 texel；开口边、拆开的顶点或
   不连通几何没有共享拓扑边。它们不能靠当前 Seam Registry 的“两侧配对”自动解决。

因此，当前 WearThreshold 较不容易产生结构性断裂，是因为它先得到**每顶点唯一数值**再烘焙；
不是因为管线中引入了额外的数据抽象。高频 AI mask、颜色残差和真实纹理采样仍需要单独检查。

### 10.2 Seam Registry

插件遍历每条恰好连接两个面的拓扑边。如果同一端点在两个 face loop 中的 UV 坐标不同，
该边就是 UV seam。Registry 保存同一条 3D 边两侧的 UV 线段：

```text
side A: uv_a(t) = (1-t)·a0 + t·a1
side B: uv_b(t) = (1-t)·b0 + t·b1
```

### 10.3 融合

沿同一拓扑位置 `t`，先在两侧各向岛内偏移半个 texel，读取最近的有效边界 texel：

```text
delta(t) = 0.5 · (sample(B,t,0.5) - sample(A,t,0.5))
A'(t,d) = A(t,d) + falloff(d) · delta(t)
B'(t,d) = B(t,d) - falloff(d) · delta(t)
```

算法只把边界差的一半分别加到 A、从 B 减去，使第一行 texel 在不改变共同能量的情况下相遇；
这份校正再在 `Seam Diffuse` 宽度内向岛内衰减。它不会把整个窄带替换成两侧平均值，因此能保留
各岛原有的划痕和颗粒。所有 seam 先从原图读取、再统一累积写回，避免 mesh 遍历顺序影响结果。
WearThreshold、WornTex RGB 和 mask alpha 都执行同样的配对校正。采样不能落在数学意义上的 UV
边界：Padding 之前，该位置的双线性 footprint 会混入岛外黑色或中性色，再把污染值写回成暗线。

### 10.4 Padding 不是 Seam Fusion

Padding 从 UV 岛有效 texel 向空白区做多轮四邻域膨胀，每轮用已有邻居均值填一个 texel。
它解决岛边界采样漏色，不负责让 seam 两侧的原始数据相等；但在实际渲染中，它往往是消除
黑边/漏色型接缝最直接的因素。Padding 不能大于资产实际保留的 island gutter；默认值因此为
2 texel。需要更高 mip 安全距离时，应先用对应 margin 重新 Pack Islands，再同步增大 Padding，
而不是只把 Padding 从 2 调到 16。

两者已经解耦，可以分别关闭：

| Seam Fusion | Padding | 作用 |
| --- | --- | --- |
| off | off | 原始烘焙结果 |
| on | off | 只让同一 3D seam 两侧一致 |
| off | on | 只扩展岛边颜色 |
| on | on | 同时处理数值差异与岛外采样 |

### 10.5 现有六视图 wear texture 的实测结果

对 `gaming_console` 的同一批 6 张 `clean/worn` 缓存运行 Replay，只改变 Seam Fusion / Padding。
该组使用 `Auto 6`、1024 工作分辨率、Custom Provider；缓存的 `views.json` 没有记录任何
`context_source`，所以不能把它当成受 Anchor 约束的一致多视图。`seam_qa` 沿 9,642 条已登记
seam 的两侧配对，在烘焙后的纹理上做双线性采样，结果如下：

| 后处理 | seam mean | seam p95 |
| --- | ---: | ---: |
| 都关闭（原始 WearThreshold） | 0.03518 | 0.07698 |
| 仅 Seam Fusion | 0.00631 | 0.02068 |
| 仅 Padding | 0.00906 | **0.01324** |
| Seam Fusion + Padding | **0.00607** | 0.01947 |

这组结果不支持“先投回 3D，所以天然无缝”的说法：原始烘焙仍测到了边界差异；只开 Padding
已经把 p95 从 `0.07698` 降到 `0.01324`，说明本例很大一部分是岛外采样问题。Seam Fusion
进一步降低了平均差异，但当前按固定半径 stamp 的实现改动范围偏宽：最终 WearThreshold 中有
`42.1%` texel 的变化超过 `0.01`。在相同 hero render 中，两组图片的全图差异 p95 为 0，
只有 `2.06%` 像素变化超过 `0.01`，差异主要集中在 seam 附近。

这里的指标有意包含 texel 中心偏移、岛外像素和双线性过滤带来的误差；它衡量的是 Shader
最终会采到的纹理边界差异，不是直接比较 mesh 顶点上的 `T(v)`。

对这份六视图结果，可见接缝更容易在以下条件出现：相邻面主要由不同 worn 视图贡献、AI 在
重叠可见区域不一致、seam 落在高频划痕/颜色变化上、某侧覆盖不足、贴图分辨率低，或 padding
小于实际过滤 footprint。反过来，如果两侧来自同一连续的每顶点数值、视图观测一致、覆盖充分，
并为目标 mip 层保留足够 padding，就可能不需要额外 seam fusion。当前实验说明应先判断接缝
属于“数值不一致”还是“过滤漏色”，再决定用配对融合还是 padding，不能把两者混成一个原因。

### 10.6 `ceiling_fan` 底座复现实验

在 Blender 5.1.2 中，对 2026-09-05 14:49 的同一批 Qwen Image 3.0 六视图缓存执行真实 Replay，
不重新调用 AI，只改变后处理。该次 `First-view Anchor` 已真实生效：V1–V5 的 `context_source`
都是 `worn_V0.png`。截图中的水平线也确实属于 Registry 能配对的 manifold UV seam；底座范围内
共统计到 556 条。

| 分支 | 最终外观跨缝 p95 | seam 周围 4px 带状差异 p95 |
| --- | ---: | ---: |
| Fusion off / Padding 2 | 0.340 | 0.313 |
| 修复前 Fusion 8 / Padding 2 | 0.192 | **0.229** |
| 修复后 Fusion 8 / Padding 2 | **0.146** | 0.286 |

这次“关闭反而更好看”不是错觉。修复前算法在 Padding 之前直接沿数学 UV 边界做双线性采样，
footprint 会混入尚未填充的岛外像素；随后又把两侧完整的 8 texel 窄带替换成共同平均轮廓。
因此它虽然把 A/B 数值差和统计指标压低，却可能制造一条两侧同样暗、肉眼反而更整齐醒目的横带。
单看跨缝差值无法发现这种 common-mode halo。

修复版改为在边界两侧各向岛内偏移半个 texel，要求配对 texel 都有效，只计算边界差并把校正量
向内衰减，不再平均整条纹理带。同一缓存下，跨缝 p95 相对 Fusion off 降低约 57%，4px 邻域带状
差异也降低约 9%，且渲染中不再出现旧版横线。当前建议仍是 `Seam Diffuse=8, Padding=2`；若使用
旧版代码，仅调小 Diffuse 或关闭 Padding 不能根治边界采样污染。

### 10.7 AI 遮挡轮廓不能直接当作表面纹理

同一批 Qwen 六视图还暴露了另一类看似接缝、实际与 UV seam 无关的白色弧线。逐视角 Replay
显示，只保留 V0 且关闭 Seam Fusion 时就能完整复现该弧线；红色 Seam Registry overlay 也不与
它重合。把 Z-test epsilon 缩小十倍后白弧仍在，因此不是后方表面错误通过深度测试。

根因在屏幕图像：V0 的 clean 与 worn 整体轮廓相近，但白色中壳遮挡黑色底座的边界移动了数个
像素。`abs(worn-clean)` 会把这条前景轮廓识别成强磨损；clean 几何的 Z-buffer 在偏移后的像素处
却认为底座可见，于是常规重投影把白色外壳残差写到了底座，形成一个在任何单张 AI 图中都找不到
的椭圆印记。

管线现在从 clean 几何深度图检测背景轮廓和深度突变，在其两侧建立约为图像宽度 `0.6%` 的保护
带。保护带内 mask 归零、带符号 RGB residual 回到中性 `0.5`；平滑曲面内部保持不变，磨损是否
靠近凸边仍由对象空间 Geometry Prior 决定。相同缓存下，底座外观跨缝 p95 从 `0.146` 降到
`0.042`，白弧消失。这个步骤解决的是 AI 编辑的 occlusion-boundary leakage，不应与 Seam Fusion
或 Island Padding 混为一类。

## 11. Shader 如何应用 AI 磨损

插件不会把 AI 的 worn render 直接当 Base Color；那样会把视图光照烘进材质，再被 Blender
灯光照第二次。它保存曝光匹配后的颜色差：

```text
delta = worn_matched - clean
encoded_rgb = 0.5 + 0.5·delta
```

写贴图前把 delta 限制在 `[-0.06, +0.28]`，抑制 AI 全局重打光。WornTex alpha 来自
`F_ai` 的平滑 mask 取值窗 `[0.06,0.55]`，不是相机 coverage。

Shader 中：

```text
gate = smoothstep(T - feather, T + feather, wear_amount)
mask = gate · worn_tex.alpha
worn_color = original_base_color + 2·(worn_tex.rgb - 0.5)
final_base_color = mix(original_base_color, worn_color, mask)
```

因此：

- `Wear Amount=30/60/100` 是同一张 WearThreshold 的阈值，天然满足集合单调增长；
- 100% 只打开所有有 AI mask 的区域，不会把全部相机覆盖区变白；
- 原材质的 Normal、Roughness、Metallic 等连接保留；当前 AI 外观叠加只修改 Base Color；
- 共享材质先复制成对象私有预览材质，不影响使用原材质的其他对象。

## 12. ComfyUI / Liblib 局部重绘

仓库提供同一张图的两种序列化：

- [`examples/comfyui/aiwear_inpaint_workflow.json`](examples/comfyui/aiwear_inpaint_workflow.json)：
  UI 格式，导入 ComfyUI / Liblib 画布；
- [`examples/comfyui/aiwear_inpaint_api.json`](examples/comfyui/aiwear_inpaint_api.json)：
  API 格式，Blender 通过 `/prompt` 调用。

工作流只使用原生节点：

```text
CheckpointLoaderSimple
  ├─ CLIPTextEncode positive / negative ──────────────┐
Clean LoadImage ─┐                                    │
Mask LoadImage ──┴─→ VAEEncodeForInpaint → KSampler → VAEDecode
Clean + decoded + mask → ImageCompositeMasked → SaveImage
```

Blender 根据轮廓内侧带和屏幕深度高梯度区域生成 `inpaint_mask_Vi.png`。RGBA alpha 按
ComfyUI `LoadImage` 的反 alpha mask 约定写入；最后再按 mask 合成，非目标像素使用原 clean。

默认节点映射：Clean `2`、Mask `3`、Prompt `4`、Seed `7`、Output `10`。Liblib 导入 UI
JSON 后只需在节点 1 选择平台已有的 inpainting checkpoint。插件安装包内也自带 API JSON。

## 13. 实验开关与输出

`N > AI Wear > Experiments / Ablation` 提供实际执行分支：

- `AI Mask`：关闭后 `w_ai=0`；
- `Geometry Prior`：关闭 convexity/exposure/cavity；
- `Topology Growth`：关闭 Dijkstra，直接由 `1-P` 生成 T；
- `Seam Fusion` 与 `Island Padding`：独立后处理开关；
- `Save Experiment Snapshot`：保存本次定性对照材料。

相机数量与 `View Context` 在 Capture 面板设置。实验快照包含最终贴图、seam/padding 前后
WearThreshold，以及完整的 clean/worn/diff/context 视图序列：

```text
.ai_wear_cache/<object>/experiments/<label>_<job_id>/
├─ config.json
├─ metrics.json                 # 只记录 elapsed_seconds
├─ M_Wear.png
├─ WearThreshold_before_seam_padding.png
├─ WearThreshold_after_seam_padding.png
├─ WearThreshold.png
├─ AIWear_WornTex.png
└─ views/
   ├─ views.json
   ├─ clean_V*.png
   ├─ worn_V*.png
   ├─ diff_mask_V*.png
   └─ inpaint_mask_V*.png
```

这里的 `views/` 是审阅快照，不是下游算法要求的第二份缓存。相机数量、Prompt 或 View Context
变化时，视图本身是实验变量，复制到实验目录可以防止缓存根目录下一次完整运行后被覆盖；但
Geometry / Topology / Seam / Padding 这些 Replay 实验复用的是同一组视图，逐目录复制没有算法
必要。当前 `gaming_console` 的 7 个实验目录共复制了 133 个、112.67 MB 的 view 文件，它们按
文件名与根目录 `views/` 全部字节相同；这是“每个实验包可独立查看”的存储取舍，也是一项可去重
的实现债务。

定性实验组合与截图重点见 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 14. 缓存、线程与代码位置

默认缓存：`<blend>/.ai_wear_cache/<object>/`。`Replay Downstream` 使用缓存的 clean/worn、
精确相机矩阵和 UV snapshot，所以改变下游 feature 不需要再次生图。

网络和 NumPy 工作运行在 worker；渲染、UV、bmesh、材质和 Blender image data 操作通过
`MainThreadBridge` 回到主线程，由 `bpy.app.timers` 驱动。这样网络等待不会锁死 Blender UI。

```text
ai_wear/render       自动相机与 clean render
ai_wear/ai           Provider、ComfyUI graph 执行
ai_wear/uv           UV QC、UV 光栅化、seam registry
ai_wear/surface      差分、重投影、融合、几何先验、Dijkstra
ai_wear/shader       WearThreshold gate 与原材质叠加
ai_wear/operators    管线、缓存、Replay、实验快照
```

模块拆解与验收映射见 `Blender_AI_Wear_Texture_Plugin_Implementation_Plan.md`。
