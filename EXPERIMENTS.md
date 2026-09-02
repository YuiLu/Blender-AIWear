# 定性实验计划

目标是用少量、变量明确的对照图说明各模块为什么存在。除运行耗时外不设置数值指标，
不做论文式参数扫表。

## 统一条件

所有对照尽量固定：

- 同一个模型和 `.blend`；
- 同一个 Wear UV；
- 同一 prompt、模型、分辨率和 seed；
- 同一 Wear Amount，建议主图用 60；
- 同一灯光、视角和截图曝光；
- 开启 `Save Experiment Snapshot`，Label 使用下文建议名称。

相机数量和上下文实验改变了 AI 输入，必须重新运行完整管线。几何先验、拓扑生长、seam、
padding 和 Amount 对照可以使用同一批缓存图运行 `Replay Downstream`。

## 实验 1：相机数量 1 / 6 / 8

使用 `Counted Auto`，只把 `Cam Count` 改成 1、6、8：

```text
cams_01
cams_06
cams_08
```

展示内容：

1. 三组最终模型使用相同观察相机并排截图；
2. 每组挑选对应的 `clean_V* / worn_V* / diff_mask_V*`；
3. 记录 `metrics.json` 中的 `elapsed_seconds`。

定性观察：

- 单视角背面和遮挡区是否缺少 AI 证据；
- 6 视角是否已经覆盖主要外壳与边缘；
- 8 视角增加的顶部/底部信息是否值得额外耗时；
- 多视角融合后是否出现互相冲突、变淡或重复纹理。

## 实验 2：逐视角上下文是否改善一致性

这组是完整 AI 实验，使用相同的 6 个相机，分别设置：

```text
context_none       View Context = Independent
context_first      View Context = First-view Anchor
context_previous   View Context = Previous View
```

三种模式含义：

- Independent：每张只看当前 clean；
- First-view Anchor：后续每张都参考第一张 worn；
- Previous View：第 i 张参考第 i-1 张 worn，形成顺序链。

当前内置 Provider 中该 feature 由 Gemini 支持。`views.json` 的 `context_source` 会记录每张
图实际收到的参考图；若 Provider 不支持，它会是 `null`，不能把这种运行当作上下文实验。

展示内容：

1. 每组把 6 张 `worn_V*` 排成一行或 contact sheet；
2. 重点圈出跨视角可见的同一条边、logo 周围和材质交界；
3. 再展示三组融合后的模型；
4. 记录三组总耗时。

定性观察：

- 磨损颜色、颗粒尺度和划痕宽度是否跨视角一致；
- 同一条边在相邻视角中是否保持类似语义；
- Previous View 是否发生风格逐帧漂移或错误累积；
- First-view Anchor 是否过度复制首张视角的局部图案；
- 无上下文是否虽然局部质量好，但融合后出现冲突。

## 实验 3：几何先验开 / 关

复用同一批 6 视角 AI 缓存：

```text
geometry_off    Geometry Prior = off
geometry_on     Geometry Prior = on
```

其他开关保持一致。展示相同观察角度下的模型，并附 `AIWear_Mask.png`，说明 AI evidence
输入没有变化，变化来自 convexity / exposure / cavity。

定性观察：

- 外壳凸边、按钮边缘和高接触区是否更容易先磨损；
- 平坦大面是否减少无意义的均匀磨白；
- 凹槽、通风孔内部是否得到保护；
- 几何先验是否压制了 AI 图中本来合理的局部磨损。

## 实验 4：拓扑传播开 / 关

继续复用相同 AI 缓存：

```text
topology_off    Topology Growth = off
topology_on     Topology Growth = on
```

关闭时 WearTime 直接来自 `1-P`；开启时从高 propensity 种子运行 multi-source Dijkstra。

定性观察：

- 开启后磨损是否形成沿真实边和曲面延伸的连续区域；
- 关闭后是否更像对 AI/几何 mask 的直接阈值，出现碎片；
- 拓扑传播是否错误跨过狭窄连接或材质边界；
- UV 岛断开处的磨损在 3D 模型上是否仍连续。

## 实验 5：UV seam 处理前 / 后

使用同一个 Replay，固定 Padding 状态，只改变 Seam Fusion：

```text
seam_off    Seam Fusion = off, Island Padding = on
seam_on     Seam Fusion = on,  Island Padding = on
```

若还要解释 padding，可额外补一张 `Seam Fusion = on, Island Padding = off`，但不必组成
四组定量矩阵。

展示内容：

1. `WearTime_before_seam_padding.png` 与 `WearTime_after_seam_padding.png`；
2. UV Editor 中圈出 seam 两侧；
3. 模型上用近距离、斜视角和 Material Preview 展示接缝；
4. 最终模型使用完全相同的 Amount 和 Feather。

定性观察：

- 同一条 3D 边两侧的 WearTime 是否跳变；
- WornTex 颜色是否出现一边亮、一边暗；
- 开启后接缝是否消失，同时有没有把细节扩散得过宽；
- padding 关闭时 mipmap / 双线性采样是否在岛边漏出黑边或中性色。

## 实验 6：Wear Amount 单调控制

同一张 WearTime 直接导出 30 / 60 / 100 三张结果，不重新运行 AI：

```text
amount_30
amount_60
amount_100
```

定性观察：

- 30 的磨损区域是否包含在 60 中，60 是否包含在 100 中；
- 增长是否从合理种子沿边缘扩展，而不是整张贴图一起变白；
- 100 是否仍受 WornTex evidence alpha 限制；
- Feather 只改变边界软硬，不应改变磨损出现顺序。

## 可选：生长形态示意

如果交付中需要解释参数，只做少量极端对照，不做网格搜索：

- `alpha=0.3` 对 `alpha=0.85`：局部 propensity 与拓扑传播主导；
- `noise_amp=0` 对 `noise_amp=0.18`：规则边界与破碎斑驳边界；
- `noise_scale=3` 对 `noise_scale=16`：大块与细碎形态。

这组用于说明控制能力，不作为主要效果优劣结论。

## 建议交付顺序

1. 主结果：Amount 30 / 60 / 100；
2. 相机 1 / 6 / 8；
3. Independent / First Anchor / Previous View；
4. 几何先验开关；
5. 拓扑传播开关；
6. seam 处理前后；
7. ComfyUI / Liblib 局部重绘作为已实现 feature 单独展示，不列入实验。
