# 实验计划

所有实验固定模型、UV、prompt、seed、分辨率和 Wear Amount；一次完整 AI 生成后，优先用
`Replay Downstream` 改消融开关。每组开启 `Save Experiment Snapshot` 并使用唯一 Label。

| 实验 | 变量 | 建议组别 | 主要指标 |
| --- | --- | --- | --- |
| 相机数量 | Counted Auto / Cam Count | 1, 2, 4, 6, 8 | coverage ratio、耗时、遮挡区缺失、视角冲突 |
| 相机分布 | Camera Preset | Turntable 4, Auto 6, Auto 8 | 顶/底覆盖、可见性、纹理一致性 |
| AI 贡献 | AI Evidence | on / off | 与 AI 目标风格相似度、边缘集中度 |
| 几何先验 | Geometry Prior | on / off | 凸边命中率、凹槽误磨损率 |
| 拓扑传播 | Topology Growth | on / off | 连续性、孤立噪点、30⊆60⊆100 |
| UV 接缝 | Seam Fusion × Padding | 00, 10, 01, 11 | seam p95、接缝可见像素、mipmap 漏色 |
| 局部重绘 | Geometry Inpaint Mask | on / off | 轮廓漂移、背景变化、非目标区域差分 |
| 生长形态 | alpha / noise amp | alpha 0.4/0.7/1.0；noise 0/0.12 | 块状度、方向性、重复性 |

## 最小交付组

1. 相机数 1/2/4/6/8：记录 `effective_view_count` 与 `coverage_ratio`。
2. UV seam 四组：用 `WearTime_before_seam_padding.png`、
   `WearTime_after_seam_padding.png` 和 `seam_before_p95/seam_after_p95`。
3. AI / Geometry / Topology 三项单独关闭：与全开 baseline 并排展示。
4. ComfyUI 全图编辑与几何局部重绘：比较 clean/worn 的非 mask 区域差异。
5. Amount 30/60/100：导出三张 mask，检查集合单调包含关系。

## 可复现命名

建议 Label 使用：

```text
cams_01, cams_02, cams_04, cams_06, cams_08
seam_00, seam_10, seam_01, seam_11
ablate_ai, ablate_geometry, ablate_topology, baseline
inpaint_off, inpaint_on
```

`config.json` 保存实际开关、权重、seed、分辨率和有效相机数；`metrics.json` 保存覆盖率、
seam p95 与 postprocess 平均变化量，足够生成实验表格而不依赖截图人工抄数。
