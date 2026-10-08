# vidpure_nodes — ComfyUI 去字幕节点

## 安装

方式 A（推荐，拷贝即用）：

```
ComfyUI/custom_nodes/vidpure_nodes/   ← 拷贝本目录
```

然后在 **ComfyUI 自带的 python 环境**里装推理依赖（LaMa）：

```
<ComfyUI>/python_embeded/python.exe -m pip install iopaint
```

方式 B（pip 安装主包后挂载）：

```
pip install "vidpure[gpu]"
```

并把 `vidpure/comfy/vidpure_nodes` 软链或拷贝到 `custom_nodes`。

重启 ComfyUI，节点管理器搜索 `VidPure` 即可看到两个节点。

## 节点

| 节点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| **VidPure 字幕清除 (LaMa)** | IMAGE + 可选 MASK | IMAGE | 逐帧 LaMa 修复；有 MASK 用 MASK，否则按 region 生成矩形 |
| **VidPure 区域 Mask** | IMAGE + region 字符串 | MASK | 比例/像素区域 → 矩形白 mask，`grow` 可外扩防描边残留 |

## 推荐工作流

```
VHS Load Video ──image─────────────────────┐
      │                                    ▼
      ├──image→ VidPure 区域 Mask ─mask→ VidPure 字幕清除 ─image→ VHS Video Combine
      └──audio──────────────────────────────────────────────────────┘
```

- 字幕位置固定：只需 `region` 参数（如底部 `0.78,0.95`），无需额外检测
- 字幕位置浮动：用 OCR 类节点产出 MASK 接入 `mask` 输入
- 大视频：`VHS Load Video` 的 `frame_load_cap` 分批载入，避免内存爆掉

## 首次运行

LaMa 模型权重（约 200MB）由 IOPaint 首次调用时自动下载到用户目录，
之后离线可用。NVIDIA 卡选 `device=cuda`；RTX 50 系请确认 ComfyUI
的 torch ≥ 2.7 (cu128)。
