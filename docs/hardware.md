# 硬件分档与方案建议

> 以「去硬字幕（台词字幕）+ 画质放大」为目标。速度按 1080p·30fps·30s ≈ 900 帧估算。

## 分档速查

| 档位 | 典型配置 | 去字幕通道 | 放大引擎 | 30s 视频全流程预估 | 说明 |
|---|---|---|---|---|---|
| 轻量 | 核显 / 无独显 | `delogo`（FFmpeg） | NCNN Vulkan tile=192 | 4~8 min（仅 delogo 级放大） | 痕迹明显；放大可跑但慢 |
| **均衡（8GB）** | **RTX 40/50 系 8GB** | **LaMa（IOPaint, cuda）** | **NCNN Vulkan tile=256** | **12~20 min** | **质量/速度平衡点，默认档** |
| 高端 | RTX 40/50 系 16GB+ | LaMa 或 ProPainter | NCNN tile=384 / torch realesrgan | 15~25 min | ProPainter 视频级修复更稳 |
| 旗舰 | RTX 4090/5090 24GB | ProPainter 全速 | torch realesrgan + GFPGAN | 20~30 min | 效果天花板的本地开源档 |

## RTX 40/50 系 · 8GB 显存（本文档默认读者）的三个注意点

1. **50 系（Blackwell sm_120）必须 torch ≥ 2.7 + cu128**，否则报
   `CUDA error: no kernel image for sm_120`。`vidpure probe` 会自动检测并提示。
2. **8GB 显存的最佳组合**是 LaMa（单帧修复，峰值 ~3GB）+ NCNN Vulkan 放大
   （不走 torch，独立用显存）。两者不同时占用，互不挤兑。
3. **ProPainter（视频级修复）在 8GB 上要降配**：先把视频缩到 720p 处理再放大回来，
   或用 `subvideo_length` 分窗；追求省心就停在 LaMa。

## 方案对比（效果 / 速度 / 硬件）

| 方案 | 效果 | 速度（900 帧） | 硬件要求 |
|---|---|---|---|
| FFmpeg delogo/blur | 有明显痕迹 | 2~3 min | 无要求 |
| LaMa 逐帧修复 | 接近无痕（静态背景优） | 4~8 fps → 5~8 min | 6GB+ 显存（cuda）或 CPU（极慢） |
| ProPainter 视频修复 | 时序一致，最优 | 1~2 fps → 10~20 min | 10GB+ 显存（8GB 需降配） |
| Real-ESRGAN x4plus | 通用最佳 | 2~5 fps → 5~10 min | 6GB+，tile 自适应 |
| Real-ESRGAN animevideov3 | **动漫/3D 渲染向，速度最快** | 8~15 fps → 2~4 min | 4GB+ |
| SwinIR / HAT-L | 学术天花板 | <0.5 fps，小时级 | 12GB+，不推荐量产 |
| Topaz Video AI（商用） | 实拍/渲染综合最稳 | 中等 | 8GB+，付费 |

> AI 短剧（3D 渲染/动漫风格）放大首选 `realesr-animevideov3`：专为动漫视频训练，
> 比通用模型快 3 倍左右，本项目的默认放大模型。

## 上游止损

生成端（H3 / Seedance 等）提示词明确写入「画面中不出现任何字幕、水印、UI 元素」，
成本为零；存量视频才需要本工具。
