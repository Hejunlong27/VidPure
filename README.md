# VidPure

**AI 视频硬字幕清除 + 画质放大 —— 一体化开源工具**

[English](README_EN.md) | 中文

`vidpure` 是一个「调度器」：它自己不实现任何 AI 模型，而是探测你的硬件、
自动拉起最合适的本地算力（FFmpeg / IOPaint-LaMa / Real-ESRGAN-NCNN / ComfyUI server），
把 **去字幕 → 放大 → 音频封装** 串成一条命令的流水线。
同时提供 **ComfyUI 自定义节点**，可把去字幕能力直接嵌进你的出片工作流。

- 核心零第三方依赖（FFmpeg 外部二进制即可跑快速通道）
- 放大引擎默认 **Real-ESRGAN NCNN Vulkan**：不依赖 torch，N/A/I 显卡通吃，模型自动下载
- 调度器模式 & ComfyUI 节点模式共享同一套核心逻辑
- 音频默认无损耗保留（AAC/MP3 直通，其他编码自动转 AAC）

---

## 功能范围

| 能力 | 通道 | 依赖 | 画质 |
|---|---|---|---|
| 去字幕 · 快速 | `delogo` / `blur` / `crop` | 仅 FFmpeg | 有痕迹 / 改构图 |
| 去字幕 · 修复 | `lama`（IOPaint 逐帧） | `pip install iopaint` + GPU | 接近无痕 |
| 画质放大 | Real-ESRGAN（NCNN Vulkan） | 自动下载引擎与模型 | 动漫/3D 渲染向最佳 |
| 批量调度 | 目录 / 文件列表 / 并发 | — | — |
| ComfyUI 节点 | 字幕清除 + 区域 Mask | iopaint | 同上 |

明确不做的事：不训练模型、不做云端 API 封装、不重分发模型权重（运行时按官方 release 下载）。

## 架构

```
                         ┌─────────────────────────────┐
   vidpure run (CLI) ───▶│        调度器 core/          │
   vidpure comfy ───────▶│  probe → pipeline → media   │
   ComfyUI 节点 ────────▶└──────────┬──────────────────┘
                                    │ 按硬件选择/拉起算力
            ┌───────────────┬───────┴────────┬──────────────┐
            ▼               ▼                ▼              ▼
     FFmpeg 子进程    IOPaint(LaMa)   Real-ESRGAN       ComfyUI server
     delogo/blur/     逐帧修复        NCNN Vulkan       （自动拉起 main.py，
     crop 快速通道                    逐帧超分           /prompt 提交工作流）
```

## 安装

```bash
# 0) 前置：FFmpeg（winget install ffmpeg）+ Python 3.9+
git clone https://github.com/Hejunlong27/VidPure.git && cd vidpure

# 1) 快速通道（零依赖，直接可用）
pip install -e .

# 2) LaMa 去字幕（推荐）
pip install -e ".[gpu]"

# 3) 硬件自检（自动给出推荐参数与警告，如 RTX 50 系需 torch cu128）
vidpure probe
```

NVIDIA RTX 50 系（Blackwell）额外要求 `torch>=2.7 cu128`：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## 使用模式零：网页版（非开发者推荐）

无需任何命令行知识，双击式操作：

```bash
vidpure web            # 自动打开浏览器 http://127.0.0.1:8787
vidpure web --lan      # 允许手机在同一 WiFi 下访问（打印局域网地址）
```

界面特性：

- **三步引导**：选视频 → 确认字幕位置 → 选模式，步骤条实时点亮
- **拖拽/点击上传**：大文件分片上传带进度条，全程本机处理
- **可视化框选字幕区域**：在视频首帧上按住拖动即可（支持手机触屏）；
  默认「字幕在画面底部」自动预设，无需手动框选
- **三种模式卡片**：⚡快速去字幕 / ✨高清修复（AI 逐帧）/ ✂️裁掉字幕区，
  硬件不满足时自动降级并提示
- **放大倍数**：不放大 / 2x / 4x 单选
- **实时进度**：排队 → 修复 → 放大 → 封装各阶段中文进度提示
- **结果页内嵌播放器**（支持拖动进度条）+ 一键保存，文件名自动带上「_去字幕」
- **全中文错误提示**：如「高清修复需要 AI 组件：请运行 pip install iopaint」
- 响应式布局，桌面 / 手机 / 主流浏览器（Chrome·Edge·Firefox·Safari）适配

技术：后端为 Python 标准库 `http.server`（零第三方依赖），前端单文件页面
无任何外部 CDN 依赖，可完全离线使用。任务在内存队列串行执行（GPU 友好），
服务重启后历史任务列表不保留。

上传稳定性设计：分片上传接口对**所有**提前返回的错误分支（会话失效/格式
拒绝/未知路径/服务异常）都会排干请求体，避免残留字节污染 HTTP keep-alive
连接导致浏览器后续请求收到空响应（「Unexpected end of JSON input」）；
前端对每个分片自动重试 3 次，网络瞬断可自愈，全部失败才给出中文指引。
排查问题可加 `--verbose` 查看服务端请求日志。

## 使用模式一：本地调度器（CLI）

```bash
# 一键流水线：LaMa 去字幕（底部 78%~95% 区域）→ 2x 放大 → 封装音频
vidpure run --input D:/drama --clean lama --region 0.78,0.95 --scale 2 -r

# 快速通道（无 GPU / 赶时间）
vidpure run --input clip.mp4 --clean delogo --scale 2

# 多行字幕：分号分隔多段区域
vidpure run --input clip.mp4 --clean lama --region "0.78,0.87;0.89,0.95"

# N 卡硬编输出
vidpure run --input clip.mp4 --clean lama --scale 2 --vcodec h264_nvenc
```

输出在 `./vidpure_output/`；首次运行自动下载 NCNN 引擎（约 60MB）与模型，
缓存于 `~/.vidpure/`，之后完全离线可用。

### 把重活派给 ComfyUI（自动拉起 server）

```bash
# 指定 ComfyUI 安装目录，未运行则自动启动，跑完自动停止
vidpure comfy --input clip.mp4 --comfy-dir D:/ComfyUI --stage both
```

需要在 ComfyUI 内安装 `ComfyUI-VideoHelperSuite` 与 `ComfyUI-IOPaint`。

## 使用模式二：ComfyUI 节点

把 `comfy/vidpure_nodes/` 整个目录拷入 `ComfyUI/custom_nodes/`，
在 ComfyUI 的 python 环境里 `pip install iopaint`，重启即用：

| 节点 | 作用 |
|---|---|
| **VidPure 字幕清除 (LaMa)** | IMAGE(±MASK) → 逐帧修复 → IMAGE |
| **VidPure 区域 Mask** | region 字符串 → 矩形 MASK（白=重绘，`grow` 外扩防描边残留） |

推荐接线：

```
VHS Load Video ─image→ VidPure 区域 Mask ─mask→ VidPure 字幕清除 ─image→ VHS Video Combine
      └─audio────────────────────────────────────────────────────────────────────┘
```

字幕位置浮动时，用 OCR 类节点产出 MASK 接入 `mask` 口，替代固定区域。
详见 [comfy/vidpure_nodes/README.md](comfy/vidpure_nodes/README.md)。

## 硬件分档与方案对比

见 **[docs/hardware.md](docs/hardware.md)**：按显卡分档给出去字幕通道、放大引擎、
tile 与速度预估；含 8GB 显存（RTX 40/50 系）的推荐组合与 ProPainter 降配说明。

实测数据沉淀在 **[docs/benchmark.md](docs/benchmark.md)**，欢迎 PR 补充：

```bash
python scripts/benchmark.py --input sample.mp4 --clean lama --scale 2
```

## 项目结构

```
vidpure/
├── vidpure/                 # 主包（pip 安装体）
│   ├── cli.py               #   调度器命令行入口
│   ├── web/                 #   网页版（server.py + static/index.html）
│   ├── core/
│   │   ├── probe.py         #   硬件探测 / 后端决策 / RTX50 检测
│   │   ├── pipeline.py      #   clean → upscale 编排与断点续跑
│   │   ├── subtitle.py      #   delogo/blur/crop/lama 四通道
│   │   ├── upscale.py       #   NCNN Vulkan 引擎 + torch realesrgan
│   │   └── media.py         #   ffmpeg/ffprobe/抽帧/合帧/音频策略
│   └── backends/
│       ├── base.py          #   Backend 抽象
│       ├── local.py         #   本地算力后端
│       ├── comfy.py         #   ComfyUI server 自动拉起 + API 提交
│       └── workflows.py     #   API 工作流 JSON 模板
├── comfy/vidpure_nodes/     # ComfyUI 自定义节点（拷贝即用，自包含）
├── docs/                    # hardware.md / benchmark.md
├── scripts/                 # bootstrap.py / benchmark.py
└── tests/                   # pytest（不依赖 GPU）
```

## 开发与测试

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Roadmap

- [ ] OCR 自动检测字幕区域（PaddleOCR，浮动字幕免手框）
- [ ] ProPainter 通道接入（8GB 降配分窗）
- [ ] RIFE 补帧阶段（放大前补帧）
- [ ] 断点续跑缓存清单 manifest
- [ ] 英文 README

## 致谢

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）及其 NCNN Vulkan 版
- [IOPaint / lama-cleaner](https://github.com/Sanster/IOPaint)（Apache-2.0）与 [LaMa](https://github.com/advimman/lama)
- [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) 思路参考
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 与 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## License

MIT，见 [LICENSE](LICENSE)。第三方模型与组件许可在该文件末尾列明。
