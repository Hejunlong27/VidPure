# VidPure

**AI 视频硬字幕批量清除 · 固定分辨率放大 · 台词识别与字幕烧录 —— 一体化开源工具**

[English](README_EN.md) | 中文

`vidpure` 是一个「调度器」：它自己不实现任何 AI 模型，而是探测你的硬件、
自动拉起最合适的本地算力（FFmpeg / IOPaint-LaMa / Real-ESRGAN-NCNN / whisper.cpp / CTC 强制对齐器），
把 **去原字幕 → 放大到固定分辨率 → 识别台词并烧录字幕** 串成一条流水线，
并支持 **一次提交最多 60 个视频（单个 ≤90 秒）** 的队列化批量处理。
同时提供 **ComfyUI 自定义节点**，可把去字幕能力直接嵌进你的出片工作流。

- 核心零第三方依赖（FFmpeg 外部二进制即可跑快速通道）
- 放大引擎默认 **Real-ESRGAN NCNN Vulkan**：不依赖 torch，N/A/I 显卡通吃，模型自动下载
- 输出支持**固定分辨率**（720p / 1080p / 1440p / 2048p / 2160p / 4096p / 自定义 WxH），不再只有 2x/4x 倍数
- 字幕烧录集成 `drama-subtitle-pipeline` 的能力：**台词识别（whisper）→ 字幕对齐（CTC 强制对齐 / 音频能量收紧）→ ASS 渲染 → 硬烧录**，
  支持**手动画框定位字幕区域**与**可调字体大小**
- 批量队列自带**并发控制**与**失败自动重试**（指数退避，单条失败不影响整批）
- 调度器模式 & ComfyUI 节点模式共享同一套核心逻辑
- 音频默认无损耗保留（AAC/MP3 直通，其他编码自动转 AAC）

---

## 功能范围

| 能力 | 通道 | 依赖 | 画质 |
|---|---|---|---|
| 去原字幕 · 快速 | `delogo` / `blur` / `crop` | 仅 FFmpeg | 有痕迹 / 改构图 |
| 去原字幕 · 修复 | `lama`（IOPaint 逐帧） | `pip install iopaint` + GPU | 接近无痕 |
| **放大 · 固定分辨率** | `720p` / `1080p` / `1440p` / `2048p` / `2160p` / `4096p` / `WxH` | 自动下载 NCNN 引擎与模型 | AI 超分 + lanczos 精确缩放 |
| **批量调度** | 队列 + 并发 + 失败重试 | — | 单批 ≤60 个 · 单个 ≤90s（可调） |
| **台词识别** | whisper.cpp（CJK 逐字时间戳） | `whisper-cli` + ggml 模型 | 识别文本与粗略时间 |
| **字幕对齐** | CTC 强制对齐 + 音频能量收紧 | `ctc-forced-aligner`（可选） | 逐条精修到实测语音 |
| **字幕烧录** | ASS 渲染（**6:4 视觉中心** / 手画框，随分辨率等比缩放）+ 可调字号 | 仅 FFmpeg（需 libass） | 硬字幕 |
| ComfyUI 节点 | 字幕清除 + 区域 Mask | iopaint | 同上 |

明确不做的事：不训练模型、不做云端 API 封装、不重分发模型权重（运行时按官方 release 下载）。

## 架构

```
                        ┌──────────────────────────────────────┐
  vidpure run (CLI) ───▶│            调度器 core/               │
  vidpure web ─────────▶│  probe → batch → resolve → pipeline   │
  ComfyUI 节点 ────────▶└──────────┬───────────────────────────┘
                                   │ 按硬件选择/拉起算力
     ┌────────────┬──────────────┬─┴───────────┬────────────────┬──────────────┐
     ▼            ▼              ▼             ▼                ▼              ▼
 FFmpeg 子进程  IOPaint(LaMa)  Real-ESRGAN  whisper.cpp  ctc-forced-    FFmpeg(libass)
 delogo/blur/   逐帧修复        NCNN Vulkan  台词识别       aligner        字幕烧录
 crop 快速通道                 逐帧超分      （-ml 1 逐字）  逐条强制对齐    subtitles=
```

`core/batch.py` 是新的编排中枢：队列 → 并发 worker → 失败重试 → 结果归集，
CLI 与 Web 共用同一条代码路径（behavior parity）。

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

### 字幕能力（可选，按需装）

| 组件 | 用途 | 缺省时的降级 |
|---|---|---|
| `whisper-cli` + ggml 模型 | 台词识别（必需，若要烧字幕） | 无字幕功能；或用 `--sub-srt` 提供现成字幕 |
| `ctc-forced-aligner` | 字幕时间轴精修（CTC 强制对齐） | 退回 whisper 时间戳（精度略低，功能不受影响） |

路径可用环境变量覆盖，无需改代码：

```bash
export VIDPURE_WHISPER_CLI="D:/whisper/whisper-cli.exe"
export VIDPURE_WHISPER_MODEL="D:/whisper/ggml-base.bin"
export VIDPURE_ALIGNER="D:/align/ctc-forced-aligner.exe"
```

`vidpure probe` 会打印这三项是否就绪。

## 使用模式零：网页版（非开发者推荐）

```bash
vidpure web                       # 自动打开浏览器 http://127.0.0.1:8787
vidpure web --concurrency 2 --retries 3   # 指定默认并发与重试
vidpure web --lan                 # 允许手机在同一 WiFi 下访问（打印局域网地址）
```

界面特性：

- **三步引导**：选视频（可多选）→ 设置 → 批量看板
- **多文件批量上传**：拖拽或点选多个视频，分片上传带进度；单批上限 60 个、单个 ≤90s
  超时长的条目会被**标记并跳过**（不影响其余条目）
- **固定分辨率下拉列表**：原始 / 720p / 1080p / 1440p / 2048p / 2160p / 4096p，
  界面实时算出「首个视频输出 ≈ W×H」，并对超大分辨率给出体量提示
- **可视化画框**：在首帧上按住拖动即可；可分别框选**原字幕区域**（橙色）与**字幕位置**（黄色）
- **字幕烧录面板**：开关 / 语言（中·西·英）/ 位置（底部 **6:4 视觉中心** 或 手动画框）/
  **字体大小滑块**（可自动或手填像素）/ 垂直摆位（等比缩放 或 固定像素）/ 时间轴精修策略
- **批量看板**：实时进度、每条的阶段中文提示、失败一键重试、外挂 `.srt` 下载
- **并发数与重试次数**可在界面直接调整
- 全中文错误提示（如「未检测到 whisper-cli：台词识别需要它」）

技术：后端为 Python 标准库 `http.server`（零第三方依赖），前端单文件页面、
无任何外部 CDN 依赖，可完全离线使用。

上传稳定性设计：分片上传接口对**所有**提前返回的错误分支（会话失效/格式
拒绝/未知路径/服务异常）都会排干请求体，避免残留字节污染 HTTP keep-alive
连接导致浏览器后续请求收到空响应（「Unexpected end of JSON input」）；
前端对每个分片自动重试 3 次。排查问题可加 `--verbose`。

## 使用模式一：本地调度器（CLI）

### 批量 + 固定分辨率

```bash
# 100 条素材目录 → 去原字幕 → 统一放大到 1080p → 并发 4、失败重试 2 次
vidpure run --input D:/drama --clean lama --region 0.78,0.95 --target-res 1080p \
            --jobs 4 --retries 2 -r

# 快速通道（无 GPU / 赶时间），输出统一 720p
vidpure run --input D:/clips --clean delogo --target-res 720p --jobs 8

# 精确指定尺寸（不写死预设）
vidpure run --input clip.mp4 --clean delogo --target-res 1920x1080

# 看有哪些预设
vidpure resolutions
```

**分辨率口径**：预设值 = **短边像素**（业界对 "1080p / 2160p" 的通用定义）。
横屏 1920×1080 → `1080p` 输出 1920×1080；竖屏 1080×1920 → `1080p` 输出 1080×1920。
长边按源宽高比**等比推算**并对齐到偶数（yuv420p 要求偶数）。

**AI 超分与精确尺寸的关系**：NCNN 引擎只支持整数倍（2/3/4），
所以实现是「选**最小可行**整数倍超分 → lanczos 缩到精确目标」，
且缩放与编码在**同一次**完成（不多编码一代）。目标 ≤ 源时跳过 AI，只做缩放。

### 字幕烧录

```bash
# 识别台词 → CTC 对齐 → 烧录（默认：底部居中，文字块视觉中心落在 6:4 处）
vidpure run --input clip.mp4 --clean delogo --target-res 1080p --burn-subtitle

# 固定距底像素（旧行为，不随分辨率缩放）
vidpure run --input clip.mp4 --burn-subtitle --sub-anchor pixel --sub-margin-bottom 400

# 调整分屏比例（7:3 更靠下）
vidpure run --input clip.mp4 --burn-subtitle --sub-anchor-top 0.7

# 手动画框定位字幕 + 自定义字号 64px
vidpure run --input clip.mp4 --clean delogo --burn-subtitle \
            --sub-mode box --sub-box 0.10,0.72,0.90,0.86 --font-size 64

# 用现成 SRT（跳过识别，仅对齐+渲染+烧录）
vidpure run --input clip.mp4 --burn-subtitle --sub-srt subs.srt --sub-lang zh

# 西班牙语 / 英语
vidpure run --input clip.mp4 --burn-subtitle --sub-lang es
```

**处理顺序铁律**：`去原字幕 → 放大 → 烧新字幕`。
烧字幕必然重编码；若先烧后放大，字幕会被 AI 超分当成画面纹理一起「重绘」
（描边发糊、笔画变形）。所以字幕一律烧在**已放大好**的母版上，
ASS 的 `PlayResX/Y` 自动等于母版真实分辨率。

**默认位置口径**：底部水平居中，文字块的**视觉中心**落在 **距顶 : 距底 = 6 : 4** 处，
即中心高度 = `0.6 × 画面高` —— **按分辨率等比缩放**，任何输出尺寸下视觉重心一致
（改用固定像素会在换分辨率时漂移：同一个 400px 在 1080p 上占 37% 高，在 2880p 上只占 14%）。

| 画布 | 视觉中心 y | 距底 |
|---|---|---|
| 1920×1080 | 648 | 432 |
| 1080×1920（竖屏） | 1152 | 768 |
| 2560×1440 | 864 | 576 |

实现用 `\an5\pos(W/2, 0.6H)`，`\an5` 以**文字块中心**为锚点，所以 1 行与 2 行字幕的
中心都落在同一条线上（不随行数浮动）。文本框宽度 ≤ 画面宽 70%。
仍可用 `--sub-anchor pixel --sub-margin-bottom N` 切回「固定距底像素」的旧行为
（矮画布会自动压低并告警，避免字幕被顶出画面）。

**字幕相关参数**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--burn-subtitle` | 开启字幕烧录 | 关 |
| `--sub-lang` | `zh` / `es` / `en`（断行算法、字体、对齐器语言码按语言分派） | `zh` |
| `--sub-srt` | 用现成 SRT，跳过台词识别 | — |
| `--sub-mode` | `bottom` 底部（6:4 视觉中心）/ `box` 手画框 | `bottom` |
| `--sub-anchor` | `ratio` 视觉中心按 6:4 等比缩放 / `pixel` 固定距底像素 | `ratio` |
| `--sub-anchor-top` | ratio 模式下视觉中心距顶占比（`0.6` = 6:4） | `0.6` |
| `--sub-box` | 手画框（归一化 `x0,y0,x1,y1`），配 `--sub-mode box` | — |
| `--font-size` | 字号（像素，相对**输出**画布）；省略=自动 | 自动 |
| `--font-pct` | 自动字号占画布高度百分比 | `5.6` |
| `--sub-margin-bottom` | pixel 模式距底像素（给出该参数即启用 pixel） | `400` |
| `--sub-refine` | `auto` / `ctc` / `none`（时间轴精修策略） | `auto` |
| `--drop-master` | 烧录成功后删除无字幕母版 | 保留母版 |

### 批量队列参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--jobs` / `-j` | 并发数。GPU 逐帧修复/超分建议 **1**；快速通道可调到 CPU 核数 | 1 |
| `--retries` | 失败自动重试次数（指数退避，退避期间释放并发槽） | 2 |
| `--max-items` | 单批上限 | 60 |
| `--max-duration` | 单条时长上限（秒），超限条目跳过而非报错 | 90 |
| `--no-duration-limit` | 关闭时长限制（只告警） | — |

其它常用参数（`--clean` / `--region` / `--region-format` / `--vcodec` / `--crf` /
`--preset` / `--upscale-model` / `--ext` / `-r`）沿用原口径。
`--scale 2/4` 作为旧式倍数保留，但会被 `--target-res` 取代。

输出结构（每个 job 一个目录）：

```
<out>/results/<job_id>/
  output.mp4           成片（已去原字幕 / 已放大 / 已烧字幕）← 交付这个
  <stem>_master.mp4    烧字幕前的无字幕母版（默认保留，--drop-master 可关）
  subs/<stem>.ass      字幕文件（可导入剪辑软件再改）
  subs/<stem>.srt      外挂字幕（与画面同源）
  subs/<stem>.asr.srt  whisper 原始转写（供复核）
  report.json          运行报告（画布 / 字号 / 安全区 / 收紧条数 / CTC 结果）
```

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

详见 [comfy/vidpure_nodes/README.md](comfy/vidpure_nodes/README.md)。

## 硬件分档与方案对比

见 **[docs/hardware.md](docs/hardware.md)**：按显卡分档给出去字幕通道、放大引擎、
tile 与速度预估。

实测数据沉淀在 **[docs/benchmark.md](docs/benchmark.md)**：

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
│   │   ├── resolve.py       #   ★ 固定分辨率预设与目标推算
│   │   ├── batch.py         #   ★ 批量队列 / 并发 / 失败重试 / 阶段编排
│   │   ├── pipeline.py      #   旧版单机流水线（保留兼容）
│   │   ├── subtitle.py      #   delogo/blur/crop/lama 四通道（去原字幕）
│   │   ├── subtitle_burn.py #   ★ 台词识别 / 字幕对齐 / ASS 渲染 / 烧录
│   │   ├── upscale.py       #   NCNN Vulkan 引擎（支持固定分辨率目标）
│   │   └── media.py         #   ffmpeg/ffprobe/抽帧/合帧/音频策略
│   └── backends/
│       ├── base.py          #   Backend 抽象
│       ├── local.py         #   本地算力后端
│       ├── comfy.py         #   ComfyUI server 自动拉起 + API 提交
│       └── workflows.py     #   API 工作流 JSON 模板
├── comfy/vidpure_nodes/     # ComfyUI 自定义节点（拷贝即用，自包含）
├── docs/                    # hardware.md / benchmark.md
├── scripts/                 # bootstrap.py / benchmark.py
└── tests/                   # pytest（不依赖 GPU）：resolve / batch / subtitle_burn / media
```

## 开发与测试

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

当前测试覆盖：分辨率预设推算、断行不越界、字号可调、**6:4 视觉中心按分辨率等比缩放**、
ASS 落点回读断言、手画框定位、时间轴单调不重叠、SRT 解析与导出、批量上限校验、
手画框定位、时间轴单调不重叠、SRT 解析与导出、批量上限校验、
失败重试与「单条失败不影响整批」。

## Roadmap

- [ ] OCR 自动检测字幕区域（PaddleOCR，浮动字幕免手框）
- [ ] ProPainter 通道接入（8GB 降配分窗）
- [ ] RIFE 补帧阶段（放大前补帧）
- [ ] 批量任务清单持久化 + 断点续跑（当前 `persist()` 已可手动快照）
- [ ] 双语同屏（两个 Style + 两条 Dialogue）

## 致谢

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause）及其 NCNN Vulkan 版
- [IOPaint / lama-cleaner](https://github.com/Sanster/IOPaint)（Apache-2.0）与 [LaMa](https://github.com/advimman/lama)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)（MIT）
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)（MIT）
- [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) 思路参考
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 与 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## License

本项目以 **Apache License 2.0** 发布，完整条款见 [LICENSE](LICENSE)，
第三方组件与模型权重的归属声明见 [NOTICE](NOTICE)。

第三方组件各自遵循其上游许可（见上方「致谢」）：
Real-ESRGAN（BSD-3-Clause）、IOPaint / LaMa（Apache-2.0）、
whisper.cpp（MIT）、ctc-forced-aligner（MIT）。
模型权重**不随本仓库分发**，由运行时按各上游官方 release 自行下载。
