# VidPure

**AI video hard-subtitle remover + quality upscaler — all-in-one open-source tool**

[中文说明](README.md) | English

`vidpure` is an **orchestrator**: it implements no AI models itself. It probes your
hardware, spins up the best local compute (FFmpeg / IOPaint-LaMa / Real-ESRGAN-NCNN /
ComfyUI server) automatically, and chains **subtitle removal → upscaling → audio muxing**
into a single command. It also ships **ComfyUI custom nodes** so you can embed
subtitle removal directly into your generation workflow.

- Zero third-party dependencies at the core (just FFmpeg for the fast lane)
- Default upscale engine: **Real-ESRGAN NCNN Vulkan** — no torch required, works on
  NVIDIA / AMD / Intel GPUs, models auto-downloaded on first run
- Orchestrator CLI & ComfyUI nodes share the same core
- Audio preserved losslessly (AAC/MP3 stream copy, others transcoded to AAC)

## Feature Scope

| Capability | Channel | Dependencies | Quality |
|---|---|---|---|
| Subtitle removal · fast | `delogo` / `blur` / `crop` | FFmpeg only | visible artifacts / reframed |
| Subtitle removal · inpaint | `lama` (IOPaint, per-frame) | `pip install iopaint` + GPU | near invisible |
| Upscaling | Real-ESRGAN (NCNN Vulkan) | engine & models auto-downloaded | best for anime / 3D-rendered |
| Batch orchestration | directory / file list / concurrency | — | — |
| ComfyUI nodes | subtitle cleaner + region mask | iopaint | same as above |

Explicitly out of scope: model training, cloud API wrappers, redistributing model
weights (downloaded from official releases at runtime).

## Architecture

```
                         ┌─────────────────────────────┐
   vidpure run (CLI) ───▶│        core/ orchestrator   │
   vidpure comfy ───────▶│  probe → pipeline → media   │
   ComfyUI nodes ───────▶└──────────┬──────────────────┘
                                    │ pick / spin up compute by hardware
            ┌───────────────┬───────┴────────┬──────────────┐
            ▼               ▼                ▼              ▼
     FFmpeg subprocess  IOPaint (LaMa)  Real-ESRGAN      ComfyUI server
     delogo/blur/       per-frame       NCNN Vulkan      (auto-launched via
     crop fast lane     inpainting      per-frame SR      /prompt workflow API)
```

## Install

```bash
# 0) Prerequisite: FFmpeg (winget install ffmpeg) + Python 3.9+
git clone https://github.com/Hejunlong27/VidPure.git && cd vidpure

# 1) Fast lane (zero deps, works immediately)
pip install -e .

# 2) LaMa subtitle removal (recommended)
pip install -e ".[gpu]"

# 3) Hardware self-check (prints recommended params & warnings, e.g. RTX 50 series needs cu128 torch)
vidpure probe
```

NVIDIA RTX 50 series (Blackwell) additionally requires `torch>=2.7 cu128`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Mode 0: Web UI (recommended for non-developers)

```bash
vidpure web            # opens http://127.0.0.1:8787 automatically
vidpure web --lan      # also reachable from phones on the same WiFi
```

Three-step guided UI: pick a video → confirm subtitle region → choose a mode.
Drag-to-select the subtitle area on the first frame (touch friendly), chunked
upload with progress, per-stage progress messages, inline result player with
Range streaming, one-click download, and fully localized Chinese error messages.
All processing stays on your machine.

## Mode 1: Local orchestrator (CLI)

```bash
# LaMa removal (bottom 78%~95% region) → 2x upscale → mux audio
vidpure run --input D:/drama --clean lama --region 0.78,0.95 --scale 2 -r

# Fast lane (no GPU / in a hurry)
vidpure run --input clip.mp4 --clean delogo --scale 2

# Multi-line subtitles: semicolon-separated regions
vidpure run --input clip.mp4 --clean lama --region "0.78,0.87;0.89,0.95"

# NVIDIA hardware encoding
vidpure run --input clip.mp4 --clean lama --scale 2 --vcodec h264_nvenc
```

Output goes to `./vidpure_output/`. The NCNN engine (~60MB) and models are
downloaded on first run into `~/.vidpure/`, fully offline afterwards.

### Delegate heavy work to ComfyUI (auto-launched server)

```bash
vidpure comfy --input clip.mp4 --comfy-dir D:/ComfyUI --stage both
```

Requires `ComfyUI-VideoHelperSuite` and `ComfyUI-IOPaint` inside ComfyUI.

## Mode 2: ComfyUI nodes

Copy `comfy/vidpure_nodes/` into `ComfyUI/custom_nodes/`, then in ComfyUI's
python environment: `pip install iopaint`. Restart and search for `VidPure`:

| Node | Purpose |
|---|---|
| **VidPure Subtitle Cleaner (LaMa)** | IMAGE(±MASK) → per-frame inpaint → IMAGE |
| **VidPure Region Mask** | region string → rectangular MASK (white = inpaint area) |

```
VHS Load Video ─image→ VidPure Region Mask ─mask→ VidPure Subtitle Cleaner ─image→ VHS Video Combine
      └─audio──────────────────────────────────────────────────────────────────────┘
```

For floating subtitles, feed an OCR-generated MASK into the `mask` input.
See [comfy/vidpure_nodes/README.md](comfy/vidpure_nodes/README.md).

## Hardware Tiers & Benchmarks

See **[docs/hardware.md](docs/hardware.md)** for per-GPU-tier channel/engine/tile
recommendations and timing estimates (including the 8GB VRAM sweet spot and
ProPainter fallback notes). Measured numbers live in
**[docs/benchmark.md](docs/benchmark.md)** — PRs welcome:

```bash
python scripts/benchmark.py --input sample.mp4 --clean lama --scale 2
```

## Project Layout

```
vidpure/
├── vidpure/                 # main package (pip installable)
│   ├── cli.py               #   orchestrator CLI entry
│   ├── web/                 #   web UI (server.py + static/index.html)
│   ├── core/
│   │   ├── probe.py         #   hardware probe / backend decision / RTX50 detection
│   │   ├── pipeline.py      #   clean → upscale staging
│   │   ├── subtitle.py      #   delogo/blur/crop/lama channels
│   │   ├── upscale.py       #   NCNN Vulkan engine + torch realesrgan
│   │   └── media.py         #   ffmpeg/ffprobe/frames/mux/audio policy
│   └── backends/
│       ├── base.py          #   Backend abstraction
│       ├── local.py         #   local compute backend
│       ├── comfy.py         #   ComfyUI server auto-launch + workflow submit
│       └── workflows.py     #   API workflow JSON templates
├── comfy/vidpure_nodes/     # ComfyUI custom nodes (self-contained, copy to install)
├── docs/                    # hardware.md / benchmark.md
├── scripts/                 # bootstrap.py / benchmark.py
└── tests/                   # pytest (no GPU required)
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Roadmap

- [ ] OCR-based subtitle region auto-detection (PaddleOCR, floating subtitles)
- [ ] ProPainter channel (windowed fallback for 8GB cards)
- [ ] RIFE frame interpolation stage (before upscaling)
- [x] English README
- [ ] Resumable run manifest

## Acknowledgements

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) and its NCNN Vulkan build
- [IOPaint / lama-cleaner](https://github.com/Sanster/IOPaint) (Apache-2.0) and [LaMa](https://github.com/advimman/lama)
- [video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover) for inspiration
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## License

MIT — see [LICENSE](LICENSE). Third-party model & component licenses are listed there.
