# VidPure

**AI video hard-subtitle batch remover · fixed-resolution upscaler · dialogue recognition & subtitle burning — all-in-one open-source tool**

[中文说明](README.md) | English

`vidpure` is an **orchestrator**: it implements no AI models itself. It probes your
hardware, spins up the best local compute (FFmpeg / IOPaint-LaMa / Real-ESRGAN-NCNN /
whisper.cpp / ctc-forced-aligner) automatically, and chains
**subtitle removal → fixed-resolution upscaling → dialogue recognition & subtitle burning**
into a single pipeline, with a queue that accepts **up to 60 clips (≤90 s each) per batch**.
It also ships **ComfyUI custom nodes** so you can embed subtitle removal directly into
your generation workflow.

- Zero third-party dependencies at the core (just FFmpeg for the fast lane)
- Default upscale engine: **Real-ESRGAN NCNN Vulkan** — no torch required, works on
  NVIDIA / AMD / Intel GPUs, models auto-downloaded on first run
- **Fixed-resolution output** (720p / 1080p / 1440p / 2048p / 2160p / 4096p / custom WxH)
  instead of bare 2x/4x multipliers
- Subtitle burning built on the `drama-subtitle-pipeline` engine:
  **transcribe (whisper) → align (CTC forced alignment + audio-energy tightening) →
  render ASS → burn**, with the text block's **visual centre anchored at the 6:4 line**
  (60 % of frame height, resolution-proportional), **manual box selection**, and an
  **adjustable font size**
- Batch queue with **configurable concurrency** and **automatic retry** (exponential
  backoff; one failing item never blocks the batch)
- Orchestrator CLI & ComfyUI nodes share the same core
- Audio preserved losslessly (AAC/MP3 stream copy, others transcoded to AAC)

## Feature Scope

| Capability | Channel | Dependencies | Quality |
|---|---|---|---|
| Subtitle removal · fast | `delogo` / `blur` / `crop` | FFmpeg only | visible artifacts / reframed |
| Subtitle removal · inpaint | `lama` (IOPaint, per-frame) | `pip install iopaint` + GPU | near invisible |
| **Upscaling · fixed resolution** | `720p` / `1080p` / `1440p` / `2048p` / `2160p` / `4096p` / `WxH` | engine & models auto-downloaded | AI super-res + exact lanczos resize |
| **Batch scheduling** | queue + concurrency + retry | — | ≤60 clips · ≤90 s each (adjustable) |
| **Dialogue recognition** | whisper.cpp (per-char timestamps for CJK) | `whisper-cli` + ggml model | text + coarse timing |
| **Subtitle alignment** | CTC forced alignment + audio-energy tightening | `ctc-forced-aligner` (optional) | per-cue, matched to real speech |
| **Subtitle burning** | ASS rendering (**6:4 visual centre**, resolution-proportional / **manual box**) + adjustable font size | FFmpeg with libass | hard subtitles |
| ComfyUI nodes | subtitle cleaner + region mask | iopaint | same as above |

Explicitly out of scope: model training, cloud API wrappers, redistributing model
weights (downloaded from official releases at runtime).

## Fixed-resolution output, batching and subtitles

```bash
# Batch: remove original subtitles, uniform 1080p output, 4 workers, 2 retries
vidpure run --input D:/drama --clean lama --region 0.78,0.95 --target-res 1080p \
            --jobs 4 --retries 2 -r

# Fixed output size, then transcribe + burn subtitles
# (default: the text block's visual centre sits at 60% of the frame height,
#  i.e. top-gap : bottom-gap = 6 : 4, scaled with the resolution)
vidpure run --input clip.mp4 --clean delogo --target-res 1080p --burn-subtitle

# Legacy fixed bottom margin (does not scale with resolution)
vidpure run --input clip.mp4 --burn-subtitle --sub-anchor pixel --sub-margin-bottom 400

# Other split ratios, e.g. 7:3 (lower)
vidpure run --input clip.mp4 --burn-subtitle --sub-anchor-top 0.7

# Manual box for subtitle placement + custom 64 px font
vidpure run --input clip.mp4 --clean delogo --burn-subtitle \
            --sub-mode box --sub-box 0.10,0.72,0.90,0.86 --font-size 64

vidpure resolutions        # list the fixed-resolution presets
```

**Subtitle placement semantics**: by default the vertical anchor is **resolution-proportional** —
the text block's visual centre is placed at `0.6 × height`, giving top-gap : bottom-gap = 6 : 4
on landscape *and* portrait. This is implemented with `\an5\pos(W/2, 0.6H)`, so 1-line and
2-line cues share the same centre line. A fixed pixel margin drifts when the resolution changes
(the same 400 px is 37 % of the height at 1080p but only 14 % at 2880p), so it is no longer the
default — pass `--sub-anchor pixel` to opt back in.

**Resolution semantics**: a preset is the **short-edge pixel count** (the common meaning
of "1080p/2160p"). Landscape 1920×1080 → `1080p` gives 1920×1080; portrait 1080×1920 →
`1080p` gives 1080×1920. The long edge is derived from the source aspect ratio and snapped
to an even number (required by yuv420p).

**Why AI + resize**: the NCNN engine only supports integer factors (2/3/4), so the
implementation picks the *smallest sufficient* integer factor, then does a lanczos resize
to the exact target — combined with encoding in a single pass. Targets ≤ source skip the AI
step entirely.

**Ordering rule**: `remove original subtitles → upscale → burn new subtitles`.
Burning always re-encodes; burning *before* upscaling would let the AI super-res redraw
your subtitles as if they were picture texture. `PlayResX/Y` is set to the real master
resolution automatically.

**Batch limits**: 60 items per submission and 90 s per clip by default, both configurable
(`--max-items`, `--max-duration`, `--no-duration-limit`). Over-length items are marked and
skipped instead of failing the whole batch.

Per-job output layout:

```
<out>/results/<job_id>/
  output.mp4           final (cleaned / upscaled / subtitled)
  <stem>_master.mp4    no-subtitle master (kept unless --drop-master)
  subs/<stem>.ass      subtitle file
  subs/<stem>.srt      sidecar SRT
  subs/<stem>.asr.srt  raw whisper transcript
  report.json          canvas / font / safe-area / tightening / CTC report
```

## Architecture

```
                         ┌──────────────────────────────────────┐
   vidpure run (CLI) ───▶│            core/ orchestrator        │
   vidpure web ─────────▶│  probe → batch → resolve → pipeline   │
   ComfyUI nodes ───────▶└──────────┬───────────────────────────┘
                                    │ pick / spin up compute by hardware
   ┌────────────┬──────────────┬────┴────────┬────────────────┬──────────────┐
   ▼            ▼              ▼             ▼                ▼              ▼
 FFmpeg      IOPaint        Real-ESRGAN   whisper.cpp   ctc-forced-   FFmpeg(libass)
 delogo/     (LaMa)         NCNN Vulkan   transcribe     aligner       burn subtitles
 blur/crop   per-frame      per-frame     (-ml 1)       per-cue CTC   subtitles=
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

Released under the **Apache License 2.0** — see [LICENSE](LICENSE) for the full text,
and [NOTICE](NOTICE) for third-party component and model-weight attribution.

Third-party components keep their own upstream licenses (see Acknowledgements above):
Real-ESRGAN (BSD-3-Clause), IOPaint / LaMa (Apache-2.0), whisper.cpp (MIT),
ctc-forced-aligner (MIT). Model weights are **not** bundled with this repository —
they are downloaded at runtime from each upstream's official release.
