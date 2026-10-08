"""流水线编排：字幕清除 → 放大 → 封装，三阶段可独立启用、断点续跑。

vidpure 作为「调度器」只负责：探测硬件 → 选择后端 → 按序拉起各阶段算力
（FFmpeg / IOPaint / NCNN Vulkan / ComfyUI server），不绑定任何具体模型实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import subtitle, upscale
from .probe import HardwareReport, probe


@dataclass
class StageResult:
    stage: str
    ok: bool
    seconds: float
    detail: str = ""


@dataclass
class PipelineResult:
    src: str
    dst: str
    stages: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"{s.stage}: {'OK' if s.ok else 'FAIL'} {s.seconds:.1f}s {s.detail}" for s in self.stages]
        return "\n".join(lines)


def _collect_inputs(inp: list[str], recursive: bool, exts: str) -> list[Path]:
    ext_set = {e if e.startswith(".") else "." + e for e in exts.split(",") if e.strip()}
    files: list[Path] = []
    for item in inp or []:
        p = Path(item.strip('"'))
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += [f for f in it if f.suffix.lower() in ext_set]
        elif p.exists():
            files.append(p)
        else:
            print(f"[警告] 输入不存在: {item}")
    seen, uniq = set(), []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return sorted(uniq)


def run_pipeline(input_items: list[str], out_dir: Path,
                 clean: str | None = "lama", clean_region: str = "0.78,0.95",
                 region_fmt: str = "auto",
                 upscale_scale: int | None = 2, upscale_model: str = "realesr-animevideov3",
                 device: str = "cuda", hw: HardwareReport | None = None,
                 recursive: bool = False, exts: str = "mp4,mov,mkv,webm,m4v",
                 jobs: int = 2, vcodec: str = "libx264", crf: int = 18,
                 preset: str = "medium", mute: bool = False,
                 keep_workdir: bool = False) -> list[PipelineResult]:
    """主流程：对每个输入视频执行 clean → upscale 链。

    clean=None 跳过去字幕；upscale_scale=None 跳过放大。
    """
    hw = hw or probe()
    rec = hw.recommendations
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_inputs(input_items, recursive, exts)
    if not files:
        raise SystemExit("未找到输入视频")

    print(f"硬件: {hw.gpu_name or 'CPU'} "
          f"{hw.vram_gb:.0f}GB | 推荐 clean={clean or rec.get('subtitle_mode')} "
          f"upscale={upscale_scale}x tile={rec.get('upscale_tile', 256)}")
    results: list[PipelineResult] = []

    for f in files:
        res = PipelineResult(src=str(f), dst="")
        work = out_dir / f"_work_{f.stem}"
        work.mkdir(parents=True, exist_ok=True)

        # ---- 阶段 1: 字幕清除 ----
        cleaned = f
        if clean and clean != "none":
            t0 = time.time()
            mid = work / f"{f.stem}_cleaned.mp4"
            try:
                if clean in ("delogo", "blur", "crop"):
                    subtitle.clean_video_fast(f, mid, clean_region, mode=clean,
                                              region_fmt=region_fmt, vcodec=vcodec,
                                              crf=crf, preset=preset, mute=mute)
                    detail = "ffmpeg 快速通道"
                else:
                    info = subtitle.clean_video_lama(f, mid, clean_region, region_fmt,
                                                     device=device, vcodec=vcodec,
                                                     crf=crf, preset=preset, mute=mute)
                    detail = f"LaMa {info['frames']} 帧"
                cleaned = mid
                res.stages.append(StageResult("clean", True, time.time() - t0, detail))
            except Exception as e:
                res.stages.append(StageResult("clean", False, time.time() - t0, str(e)[:200]))
                results.append(res)
                continue

        # ---- 阶段 2: 放大 ----
        final = cleaned
        if upscale_scale:
            t0 = time.time()
            final = work / f"{f.stem}_x{upscale_scale}.mp4"
            try:
                upscale.upscale_video_ncnn(cleaned, final, scale=upscale_scale,
                                           model=upscale_model,
                                           tile=rec.get("upscale_tile", 256))
                res.stages.append(StageResult("upscale", True, time.time() - t0,
                                              f"{upscale_model} x{upscale_scale}"))
            except Exception as e:
                res.stages.append(StageResult("upscale", False, time.time() - t0, str(e)[:200]))
                final = cleaned  # 放大失败时保留已去字幕版本

        # ---- 归位输出 ----
        dst = out_dir / f"{f.stem}_vidpure{final.suffix}"
        if final != dst:
            if final == f:  # 两阶段均未产出中间文件：复制而非移动，保护源文件
                import shutil
                shutil.copyfile(final, dst)
            else:
                final.replace(dst)
        res.dst = str(dst)
        results.append(res)
        print(f"[完成] {f.name} -> {dst.name}")
        for s in res.stages:
            print(f"    {s.stage}: {'OK' if s.ok else 'FAIL'} {s.seconds:.1f}s {s.detail}")
        if not keep_workdir:
            import shutil
            shutil.rmtree(work, ignore_errors=True)

    return results
