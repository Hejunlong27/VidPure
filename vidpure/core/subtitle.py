"""字幕清除通道：delogo / blur（FFmpeg 快速通道）与 lama（IOPaint 逐帧修复）。

设计为「帧级」与「视频级」双层 API：
  clean_video()  面向调度器的整段视频处理
  clean_frame()  面向 ComfyUI 节点的单帧/批帧处理（comfy_nodes 复用）
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import media


# ---------------------------------------------------------------- 快速通道（纯 FFmpeg）
def clean_video_fast(src: Path, dst: Path, region: str, mode: str = "delogo",
                     region_fmt: str = "auto", blur_sigma: int = 24,
                     vcodec: str = "libx264", crf: int = 18, preset: str = "medium",
                     mute: bool = False) -> Path:
    info = media.ffprobe(src)
    rects = media.parse_region(region, region_fmt, info["w"], info["h"])
    key, filt = media.build_clean_filter(mode, rects, info["w"], info["h"], blur_sigma=blur_sigma)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if key == "filter_complex":
        cmd += ["-filter_complex", filt, "-map", "[vout]"]
    else:
        cmd += ["-vf", filt, "-map", "0:v:0"]
    if info["has_audio"] and not mute:
        if dst.suffix in (".mp4", ".m4v") and info["audio_codec"] in media.MP4_OK_AUDIO:
            cmd += ["-map", "0:a:0?", "-c:a", "copy"]
        else:
            cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    media.run(cmd)
    return dst


# ---------------------------------------------------------------- LaMa 逐帧修复
def _iopaint_cmd() -> list:
    exe = shutil.which("iopaint")
    if exe:
        return [exe]
    import sys
    return [sys.executable, "-m", "iopaint"]


def clean_frame(image, mask, device: str = "cuda", model: str = "lama"):
    """单帧修复（ComfyUI 节点入口）。image/mask: PIL.Image（mask 白=重绘）。返回 np.ndarray HxWx3。"""
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy
    except ImportError as e:
        raise RuntimeError("需要 IOPaint：pip install iopaint") from e
    global _MM_CACHE
    try:
        mm = _MM_CACHE
    except NameError:
        mm = _MM_CACHE = ModelManager(name=model, device=device)
    return mm(image, mask,
              hd_strategy=HDStrategy.CROP,
              hd_strategy_crop_margin=64,
              hd_strategy_crop_trigger_size=800,
              hd_strategy_resize_limit=1600)


def clean_video_lama(src: Path, dst: Path, region: str, region_fmt: str = "auto",
                     device: str = "cuda", keep_tmp: bool = False, tmp_root: str | None = None,
                     vcodec: str = "libx264", crf: int = 18, preset: str = "medium",
                     mute: bool = False) -> dict:
    """拆帧 → 区域 mask → IOPaint 批量修复 → 合帧封装。"""
    info = media.ffprobe(src)
    rects = media.parse_region(region, region_fmt, info["w"], info["h"])
    tmp = Path(tempfile.mkdtemp(prefix=f"vidpure_{src.stem}_",
                                dir=tmp_root or tempfile.gettempdir()))
    frames, masks, clean = tmp / "frames", tmp / "masks", tmp / "clean"
    for d in (frames, masks, clean):
        d.mkdir(parents=True)
    try:
        n = media.extract_frames(src, frames, "png")
        if n == 0:
            raise RuntimeError("抽帧失败")
        mask_png = tmp / "mask.png"
        media.make_mask_png(info["w"], info["h"], rects, mask_png)
        for i in range(1, n + 1):
            shutil.copyfile(mask_png, masks / f"f{i:06d}.png")
        media.run(_iopaint_cmd() + ["run", "--model", "lama", "--device", device,
                                    "--image", str(frames), "--mask", str(masks),
                                    "--output", str(clean)])
        if not any(clean.glob("f*.png")):
            raise RuntimeError("IOPaint 未产出修复帧（检查 pip install iopaint / device）")
        dst.parent.mkdir(parents=True, exist_ok=True)
        media.mux(clean, src, dst, info["fps"], vcodec, crf, preset, mute)
        return {"frames": n, "rects": rects}
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def clean_video(src: Path, dst: Path, mode: str = "lama", **kw) -> dict | Path:
    """统一入口。mode: delogo / blur / lama（crop 请用 clean_video_fast 的 crop）。"""
    if mode in ("delogo", "blur", "crop"):
        return clean_video_fast(src, dst, kw.pop("region"), mode=mode, **kw)
    if mode == "lama":
        return clean_video_lama(src, dst, kw.pop("region"), **kw)
    raise ValueError(mode)


def ffprobe_video(src: Path) -> dict:  # 对外便捷转发
    return media.ffprobe(src)


def _ffmpeg_ok() -> bool:  # 供 CLI 预检
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False
