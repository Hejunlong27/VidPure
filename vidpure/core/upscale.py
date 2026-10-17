"""视频放大引擎：Real-ESRGAN（NCNN Vulkan 版，主引擎）+ torch realesrgan（可选）。

NCNN Vulkan 版不依赖 torch，N/A/I 三家显卡通吃，正好作为「自动拉起本地算力」
的默认放大引擎：首次运行自动下载引擎与模型到 ~/.vidpure/，之后离线可用。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .. import APP_DIR_NAME
from . import media

HOME_CACHE = Path.home() / APP_DIR_NAME
ENGINE_DIR = HOME_CACHE / "engines"
MODEL_DIR = HOME_CACHE / "models"

NCNN_RELEASE_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                    "realesrgan-ncnn-vulkan-20220424-windows.zip")
MODELS = {
    # name -> (url, 适用场景)
    "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-x4plus.param.zip",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.zip",
}


def ensure_ncnn_engine(progress=print) -> Path:
    """确保 real-esrgan-ncnn-vulkan 可执行文件就绪，返回其路径（兼容包内有无子目录层）。"""
    if platform.system() != "Windows":
        raise RuntimeError("请从 https://github.com/xinntao/Real-ESRGAN/releases 下载对应平台 NCNN 包并解压至 "
                           f"{ENGINE_DIR}")
    exe = ENGINE_DIR / "realesrgan-ncnn-vulkan.exe"
    if not exe.exists():
        progress("[下载] Real-ESRGAN NCNN Vulkan 引擎（约 60MB，仅首次）...")
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = ENGINE_DIR / "ncnn.zip"
        urllib.request.urlretrieve(NCNN_RELEASE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(ENGINE_DIR)  # 官方 zip 直接平铺（exe + models/ + dll）
        zip_path.unlink(missing_ok=True)
        candidates = list(ENGINE_DIR.rglob("realesrgan-ncnn-vulkan.exe"))
        if not candidates:
            raise RuntimeError(f"引擎解压后未找到可执行文件（{ENGINE_DIR}）")
        exe = candidates[0]
    return exe


def _ncnn_model_dir(progress=print) -> Path:
    """模型目录：优先引擎包自带 models/（含 animevideov3-x2/3/4 与 x4plus 全家）。"""
    bundled = ENGINE_DIR / "models"
    if bundled.exists() and any(bundled.glob("*.bin")):
        return bundled
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not any(MODEL_DIR.glob("*.bin")):
        progress("[下载] 模型（仅首次）...")
        for name, url in MODELS.items():
            zip_path = MODEL_DIR / f"{name}.zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(MODEL_DIR)
            zip_path.unlink(missing_ok=True)
    return MODEL_DIR


def _ncnn_model_name(model: str, scale: int) -> str:
    """NCNN 模型名带刻度后缀：realesr-animevideov3-x2/x3/x4。"""
    if model == "realesr-animevideov3" and scale in (2, 3, 4):
        return f"realesr-animevideov3-x{scale}"
    return model


# 各模型支持的**整数** AI 放大倍率（NCNN 版只提供这几档模型文件）
MODEL_SCALES = {
    "realesr-animevideov3": (2, 3, 4),
    "realesrgan-x4plus": (4,),
}


def plan_scale(src_w: int, src_h: int, target_w: int, target_h: int,
               model: str = "realesr-animevideov3") -> tuple[int, bool]:
    """给定源尺寸与**固定目标分辨率**，算出该用几倍 AI 超分、是否需要补一次精确缩放。

    为什么需要这一步：AI 引擎只认整数倍（2/3/4），而交付要的是**固定尺寸**
    （如 1080p / 2048p）。所以流程是「整数倍 AI 超分到 ≥ 目标 → lanczos 缩到精确目标」。
    取**最小可行倍率**（够用就好）以省显存与时间；目标小于等于源时不跑 AI。

    返回 `(n, need_resize)`：`n=1` 表示不做 AI 超分（仅缩放）。
    """
    if target_w <= 0 or target_h <= 0:
        raise ValueError("目标分辨率无效")
    if src_w >= src_h:
        need = max(target_w / src_w, target_h / src_h)
    else:
        need = max(target_w / src_w, target_h / src_h)
    allowed = MODEL_SCALES.get(model, (2, 3, 4))
    if need <= 1.0:
        return 1, (target_w, target_h) != (src_w, src_h)
    n = next((c for c in allowed if c >= need - 1e-9), allowed[-1])
    # AI 放大后的实际尺寸（NCNN 逐帧超分严格按 n 倍）
    up_w, up_h = src_w * n, src_h * n
    need_resize = (up_w, up_h) != (target_w, target_h)
    return n, need_resize


def auto_tile(target_w: int, target_h: int, base_tile: int = 256) -> int:
    """按目标像素量自动收敛 tile，降低大分辨率下的显存峰值（8GB 卡更稳）。"""
    mp = (target_w * target_h) / 1_000_000.0
    if mp >= 16:
        return min(base_tile, 96)
    if mp >= 9:
        return min(base_tile, 128)
    if mp >= 6:
        return min(base_tile, 192)
    return base_tile


def resize_video_ffmpeg(src: Path, dst: Path, w: int, h: int, vcodec: str = "libx264",
                        crf: int = 18, preset: str = "medium", mute: bool = False,
                        progress=print) -> Path:
    """纯 FFmpeg 精确缩放（不做 AI 超分）。用于目标 ≤ 源，或倍率为 1 的场景。"""
    info = media.ffprobe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={w}:{h}:flags=lanczos"
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vf", vf]
    if info["has_audio"] and not mute:
        if dst.suffix in (".mp4", ".m4v") and info["audio_codec"] in media.MP4_OK_AUDIO:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    progress(f"[缩放] {info['w']}x{info['h']} -> {w}x{h}（lanczos，不做 AI 超分）")
    media.run(cmd)
    return dst


def upscale_video_ncnn(src: Path, dst: Path, scale: int = 2, model: str = "realesr-animevideov3",
                       tile: int = 256, target: tuple[int, int] | None = None,
                       vcodec: str = "libx264", crf: int = 18, preset: str = "medium",
                       mute: bool = False, progress=print) -> Path:
    """视频放大（NCNN Vulkan）：拆帧 → 逐帧超分 → 合帧封装。

    `target=(W,H)`：输出**固定分辨率**。此时 `scale` 被忽略，改由
    `plan_scale()` 选出最小可行整数倍率，再补一次 lanczos 精确缩放（与编码同一次完成）。
    `target=None`：沿用旧行为，按 `scale` 整数倍放大、不改变目标尺寸。
    """
    info = media.ffprobe(src)
    src_w, src_h = info["w"], info["h"]
    vf: str | None = None

    if target:
        tw, th = target
        n, need_resize = plan_scale(src_w, src_h, tw, th, model)
        if n == 1 and not need_resize:
            # 目标 == 源分辨率：不做任何处理，直接流复制（无损、瞬时）
            progress(f"[放大] 目标 {tw}x{th} 等于源分辨率，跳过缩放")
            return _stream_copy(src, dst)
        if n == 1:
            return resize_video_ffmpeg(src, dst, tw, th, vcodec=vcodec, crf=crf,
                                       preset=preset, mute=mute, progress=progress)
        scale = n
        if need_resize:
            vf = f"scale={tw}:{th}:flags=lanczos"
        tile = auto_tile(src_w * n, src_h * n, tile)
        progress(f"[放大] 目标 {tw}x{th} → AI x{n} → " +
                 (f"缩放到 {tw}x{th}" if need_resize else "尺寸已精确吻合"))

    exe = ensure_ncnn_engine(progress)
    mdir = _ncnn_model_dir(progress)
    ncnn_name = _ncnn_model_name(model, scale)
    if not (mdir / f"{ncnn_name}.bin").exists():
        raise FileNotFoundError(f"模型缺失: {mdir}/{ncnn_name}.bin")
    tmp = Path(tempfile.mkdtemp(prefix=f"vidpure_up_{src.stem}_"))
    frames, done = tmp / "in", tmp / "out"
    frames.mkdir(), done.mkdir()
    try:
        progress(f"[放大] {ncnn_name} tile={tile}")
        media.extract_frames(src, frames, "png")
        media.run([str(exe), "-i", str(frames), "-o", str(done),
                   "-n", ncnn_name, "-s", str(scale), "-t", str(tile),
                   "-f", "png", "-m", str(mdir)])
        n_out = len(list(done.glob("*.png")))
        if n_out == 0:
            raise RuntimeError("NCNN 超分未产出图像（检查显卡 Vulkan 驱动）")
        dst.parent.mkdir(parents=True, exist_ok=True)
        media.mux(done, src, dst, info["fps"], vcodec=vcodec, crf=crf,
                  preset=preset, mute=mute, vf=vf)
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stream_copy(src: Path, dst: Path) -> Path:
    """无损流复制（目标分辨率与源一致时用，避免无谓重编码）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    media.run(["ffmpeg", "-y", "-i", str(src), "-c", "copy",
               "-movflags", "+faststart", str(dst)])
    return dst


def upscale_frames_torch(images, scale: int = 2, model: str = "realesr-animevideov3",
                         tile: int = 256, device: str = "cuda"):
    """torch 路线（ComfyUI 节点用）：输入/输出均为 list[PIL.Image]。需 pip install realesrgan。

    注意：basicsr 1.4.2 与 torchvision>=0.17 存在 functional_tensor 兼容问题，
    若报 ImportError 请降级 torchvision 或改用 NCNN 引擎。
    """
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import numpy as np
        import torch
    except ImportError as e:
        raise RuntimeError("需要 pip install realesrgan basicsr（或改用 NCNN 引擎）") from e

    arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                   num_grow_ch=32, scale=4)
    weight_dir = MODEL_DIR / "torch"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight = weight_dir / f"{model}.pth"
    if not weight.exists():
        urls = {
            "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
            "realesrgan-x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        }
        urllib.request.urlretrieve(urls[model], weight)
    ups = RealESRGANer(scale=4, model_path=str(weight), model=arch,
                       tile=tile, tile_pad=10, pre_pad=0, half=(device != "cpu"),
                       device=torch.device(device))
    outs = []
    for img in images:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
        out, _ = ups.enhance(arr, outscale=scale)
        outs.append(__import__("PIL.Image", fromlist=["Image"]).fromarray(out[:, :, ::-1]))
    return outs


def upscale_video(src: Path, dst: Path, engine: str = "ncnn", **kw) -> Path:
    if engine == "ncnn":
        return upscale_video_ncnn(src, dst, **kw)
    raise ValueError(f"未知引擎: {engine}")
