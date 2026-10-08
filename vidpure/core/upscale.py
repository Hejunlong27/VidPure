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


def upscale_video_ncnn(src: Path, dst: Path, scale: int = 2, model: str = "realesr-animevideov3",
                       tile: int = 256, progress=print) -> Path:
    """视频放大（NCNN Vulkan）：拆帧 → 逐帧超分 → 合帧封装。"""
    exe = ensure_ncnn_engine(progress)
    mdir = _ncnn_model_dir(progress)
    ncnn_name = _ncnn_model_name(model, scale)
    if not (mdir / f"{ncnn_name}.bin").exists():
        raise FileNotFoundError(f"模型缺失: {mdir}/{ncnn_name}.bin")
    info = media.ffprobe(src)
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
        media.mux(done, src, dst, info["fps"])
        return dst
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
