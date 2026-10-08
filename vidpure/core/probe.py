"""硬件探测与算力后端决策。

探测内容：操作系统 / NVIDIA GPU 型号与显存 / torch CUDA 可用性 / FFmpeg。
决策输出：推荐的去字幕通道、放大引擎、tile 大小、并发数，以及
关键警告（如 RTX 50 系需要 torch>=2.7+cu128）。
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field, asdict


def _nvidia_smi() -> dict | None:
    """通过 nvidia-smi 读取 GPU 名与显存（MB），无 NVIDIA 卡返回 None。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        name, vram, driver = [c.strip() for c in out.split(",")]
        return {"name": name, "vram_mb": int(float(vram)), "driver": driver}
    except Exception:
        return None


def _torch_info() -> dict:
    try:
        import torch  # noqa
        cuda = torch.cuda.is_available()
        cap = None
        if cuda:
            major, minor = torch.cuda.get_device_capability(0)
            cap = [major, minor]
        return {"installed": True, "cuda": cuda, "capability": cap,
                "torch_version": torch.__version__}
    except ImportError:
        return {"installed": False, "cuda": False, "capability": None,
                "torch_version": None}


@dataclass
class HardwareReport:
    os_name: str
    gpu_name: str | None = None
    vram_mb: int | None = None
    driver: str | None = None
    torch_installed: bool = False
    torch_cuda: bool = False
    torch_version: str | None = None
    ffmpeg_ok: bool = False
    warnings: list = field(default_factory=list)
    recommendations: dict = field(default_factory=dict)

    @property
    def vram_gb(self) -> float:
        return (self.vram_mb or 0) / 1024

    def to_dict(self) -> dict:
        return asdict(self)


def probe() -> HardwareReport:
    r = HardwareReport(os_name=f"{platform.system()} {platform.release()}")
    gpu = _nvidia_smi()
    if gpu:
        r.gpu_name, r.vram_mb, r.driver = gpu["name"], gpu["vram_mb"], gpu["driver"]
    t = _torch_info()
    r.torch_installed, r.torch_cuda = t["installed"], t["cuda"]
    r.torch_version = t["torch_version"]
    r.ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    # ---- 警告 ----
    gb = r.vram_gb
    if not r.ffmpeg_ok:
        r.warnings.append("未检测到 FFmpeg/ffprobe，请安装并加入 PATH（winget install ffmpeg）")
    if r.gpu_name and "RTX 50" in r.gpu_name and r.torch_cuda:
        cap = t.get("capability") or [0, 0]
        if cap[0] < 12:  # Blackwell = sm_120，旧版 torch 无法在 50 系上跑 CUDA
            r.warnings.append(
                "检测到 RTX 50 系（Blackwell sm_120），当前 torch 不支持其 CUDA 架构；"
                "请安装 torch>=2.7 的 cu128 版本：pip install torch --index-url https://download.pytorch.org/whl/cu128"
            )
    if r.gpu_name and r.torch_installed and not r.torch_cuda:
        r.warnings.append("已装 torch 但 CUDA 不可用（可能装成 CPU 版），GPU 推理将回退 CPU")
    if r.gpu_name and gb >= 6 and not r.torch_cuda:
        r.warnings.append(
            f"检测到 NVIDIA GPU（{r.vram_gb:.0f}GB）但 torch CUDA 不可用："
            "pip install torch iopaint 后，去字幕通道可升级为 lama（接近无痕）"
        )

    # ---- 8GB 档推荐参数 ----
    if r.torch_cuda and gb >= 6:
        r.recommendations = {
            "subtitle_mode": "lama" if gb < 10 else "propainter",
            "upscale_engine": "realesrgan-ncnn-vulkan",   # 免 torch，Vulkan 通用
            "upscale_tile": 256 if gb <= 8 else 384,
            "clean_jobs": 1,                              # GPU 推理串行
            "encode": "h264_nvenc" if r.gpu_name else "libx264",
            "frame_format": "png",
        }
        if gb <= 8:
            r.warnings.append("8GB 显存：LaMa/Real-ESRGAN 可全速运行；ProPainter 建议先缩到 720p 或分窗处理")
    elif r.ffmpeg_ok:
        r.recommendations = {
            "subtitle_mode": "delogo",  # 无 GPU：FFmpeg 快速通道
            "upscale_engine": "realesrgan-ncnn-vulkan",   # Vulkan 可用核显/A 卡
            "upscale_tile": 192,
            "clean_jobs": max(1, (os_cpu := __import__("os").cpu_count() or 4) - 2),
            "encode": "libx264",
            "frame_format": "png",
        }
    return r


if __name__ == "__main__":
    print(json.dumps(probe().to_dict(), ensure_ascii=False, indent=2))
