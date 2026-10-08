"""本地后端：直接以子进程/进程内方式拉起 FFmpeg、IOPaint、NCNN Vulkan。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core import subtitle, upscale
from .base import Backend


class LocalBackend(Backend):
    name = "local"

    def __init__(self, device: str = "cuda", vcodec: str = "libx264",
                 crf: int = 18, preset: str = "medium", mute: bool = False):
        self.device = device
        self.vcodec = vcodec
        self.crf = crf
        self.preset = preset
        self.mute = mute

    def health_check(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def clean(self, src: Path, dst: Path, region: str, mode: str = "lama") -> Path:
        subtitle.clean_video(src, dst, mode=mode, region=region, device=self.device,
                             vcodec=self.vcodec, crf=self.crf, preset=self.preset,
                             mute=self.mute)
        return dst

    def upscale(self, src: Path, dst: Path, scale: int, model: str) -> Path:
        upscale.upscale_video_ncnn(src, dst, scale=scale, model=model)
        return dst
