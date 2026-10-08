"""算力后端抽象。调度器根据硬件探测结果选择后端：

  LocalBackend   本地直接拉起 FFmpeg / IOPaint / NCNN Vulkan 进程
  ComfyBackend   连接或自动启动本机 ComfyUI server，把阶段任务派给它执行
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def clean(self, src: Path, dst: Path, region: str, mode: str = "lama") -> Path: ...

    @abstractmethod
    def upscale(self, src: Path, dst: Path, scale: int, model: str) -> Path: ...
