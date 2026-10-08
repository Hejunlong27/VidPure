#!/usr/bin/env python3
"""一键环境自检 + 可选依赖安装引导。

  python scripts/bootstrap.py            # 只做体检
  python scripts/bootstrap.py --fix-iopaint   # 自动在当前环境装 iopaint
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vidpure.core.probe import probe  # noqa: E402


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-iopaint", action="store_true", help="自动安装 iopaint（LaMa 推理）")
    args = ap.parse_args()

    r = probe()
    print(f"OS        : {r.os_name}")
    print(f"GPU       : {r.gpu_name or '未检测到 NVIDIA'}"
          f"  VRAM={r.vram_gb:.0f}GB" if r.gpu_name else "GPU       : 未检测到 NVIDIA")
    print(f"FFmpeg    : {'OK' if r.ffmpeg_ok else '缺失（winget install ffmpeg）'}")
    print(f"torch     : {r.torch_version or '未安装'}  cuda={r.torch_cuda}")
    print(f"iopaint   : {'OK' if have('iopaint') else '未安装（LaMa 去字幕需要）'}")
    print(f"realesrgan: {'OK' if have('realesrgan') else '未安装（可选，默认走 NCNN 引擎）'}")

    if args.fix_iopaint and not have("iopaint"):
        print("\n[安装] pip install iopaint ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "iopaint"], check=True)

    for w in r.warnings:
        print(f"[警告] {w}")

    ready_fast = r.ffmpeg_ok
    ready_lama = ready_fast and have("iopaint") and r.torch_cuda
    print(f"\n快速通道（delogo/blur/crop）: {'就绪' if ready_fast else '未就绪'}")
    print(f"LaMa 去字幕 + NCNN 放大      : {'就绪' if ready_lama else '未就绪'}")
    print("下一步: vidpure probe && vidpure run --input <目录> --clean lama --scale 2")


if __name__ == "__main__":
    main()
