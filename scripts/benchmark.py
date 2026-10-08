#!/usr/bin/env python3
"""分阶段计时基准脚本，输出可直接贴进 docs/benchmark.md 的 Markdown 行。

  python scripts/benchmark.py --input sample.mp4 --clean lama --scale 2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidpure.core import subtitle, upscale  # noqa: E402
from vidpure.core.media import ffprobe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--clean", default="lama", choices=["lama", "delogo", "blur", "crop", "none"])
    ap.add_argument("--region", default="0.78,0.95")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--model", default="realesr-animevideov3")
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = ffprobe(src)
    print(f"输入: {src.name} {info['w']}x{info['h']} {info['fps']:.2f}fps {info['duration']:.1f}s")

    t_clean = None
    f = src
    if args.clean != "none":
        t0 = time.time()
        f = out / f"{src.stem}_cleaned.mp4"
        subtitle.clean_video(src, f, mode=args.clean, region=args.region)
        t_clean = time.time() - t0

    t_up = None
    if args.scale:
        t0 = time.time()
        dst = out / f"{f.stem}_x{args.scale}.mp4"
        upscale.upscale_video_ncnn(f, dst, scale=args.scale, model=args.model)
        t_up = time.time() - t0

    total = (t_clean or 0) + (t_up or 0)
    print("\n| 平台 | clean 通道 | upscale | clean 耗时 | upscale 耗时 | 合计 | 备注 |")
    print("|---|---|---|---|---|---|---|")
    print(f"| _填写_ | {args.clean} | {args.model} x{args.scale} | "
          f"{t_clean and f'{t_clean:.0f}s' or '—'} | {t_up and f'{t_up:.0f}s' or '—'} | "
          f"{total:.0f}s | {info['w']}x{info['h']}/{info['duration']:.0f}s |")


if __name__ == "__main__":
    main()
