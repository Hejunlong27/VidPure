"""vidpure 命令行 —— 调度器模式的入口。

  vidpure probe    硬件自检与方案推荐
  vidpure run      一键流水线：去字幕 → 放大（自动选择/拉起本地算力）
  vidpure comfy    把任务派给本机 ComfyUI server（可自动拉起）
  vidpure workflow 导出 API 格式工作流 JSON，供在 ComfyUI 网页里导入/改参
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core.probe import probe
from .core.pipeline import run_pipeline
from .backends import ComfyBackend, LocalBackend
from .backends import workflows as wf


def cmd_probe(args) -> int:
    r = probe()
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    rec = r.recommendations
    if rec:
        print("\n推荐配置：")
        print(f"  去字幕通道   {rec.get('subtitle_mode')}")
        print(f"  放大引擎     {rec.get('upscale_engine')} (tile={rec.get('upscale_tile')})")
        print(f"  视频编码     {rec.get('encode')}")
    for w in r.warnings:
        print(f"  [警告] {w}")
    return 0


def cmd_run(args) -> int:
    results = run_pipeline(
        input_items=args.input, out_dir=Path(args.out), clean=args.clean,
        clean_region=args.region, region_fmt=args.region_format,
        upscale_scale=args.scale, upscale_model=args.upscale_model,
        device=args.device, recursive=args.recursive, exts=args.ext,
        jobs=args.jobs, vcodec=args.vcodec, crf=args.crf, preset=args.preset,
        mute=args.mute, keep_workdir=args.keep_workdir)
    failed = [r for r in results if any(not s.ok for s in r.stages)]
    print(f"\n批量完成: 成功 {len(results) - len(failed)}/{len(results)}")
    return 1 if failed else 0


def cmd_comfy(args) -> int:
    be = ComfyBackend(comfy_dir=args.comfy_dir, host=args.host, port=args.port)
    be.start()
    try:
        be_local = LocalBackend()
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        from .core.pipeline import _collect_inputs
        for f in _collect_inputs([args.input], args.recursive, args.ext):
            if args.stage in ("clean", "both"):
                print(f"[去字幕] {f.name}")
                dst = out_dir / f"{f.stem}_comfy_clean.mp4"
                be.clean(f, dst, region=args.region)
                print(f"    -> {dst}")
                f = dst
            if args.stage in ("upscale", "both"):
                print(f"[放大] {f.name}")
                dst = out_dir / f"{f.stem}_comfy_x{args.scale}.mp4"
                be.upscale(f, dst, args.scale, args.upscale_model)
                print(f"    -> {dst}")
    finally:
        be.shutdown()
    return 0


def cmd_workflow(args) -> int:
    w = (wf.subtitle_workflow(video_path=args.video, region=args.region)
         if args.kind == "subtitle"
         else wf.upscale_workflow(video_path=args.video, model_name=args.upscale_model))
    out = Path(args.out or f"vidpure_{args.kind}_workflow_api.json")
    wf.save(w, out)
    print(f"已导出 {out}（在 ComfyUI 网页「加载」API JSON 或用脚本提交 /prompt）")
    return 0


def cmd_web(args) -> int:
    from .web import serve
    serve(host=args.host, port=args.port, open_browser=not args.no_browser,
          verbose=args.verbose)
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vidpure",
                                 description="AI 视频去字幕 + 放大调度器（本地 / ComfyUI 双模式）")
    ap.add_argument("--version", action="version", version=f"vidpure {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="硬件自检与方案推荐").set_defaults(fn=cmd_probe)

    p = sub.add_parser("run", help="一键流水线：去字幕 → 放大（本地算力）")
    p.add_argument("--input", nargs="+", required=True, help="视频文件或目录")
    p.add_argument("--out", default="./vidpure_output")
    p.add_argument("--clean", default="lama", choices=["lama", "delogo", "blur", "crop", "none"])
    p.add_argument("--region", default="0.78,0.95", help="字幕区域（比例或像素，多段分号分隔）")
    p.add_argument("--region-format", dest="region_format", default="auto",
                   choices=["auto", "ratio", "pixel"])
    p.add_argument("--scale", type=int, default=2, help="放大倍数（0/none 跳过放大）")
    p.add_argument("--upscale-model", dest="upscale_model", default="realesr-animevideov3",
                   choices=["realesr-animevideov3", "realesrgan-x4plus"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--ext", default="mp4,mov,mkv,webm,m4v")
    p.add_argument("--jobs", "-j", type=int, default=2)
    p.add_argument("--vcodec", default="libx264", help="libx264 / h264_nvenc")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--keep-workdir", dest="keep_workdir", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("comfy", help="把任务派给本机 ComfyUI server")
    p.add_argument("--input", required=True, help="视频文件或目录")
    p.add_argument("--out", default="./vidpure_output")
    p.add_argument("--stage", default="both", choices=["clean", "upscale", "both"])
    p.add_argument("--region", default="0.78,0.95")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--upscale-model", dest="upscale_model", default="4x-UltraSharp.pth")
    p.add_argument("--comfy-dir", dest="comfy_dir", help="ComfyUI 安装目录（含 main.py），用于自动拉起")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--ext", default="mp4,mov,mkv,webm,m4v")
    p.set_defaults(fn=cmd_comfy)

    p = sub.add_parser("workflow", help="导出 API 工作流 JSON")
    p.add_argument("kind", choices=["subtitle", "upscale"])
    p.add_argument("--video", default="input.mp4")
    p.add_argument("--region", default="0.78,0.95")
    p.add_argument("--upscale-model", dest="upscale_model", default="4x-UltraSharp.pth")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_workflow)

    p = sub.add_parser("web", help="启动网页版（非开发者推荐）")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 允许手机同 WiFi 访问")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--verbose", action="store_true", help="输出诊断日志（排查问题时使用）")
    p.set_defaults(fn=cmd_web)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
