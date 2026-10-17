"""vidpure 命令行 —— 调度器模式入口。

  vidpure probe        硬件自检 / 字幕能力 / 分辨率列表
  vidpure resolutions  只列出固定分辨率预设
  vidpure run          批量流水线：去原字幕 → 放大到固定分辨率 → 烧录字幕（队列 + 并发 + 重试）
  vidpure comfy        把任务派给本机 ComfyUI server（可自动拉起）
  vidpure workflow     导出 API 格式工作流 JSON
  vidpure web          网页版（批量上传 + 可视化框选 + 看板）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .core import resolve, subtitle_burn
from .core.batch import BatchLimits, BatchRunner, JobOptions
from .core.pipeline import _collect_inputs
from .core.probe import probe
from .backends import ComfyBackend, LocalBackend
from .backends import workflows as wf


def cmd_probe(args) -> int:
    r = probe()
    d = r.to_dict()
    d["subtitle_caps"] = subtitle_burn.capabilities()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    rec = r.recommendations
    if rec:
        print("\n推荐配置：")
        print(f"  去字幕通道   {rec.get('subtitle_mode')}")
        print(f"  放大引擎     {rec.get('upscale_engine')} (tile={rec.get('upscale_tile')})")
        print(f"  视频编码     {rec.get('encode')}")
    caps = d["subtitle_caps"]
    print("\n字幕能力：")
    print(f"  台词识别(whisper)  {'可用' if caps['transcribe'] else '缺失'}"
          + (f"  {caps['whisper_cli']}" if caps.get("whisper_cli") else ""))
    print(f"  CTC 强制对齐       {'可用' if caps['ctc_align'] else '缺失（改用语音识别时间轴）'}")
    for w in r.warnings:
        print(f"  [警告] {w}")
    return 0


def cmd_resolutions(args) -> int:
    rows = resolve.list_presets()
    print(f"{'id':<8} {'名称':<16} 说明")
    for p in rows:
        print(f"{p['id']:<8} {p['label']:<16} {p['desc']}")
    print("\n备注：预设按【短边像素】定义，长边按源宽高比等比推算并对齐偶数；"
          "也可用 --target-res 1920x1080 精确指定。")
    return 0


def cmd_run(args) -> int:
    """批量流水线（队列 + 并发 + 自动重试）。"""
    files = _collect_inputs(args.input, args.recursive, args.ext)
    if not files:
        print("[错误] 未找到输入视频", file=sys.stderr)
        return 1
    limits = BatchLimits(max_items=args.max_items, max_duration_s=args.max_duration,
                         enforce_limits=not args.no_duration_limit)
    # 给了固定距底像素 ⇒ 自动切到 pixel 摆位（与核心 compute_layout 的判定一致）
    sub_anchor = args.sub_anchor
    if args.sub_margin_bottom is not None and sub_anchor == "ratio":
        sub_anchor = "pixel"
    opts = JobOptions(
        clean=args.clean, region=args.region, region_fmt=args.region_format,
        scale=args.scale, target_res=args.target_res, upscale_model=args.upscale_model,
        vcodec=args.vcodec, crf=args.crf, preset=args.preset,
        subtitle=args.burn_subtitle, sub_lang=args.sub_lang,
        sub_srt=args.sub_srt, sub_mode=args.sub_mode,
        sub_box=_parse_box(args.sub_box), sub_anchor=sub_anchor,
        sub_anchor_top=args.sub_anchor_top,
        sub_font_size_px=args.font_size,
        sub_font_size_pct=args.font_pct, sub_margin_bottom_px=args.sub_margin_bottom,
        sub_refine=args.sub_refine, keep_burn=not args.drop_master,
    )
    out_dir = Path(args.out).resolve()
    runner = BatchRunner(out_dir=out_dir, work_dir=out_dir / "_work",
                         concurrency=args.jobs, retries=args.retries, limits=limits)

    if args.target_res:
        first = files[0]
        from .core.media import ffprobe
        inf = ffprobe(first)
        for w in resolve.warnings_for(inf["w"], inf["h"], args.target_res):
            print(f"  [提示] {w}")

    items = [{"src": str(f), "name": f.name} for f in files]
    print(f"批量提交 {len(items)} 个视频 | 并发 {args.jobs} | 失败重试 {args.retries} 次"
          f" | 单条上限 {args.max_duration:.0f}s")
    try:
        jobs = runner.submit(items, default_opts=opts)
    except Exception as e:                                       # noqa: BLE001
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    # 轮询等待（CLI 下给出简洁进度）
    seen: dict[str, str] = {}
    while True:
        active = 0
        for j in jobs:
            line = f"{j.status}/{j.stage}"
            if seen.get(j.jid) != line:
                seen[j.jid] = line
                txt = j.message or j.stage
                print(f"  [{j.name}] {txt}"
                      + (f"  {j.error}" if j.error else ""))
            if j.status in ("queued", "running"):
                active += 1
        if not active:
            break
        time.sleep(1.0)

    ok = sum(1 for j in jobs if j.status == "done")
    err = sum(1 for j in jobs if j.status == "error")
    rej = sum(1 for j in jobs if j.status == "rejected")
    print(f"\n批量完成: 成功 {ok}/{len(jobs)}" + (f" · 失败 {err}" if err else "")
          + (f" · 跳过 {rej}" if rej else ""))
    print(f"输出目录: {out_dir / 'results'}")
    return 1 if (err or rej) else 0


def _parse_box(spec: str | None):
    if not spec:
        return None
    parts = [float(x) for x in spec.replace(";", ",").split(",") if x.strip()]
    if len(parts) != 4:
        raise SystemExit("--sub-box 需要 4 个 0~1 的归一化坐标：x0,y0,x1,y1")
    return tuple(parts)


def cmd_comfy(args) -> int:
    be = ComfyBackend(comfy_dir=args.comfy_dir, host=args.host, port=args.port)
    be.start()
    try:
        out_dir = Path(args.out).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
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
          verbose=args.verbose, concurrency=args.concurrency, retries=args.retries)
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="vidpure",
        description="AI 视频去字幕 + 固定分辨率放大 + 字幕烧录调度器（本地 / ComfyUI 双模式）")
    ap.add_argument("--version", action="version", version=f"vidpure {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="硬件自检 / 字幕能力 / 方案推荐").set_defaults(fn=cmd_probe)
    sub.add_parser("resolutions", help="列出固定分辨率预设").set_defaults(fn=cmd_resolutions)

    p = sub.add_parser("run", help="批量流水线：去字幕 → 固定分辨率放大 → 烧录字幕")
    p.add_argument("--input", nargs="+", required=True, help="视频文件或目录（可多个）")
    p.add_argument("--out", default="./vidpure_output")
    p.add_argument("--clean", default="lama",
                   choices=["lama", "delogo", "blur", "crop", "none"])
    p.add_argument("--region", default="0.78,0.95", help="原字幕区域（比例或像素，多段分号分隔）")
    p.add_argument("--region-format", dest="region_format", default="auto",
                   choices=["auto", "ratio", "pixel"])
    p.add_argument("--target-res", dest="target_res", default=None,
                   help="固定分辨率：预设 id（1080p/1440p/2048p/2160p/4096p）或 WxH；"
                        "省略=不缩放")
    p.add_argument("--scale", type=int, default=0,
                   help="[旧式] 放大倍数 2/3/4；被 --target-res 取代，仅作兼容")
    p.add_argument("--upscale-model", dest="upscale_model", default="realesr-animevideov3",
                   choices=["realesr-animevideov3", "realesrgan-x4plus"])
    p.add_argument("--vcodec", default="libx264", help="libx264 / h264_nvenc")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="medium")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--ext", default="mp4,mov,mkv,webm,m4v")
    p.add_argument("--jobs", "-j", type=int, default=1, help="并发数（GPU 逐帧处理建议 1）")
    p.add_argument("--retries", type=int, default=2, help="失败自动重试次数")
    p.add_argument("--max-items", dest="max_items", type=int, default=60, help="单批上限（默认 60）")
    p.add_argument("--max-duration", dest="max_duration", type=float, default=90.0,
                   help="单条时长上限秒数（默认 90）")
    p.add_argument("--no-duration-limit", dest="no_duration_limit", action="store_true",
                   help="超时长不跳过，只告警")
    # 字幕烧录
    p.add_argument("--burn-subtitle", dest="burn_subtitle", action="store_true",
                   help="识别台词并烧录字幕")
    p.add_argument("--sub-lang", dest="sub_lang", default="zh", choices=["zh", "es", "en"])
    p.add_argument("--sub-srt", dest="sub_srt", default=None, help="用现成 SRT（跳过识别）")
    p.add_argument("--sub-mode", dest="sub_mode", default="bottom", choices=["bottom", "box"],
                   help="字幕位置：底部（按 6:4 视觉中心）/ 手画框")
    p.add_argument("--sub-box", dest="sub_box", default=None,
                   help="手画框区域（归一化 x0,y0,x1,y1，配 --sub-mode box）")
    p.add_argument("--sub-anchor", dest="sub_anchor", default="ratio",
                   choices=["ratio", "pixel"],
                   help="垂直摆位（仅底部模式）：ratio=文字块视觉中心按 6:4（距顶:距底）分屏，"
                        "随分辨率等比缩放（默认）；pixel=固定距底像素")
    p.add_argument("--sub-anchor-top", dest="sub_anchor_top", type=float, default=0.6,
                   help="ratio 模式下视觉中心距画面顶部占比（默认 0.6，即 6:4）")
    p.add_argument("--font-size", dest="font_size", type=int, default=None,
                   help="字幕字号（像素，相对输出画布）；省略=自动")
    p.add_argument("--font-pct", dest="font_pct", type=float, default=5.6,
                   help="自动字号占画布高度百分比（默认 5.6）")
    p.add_argument("--sub-margin-bottom", dest="sub_margin_bottom", type=int, default=None,
                   help="pixel 模式距画面底部像素（默认 400）；给出该参数即启用 pixel 模式")
    p.add_argument("--sub-refine", dest="sub_refine", default="auto",
                   choices=["auto", "ctc", "none"], help="时间轴精修策略")
    p.add_argument("--drop-master", dest="drop_master", action="store_true",
                   help="烧字幕成功后删除无字幕母版（默认保留）")
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

    p = sub.add_parser("web", help="启动网页版（批量上传 + 画框 + 看板）")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 允许手机同 WiFi 访问")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--verbose", action="store_true", help="输出诊断日志")
    p.add_argument("--concurrency", type=int, default=1, help="默认并发数")
    p.add_argument("--retries", type=int, default=2, help="默认失败重试次数")
    p.set_defaults(fn=cmd_web)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as e:                                       # noqa: BLE001
        print(f"[错误] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
