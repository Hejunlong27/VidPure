"""FFmpeg / ffprobe 薄封装：探测、抽帧、生成 mask、合帧封装、音频保留策略。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

MP4_OK_AUDIO = {"aac", "mp3"}


def ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
           "-of", "json", str(path)]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"无视频流: {path}")
    v = streams[0]
    from fractions import Fraction
    fps = 30.0
    try:
        f = Fraction(v.get("r_frame_rate", "30/1"))
        if f.numerator:
            fps = float(f)
    except Exception:
        pass
    a_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "json", str(path)]
    a = json.loads(subprocess.run(a_cmd, capture_output=True, text=True, check=True).stdout)
    a_streams = a.get("streams") or []
    dur = v.get("duration") or (info.get("format") or {}).get("duration") or 0
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps,
            "duration": float(dur), "has_audio": bool(a_streams),
            "audio_codec": a_streams[0]["codec_name"] if a_streams else ""}


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


def extract_frames(src: Path, out_dir: Path, fmt: str = "png") -> int:
    """视频 → 帧序列 f%06d.<fmt>，返回帧数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(src), "-start_number", "0", "-vsync", "0",
         str(out_dir / f"f%06d.{fmt}")])
    return len(list(out_dir.glob(f"f*.{fmt}")))


def make_mask_png(w: int, h: int, rects: list, out: Path) -> None:
    """黑底 + 白色矩形（白 = 需重绘区域）。rects: [(x,y,w,h), ...]"""
    draw = ":".join(f"drawbox=x={x}:y={y}:w={rw}:h={rh}:color=white:t=fill"
                    for x, y, rw, rh in rects)
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=black:s={w}x{h}",
         "-vf", draw, "-frames:v", "1", str(out)])


def mux(frames_dir: Path, src: Path, dst: Path, fps: float,
        vcodec: str = "libx264", crf: int = 18, preset: str = "medium",
        mute: bool = False) -> None:
    """帧序列 + 原视频音频 → 成片。音频优先无损耗 copy。"""
    info = ffprobe(src)
    cmd = ["ffmpeg", "-y", "-framerate", f"{fps:.6f}",
           "-i", str(frames_dir / "f%06d.png"), "-i", str(src), "-map", "0:v:0"]
    if info["has_audio"] and not mute:
        if dst.suffix in (".mp4", ".m4v") and info["audio_codec"] in MP4_OK_AUDIO:
            cmd += ["-map", "1:a:0?", "-c:a", "copy"]
        else:
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(dst)]
    run(cmd)


def build_clean_filter(mode: str, rects: list, w: int, h: int,
                       upscale_back: bool = False, blur_sigma: int = 24) -> tuple[str, str]:
    """返回 (filter_arg_key, filter_str)：blur 用 filter_complex，其余用 vf。"""
    if mode == "crop":
        y0 = min(r[1] for r in rects)
        crop_h = max(16, int(y0) // 2 * 2)
        vf = f"crop={w}:{crop_h}:0:0"
        if upscale_back:
            vf += f",scale={w}:{h}:flags=lanczos"
        return "vf", vf
    if mode == "delogo":
        return "vf", ",".join(f"delogo=x={x}:y={y}:w={rw}:h={rh}" for x, y, rw, rh in rects)
    if mode == "blur":
        rebuilt, src = [], "0:v"
        for i, (x, y, rw, rh) in enumerate(rects):
            last = "vout" if i == len(rects) - 1 else f"v{i}"
            rebuilt.append(f"[{src}]split=2[bg{i}][sub{i}];"
                           f"[sub{i}]crop={rw}:{rh}:{x}:{y},gblur=sigma={blur_sigma}[bl{i}];"
                           f"[bg{i}][bl{i}]overlay={x}:{y}[{last}]")
            src = last
        return "filter_complex", ";".join(rebuilt)
    raise ValueError(mode)


def parse_region(spec: str, fmt: str, w: int, h: int) -> list:
    """'0.78,0.95'（比例）/ 'x0,y0,x1,y1' / 'x,y,w,h' 像素，多段分号分隔 → [(x,y,w,h)]"""
    rects = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        nums = [float(p) for p in parts]
        use = fmt if fmt != "auto" else ("ratio" if (len(nums) == 2 or (len(nums) == 1 and nums[0] <= 1)) else "pixel")
        if use == "ratio":
            if len(parts) == 1:
                x0, y0, x1, y1 = 0.0, nums[0], 1.0, 1.0
            elif len(parts) == 2:
                x0, y0, x1, y1 = 0.0, nums[0], 1.0, nums[1]
            elif len(parts) == 4:
                x0, y0, x1, y1 = nums
            else:
                raise ValueError(chunk)
            rects.append((x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h))
        else:
            if len(parts) != 4:
                raise ValueError(chunk)
            x, y, rw, rh = (int(float(p)) for p in parts)
            rects.append((x, y, rw, rh))
    out = []
    for x, y, rw, rh in rects:
        x, y = max(1, int(round(x))), max(1, int(round(y)))
        rw, rh = min(int(round(rw)), w - 1 - x), min(int(round(rh)), h - 1 - y)
        if rw > 0 and rh > 0:
            out.append((x, y, rw, rh))
    if not out:
        raise ValueError(f"区域解析后为空: {spec}")
    return out
