"""Web 服务端：REST API + 静态页面 + 后台任务队列。

设计要点：
  - 零第三方依赖（http.server + threading）
  - 分片上传（4MB/片，流式追加写，支持大视频与断点提示）
  - 单 GPU 任务串行队列（ThreadPoolExecutor max_workers=1）
  - 视频结果支持 HTTP Range（浏览器可拖动进度条）
  - 所有面向用户的错误翻译为中文
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..core import subtitle, upscale
from ..core.media import ffprobe
from ..core.probe import probe

STATIC_DIR = Path(__file__).parent / "static"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv"}
CHUNK_LIMIT = 32 * 1024 * 1024  # 单请求体上限 32MB（分片 4MB 足够）

# 进度阶段文案（中文）
STAGE_TEXT = {
    "queued": "排队等待中…",
    "clean_fast": "正在抹除字幕（快速模式）…",
    "clean_lama": "AI 正在逐帧修复字幕区域，高清模式较慢，请耐心等待…",
    "upscale": "正在提升画质（放大中）…",
    "mux": "正在封装输出…",
    "done": "处理完成",
}


@dataclass
class Task:
    tid: str
    src: Path
    name: str
    clean: str            # fast / lama / crop
    region: str
    scale: int
    status: str = "running"     # running / done / error
    stage: str = "queued"
    message: str = STAGE_TEXT["queued"]
    progress: float = 0.0       # 0~1（阶段粗进度）
    dst: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)

    def snapshot(self) -> dict:
        return {
            "id": self.tid, "name": self.name, "status": self.status,
            "stage": self.stage, "stage_text": STAGE_TEXT.get(self.stage, self.message),
            "progress": round(self.progress, 3), "message": self.message,
            "dst": self.dst, "error": self.error,
            "video_url": f"/api/result/{self.tid}/output.mp4" if self.status == "done" else "",
            "download_url": f"/api/download/{self.tid}" if self.status == "done" else "",
        }


class TaskBoard:
    """任务表 + 单 worker 队列（GPU 任务串行）。"""

    def __init__(self, work_dir: Path):
        self.tasks: dict[str, Task] = {}
        self.work_dir = work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vidpure-task")
        self.lock = threading.Lock()

    def create(self, fid: str, clean: str, region: str, scale: int) -> Task:
        src = self.work_dir / "uploads" / fid / "input.mp4"
        if not src.exists():
            raise FileNotFoundError("上传文件不存在，请重新上传")
        clean_map = {"fast": "delogo", "lama": "lama", "crop": "crop"}
        t = Task(tid=uuid.uuid4().hex[:12], src=src,
                 name=src.parent.parent.name or "视频",
                 clean=clean_map.get(clean, "delogo"),
                 region=region, scale=int(scale or 0))
        with self.lock:
            self.tasks[t.tid] = t
        self.pool.submit(self._run, t)
        return t

    # ---------------------------------------------------------------- worker
    def _set(self, t: Task, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(t, k, v)

    def _run(self, t: Task):
        out_dir = self.work_dir / "results" / t.tid
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            info = ffprobe(t.src)
            cleaned = t.src
            if t.clean != "none":
                self._set(t, stage="clean_lama" if t.clean == "lama" else "clean_fast",
                          progress=0.15,
                          message=STAGE_TEXT["clean_lama" if t.clean == "lama" else "clean_fast"])
                mid = out_dir / "cleaned.mp4"
                if t.clean == "lama":
                    subtitle.clean_video_lama(t.src, mid, t.region, device=_best_device(),
                                              vcodec=_vcodec())
                else:
                    subtitle.clean_video_fast(t.src, mid, t.region, mode=t.clean,
                                              vcodec=_vcodec())
                cleaned = mid

            final = cleaned
            if t.scale:
                self._set(t, stage="upscale", progress=0.6, message=STAGE_TEXT["upscale"])
                final = out_dir / "upscaled.mp4"
                upscale.upscale_video_ncnn(cleaned, final, scale=t.scale,
                                           tile=_tile(), progress=lambda m: None)

            self._set(t, stage="mux", progress=0.92, message=STAGE_TEXT["mux"])
            dst = out_dir / "output.mp4"
            if final != dst:
                shutil.copyfile(final, dst)
            self._set(t, status="done", stage="done", progress=1.0, dst=str(dst))
        except Exception as e:
            self._set(t, status="error", error=friendly_error(e), message="处理失败")
        finally:
            # 清理中间文件，保留 output
            for f in ("cleaned.mp4", "upscaled.mp4"):
                p = out_dir / f
                if p.exists() and p.name != "output.mp4":
                    p.unlink(missing_ok=True)


# ---------------------------------------------------------------- 环境探测缓存
_probe_cache: dict = {}


def best_env() -> dict:
    if not _probe_cache:
        r = probe()
        _probe_cache.update(r.to_dict())
        _probe_cache["recommendations"] = r.recommendations
        _probe_cache["warnings"] = r.warnings
    return _probe_cache


def _best_device() -> str:
    return "cuda" if best_env().get("torch_cuda") else "cpu"


def _vcodec() -> str:
    return "h264_nvenc" if best_env().get("gpu_name") else "libx264"


def _tile() -> int:
    return int(best_env().get("recommendations", {}).get("upscale_tile", 256))


def friendly_error(e: Exception) -> str:
    """把常见异常翻译为普通用户能懂的中文。"""
    s = str(e)
    pairs = [
        ("No module named 'iopaint'", "高清修复需要 AI 组件：请在命令行运行 pip install iopaint 后重启服务，或改用「快速去字幕」模式"),
        ("no kernel image", "当前 PyTorch 版本不支持您的显卡（RTX 50 系需要新版 PyTorch），建议改用「快速去字幕」模式"),
        ("CUDA", "显卡计算出错：请确认显卡驱动为最新版本，或改用「快速去字幕」模式"),
        ("Vulkan", "显卡超分引擎启动失败：请更新显卡驱动后重试"),
        ("未产出", "处理引擎没有输出结果，请缩小字幕选区后重试"),
        ("抽帧失败", "视频文件无法读取：请确认文件为标准视频格式（mp4/mov/mkv 等）"),
    ]
    for key, msg in pairs:
        if key.lower() in s.lower():
            return msg
    return f"处理失败：{s[:180]}"


# ---------------------------------------------------------------- HTTP Handler
class Handler(BaseHTTPRequestHandler):
    server_version = "VidPureWeb/0.1"
    protocol_version = "HTTP/1.1"   # 所有响应均带 Content-Length，支持 keep-alive
    timeout = 300                   # 单请求读超时，防止死线程长期占用
    board: TaskBoard = None  # 注入
    verbose: bool = False    # 诊断日志开关

    def _log(self, msg: str):
        if self.verbose:
            import sys
            print(f"[web] {self.command} {self.path} -> {msg}", file=sys.stderr, flush=True)

    def log_message(self, fmt, *args):  # 默认静默，verbose 时输出
        if self.verbose:
            import sys
            print(f"[web] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)
    uploads: dict = {}       # fid -> {path, handle, received, name}

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    # ---- 基础响应 ----
    def _json(self, obj, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self._headers_sent = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
        self._log(f"{code} json {len(data)}B")

    def _file(self, path: Path, ctype: str, download_name: str = ""):
        if not path.exists():
            return self._json({"ok": False, "error": "文件不存在或已被清理"}, 404)
        self._headers_sent = True
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1) or 0)
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            code, extra = 206, [("Content-Range", f"bytes {start}-{end}/{size}")]
            length = end - start + 1
        else:
            start, length, code, extra = 0, size, 200, []
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if download_name:
        # RFC 5987：中文文件名兼容
            from urllib.parse import quote
            q = quote(download_name)
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{download_name.encode('ascii','ignore').decode() or 'download'}\"; filename*=UTF-8''{q}")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _body(self) -> bytes:
        """读取请求体（带缓存：同一请求内只读一次，保证 keep-alive 连接不被残留字节污染）。"""
        if getattr(self, "_body_cache", None) is not None:
            return self._body_cache
        length = int(self.headers.get("Content-Length") or 0)
        if length > CHUNK_LIMIT:
            # 超限时也要把 body 读完丢弃，再抛错（否则污染连接）
            self._drain_body()
            raise ValueError(f"请求体过大: {length}")
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            self._log(f"body 截断! 声明 {length} 实读 {len(data)}")
        self._body_cache = data
        self._body_taken = True
        return data

    # ---- 路由 ----
    def do_HEAD(self):
        self._route()

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()

    def _route(self):
        self._body_cache = None
        self._body_taken = False
        try:
            path = self.path.split("?")[0]
            qs = dict(p.split("=", 1) for p in (self.path.split("?")[1:] or [""])[0].split("&") if "=" in p) if "?" in self.path else {}
            if self.command == "GET":
                if path in ("/", "/index.html"):
                    return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                if path == "/api/probe":
                    env = best_env()
                    return self._json({
                        "gpu": env.get("gpu_name") or "未检测到独立显卡",
                        "vram_gb": round(env.get("vram_mb", 0) / 1024, 1),
                        "os": env.get("os_name"), "ffmpeg": env.get("ffmpeg_ok"),
                        "lama_ready": env.get("torch_cuda"),
                        "recommend": ("lama" if env.get("torch_cuda") else "fast"),
                        "warnings": env.get("warnings", []),
                    })
                if path == "/api/tasks":
                    return self._json({"tasks": [t.snapshot() for t in
                                                 sorted(self.board.tasks.values(),
                                                        key=lambda t: -t.created)]})
                if path.startswith("/api/task/"):
                    t = self.board.tasks.get(path.rsplit("/", 1)[1])
                    return self._json(t.snapshot() if t else {"status": "unknown"}, 404 if not t else 200)
                if path.startswith("/api/thumb/"):
                    fid = path.split("/")[3].replace(".jpg", "")
                    return self._file(self.board.work_dir / "uploads" / fid / "thumb.jpg", "image/jpeg")
                if path.startswith("/api/input/"):
                    fid = path.split("/")[3]
                    return self._file(self.board.work_dir / "uploads" / fid / "input.mp4", "video/mp4")
                if path.startswith("/api/result/"):
                    tid = path.split("/")[3]
                    t = self.board.tasks.get(tid)
                    if not t or t.status != "done":
                        return self._json({"ok": False, "error": "任务未完成"}, 404)
                    return self._file(Path(t.dst), "video/mp4")
                if path.startswith("/api/download/"):
                    tid = path.split("/")[3]
                    t = self.board.tasks.get(tid)
                    if not t or t.status != "done":
                        return self._json({"ok": False, "error": "任务未完成"}, 404)
                    return self._file(Path(t.dst), "video/mp4", download_name=t.name.replace(".mp4", "") + "_去字幕.mp4")
            else:  # POST
                if path == "/api/upload/start":
                    body = json.loads(self._body() or b"{}")
                    name = body.get("name", "video.mp4")
                    if Path(name).suffix.lower() not in VIDEO_EXTS:
                        return self._json({"ok": False,
                                           "error": "暂不支持该格式，请上传 mp4 / mov / mkv / webm 等常见视频"}, 400)
                    fid = uuid.uuid4().hex[:12]
                    d = self.board.work_dir / "uploads" / fid
                    d.mkdir(parents=True, exist_ok=True)
                    self.uploads[fid] = {"path": d / "input.mp4", "received": 0,
                                         "name": name, "total": int(body.get("size", 0))}
                    return self._json({"ok": True, "fid": fid, "chunk": 4 * 1024 * 1024})
                if path == "/api/upload/chunk":
                    fid = qs.get("fid", "")
                    info = self.uploads.get(fid)
                    if not info:
                        return self._json({"ok": False, "error": "上传会话已失效，请重新选择文件"}, 410)
                    data = self._body()
                    with open(info["path"], "ab") as f:
                        f.write(data)
                    info["received"] += len(data)
                    return self._json({"ok": True, "received": info["received"]})
                if path == "/api/upload/end":
                    fid = qs.get("fid", "")
                    info = self.uploads.pop(fid, None)
                    if not info:
                        return self._json({"ok": False, "error": "上传会话已失效，请重新选择文件"}, 410)
                    src = info["path"]
                    try:
                        ffprobe(src)
                    except Exception:
                        src.unlink(missing_ok=True)
                        return self._json({"ok": False,
                                           "error": "视频文件无法识别，请确认文件完整且为常见视频格式"}, 400)
                    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ss", "0.1",
                                    "-frames:v", "1", "-vf", "scale=640:-2",
                                    str(src.parent / "thumb.jpg")],
                                   capture_output=True, text=True)
                    return self._json({"ok": True, "fid": fid, "name": info["name"]})
                if path == "/api/task":
                    body = json.loads(self._body() or b"{}")
                    t = self.board.create(body.get("fid", ""), body.get("clean", "fast"),
                                          body.get("region", "0.78,0.95"),
                                          body.get("scale", 0))
                    return self._json({"ok": True, "id": t.tid, "task": t.snapshot()})
            return self._json({"ok": False, "error": "接口不存在"}, 404)
        except BrokenPipeError:
            self.close_connection = True
        except Exception as e:
            try:
                if getattr(self, "_headers_sent", False):
                    self.close_connection = True  # 响应头已发出，无法再发 JSON，只能断连
                    self._log(f"500 headers-already-sent: {type(e).__name__}: {e}")
                else:
                    self._json({"ok": False, "error": friendly_error(e)}, 500)
            except Exception:
                self.close_connection = True
        finally:
            # 关键兜底：POST 请求无论走哪个分支（410/400/404/异常），
            # 只要 body 未被业务读取，都必须排干——否则残留字节会污染
            # keep-alive 连接，导致浏览器复用此连接时收到空/错乱响应，
            # 表现为前端 "Unexpected end of JSON input"。
            if self.command == "POST" and not self._body_taken:
                self._drain_body()

    def _drain_body(self):
        self._body_taken = True
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        remaining = min(max(length, 0), 64 * 1024 * 1024)
        while remaining > 0:
            chunk = self.rfile.read(min(1 << 16, remaining))
            if not chunk:
                break
            remaining -= len(chunk)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """客户端正常关闭 keep-alive 连接时的 ConnectionReset 属常见现象，静音。"""
        import sys
        etype = sys.exception()
        if etype is not None and etype.__name__ in ("ConnectionResetError", "BrokenPipeError", "TimeoutError"):
            return
        super().handle_error(request, client_address)


def serve(host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True,
          work_dir: Path | None = None, verbose: bool = False) -> None:
    work = work_dir or Path("./vidpure_output/web").resolve()
    board = TaskBoard(work)
    Handler.board = board
    Handler.verbose = verbose

    httpd = QuietThreadingHTTPServer((host, port), Handler)
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    print(f"VidPure 网页版已启动: {url}")
    if host == "0.0.0.0":
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            print(f"手机访问（同一 WiFi）: http://{s.getsockname()[0]}:{port}")
            s.close()
        except Exception:
            pass
    print("按 Ctrl+C 停止服务")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
