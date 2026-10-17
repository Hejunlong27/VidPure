"""Web 服务端：REST API + 静态页面 + 后台批量任务队列。

设计要点
  - 零第三方依赖（http.server + threading）
  - 分片上传（4MB/片，流式追加写，支持大视频与断点提示）
  - 批量作业队列（core.batch.BatchRunner）：并发可调、失败自动重试、单批上限 60 条
  - 固定分辨率输出（core.resolve）：前端口径与 CLI 完全一致
  - 字幕烧录（core.subtitle_burn）：台词识别 + 对齐 + 可手画框 / 可调字号
  - 视频结果支持 HTTP Range（浏览器可拖动进度条）
  - 所有面向用户的错误翻译为中文
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..core import resolve, subtitle_burn
from ..core.batch import BatchError, BatchLimits, BatchRunner, JobOptions
from ..core.media import ffprobe
from ..core.probe import probe

STATIC_DIR = Path(__file__).parent / "static"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv"}
CHUNK_LIMIT = 32 * 1024 * 1024       # 单请求体上限 32MB（分片 4MB 足够）
MAX_ITEMS = 60                       # 单批最多视频数（需求硬指标）

# 常见中文字幕字体（供前端下拉；libass 需系统已安装）
FONT_CHOICES = ["Microsoft YaHei", "SimHei", "SimSun", "DengXian", "KaiTi", "Arial"]


# ---------------------------------------------------------------- 环境探测缓存
_probe_cache: dict = {}
_probe_lock = threading.Lock()


def best_env() -> dict:
    with _probe_lock:
        if not _probe_cache:
            r = probe()
            d = r.to_dict()
            d["recommendations"] = r.recommendations
            d["warnings"] = r.warnings
            d["subtitle_caps"] = subtitle_burn.capabilities()
            _probe_cache.update(d)
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
        ("未找到 whisper-cli", "台词识别需要 whisper-cli：请设置环境变量 VIDPURE_WHISPER_CLI 指向可执行文件"),
        ("未找到 whisper 模型", "台词识别需要 whisper 模型：请设置环境变量 VIDPURE_WHISPER_MODEL 指向 ggml 模型"),
        ("未得到任何字幕", "未识别到任何人声台词，已跳过字幕烧录"),
        ("CUDA", "显卡计算出错：请确认显卡驱动为最新版本，或改用「快速去字幕」模式"),
        ("Vulkan", "显卡超分引擎启动失败：请更新显卡驱动后重试"),
        ("模型缺失", "AI 超分模型缺失：请联网后重试以自动下载"),
        ("未产出", "处理引擎没有输出结果，请缩小字幕选区后重试"),
        ("抽帧失败", "视频文件无法读取：请确认文件为标准视频格式（mp4/mov/mkv 等）"),
    ]
    for key, msg in pairs:
        if key.lower() in s.lower():
            return msg
    return f"处理失败：{s[:180]}"


# ---------------------------------------------------------------- HTTP Handler
class Handler(BaseHTTPRequestHandler):
    server_version = "VidPureWeb/0.2"
    protocol_version = "HTTP/1.1"   # 所有响应均带 Content-Length，支持 keep-alive
    timeout = 300                   # 单请求读超时，防止死线程长期占用
    board: BatchRunner = None       # 注入
    verbose: bool = False
    # 上传会话（fid -> {path, received, name, total}）
    uploads: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            import sys
            print(f"[web] {self.command} {self.path} -> {msg}", file=sys.stderr, flush=True)

    def log_message(self, fmt, *args):  # 静默访问日志（verbose 时输出）
        if self.verbose:
            import sys
            print(f"[web] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

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
            ascii_name = download_name.encode("ascii", "ignore").decode() or "download"
            q = quote(download_name)
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{q}")
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
            self._drain_body()       # 超限也要把 body 读完丢弃，否则污染连接
            raise ValueError(f"请求体过大: {length}")
        data = self.rfile.read(length) if length else b""
        if len(data) != length:
            self._log(f"body 截断! 声明 {length} 实读 {len(data)}")
        self._body_cache = data
        self._body_taken = True
        return data

    def _json_body(self) -> dict:
        raw = self._body()
        return json.loads(raw or b"{}")

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
            qs: dict[str, str] = {}
            if "?" in self.path:
                for p in self.path.split("?", 1)[1].split("&"):
                    if "=" in p:
                        k, v = p.split("=", 1)
                        qs[k] = v
            if self.command in ("GET", "HEAD"):
                return self._get(path)
            return self._post(path, qs)
        except BrokenPipeError:
            self.close_connection = True
        except Exception as e:                                   # noqa: BLE001
            try:
                if getattr(self, "_headers_sent", False):
                    self.close_connection = True
                    self._log(f"500 headers-already-sent: {type(e).__name__}: {e}")
                else:
                    self._json({"ok": False, "error": friendly_error(e)}, 500)
            except Exception:                                    # noqa: BLE001
                self.close_connection = True
        finally:
            # 关键兜底：POST 请求无论走哪个分支，只要 body 未被业务读取都必须排干，
            # 否则残留字节会污染 keep-alive 连接（前端报 "Unexpected end of JSON input"）。
            if self.command == "POST" and not self._body_taken:
                self._drain_body()

    # ------------------------------------------------------------ GET
    def _get(self, path: str):
        if path in ("/", "/index.html"):
            return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/probe":
            env = best_env()
            caps = env.get("subtitle_caps", {})
            return self._json({
                "gpu": env.get("gpu_name") or "未检测到独立显卡",
                "vram_gb": round((env.get("vram_mb") or 0) / 1024, 1),
                "os": env.get("os_name"), "ffmpeg": env.get("ffmpeg_ok"),
                "lama_ready": env.get("torch_cuda"),
                "recommend": ("lama" if env.get("torch_cuda") else "fast"),
                "warnings": env.get("warnings", []),
                "subtitle": caps,
                "resolutions": resolve.list_presets(),
                "fonts": FONT_CHOICES,
                "limits": {"max_items": MAX_ITEMS,
                           "max_duration_s": BatchLimits().max_duration_s},
                "concurrency": self.board.concurrency,
                "retries": self.board.retries,
            })
        if path == "/api/resolutions":
            return self._json({"presets": resolve.list_presets()})
        if path == "/api/batch" or path == "/api/tasks":
            return self._json(self.board.snapshot())
        if path.startswith("/api/task/") or path.startswith("/api/job/"):
            jid = path.rsplit("/", 1)[1]
            j = self.board.job(jid)
            return self._json(j.snapshot() if j else {"status": "unknown"}, 404 if not j else 200)
        if path.startswith("/api/srt/"):
            jid = path.split("/")[3]
            j = self.board.job(jid)
            if not j or not j.srt:
                return self._json({"ok": False, "error": "暂无外挂字幕"}, 404)
            return self._file(Path(j.srt), "text/plain; charset=utf-8",
                              download_name=Path(j.srt).name)
        if path.startswith("/api/thumb/"):
            fid = path.split("/")[3].replace(".jpg", "")
            return self._file(self.board.work_dir / "uploads" / fid / "thumb.jpg", "image/jpeg")
        if path.startswith("/api/input/"):
            fid = path.split("/")[3]
            return self._file(self.board.work_dir / "uploads" / fid / "input.mp4", "video/mp4")
        if path.startswith("/api/result/"):
            jid = path.split("/")[3]
            j = self.board.job(jid)
            if not j or j.status != "done" or not j.dst:
                return self._json({"ok": False, "error": "任务未完成"}, 404)
            return self._file(Path(j.dst), "video/mp4")
        if path.startswith("/api/download/"):
            jid = path.split("/")[3]
            j = self.board.job(jid)
            if not j or j.status != "done" or not j.dst:
                return self._json({"ok": False, "error": "任务未完成"}, 404)
            stem = Path(j.name).stem
            return self._file(Path(j.dst), "video/mp4", download_name=f"{stem}_vidpure.mp4")
        return self._json({"ok": False, "error": "接口不存在"}, 404)

    # ------------------------------------------------------------ POST
    def _post(self, path: str, qs: dict):
        if path == "/api/upload/start":
            body = self._json_body()
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
                meta = ffprobe(src)
            except Exception:                                    # noqa: BLE001
                src.unlink(missing_ok=True)
                return self._json({"ok": False,
                                   "error": "视频文件无法识别，请确认文件完整且为常见视频格式"}, 400)
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ss", "0.1",
                            "-frames:v", "1", "-vf", "scale=640:-2",
                            str(src.parent / "thumb.jpg")],
                           capture_output=True, text=True)
            return self._json({"ok": True, "fid": fid, "name": info["name"],
                               "duration": round(meta["duration"], 2),
                               "w": meta["w"], "h": meta["h"],
                               "has_audio": meta["has_audio"]})
        if path == "/api/batch":
            body = self._json_body()
            items = body.get("items") or []
            if not items:
                return self._json({"ok": False, "error": "没有收到视频"}, 400)
            if len(items) > MAX_ITEMS:
                return self._json({"ok": False,
                                   "error": f"一次最多提交 {MAX_ITEMS} 个视频，当前 {len(items)} 个"}, 400)
            opts = _opts_from_body(body)
            resolved = []
            for it in items:
                fid = str(it.get("fid", ""))
                src = self.board.work_dir / "uploads" / fid / "input.mp4"
                if not src.exists():
                    return self._json({"ok": False, "error": f"上传文件不存在（{fid}），请重新上传"}, 400)
                resolved.append({"src": str(src), "name": it.get("name") or f"{fid}.mp4"})
            try:
                jobs = self.board.submit(resolved, default_opts=opts,
                                         concurrency=body.get("concurrency"),
                                         retries=body.get("retries"))
            except BatchError as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            return self._json({"ok": True, "jobs": [j.snapshot() for j in jobs],
                               "batch": self.board.snapshot()})
        if path == "/api/settings":
            body = self._json_body()
            if body.get("concurrency") is not None:
                self.board.set_concurrency(int(body["concurrency"]))
            if body.get("retries") is not None:
                self.board.retries = max(0, int(body["retries"]))
            return self._json({"ok": True, "concurrency": self.board.concurrency,
                               "retries": self.board.retries})
        m = re.match(r"^/api/retry/(.+)$", path)
        if m:
            try:
                j = self.board.retry(m.group(1))
            except BatchError as e:
                return self._json({"ok": False, "error": str(e)}, 400)
            return self._json({"ok": True, "job": j.snapshot()})
        return self._json({"ok": False, "error": "接口不存在"}, 404)

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


# ---------------------------------------------------------------- 请求 → 选项
def _opts_from_body(body: dict) -> JobOptions:
    """把前端提交的 JSON 映射成 JobOptions（字段名与前端保持一致）。"""
    sub = body.get("subtitle") or {}
    box = sub.get("box")
    if box:
        box = tuple(float(x) for x in box)
    anchor = str(sub.get("anchor", "ratio")).lower()
    if anchor not in ("ratio", "pixel"):
        anchor = "ratio"
    margin_px = sub.get("margin_bottom_px")
    # anchor=ratio 时忽略固定像素，避免「既给比例又给像素」的歧义
    margin_px = (int(margin_px) if (anchor == "pixel" and margin_px) else None)
    return JobOptions(
        clean=str(body.get("clean", "delogo")),
        region=str(body.get("region", "0.78,0.95")),
        region_fmt=str(body.get("region_format", "auto")),
        target_res=body.get("target_res"),
        upscale_model=str(body.get("upscale_model", "realesr-animevideov3")),
        vcodec=str(body.get("vcodec") or _vcodec()),
        crf=int(body.get("crf", 18)),
        preset=str(body.get("preset", "medium")),
        keep_burn=bool(body.get("keep_burn", True)),
        subtitle=bool(sub.get("enabled")),
        sub_lang=str(sub.get("lang", "zh")),
        sub_mode=str(sub.get("mode", "bottom")),
        sub_box=box,
        sub_anchor=anchor,
        sub_anchor_top=float(sub.get("anchor_top", 0.6)),
        sub_font_size_px=(int(sub["font_size_px"]) if sub.get("font_size_px") else None),
        sub_font_size_pct=float(sub.get("font_size_pct", 5.6)),
        sub_margin_bottom_px=margin_px,
        sub_refine=str(sub.get("refine", "auto")),
    )


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
          work_dir: Path | None = None, verbose: bool = False,
          concurrency: int = 1, retries: int = 2) -> None:
    work = Path(work_dir or "./vidpure_output/web").resolve()
    board = BatchRunner(out_dir=work / "results", work_dir=work,
                        concurrency=concurrency, retries=retries)
    Handler.board = board
    Handler.verbose = verbose

    httpd = QuietThreadingHTTPServer((host, port), Handler)
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    print(f"VidPure 网页版已启动: {url}")
    caps = subtitle_burn.capabilities()
    print(f"  字幕能力: 台词识别={'可用' if caps['transcribe'] else '缺失'} · "
          f"CTC 对齐={'可用' if caps['ctc_align'] else '缺失（用 whisper 时间轴）'}")
    print(f"  批量: 单批最多 {MAX_ITEMS} 个 · 并发 {board.concurrency} · 失败重试 {board.retries} 次")
    if host == "0.0.0.0":
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            print(f"手机访问（同一 WiFi）: http://{s.getsockname()[0]}:{port}")
            s.close()
        except Exception:                                        # noqa: BLE001
            pass
    print("按 Ctrl+C 停止服务")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        board.shutdown()
