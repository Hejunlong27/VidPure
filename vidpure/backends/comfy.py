"""ComfyUI 后端：自动拉起本机 ComfyUI server 并把任务派给它执行。

职责：
  1. start(): 以子进程方式启动 ComfyUI（python main.py --port 8188），轮询 /system_stats 直到就绪
  2. submit(): 把 vidpure 的去字幕/放大意图渲染成 API 工作流 JSON，POST /prompt
  3. wait(): 轮询 /history/{prompt_id} 直到完成（零额外依赖，不依赖 websocket）
  4. fetch(): 从 /history outputs 提取输出文件并经 /view 下载

依赖的 ComfyUI 自定义节点（由用户装在 ComfyUI 内）：
  ComfyUI-VideoHelperSuite（VHS_LoadVideo / VHS_VideoCombine）
  ComfyUI-IOPaint（IOPaint 节点，LaMa 去字幕）
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from .base import Backend

DEFAULT_PORT = 8188


def _http_json(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class ComfyBackend(Backend):
    name = "comfy"

    def __init__(self, comfy_dir: str | None = None, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, python: str | None = None,
                 auto_start: bool = True):
        self.comfy_dir = Path(comfy_dir) if comfy_dir else None
        self.host = host
        self.port = port
        self.base = f"http://{host}:{port}"
        self.python = python or sys.executable
        self.auto_start = auto_start
        self._proc: subprocess.Popen | None = None

    # ------------------------------------------------ 就绪 / 启动
    def health_check(self) -> bool:
        try:
            _http_json(f"{self.base}/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def start(self, wait_sec: int = 120) -> None:
        """server 已跑则直接复用；否则需要 comfy_dir 并自动拉起。"""
        if self.health_check():
            print(f"[ComfyUI] 复用运行中的 server: {self.base}")
            return
        if not self.auto_start or not self.comfy_dir:
            raise RuntimeError(f"ComfyUI 未运行于 {self.base}，且未提供 --comfy-dir 自动拉起")
        main_py = self.comfy_dir / "main.py"
        if not main_py.exists():
            raise RuntimeError(f"未找到 {main_py}，请确认 ComfyUI 安装目录")
        print(f"[ComfyUI] 自动拉起: {self.python} {main_py} --port {self.port}")
        self._proc = subprocess.Popen(
            [self.python, str(main_py), "--port", str(self.port)],
            cwd=str(self.comfy_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        t0 = time.time()
        while time.time() - t0 < wait_sec:
            if self.health_check():
                print(f"[ComfyUI] 就绪 ({time.time() - t0:.0f}s)")
                return
            if self._proc.poll() is not None:
                raise RuntimeError("ComfyUI 进程提前退出，请检查其环境依赖")
            time.sleep(2)
        raise TimeoutError(f"ComfyUI {wait_sec}s 内未就绪")

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            print("[ComfyUI] 自动拉起的 server 已停止")

    # ------------------------------------------------ 任务提交
    def submit(self, workflow_api: dict) -> str:
        """提交 API 格式工作流，返回 prompt_id。"""
        resp = _http_json(f"{self.base}/prompt", {"prompt": workflow_api})
        return resp["prompt_id"]

    def wait(self, prompt_id: str, timeout: int = 3600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            hist = _http_json(f"{self.base}/history/{prompt_id}")
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("completed"):
                    return entry
                if status.get("status_str") == "error":
                    raise RuntimeError(f"工作流执行出错: {json.dumps(status, ensure_ascii=False)[:500]}")
            time.sleep(2)
        raise TimeoutError("等待超时")

    def run_workflow(self, workflow_api: dict, timeout: int = 3600) -> dict:
        pid = self.submit(workflow_api)
        print(f"[ComfyUI] 提交工作流 prompt_id={pid}")
        return self.wait(pid, timeout)

    # ------------------------------------------------ 输出取回
    def fetch_outputs(self, entry: dict, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for node_out in (entry.get("outputs") or {}).values():
            for key in ("images", "videos", "gifs"):
                for item in node_out.get(key) or []:
                    fname = item["filename"]
                    sub = item.get("subfolder", "")
                    url = (f"{self.base}/view?filename={urllib.request.quote(fname)}"
                           f"&subfolder={urllib.request.quote(sub)}&type={item.get('type', 'output')}")
                    dst = out_dir / fname
                    urllib.request.urlretrieve(url, dst)
                    saved.append(dst)
        return saved

    # ------------------------------------------------ Backend 接口
    def clean(self, src: Path, dst: Path, region: str, mode: str = "lama") -> Path:
        from .workflows import subtitle_workflow
        self.start()
        wf = subtitle_workflow(video_path=str(src.resolve()),
                               region=region, model="lama" if mode == "lama" else "sttn",
                               out_name=dst.name)
        entry = self.run_workflow(wf)
        files = self.fetch_outputs(entry, dst.parent)
        if not files:
            raise RuntimeError("ComfyUI 未返回输出文件")
        if files[0] != dst:
            files[0].replace(dst)
        return dst

    def upscale(self, src: Path, dst: Path, scale: int, model: str) -> Path:
        from .workflows import upscale_workflow
        self.start()
        wf = upscale_workflow(video_path=str(src.resolve()), model_name=model,
                              out_name=dst.name)
        entry = self.run_workflow(wf)
        files = self.fetch_outputs(entry, dst.parent)
        if not files:
            raise RuntimeError("ComfyUI 未返回输出文件")
        if files[0] != dst:
            files[0].replace(dst)
        return dst
