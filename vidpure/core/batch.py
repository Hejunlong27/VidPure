"""批量任务队列：队列 → 并发控制 → 失败重试。

针对需求「一次性提交并处理最多 60 个 90 秒视频」：
  · 单次提交上限 `BatchLimits.max_items`（默认 60）
  · 单条时长上限 `BatchLimits.max_duration_s`（默认 90s）；超限条目按 `enforce_limits`
    决定是「拒绝该条、其余照跑」还是「直接报错」
  · 并发度 `concurrency`（默认 1：GPU 逐帧修复/超分本就吃满显卡；快速通道可调到 CPU 核数）
  · 失败重试 `retries` 次，指数退避；重试期间**释放工作槽**（用定时器重新入队）

设计取舍
  · 用 `ThreadPoolExecutor` + 定时器实现重试，而不是「线程内 sleep 重试」——
    后者会在退避期间占住并发槽，让整批吞吐骤降。
  · 每个作业独立 try/except：**单条失败绝不影响整批**（对应原 skill 的「批量跑到某集就全挂」教训）。
  · 作业表可快照成 JSON，供 Web 轮询与断点查看。
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import resolve, subtitle, subtitle_burn, upscale
from .media import ffprobe
from .probe import probe

# ------------------------------------------------------------------ 阶段文案
STAGE_TEXT = {
    "queued": "排队等待中…",
    "clean": "正在抹除原字幕…",
    "upscale": "正在提升画质（AI 超分）…",
    "subtitle": "正在识别台词并烧录字幕…",
    "mux": "正在封装输出…",
    "retry": "处理失败，等待重试…",
    "done": "处理完成",
    "rejected": "已跳过（不满足批量条件）",
}

DEFAULT_MAX_ITEMS = 60
DEFAULT_MAX_DURATION_S = 90.0


@dataclass
class BatchLimits:
    max_items: int = DEFAULT_MAX_ITEMS
    max_duration_s: float = DEFAULT_MAX_DURATION_S
    enforce_limits: bool = True          # False = 只告警不拒绝


@dataclass
class JobOptions:
    """单条作业的全部处理参数（与 Web / CLI 一一对应）。"""
    clean: str = "delogo"                # delogo / blur / crop / lama / none
    region: str = "0.78,0.95"
    region_fmt: str = "auto"
    scale: int = 0                       # 旧式倍数（0 = 不用倍数模式）
    target_res: str | None = None         # 固定分辨率预设 id / 'WxH' / None=不缩放
    upscale_model: str = "realesr-animevideov3"
    vcodec: str = "libx264"
    crf: int = 18
    preset: str = "medium"
    keep_burn: bool = True                # 保留「烧字幕前」的无字幕母版
    subtitle: bool = False                # 是否烧录字幕
    sub_lang: str = "zh"
    sub_srt: str | None = None            # 用现成 SRT（跳过识别）
    sub_mode: str = "bottom"              # bottom / box
    sub_box: tuple[float, float, float, float] | None = None
    sub_anchor: str = "ratio"             # ratio（6:4 视觉中心，默认）/ pixel（固定距底）
    sub_anchor_top: float = 0.6           # ratio 模式：文字块视觉中心距顶占比
    sub_font_size_px: int | None = None
    sub_font_size_pct: float = 5.6
    sub_margin_bottom_px: int | None = None
    sub_refine: str = "auto"              # auto / ctc / none


@dataclass
class Job:
    jid: str
    src: str
    name: str
    opts: JobOptions
    duration: float = 0.0
    status: str = "queued"                # queued / running / done / error / rejected
    stage: str = "queued"
    message: str = STAGE_TEXT["queued"]
    progress: float = 0.0
    attempts: int = 0
    max_attempts: int = 1
    dst: str = ""
    subbed: str = ""
    master: str = ""                       # 烧字幕前的无字幕母版（keep_burn 时保留）
    srt: str = ""
    error: str = ""
    warnings: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    started: float = 0.0
    finished: float = 0.0

    def snapshot(self) -> dict:
        return {
            "id": self.jid, "name": self.name, "status": self.status,
            "stage": self.stage, "stage_text": STAGE_TEXT.get(self.stage, self.message),
            "progress": round(self.progress, 3), "message": self.message,
            "attempts": self.attempts, "max_attempts": self.max_attempts,
            "duration": round(self.duration, 2),
            "dst": self.dst, "error": self.error, "warnings": self.warnings[:12],
            "video_url": f"/api/result/{self.jid}/output.mp4" if self.status == "done" else "",
            "download_url": f"/api/download/{self.jid}" if self.status == "done" else "",
            "srt_url": f"/api/srt/{self.jid}" if self.srt else "",
            "has_master": bool(self.master),
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
        }


class BatchError(Exception):
    pass


class BatchRunner:
    """批量执行器：提交 → 队列 → 并发 worker（含重试）。"""

    def __init__(self, out_dir: Path, work_dir: Path | None = None, concurrency: int = 1,
                 retries: int = 2, limits: BatchLimits | None = None,
                 retry_backoff: float = 3.0, on_update=None, verbose: bool = False):
        self.out_dir = Path(out_dir).resolve()
        self.work_dir = Path(work_dir).resolve() if work_dir else self.out_dir / "_work"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = max(1, int(concurrency))
        self.retries = max(0, int(retries))
        self.retry_backoff = max(0.0, float(retry_backoff))   # 退避基数（秒），可调便于测试
        self.limits = limits or BatchLimits()
        self.on_update = on_update
        self.verbose = verbose
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.RLock()
        self._timers: list[threading.Timer] = []
        self._pool = ThreadPoolExecutor(max_workers=self.concurrency,
                                        thread_name_prefix="vidpure-batch")
        self._hw = None
        self._active = 0

    # -------------------------------------------------------------- 硬件（懒探测）
    def hw(self):
        if self._hw is None:
            self._hw = probe()
        return self._hw

    def _vcodec(self) -> str:
        return "h264_nvenc" if self.hw().gpu_name else "libx264"

    def _device(self) -> str:
        return "cuda" if self.hw().torch_cuda else "cpu"

    # -------------------------------------------------------------- 提交
    def submit(self, items: list[dict], default_opts: JobOptions | None = None,
               concurrency: int | None = None, retries: int | None = None) -> list[Job]:
        """提交一批作业。items: [{'src': 路径, 'name': 可省, **opts 覆盖}]。

        返回新建的 Job 列表（含被拒绝的条目，其 status='rejected'）。
        """
        if concurrency is not None:
            self.set_concurrency(int(concurrency))
        if retries is not None:
            self.retries = max(0, int(retries))
        if not items:
            raise BatchError("没有可提交的视频")
        if len(items) > self.limits.max_items:
            raise BatchError(f"一次最多提交 {self.limits.max_items} 个视频，"
                             f"当前 {len(items)} 个；请分批提交")
        created: list[Job] = []
        for it in items:
            src = Path(str(it.get("src", "")).strip('"'))
            name = it.get("name") or (src.name if src.name else "视频")
            opts = _merge_opts(default_opts or JobOptions(), it)
            j = Job(jid=uuid.uuid4().hex[:12], src=str(src), name=name, opts=opts,
                    max_attempts=self.retries + 1)
            if not src.exists():
                j.status, j.stage, j.error = "rejected", "rejected", "文件不存在"
                j.message = j.error
            else:
                try:
                    info = ffprobe(src)
                    j.duration = float(info.get("duration") or 0.0)
                    if self.limits.enforce_limits and j.duration > self.limits.max_duration_s + 0.5:
                        j.status, j.stage = "rejected", "rejected"
                        j.error = (f"时长 {j.duration:.0f}s 超过单条上限 "
                                   f"{self.limits.max_duration_s:.0f}s")
                        j.message = j.error
                except Exception as e:                # noqa: BLE001
                    j.status, j.stage = "rejected", "rejected"
                    j.error = f"无法读取视频信息：{str(e)[:120]}"
                    j.message = j.error
            with self.lock:
                self.jobs[j.jid] = j
                self.order.append(j.jid)
            created.append(j)
            self._notify(j)
        for j in created:
            if j.status == "queued":
                self._pool.submit(self._attempt, j)
        return created

    def set_concurrency(self, n: int):
        """动态调整并发度（对已在跑的作业即时生效；未启动的按新并发排队）。"""
        n = max(1, int(n))
        if n == self.concurrency:
            return
        self.concurrency = n
        old = self._pool
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="vidpure-batch")
        old.shutdown(wait=False)

    # -------------------------------------------------------------- 重试/手动
    def retry(self, jid: str) -> Job:
        with self.lock:
            j = self.jobs.get(jid)
        if not j:
            raise BatchError("任务不存在")
        if j.status not in ("error", "rejected"):
            raise BatchError("仅失败的作业可以重试")
        if j.status == "rejected":
            j.status = "queued"
        j.status = "queued"
        j.stage, j.message, j.progress = "queued", STAGE_TEXT["queued"], 0.0
        j.error = ""
        j.attempts = 0
        j.max_attempts = max(j.max_attempts, self.retries + 1)
        self._pool.submit(self._attempt, j)
        self._notify(j)
        return j

    # -------------------------------------------------------------- worker
    def _attempt(self, j: Job):
        with self.lock:
            if j.status not in ("queued",):
                return
            j.status = "running"
            j.attempts += 1
            j.started = j.started or time.time()
            j.stage, j.progress = "clean", 0.02
            j.message = STAGE_TEXT["clean"]
        self._notify(j)
        try:
            self._run_job(j)
            with self.lock:
                j.status, j.stage, j.progress = "done", "done", 1.0
                j.message = STAGE_TEXT["done"]
                j.finished = time.time()
            self._write_report(j, self.out_dir / "results" / j.jid)
            self._notify(j)
        except Exception as e:                        # noqa: BLE001
            msg = _friendly(e)
            with self.lock:
                j.error = msg
                if j.attempts < j.max_attempts:
                    j.status = "queued"
                    j.stage = "retry"
                    j.message = f"{STAGE_TEXT['retry']}（第 {j.attempts}/{j.max_attempts} 次）"
                else:
                    j.status, j.stage, j.progress = "error", "error", 1.0
                    j.message = "处理失败"
                j.finished = time.time()
            self._notify(j)
            if j.status == "queued":
                backoff = min(30.0, self.retry_backoff * (2 ** (j.attempts - 1)))
                t = threading.Timer(backoff, lambda: self._pool.submit(self._attempt, j))
                t.daemon = True
                with self.lock:
                    self._timers.append(t)
                t.start()
        finally:
            with self.lock:
                self._timers = [t for t in self._timers if t.is_alive()]

    def _set(self, j: Job, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(j, k, v)
        self._notify(j)

    def _run_job(self, j: Job):
        """clean → upscale（固定分辨率）→ 字幕烧录。先放大后烧字幕（原 skill 铁律）。"""
        o = j.opts
        src = Path(j.src)
        out_dir = self.out_dir / "results" / j.jid
        out_dir.mkdir(parents=True, exist_ok=True)
        stages = [bool(o.clean and o.clean != "none"),
                  bool(o.target_res or (o.scale and int(o.scale) > 0)),
                  bool(o.subtitle)]
        weights = _weights(stages)
        acc = 0.0

        cur = src
        # ---- 阶段 1：去原字幕
        if stages[0]:
            self._set(j, stage="clean", progress=0.05, message=STAGE_TEXT["clean"])
            mid = out_dir / f"{src.stem}_cleaned.mp4"
            if o.clean in ("delogo", "blur", "crop"):
                subtitle.clean_video_fast(cur, mid, o.region, mode=o.clean,
                                          region_fmt=o.region_fmt, vcodec=o.vcodec,
                                          crf=o.crf, preset=o.preset)
            elif o.clean == "lama":
                subtitle.clean_video_lama(cur, mid, o.region, o.region_fmt,
                                          device=self._device(), vcodec=o.vcodec,
                                          crf=o.crf, preset=o.preset)
            else:
                raise BatchError(f"未知去字幕方式: {o.clean}")
            cur = mid
            acc += weights[0]
            self._set(j, progress=acc)

        # ---- 阶段 2：放大（固定分辨率优先；`scale` 为旧式倍数，兼容保留）
        if stages[1]:
            self._set(j, stage="upscale", progress=min(0.9, acc + 0.02),
                      message=STAGE_TEXT["upscale"])
            info = ffprobe(cur)
            dst = out_dir / f"{src.stem}_up.mp4"
            if o.target_res:
                target = resolve.resolve_target(info["w"], info["h"], o.target_res)
            else:
                target = None
            if target is not None:
                upscale.upscale_video_ncnn(cur, dst, model=o.upscale_model,
                                           tile=self.hw().recommendations.get("upscale_tile", 256),
                                           target=target, vcodec=o.vcodec, crf=o.crf,
                                           preset=o.preset, progress=lambda m: None)
                cur = dst
            elif o.scale and int(o.scale) > 0:
                upscale.upscale_video_ncnn(cur, dst, scale=int(o.scale),
                                           model=o.upscale_model,
                                           tile=self.hw().recommendations.get("upscale_tile", 256),
                                           vcodec=o.vcodec, crf=o.crf,
                                           preset=o.preset, progress=lambda m: None)
                cur = dst
            acc += weights[1]
            self._set(j, progress=acc)

        # ---- 阶段 3：字幕识别 + 烧录
        if stages[2]:
            self._set(j, stage="subtitle", progress=min(0.95, acc + 0.02),
                      message=STAGE_TEXT["subtitle"])
            placement = subtitle_burn.Placement(
                mode=o.sub_mode, box=o.sub_box, font_size_px=o.sub_font_size_px,
                font_size_pct=o.sub_font_size_pct,
                anchor=o.sub_anchor, anchor_top=o.sub_anchor_top,
                margin_bottom_px=o.sub_margin_bottom_px)
            dst = out_dir / "output.mp4"
            rep = subtitle_burn.add_subtitles(
                cur, dst, lang=o.sub_lang, placement=placement,
                srt=Path(o.sub_srt) if o.sub_srt else None, refine=o.sub_refine,
                work_dir=out_dir / "subs", report={}, progress=lambda m: None)
            j.report = rep
            j.subbed = str(dst)
            j.srt = rep.get("srt", "")
            cur = dst
            acc += weights[2]
            self._set(j, progress=acc)
        else:
            dst = out_dir / "output.mp4"
            if cur != dst:
                if cur == src:                       # 两阶段都没产出中间文件：复制保护源文件
                    shutil.copyfile(cur, dst)
                else:
                    cur.replace(dst)
            cur = dst

        j.dst = str(cur)
        # 无字幕母版语义：keep_burn=True 时把「烧字幕前」的母版另存，供用户再出无字幕版
        master = out_dir / f"{src.stem}_master.mp4"
        if stages[2] and o.keep_burn:
            burned_src = out_dir / f"{src.stem}_up.mp4"
            if not burned_src.exists():
                burned_src = out_dir / f"{src.stem}_cleaned.mp4"
            if burned_src.exists():
                try:
                    burned_src.replace(master)
                    j.master = str(master)
                except OSError:
                    pass
        # 清理中间大文件：永不让删除异常冒泡（原 skill 的批量中断教训）
        for f in ("_cleaned.mp4", "_up.mp4", "_master.mp4"):
            p = out_dir / f"{src.stem}{f}"
            keep = (f == "_master.mp4" and o.keep_burn and stages[2])
            if p.exists() and not keep:
                try:
                    p.unlink()
                except OSError:
                    pass
        self._set(j, stage="mux", progress=0.98, message=STAGE_TEXT["mux"])

    def _write_report(self, j: Job, out_dir: Path):
        """落盘运行报告（便于复核字幕条数 / 画布 / 收紧结果）。"""
        try:
            (out_dir / "report.json").write_text(
                json.dumps({"job": j.snapshot(), "opts": j.opts.__dict__,
                            "subtitle_report": j.report}, ensure_ascii=False, indent=2,
                           default=str), encoding="utf-8")
        except OSError:
            pass

    # -------------------------------------------------------------- 快照 / 落盘
    def _notify(self, j: Job):
        if self.on_update:
            try:
                self.on_update(j)
            except Exception:                          # noqa: BLE001
                pass

    def snapshot(self) -> dict:
        with self.lock:
            jobs = [self.jobs[i].snapshot() for i in self.order]
        counts = {}
        for j in jobs:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        return {"jobs": jobs, "counts": counts, "concurrency": self.concurrency,
                "retries": self.retries, "total": len(jobs)}

    def job(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def persist(self, path: Path | None = None):
        path = Path(path) if path else self.out_dir / "batch_jobs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def shutdown(self):
        with self.lock:
            for t in self._timers:
                t.cancel()
        self._pool.shutdown(wait=False)


# ------------------------------------------------------------------ 工具
def _weights(stages: list[bool]) -> list[float]:
    """按启用的阶段分配进度权重（总长 1.0）。"""
    base = {0: 0.45, 1: 0.40, 2: 0.15}
    active = [i for i, s in enumerate(stages) if s]
    if not active:
        return [0.0, 0.0, 0.0]
    total = sum(base[i] for i in active)
    return [(base[i] / total) if stages[i] else 0.0 for i in range(3)]


def _merge_opts(base: JobOptions, item: dict) -> JobOptions:
    """用提交项里的键覆盖默认参数（只认已知字段，避免脏键）。"""
    known = set(JobOptions.__dataclass_fields__.keys())
    kw = {k: v for k, v in item.items() if k in known and k != "src"}
    if "sub_box" in kw and kw["sub_box"] is not None:
        kw["sub_box"] = tuple(float(x) for x in kw["sub_box"])
    return JobOptions(**{**base.__dict__, **kw})


def _friendly(e: Exception) -> str:
    s = str(e)
    pairs = [
        ("No module named 'iopaint'", "高清修复需要 AI 组件：请运行 pip install iopaint，或改用「快速去字幕」"),
        ("no kernel image", "当前 PyTorch 不支持该显卡（RTX 50 系需新版 PyTorch），建议改用「快速去字幕」"),
        ("Vulkan", "显卡超分引擎启动失败：请更新显卡驱动后重试"),
        ("未找到 whisper-cli", "未检测到 whisper-cli：台词识别需要它，请设置 VIDPURE_WHISPER_CLI"),
        ("未找到 whisper 模型", "未检测到 whisper 模型：请设置 VIDPURE_WHISPER_MODEL 指向 ggml 文件"),
        ("未得到任何字幕条目", "未识别到任何人声台词，已跳过字幕烧录"),
        ("抽帧失败", "视频文件无法读取：请确认文件完整且为标准视频格式"),
        ("模型缺失", "AI 超分模型缺失：请联网重试以自动下载模型"),
    ]
    for k, msg in pairs:
        if k.lower() in s.lower():
            return msg
    return f"处理失败：{s[:200]}"
