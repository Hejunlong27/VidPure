"""字幕识别 / 对齐 / 烧录 —— 移植自 `drama-subtitle-pipeline`，改写为 vidpure 自包含实现。

三件事（对应需求「台词识别、字幕对齐、烧录」）
------------------------------------------------------------------
① 台词识别 `transcribe()`        whisper-cli ASR → 带时间戳的字幕条目（词级对齐后合并成句）
② 字幕对齐 `refine_alignment()`  逐条 CTC 强制对齐（可选）+ 音频能量收紧（默认）
③ 烧录     `burn_subtitles()`    ASS 渲染（支持**手画框**定位 + **可调字号**）→ ffmpeg 烧录

与原 skill 的差异（刻意的）
  · 原 skill 的台词真源是剧本 `<d>` 标签（有稿对齐）；vidpure 的输入是**任意视频**，
    没有剧本 ⇒ 台词由 whisper **识别**得到，再用 CTC 对齐器在时间轴上精修。
    因此本模块的方向是「识别 → 精确时间 → 渲染」，而不是「已知文本 → 找位置」。
  · 不依赖 `config.subtitle.json`：全部样式参数走函数入参 / `SubtitleStyle` dataclass，
    便于 Web 与 CLI 直接透传用户选择。
  · 最小可读时长 / 语速护栏沿用原引擎的口径（宁可字幕略紧，也不产出念不完的字幕）。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import media

# ------------------------------------------------------------------ 语言样式预设
# char_width_em：一个字符的宽度相对字号的比例（中文等宽 1.0，拉丁约 0.52）
LANG_PRESETS: dict[str, dict] = {
    "zh": {
        "font": "Microsoft YaHei", "wrap": "char", "char_width_em": 1.0,
        "break_punct": "，。！？；：…—、",
        "forbid_line_start": "，。！？；：、）》」』…—",
        "forbid_line_end": "（《「『",
        "max_line_chars": 16, "line_tolerance_chars": 3, "max_lines": 2,
        "max_cue_chars": 32, "min_duration_s": 0.8, "max_duration_s": 6.0,
        "whisper_lang": "zh", "aligner_iso": "cmn", "word_level": True,
        "join_sep": "",
    },
    "es": {
        "font": "Microsoft YaHei", "wrap": "word", "char_width_em": 0.52,
        "break_punct": ".,!?;:…—",
        "forbid_line_start": ".,!?;:)]}»…—",
        "forbid_line_end": "([{«",
        "max_line_chars": 42, "line_tolerance_chars": 6, "max_lines": 2,
        "max_cue_chars": 84, "min_duration_s": 0.8, "max_duration_s": 6.0,
        "whisper_lang": "es", "aligner_iso": "spa", "word_level": False,
        "join_sep": " ",
    },
    "en": {
        "font": "Microsoft YaHei", "wrap": "word", "char_width_em": 0.52,
        "break_punct": ".,!?;:…—",
        "forbid_line_start": ".,!?;:)]}»…—",
        "forbid_line_end": "([{«",
        "max_line_chars": 42, "line_tolerance_chars": 6, "max_lines": 2,
        "max_cue_chars": 84, "min_duration_s": 0.8, "max_duration_s": 6.0,
        "whisper_lang": "en", "aligner_iso": "eng", "word_level": False,
        "join_sep": " ",
    },
}

# ------------------------------------------------------------------ 字幕摆位口径
# ★ 默认（anchor="ratio"）：把文字块的**视觉中心**放在 距顶 : 距底 = 6 : 4 处，
#   即中心 y = 0.6 × 画面高。**按分辨率等比缩放** ⇒ 任何输出尺寸下视觉重心一致；
#   固定像素（旧做法）会在换分辨率时漂移（同一个 400px 在 1080p 上占 37% 高，
#   在 2880p 上只占 14%），所以不再作为默认。
#   实现：`\an5\pos(W/2, round(0.6H))` —— `\an5` 以文字块中心为锚点，逐条精确居中，
#   因此 1 行与 2 行字幕的中心都落在同一条线上（不随行数浮动）。
ANCHOR_TOP_DEFAULT = 0.6        # = 距顶 6 : 距底 4
LINE_HEIGHT_EM = 1.2            # 估算文字块高度用的行高系数（仅用于报告，不入 ASS）
# anchor="pixel"（旧行为，保留为可选项）：距画面底部固定像素
DEFAULT_MARGIN_BOTTOM_PX = 400
SAFE_MAX_TEXT_WIDTH_PCT = 70.0
# 仅 pixel 模式的兜底：距底不得超过「画面高 − 两行字幕高」（否则字幕会被顶出画面）
SAFE_MAX_BOTTOM_FRAC = 0.85

# whisper-cli / 模型 / 对齐器的常见落点。优先级：
#   ① 环境变量（推荐，跨机器可配）② 用户主目录下的约定路径 ③ PATH（shutil.which）
# 刻意不写死任何绝对路径：本文件要能直接进开源仓库，不绑定某一台机器。
WHISPER_CLI_HINTS = [
    os.environ.get("VIDPURE_WHISPER_CLI") or "",
    str(Path.home() / "whisper-cli" / "Release" / "whisper-cli.exe"),
]
WHISPER_MODEL_HINTS = [
    os.environ.get("VIDPURE_WHISPER_MODEL") or "",
    str(Path.home() / "whisper-models" / "ggml-base.bin"),
]
ALIGNER_HINTS = [
    os.environ.get("VIDPURE_ALIGNER") or "",
    str(Path.home() / "align-env" / "Scripts" / "ctc-forced-aligner.exe"),
]


def _first_existing(hints: list[str]) -> str:
    for h in hints:
        if h and Path(h).exists():
            return h
    return ""


def capabilities() -> dict:
    """探测字幕相关能力（供 Web `/api/probe` 与 CLI 展示）。"""
    wc = _first_existing(WHISPER_CLI_HINTS) or (shutil.which("whisper-cli") or "")
    model = _first_existing(WHISPER_MODEL_HINTS)
    al = _first_existing(ALIGNER_HINTS) or (shutil.which("ctc-forced-aligner") or "")
    return {
        "transcribe": bool(wc and model),      # 台词识别（whisper）
        "whisper_cli": wc,
        "whisper_model": model,
        "ctc_align": bool(al),                 # 字幕对齐（CTC 强制对齐，可选精修）
        "aligner": al,
        "fonts": _has_font("Microsoft YaHei"),
        # 默认摆位口径（单一真源，供 Web/CLI 展示，避免各处硬编码）
        "anchor": "ratio",
        "anchor_top": ANCHOR_TOP_DEFAULT,
        "anchor_label": ratio_label(ANCHOR_TOP_DEFAULT),
        "anchor_desc": f"文字块视觉中心位于画面 {ANCHOR_TOP_DEFAULT:.0%} 高度处"
                       f"（距顶:距底 = {ratio_label(ANCHOR_TOP_DEFAULT)}）",
        "margin_bottom_px_default": DEFAULT_MARGIN_BOTTOM_PX,
        "safe_max_text_width_pct": SAFE_MAX_TEXT_WIDTH_PCT,
    }


def _has_font(name: str) -> bool:
    """粗略判断字体是否可用（Windows 常见字体目录）。"""
    try:
        root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        if not root.exists():
            return True
        blob = " ".join(p.name.lower() for p in root.iterdir()).lower()
        key = name.replace(" ", "").lower()
        return key in blob or "msyh" in blob
    except Exception:            # noqa: BLE001
        return True


# ------------------------------------------------------------------ 数据结构
@dataclass
class Cue:
    start: float
    end: float
    text: str
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "lines": self.lines}


@dataclass
class Placement:
    """字幕**摆在哪、多大字** —— 对应需求「按分辨率摆位」+「手动画框」+「可调字体大小」。

    mode="bottom"（默认）：水平居中；垂直位置由 `anchor` 决定 ——
      · `anchor="ratio"`（默认）：文字块**视觉中心**落在 `anchor_top × 画面高` 处。
        默认 `anchor_top=0.6`，即 距顶:距底 = **6:4**，随分辨率等比缩放。
      · `anchor="pixel"`（旧行为 / 显式指定）：`margin_bottom_px` 为距画面底部像素；
        省略时取 `DEFAULT_MARGIN_BOTTOM_PX`。
    mode="box"：在用户手画的矩形内居中（`box` 为归一化 x0,y0,x1,y1），不受 anchor 影响。

    `font_size_px=None` 表示按画布高度自动（`font_size_pct`），并在可用宽度内收窄。
    """
    mode: str = "bottom"
    box: tuple[float, float, float, float] | None = None
    anchor: str = "ratio"                     # ratio（6:4 视觉中心）/ pixel（固定距底）
    anchor_top: float = ANCHOR_TOP_DEFAULT     # ratio 模式：视觉中心距顶占比
    margin_bottom_px: int | None = None        # pixel 模式：距底像素
    font_size_px: int | None = None
    font_size_pct: float = 5.6
    margin_lr_pct: float = 4.2
    font: str | None = None
    outline: int = 3
    shadow: int = 1
    bold: bool = True
    primary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    back_colour: str = "&H80000000"

    def to_dict(self) -> dict:
        return {"mode": self.mode, "box": list(self.box) if self.box else None,
                "anchor": self.anchor, "anchor_top": self.anchor_top,
                "font_size_px": self.font_size_px, "font_size_pct": self.font_size_pct,
                "margin_bottom_px": self.margin_bottom_px, "font": self.font}


# ------------------------------------------------------------------ 小工具
def lang_cfg(lang: str) -> dict:
    return dict(LANG_PRESETS.get(lang, LANG_PRESETS["zh"]))


def harden_env() -> dict:
    """whisper / 对齐器在 Windows 上需要 UTF-8，否则读中文文本崩（原 skill 铁律 4）。"""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run(cmd: list, env: dict | None = None, timeout: int | None = None) -> tuple[int, str]:
    """执行外部命令；超时抛 RuntimeError（对齐器实测会死锁，必须带超时）。"""
    try:
        p = subprocess.run(cmd, env=env or harden_env(), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"命令超时（{timeout}s 未返回，已强杀）: {' '.join(map(str, cmd))}")
    return p.returncode, p.stdout.decode("utf-8", errors="replace")


def esc_filter_path(p) -> str:
    """ffmpeg 滤镜里的绝对路径：必须『转义冒号 + 整值加单引号』（已实测）。"""
    s = str(p).replace("\\", "/")
    s = re.sub(r"^([A-Za-z]):", r"\1\\:", s)
    return "filename='%s'" % s


def _to_wav16k(src: Path, wav: Path) -> Path:
    media.run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ar", "16000", "-ac", "1",
               "-c:a", "pcm_s16le", str(wav)])
    return wav


# ------------------------------------------------------------------ SRT 解析 / 导出
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(path: Path | str) -> list[dict]:
    """解析 SRT → [{'start','end','text'}]；编码容错，缺失返回 []。"""
    try:
        txt = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    out: list[dict] = []
    for blk in re.split(r"\n\s*\n", txt.replace("\r", "").strip()):
        ls = [x for x in blk.splitlines() if x.strip()]
        if len(ls) < 3:
            continue
        m = _SRT_TIME.match(ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        out.append({"start": a, "end": b, "text": " ".join(ls[2:]).strip()})
    return out


def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def export_srt(cues: list[Cue], out: Path) -> Path:
    """导出外挂 SRT（同时供用户自取 / 二次编辑）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i, c in enumerate(cues, 1):
        txt = "\n".join(c.lines) if c.lines else c.text
        blocks.append("%d\n%s --> %s\n%s" % (i, _srt_time(c.start), _srt_time(c.end), txt))
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------ ① 台词识别（ASR）
def transcribe(src: Path, lang: str = "zh", model: str = "", cli: str = "",
               out_dir: Path | None = None, progress=print) -> dict:
    """用 whisper-cli 识别台词 → 字幕条目。

    CJK 语言用 `-ml 1` 取**逐字**时间戳再合并成句（时间更准）；
    拉丁语系用 whisper 自然分句。结果按句聚合后断行。
    """
    lc = lang_cfg(lang)
    wc = cli or _first_existing(WHISPER_CLI_HINTS) or (shutil.which("whisper-cli") or "")
    md = model or _first_existing(WHISPER_MODEL_HINTS)
    if not wc or not Path(wc).exists():
        raise RuntimeError("未找到 whisper-cli：请设置环境变量 VIDPURE_WHISPER_CLI 或安装 whisper.cpp")
    if not md or not Path(md).exists():
        raise RuntimeError("未找到 whisper 模型：请设置环境变量 VIDPURE_WHISPER_MODEL 指向 ggml 模型")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"vidpure_asr_{src.stem}_",
                                   dir=str(out_dir) if out_dir else None))
    try:
        wav = _to_wav16k(src, tmpdir / "a.wav")
        base = tmpdir / "out"
        cmd = [wc, "-m", md, "-f", str(wav), "-l", lc["whisper_lang"],
               "-osrt", "-of", str(base), "-np"]
        if lc["word_level"]:
            cmd += ["-ml", "1"]
        progress(f"[台词识别] whisper({lc['whisper_lang']}) 转写中…")
        rc, out = _run(cmd, timeout=1800)
        srt = Path(str(base) + ".srt")
        if not srt.exists():
            raise RuntimeError(f"whisper 未产出字幕（rc={rc}）: {out[-400:]}")
        entries = parse_srt(srt)
        if not entries:
            raise RuntimeError("whisper 未识别到任何语音（可能无音轨或纯音乐）")
        cues = group_entries(entries, lc)
        result = {"cues": cues, "entries": len(entries), "lang": lang,
                  "whisper_cli": wc, "model": Path(md).name}
        # 保留 whisper 原始 SRT 供复核
        if out_dir:
            dst = Path(out_dir) / f"{src.stem}.asr.srt"
            shutil.copyfile(srt, dst)
            result["asr_srt"] = str(dst)
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def group_entries(entries: list[dict], lc: dict, max_gap: float = 0.7) -> list[Cue]:
    """把（可能碎到逐字的）ASR 条目聚合成句级字幕条目。

    切句规则：间隔 > max_gap / 累计超 max_cue_chars / 累计时长超 max_duration /
    上一句已以句末标点收尾且长度已够 —— 取「任一命中即切」。
    """
    sep = lc.get("join_sep", "")
    mc = lc["max_cue_chars"]
    maxdur = lc["max_duration_s"]
    end_punct = "。！？!?…" if lc["wrap"] == "char" else ".!?…"
    min_len_for_punct = 6 if lc["wrap"] == "char" else 18

    cues: list[Cue] = []
    cur: dict | None = None

    def flush():
        nonlocal cur
        if cur and cur["text"].strip():
            txt = re.sub(r"\s+", " ", cur["text"]).strip()
            if txt:
                cues.append(Cue(start=cur["start"], end=cur["end"], text=txt))
        cur = None

    for e in entries:
        t = (e.get("text") or "").strip()
        if not t or t.startswith("[") or t.startswith("("):     # 丢弃 [音乐]/[掌声] 类标记
            continue
        if cur is None:
            cur = {"start": e["start"], "end": e["end"], "text": t}
            continue
        new_text = (cur["text"] + sep + t) if sep else (cur["text"] + t)
        gap = e["start"] - cur["end"]
        over_len = len(new_text) > mc
        over_dur = (e["end"] - cur["start"]) > maxdur
        punct_break = (cur["text"][-1] in end_punct) and len(cur["text"]) >= min_len_for_punct
        if gap > max_gap or over_len or over_dur or punct_break:
            flush()
            cur = {"start": e["start"], "end": e["end"], "text": t}
        else:
            cur["text"] = new_text
            cur["end"] = e["end"]
    flush()
    return cues


def cues_from_srt(path: Path | str, lang: str = "zh") -> list[Cue]:
    """已有字幕文件（SRT）→ 条目（用于「跳过识别、直接用现成字幕」）。"""
    lc = lang_cfg(lang)
    return group_entries(parse_srt(path), lc)


# ------------------------------------------------------------------ ② 字幕对齐
def _frame_rms(wav: Path, frame_s: float = 0.02) -> tuple[list[float] | None, float]:
    """20ms 帧 RMS 序列（纯标准库 wave + array，零依赖）。"""
    import array
    import math
    import wave
    try:
        with wave.open(str(wav), "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            if sw != 2:
                return None, frame_s
            raw = w.readframes(w.getnframes())
    except Exception:            # noqa: BLE001
        return None, frame_s
    a = array.array("h")
    a.frombytes(raw)
    if ch == 2:
        a = a[0::2]
    per = max(1, int(sr * frame_s))
    out = []
    for i in range(0, len(a), per):
        seg = a[i:i + per]
        if not seg:
            break
        s = 0.0
        for x in seg:
            s += float(x) * float(x)
        out.append(math.sqrt(s / len(seg)))
    return (out or None), frame_s


def loud_intervals(wav: Path, rel_frac: float = 0.30, abs_floor: float = 600.0,
                   min_dur: float = 0.20, merge_gap: float = 0.15) -> list[tuple[float, float]] | None:
    """音频能量给出的「真实有声区间」；解析失败返回 None（调用方退回不收紧）。"""
    fr, fs = _frame_rms(wav)
    if not fr:
        return None
    vals = sorted(fr)
    p90 = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
    if p90 <= 0:
        return None
    thr = max(abs_floor, rel_frac * p90)
    if not any(v >= thr for v in fr):
        return None
    regs, i, n = [], 0, len(fr)
    while i < n:
        if fr[i] < thr:
            i += 1
            continue
        j, k, gap = i, i, 0
        while k < n:
            if fr[k] >= thr:
                j, gap = k, 0
            else:
                gap += 1
                if gap > merge_gap / fs:
                    break
            k += 1
        a, b = i * fs, (j + 1) * fs
        if b - a >= min_dur:
            regs.append((round(a, 3), round(b, 3)))
        i = k + 1
    return regs or None


def tighten_to_audio(cues: list[Cue], wav: Path, lc: dict,
                     head_keep: float = 0.06, tail_keep: float = 0.06,
                     min_loud: float = 0.30) -> int:
    """把每条字幕收紧到其覆盖范围内的真实有声区间（去头尾静音）。

    护栏：收紧后若短于「发音字数所需时长」则**整条放弃收紧** —— 宁可多停留，
    绝不产出念不完的字幕（原 skill 的实测教训）。
    """
    loud = loud_intervals(wav)
    if not loud:
        return 0
    safe_cps = max(12.0, lc.get("cps_hard_max", 12.0) * 0.85)
    changed = 0
    for c in cues:
        seg = [(a, b) for (a, b) in loud if b > c.start + 0.02 and a < c.end - 0.02]
        if not seg:
            continue
        a = max(c.start, min(x[0] for x in seg)) - head_keep
        b = min(c.end, max(x[1] for x in seg)) + tail_keep
        if b - a < min_loud:
            continue
        spoken = len(re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", c.text))
        if b - a < spoken / safe_cps:
            continue
        if a > c.start + 0.05 or b < c.end - 0.05:
            c.start, c.end = round(max(0.0, a), 3), round(b, 3)
            changed += 1
    return changed


def refine_with_ctc(cues: list[Cue], src: Path, lang: str = "zh", aligner: str = "",
                    progress=print) -> dict:
    """可选的 CTC 强制对齐精修：对每条字幕，在 [起点-余量, 终点+余量] 窗口内重对齐。

    失败（对齐器缺失/死锁/模型未缓存/断言）**不抛异常**，逐条跳过 —— 字幕时间退回识别结果。
    ★ 连续 2 条失败即**整体短路**：否则每条都要等一次超时/网络报错，批量会被拖垮。
    """
    al = aligner or _first_existing(ALIGNER_HINTS) or (shutil.which("ctc-forced-aligner") or "")
    report = {"engine": "ctc", "used": 0, "failed": 0, "available": bool(al)}
    if not al or not Path(al).exists():
        report["note"] = "未找到 ctc-forced-aligner，跳过精修（使用语音识别时间轴）"
        return report
    lc = lang_cfg(lang)
    # 模型已缓存时用离线模式：避免每次联网校验版本（本机实测：联网被墙 → 502 全失败）
    env = harden_env()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tmpdir = Path(tempfile.mkdtemp(prefix=f"vidpure_align_{src.stem}_"))
    consecutive_fail = 0
    try:
        wav = _to_wav16k(src, tmpdir / "full.wav")
        cursor = 0.0
        for i, c in enumerate(cues):
            win = tmpdir / f"w{i}.wav"
            txt = tmpdir / f"t{i}.txt"
            out = win.with_suffix(".txt")            # 对齐器把结果写在音频同名 .txt
            win_end = max(c.end + 0.6, cursor + 0.3)
            media.run(["ffmpeg", "-y", "-i", str(wav), "-ss", "%.3f" % cursor,
                       "-t", "%.3f" % max(win_end - cursor, 0.3), "-ac", "1",
                       "-c:a", "pcm_s16le", str(win)])
            txt.write_text(c.text, encoding="utf-8")
            try:
                rc, log = _run([al, "--audio_path", str(win), "--text_path", str(txt),
                                "--language", lc["aligner_iso"], "--split_size", "char",
                                "--romanize", "--star_frequency", "edges"],
                               env=env, timeout=90)
            except RuntimeError:
                rc, log = 1, "超时"
            rel = _parse_align_txt(out) if (rc == 0 and out.exists()) else []
            if rel:
                s = cursor + rel[0][0]
                e = cursor + rel[-1][1]
                if e > s:
                    c.start, c.end = round(s, 3), round(e, 3)
                    cursor = e
                    report["used"] += 1
                    consecutive_fail = 0
                    continue
            report["failed"] += 1
            consecutive_fail += 1
            if consecutive_fail >= 2 and report["used"] == 0:
                report["note"] = ("对齐器连续失败（多半缺模型或需联网），已整体跳过精修；"
                                  "字幕使用语音识别时间轴")
                break
        return report
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_align_txt(path: Path) -> list[tuple[float, float]]:
    """解析对齐器输出：每行形如 `0.20-0.24: H`（也兼容空白分隔的时间对）。"""
    spans: list[tuple[float, float]] = []
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return spans
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:", txt):
        a, b = float(m.group(1)), float(m.group(2))
        if b >= a:
            spans.append((a, b))
    if not spans:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", txt):
            a, b = float(m.group(1)), float(m.group(2))
            if b >= a:
                spans.append((a, b))
    return spans


def normalize_cues(cues: list[Cue], lc: dict) -> list[Cue]:
    """排序 → 保证单调不重叠 → 补齐最小可读时长。

    取舍：优先「不与下一条重叠」（重叠是肉眼可见的缺陷），其次才考虑可读时长；
    只有在极端紧贴（下一条起点早于本条所需时长）时才允许 ≤0.1s 的极小重叠。
    """
    cues = sorted([c for c in cues if c.text.strip()], key=lambda c: c.start)
    mind = lc["min_duration_s"]
    safe_cps = max(12.0, lc.get("cps_hard_max", 12.0) * 0.85)
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        # ① 先**截断重叠**：两条字幕同屏是肉眼可见的缺陷（实测 CTC 精修后曾出现 0.36s 重叠）
        if nxt is not None and c.end > nxt:
            c.end = nxt
        # ② 再补最小可读时长（延长必须受下一条起点约束，否则又把重叠放回来）
        spoken = len(re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", c.text))
        need = max(mind, spoken / safe_cps)
        if c.end - c.start < need:
            limit = nxt if nxt is not None else c.start + need
            c.end = max(c.end, min(c.start + need, limit))
        if c.end <= c.start:
            # 下一条起点紧贴本条起点（罕见）：给 0.1s 存在感，重叠极小、肉眼不可辨
            c.end = c.start + 0.1
    return cues


# ------------------------------------------------------------------ 断行（移植原引擎）
def split_two_lines(text: str, lc: dict) -> list[str]:
    """折成 ≤max_lines 行、每行 ≤max_line_chars；中文按字符（标点优先 + 容差），拉丁按单词。"""
    ml = lc["max_line_chars"]
    tol = lc.get("line_tolerance_chars", 0)
    n = len(text)
    if n <= ml:
        return [text]
    if lc.get("wrap") == "word":
        words = text.split(" ")
        for k in range(1, len(words)):
            left, right = " ".join(words[:k]), " ".join(words[k:])
            if len(left) <= ml and len(right) <= ml:
                return [left, right]
        best_k, best_w = 1, None
        for k in range(1, len(words)):
            w = max(len(" ".join(words[:k])), len(" ".join(words[k:])))
            if best_w is None or w < best_w:
                best_k, best_w = k, w
        return [" ".join(words[:best_k]), " ".join(words[best_k:])]
    punct = lc.get("break_punct", "")
    mid = n // 2
    best = None
    for i in range(1, n):
        if text[i - 1] in punct and max(i, n - i) <= ml + tol:
            score = abs(i - mid)
            if best is None or score < best[0]:
                best = (score, i)
    if best:
        return [text[:best[1]], text[best[1]:]]
    cut = min(ml + tol, n - 1)
    return [text[:cut], text[cut:]]


def split_long_cue(text: str, lc: dict) -> list[str]:
    """整句超 max_cue_chars 时按标点/词边界拆多条。"""
    mc = lc["max_cue_chars"]
    if len(text) <= mc:
        return [text]
    punct = lc.get("break_punct", "")
    chunks, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= mc or (ch in punct and len(cur) >= mc * 0.55):
            if ch not in punct and ch != " " and len(cur) >= 2:
                cut = cur.rfind(" ", 0, len(cur) - 1)
                if cut >= int(mc * 0.5):
                    chunks.append(cur[:cut + 1])
                    cur = cur[cut + 1:]
                    continue
            chunks.append(cur)
            cur = ""
    if cur:
        if chunks and len(cur) < 6:
            chunks[-1] += cur
        else:
            chunks.append(cur)
    return chunks


def layout_chars(lc: dict, max_line_chars: int) -> int:
    """断行的便捷包装（返回行宽上限）。"""
    return max_line_chars


def read_ass_anchor(ass_path: Path) -> dict:
    """从**写好的 ASS** 回读几何：PlayRes、样式 MarginV、以及每条 Dialogue 的 `\\pos` 落点。

    断言看产物、不看意图（沿用原 skill 的一贯做法）：日后谁改了写入逻辑，
    这道回读都会把「ASS 实际写的位置」与「我们要的位置」的偏差暴露出来。
    """
    try:
        txt = Path(ass_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"error": "ASS 不可读"}
    W = H = None
    style = None
    ys: list[int] = []
    for ln in txt.splitlines():
        if ln.startswith("PlayResX:"):
            W = int(ln.split(":", 1)[1].strip())
        elif ln.startswith("PlayResY:"):
            H = int(ln.split(":", 1)[1].strip())
        elif ln.startswith("Style: Default,"):
            style = ln
        elif ln.startswith("Dialogue:"):
            m = re.search(r"\\pos\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", ln)
            if m:
                ys.append(int(m.group(2)))
    out: dict = {"PlayResX": W, "PlayResY": H,
                 "style_margin_v": (int(style.split(",")[-2]) if style else None),
                 "pos_ys": ys}
    if ys:
        out["pos_y_min"], out["pos_y_max"] = min(ys), max(ys)
    return out


# ------------------------------------------------------------------ ③ ASS 渲染
def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return "%d:%02d:%02d.%02d" % (h, m, s, cs)


def compute_layout(canvas: tuple[int, int], p: Placement, lc: dict) -> dict:
    """按画布 + 摆位 + 语言算出字号、锚点、行宽上限与落点。

    · 底部模式 · ratio（默认）：视觉中心落在 `anchor_top × H`（默认 0.6 = 距顶:距底 6:4），
      随分辨率等比缩放；用 `\\an5\\pos` 实现，逐条精确居中。
    · 底部模式 · pixel：距画面底部固定像素（旧行为），受「不得顶出画面」兜底。
    · 画框模式：字号受画框高度与宽度约束，落点为框心（`\\an5\\pos`）。
    """
    W, H = canvas
    font = p.font or lc["font"]
    cw = float(lc.get("char_width_em", 1.0))

    if p.mode == "box" and p.box:
        x0, y0, x1, y1 = p.box
        bx0, by0 = int(x0 * W), int(y0 * H)
        bx1, by1 = int(x1 * W), int(y1 * H)
        box_w = max(40, bx1 - bx0)
        box_h = max(24, by1 - by0)
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        # 画框只界定「位置 + 最大行宽」；字号是独立旋钮 —— **用户显式给的字号一律照用**
        # （否则「可调字号」会被画框高度静默改掉，实测用户设 44px 被压成 15px）。
        if p.font_size_px:
            font_size = max(8, int(p.font_size_px))
            font_size_by = "explicit"
        else:
            auto_h = int(H * p.font_size_pct / 100.0)
            auto_box = int(box_h / (lc["max_lines"] * 1.35))
            font_size = max(10, min(auto_h, auto_box) if auto_box > 0 else auto_h)
            font_size_by = "auto(box)"
        max_line_chars = max(4, int(box_w / max(1.0, font_size * cw)))
        need_h = int(font_size * lc["max_lines"] * 1.35)
        out = {"mode": "box", "canvas": (W, H), "font": font,
               "font_size": font_size, "font_size_by": font_size_by,
               "anchor": "box",
               "box_px": (bx0, by0, box_w, box_h), "pos": (cx, cy),
               "center_y": cy, "bottom_to_center_px": H - cy,
               "block_h_px": need_h,
               "max_line_chars": max_line_chars, "usable_w": box_w,
               "text_w_pct": round(100.0 * box_w / W, 2),
               "margin_lr": 0, "margin_v": 0,
               "box_overflow": need_h > box_h}
        _apply_tolerance(out, lc, box_w, font_size, cw, cap=lc["max_line_chars"])
        return out

    # ---- 底部居中 ----
    margin_lr = int(round(W * p.margin_lr_pct / 100.0))
    need_lr = int(round(W * (100.0 - SAFE_MAX_TEXT_WIDTH_PCT) / 200.0))
    margin_lr = max(margin_lr, need_lr)
    usable_w = W - 2 * margin_lr
    font_h = int(round(H * p.font_size_pct / 100.0))
    # 字号拟合时预留 tolerance 的宽度，保住「宁可行长多一点、也不切词」的断行容差
    _budget = lc["max_line_chars"] + lc.get("line_tolerance_chars", 0)
    font_fit = int(usable_w / max(1.0, _budget * cw))
    font_size = int(p.font_size_px) if p.font_size_px else max(1, min(font_h, font_fit))

    # 显式给了固定像素 ⇒ 按 pixel 处理；否则走 ratio（默认 6:4）
    use_pixel = (str(p.anchor).lower() == "pixel") or (p.margin_bottom_px is not None)
    block_h = max(1, int(round(font_size * LINE_HEIGHT_EM))) * int(lc["max_lines"])
    safe = {"max_text_width_pct": SAFE_MAX_TEXT_WIDTH_PCT}

    if use_pixel:
        want = int(p.margin_bottom_px) if p.margin_bottom_px else DEFAULT_MARGIN_BOTTOM_PX
        # 画面兜底：字幕块必须留在画面内（距底 ≤ 画面高 − 两行字幕高）
        cap_v = max(8, min(int(H * SAFE_MAX_BOTTOM_FRAC),
                           H - int(math.ceil(lc["max_lines"] * 1.4 * font_size))))
        margin_v = max(8, min(want, cap_v))
        pos = None
        center_y = H - margin_v - block_h // 2
        safe.update({"mode": "pixel", "margin_bottom_px": margin_v,
                     "requested_bottom_px": want,
                     "clamped_by_canvas": want > cap_v})
    else:
        top = min(max(float(p.anchor_top), 0.05), 0.95)
        center_y = int(round(H * top))
        margin_v = 0                      # 用 \pos 定位，不使用样式 MarginV
        pos = (W // 2, center_y)
        safe.update({"mode": "ratio", "anchor_top": round(top, 4),
                     "anchor_label": ratio_label(top),
                     "margin_bottom_px": H - center_y,
                     "clamped_by_canvas": False})

    out = {"mode": "bottom", "canvas": (W, H), "font": font, "font_size": font_size,
           "font_size_by": ("explicit" if p.font_size_px
                            else ("height" if font_size == font_h else "width-fit")),
           "anchor": "pixel" if use_pixel else "ratio",
           "pos": pos, "center_y": center_y,
           "bottom_to_center_px": H - center_y, "block_h_px": block_h,
           "margin_lr": margin_lr, "margin_v": margin_v, "usable_w": usable_w,
           "text_w_pct": round(100.0 * usable_w / W, 2),
           "max_line_chars": max(4, int(usable_w / max(1.0, font_size * cw))),
           "safe_area": safe}
    _apply_tolerance(out, lc, usable_w, font_size, cw, cap=lc["max_line_chars"])
    return out


def ratio_label(top: float) -> str:
    """把 `0.6` 表述为 `6:4`（距顶:距底，十分制近似）。"""
    a = int(round(min(max(top, 0.0), 1.0) * 10))
    return f"{a}:{10 - a}"


def _apply_tolerance(out: dict, lc: dict, usable_w: int, font_size: int, cw: float,
                     cap: int) -> None:
    """把「每行字符上限」与「断行容差」约束在**可用宽度**之内。

    为什么要这一步：原 skill 允许 +3 字容差以「宁可行长一点、也不把中文词切开」。
    但当字号较大（或画框较窄）时，`max_line_chars + 容差` 会**超出可用宽度** ⇒
    字幕溢出画框 / 突破平台 70% 宽安全区。这里把容差收敛到宽度允许的余额：
    `容差_有效 = min(容差, 硬上限 - 软上限)`。字号小时余额充足，容差保持原值（行为不变）。
    """
    hard = max(4, int(usable_w / max(1.0, font_size * cw)))
    soft = min(cap, hard)
    tol = min(int(lc.get("line_tolerance_chars", 0)), max(0, hard - soft))
    out["max_line_chars"] = soft
    out["line_tolerance"] = tol
    out["hard_line_chars"] = hard
    out["tolerance_collapsed"] = (tol == 0 and int(lc.get("line_tolerance_chars", 0)) > 0)


def render_ass(cues: list[Cue], out: Path, canvas: tuple[int, int], placement: Placement,
               lang: str = "zh") -> dict:
    """生成 ASS 字幕文件并返回几何度量（供断言 / 前端展示）。"""
    lc = lang_cfg(lang)
    lay = compute_layout(canvas, placement, lc)
    W, H = canvas
    fs = lay["font_size"]
    bold = -1 if placement.bold else 0
    # 有显式落点（画框 / 6:4 视觉中心）就用 `\an5\pos` 逐条定位；pixel 模式改用样式 MarginV
    pos_tag = ""
    if lay.get("pos"):
        cx, cy = lay["pos"]
        pos_tag = "{\\an5\\pos(%d,%d)}" % (cx, cy)

    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Title: %s" % out.name,
        "PlayResX: %d" % W,
        "PlayResY: %d" % H,
        "WrapStyle: 2",                      # 2 = 不自动折行，换行全由我们控制
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
         "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
         "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"),
        "Style: Default,%s,%d,%s,&H000000FF,%s,%s,%d,0,0,0,100,100,0,0,1,%d,%d,%s,%d,%d,%d,%d"
        % (lay["font"], fs, placement.primary_colour, placement.outline_colour,
           placement.back_colour, bold, placement.outline, placement.shadow,
           "5" if lay.get("pos") else "2",
           lay["margin_lr"], lay["margin_lr"], lay["margin_v"], 1),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = []
    rendered_lines: list[str] = []
    for c in cues:
        parts = split_long_cue(c.text, lc)
        total = sum(len(x) for x in parts) or 1
        cur = c.start
        for part in parts:
            span = (c.end - c.start) * (len(part) / total)
            pst, pen = cur, min(cur + span, c.end)
            if pen - pst < lc["min_duration_s"]:
                pen = min(pst + lc["min_duration_s"], c.end)
            lines = split_two_lines(part, {
                **lc,
                "max_line_chars": lay["max_line_chars"],
                "line_tolerance_chars": lay.get("line_tolerance",
                                                lc.get("line_tolerance_chars", 0)),
            })
            txt = "\\N".join(lines).replace("{", "(").replace("}", ")")
            body.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s%s"
                        % (_ass_time(pst), _ass_time(pen), pos_tag, txt))
            rendered_lines.extend(lines)
            cur = pen
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + body) + "\n", encoding="utf-8")

    metrics = dict(lay)
    metrics["cues_in"] = len(cues)
    metrics["dialogue_lines"] = len(body)
    metrics["longest_line_chars"] = max((len(x) for x in rendered_lines), default=0)
    metrics["ass"] = str(out)
    metrics["font"] = lay["font"]
    metrics["safe_area_ok"] = _safe_area_check(metrics, W)
    return metrics


def _safe_area_check(m: dict, W: int) -> bool:
    """底部模式的合规复核。

    · ratio 模式：断言**实际落点**等于设计值 `round(anchor_top × H)`（容 1px 舍入）。
      即「产物里写的位置」与「要的 6:4 位置」一致 —— 断言看产物、不看意图。
    · pixel 模式：断言文字块仍完整落在画面内（不得被顶出画面）。
    · 两种模式都要求文本框 ≤70% 画面宽。
    画框模式是用户显式指定的位置，不参与本项判定。
    """
    if m["mode"] == "box":
        return True
    if m["text_w_pct"] > SAFE_MAX_TEXT_WIDTH_PCT + 0.05:
        return False
    if m.get("anchor") == "pixel":
        return (m["center_y"] + m["block_h_px"] / 2.0) <= m["canvas"][1] + 0.5
    top = float((m.get("safe_area") or {}).get("anchor_top", ANCHOR_TOP_DEFAULT))
    return abs(m["center_y"] - round(m["canvas"][1] * top)) <= 1


def burn_subtitles(video: Path, ass: Path, out: Path, vcodec: str = "libx264", crf: int = 18,
                   preset: str = "medium", mute: bool = False, progress=print) -> Path:
    """把 ASS 硬烧进视频（ffmpeg subtitles 滤镜）。"""
    info = media.ffprobe(video)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = "subtitles=%s" % esc_filter_path(ass)
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vf", vf, "-map", "0:v:0"]
    if info["has_audio"] and not mute:
        if out.suffix in (".mp4", ".m4v") and info["audio_codec"] in media.MP4_OK_AUDIO:
            cmd += ["-map", "0:a:0?", "-c:a", "copy"]
        else:
            cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", vcodec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    progress("[烧录字幕] 渲染中…")
    media.run(cmd)
    return out


def add_subtitles(src: Path, out: Path, lang: str = "zh", placement: Placement | None = None,
                  canvas: tuple[int, int] | None = None, srt: Path | None = None,
                  refine: str = "auto", whisper_cli: str = "", whisper_model: str = "",
                  aligner: str = "", work_dir: Path | None = None, report: dict | None = None,
                  progress=print) -> dict:
    """一体化入口：识别（或读取现成 SRT）→ 对齐精修 → 收紧 → 渲染 ASS → 烧录。

    `refine`：`"auto"` 有对齐器就精修；`"ctc"` 强制；`"none"` 只靠 whisper 时间轴。
    """
    src = Path(src)
    out = Path(out)
    placement = placement or Placement()
    info = media.ffprobe(src)
    canvas = canvas or (info["w"], info["h"])
    wd = Path(work_dir) if work_dir else out.parent
    wd.mkdir(parents=True, exist_ok=True)
    rep: dict = report if report is not None else {}

    # ① 台词来源
    if srt:
        cues = cues_from_srt(srt, lang)
        rep["source"] = f"srt:{Path(srt).name}"
        rep["asr_entries"] = len(cues)
    else:
        tr = transcribe(src, lang=lang, model=whisper_model, cli=whisper_cli,
                        out_dir=wd, progress=progress)
        cues = tr["cues"]
        rep["source"] = "whisper-asr"
        rep["asr_entries"] = tr["entries"]
        rep["whisper"] = {"cli": tr["whisper_cli"], "model": tr["model"]}
        if tr.get("asr_srt"):
            rep["asr_srt"] = tr["asr_srt"]
    if not cues:
        raise RuntimeError("未得到任何字幕条目（视频可能无人声）")

    lc = lang_cfg(lang)

    # ② 对齐精修（可选）+ 能量收紧
    tmpdir = Path(tempfile.mkdtemp(prefix=f"vidpure_sub_{src.stem}_"))
    try:
        wav = _to_wav16k(src, tmpdir / "a.wav")
        do_ctc = refine == "ctc" or (refine == "auto" and capabilities()["ctc_align"])
        if do_ctc:
            rep["ctc"] = refine_with_ctc(cues, src, lang=lang, aligner=aligner,
                                         progress=progress)
        rep["tightened"] = tighten_to_audio(cues, wav, lc)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    cues = normalize_cues(cues, lc)

    # ③ 渲染 + 烧录
    ass = wd / f"{src.stem}.ass"
    metrics = render_ass(cues, ass, canvas, placement, lang=lang)
    rep["layout"] = {k: metrics[k] for k in
                     ("mode", "anchor", "font", "font_size", "font_size_by",
                      "margin_v", "margin_lr", "max_line_chars", "text_w_pct",
                      "safe_area_ok", "center_y", "bottom_to_center_px",
                      "block_h_px", "box_overflow", "safe_area")
                     if k in metrics}
    warns: list[str] = []
    # ★ 回读写好的 ASS：确认落点确实是设计值（断言看产物）
    qa = read_ass_anchor(ass)
    if metrics.get("anchor") == "ratio" and qa.get("pos_ys"):
        want_y = int(round(canvas[1] * float(placement.anchor_top)))
        qa["anchor_expected_y"] = want_y
        qa["anchor_ok"] = all(abs(y - want_y) <= 1 for y in qa["pos_ys"])
        if not qa["anchor_ok"]:
            warns.append(f"ASS 落点与目标 {ratio_label(placement.anchor_top)} 位置不符"
                         f"（应为 y={want_y}），请复核 report.json 的 ass_qa")
    rep["ass_qa"] = qa
    if metrics.get("tolerance_collapsed"):
        warns.append("画框/字号组合较紧，已收紧断行容差；为不超框可能按字符断行（词可能被切开），"
                     "可减小字号或把框拉宽")
    if metrics.get("box_overflow"):
        warns.append(f"字号 {metrics['font_size']}px 在画框内两行会超出框高，"
                     "建议减小字号或把框拉高")
    if not metrics.get("safe_area_ok"):
        if metrics.get("anchor") == "pixel":
            warns.append("字幕块已超出画面下边界，请调小「固定距底」像素或改用「6:4 视觉中心」")
        else:
            warns.append(f"字幕文本框超过画面宽 {SAFE_MAX_TEXT_WIDTH_PCT:.0f}%")
    if (metrics.get("safe_area") or {}).get("clamped_by_canvas"):
        warns.append(f"画布高度不足以放下固定距底 "
                     f"{(metrics.get('safe_area') or {}).get('requested_bottom_px')}px，"
                     f"已压到 {metrics['margin_v']}px 以免字幕顶出画面；"
                     "如需固定距底请调高输出分辨率，或把位置改回「6:4 视觉中心」")
    if metrics.get("longest_line_chars", 0) > metrics.get("max_line_chars", 999) + 4:
        warns.append(f"存在较长字幕行（{metrics['longest_line_chars']} 字），"
                     "如需更小字号可调大「每行字符」上限")
    rep["warnings"] = warns
    rep["canvas"] = "%dx%d" % canvas
    rep["cues"] = len(cues)
    rep["dialogue_lines"] = metrics["dialogue_lines"]
    srt_out = export_srt(cues, wd / f"{src.stem}.srt")
    rep["srt"] = str(srt_out)
    rep["ass"] = str(ass)

    burn_subtitles(src, ass, out, mute=False, progress=progress)
    rep["subbed"] = str(out)
    return rep
