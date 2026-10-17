"""固定分辨率预设 —— 把「放大倍数」换成「目标分辨率」。

口径（重要，避免歧义）
--------------------------------------------------------------
预设值 = **短边像素**，即业界对 "1080p / 2160p" 的通用定义：
  · 横屏 1920x1080 → `1080p` 指短边 1080，输出 1920x1080
  · 竖屏 1080x1920 → `1080p` 仍指短边 1080，输出 1080x1920

为什么用固定分辨率而不是「2x / 4x」倍数
  · 倍数是相对量，源分辨率一变，输出尺寸就漂移（竖屏 768x1376 的 2x = 1536x2752，
    既不是标准尺寸、也不是任何平台的上传规格）。
  · 平台（抖音/YouTube/发行方）要的是**固定尺寸**；固定目标能让批量任务产出一致的规格。
  · 长边按源宽高比**等比推算**并**对齐到偶数**（yuv420p 编码要求宽高均为偶数，
    奇数会让 libx264 静默失败或自动砍一列像素）。

不放大即 `source`（返回 None），与 vidpure 既有「默认按源分辨率出片」口径一致。
"""

from __future__ import annotations

# 预设表：id / 展示名 / 短边像素（0 = 保持原始，不缩放）
PRESETS: list[dict] = [
    {"id": "source", "label": "原始分辨率", "short_px": 0,
     "desc": "不做任何缩放，保持源尺寸"},
    {"id": "720p", "label": "720p", "short_px": 720,
     "desc": "短边 720 · 体积小、出片快"},
    {"id": "1080p", "label": "1080p", "short_px": 1080,
     "desc": "短边 1080 · 主流平台标准"},
    {"id": "1440p", "label": "1440p (2K)", "short_px": 1440,
     "desc": "短边 1440 · 高清交付"},
    {"id": "2048p", "label": "2048p (2K DCI)", "short_px": 2048,
     "desc": "短边 2048 · DCI 2K 档"},
    {"id": "2160p", "label": "2160p (4K)", "short_px": 2160,
     "desc": "短边 2160 · UHD 4K"},
    {"id": "4096p", "label": "4096p (4K DCI)", "short_px": 4096,
     "desc": "短边 4096 · DCI 4K；显存/耗时压力极大"},
]

_BY_ID = {p["id"]: p for p in PRESETS}

# 超过该像素总数（约 12MP）时给出显存/耗时警告
HUGE_MP = 12.0


def list_presets() -> list[dict]:
    """返回可直接喂给前端/CLI 的预设列表（深拷贝，调用方随便改）。"""
    return [dict(p) for p in PRESETS]


def even(n: float) -> int:
    """向下取最近的**偶数**（≥2）。yuv420p 要求宽高为偶数。"""
    n = int(round(n))
    n += n & 1          # 奇数 +1 → 偶数（向上取，避免意外缩小）
    return max(2, n)


def megapixels(w: int, h: int) -> float:
    return round((w * h) / 1_000_000.0, 2)


def parse_spec(spec) -> dict:
    """把用户输入解析成目标描述。

    接受：
      · `"source"` / `"none"` / `"0"` / `None` / `""`  → 不缩放
      · `"1080p"` / `"2160p"` / `"1440"` / `1080`（int） → 短边像素
      · `"1920x1080"` / `"1920*1080"` / `"1920:1080"`  → 精确尺寸

    返回 `{"kind": "source"}` 或
        `{"kind": "short", "short_px": int, "id": str|None}` 或
        `{"kind": "exact", "w": int, "h": int}`。
    """
    if spec is None:
        return {"kind": "source"}
    if isinstance(spec, (int, float)):
        v = int(spec)
        return {"kind": "source"} if v <= 0 else {"kind": "short", "short_px": v, "id": None}

    s = str(spec).strip().lower()
    if s in ("", "source", "none", "orig", "original", "0"):
        return {"kind": "source"}

    # 精确 WxH
    for sep in ("x", "*", ":", "×"):
        if sep in s:
            a, _, b = s.partition(sep)
            try:
                w, h = int(float(a)), int(float(b))
            except ValueError:
                raise ValueError(f"无法解析分辨率: {spec}")
            if w <= 0 or h <= 0:
                raise ValueError(f"分辨率必须为正: {spec}")
            return {"kind": "exact", "w": even(w), "h": even(h)}

    if s in _BY_ID:
        p = _BY_ID[s]
        return {"kind": "source"} if not p["short_px"] else {
            "kind": "short", "short_px": p["short_px"], "id": p["id"]}

    if s.endswith("p"):
        s = s[:-1]
    try:
        v = int(float(s))
    except ValueError:
        raise ValueError(f"无法解析分辨率: {spec}")
    if v <= 0:
        return {"kind": "source"}
    return {"kind": "short", "short_px": v, "id": None}


def resolve_target(src_w: int, src_h: int, spec) -> tuple[int, int] | None:
    """按源尺寸与目标描述算出输出尺寸 `(W, H)`；`None` = 不缩放。

    短边模式：把**短边**顶到目标值，长边按源宽高比等比推算，两者都对齐偶数。
    """
    t = parse_spec(spec)
    if t["kind"] == "source":
        return None
    if t["kind"] == "exact":
        return t["w"], t["h"]

    short = int(t["short_px"])
    if src_w <= 0 or src_h <= 0:
        raise ValueError("源分辨率无效")
    if src_w >= src_h:                      # 横屏：短边是高度
        new_h = short
        new_w = even(src_w * short / src_h)
    else:                                   # 竖屏：短边是宽度
        new_w = short
        new_h = even(src_h * short / src_w)
    return new_w, new_h


def describe(src_w: int, src_h: int, spec) -> dict:
    """给前端/日志用：目标尺寸 + 是否需要缩放 + 体量提示。"""
    target = resolve_target(src_w, src_h, spec)
    out: dict = {"source": f"{src_w}x{src_h}", "target": None, "noop": target is None}
    if target is None:
        out["target"] = f"{src_w}x{src_h}"
        out["noop"] = True
        return out
    tw, th = target
    out["target"] = f"{tw}x{th}"
    out["mp"] = megapixels(tw, th)
    out["huge"] = out["mp"] > HUGE_MP
    out["up"] = tw * th > src_w * src_h
    return out


def warnings_for(src_w: int, src_h: int, spec) -> list[str]:
    """对「过大 / 反而缩小」的目标给出中文提示（不阻断，只提醒）。"""
    d = describe(src_w, src_h, spec)
    w: list[str] = []
    if d.get("noop"):
        return w
    if d.get("huge"):
        w.append(
            f"目标 {d['target']}（{d['mp']}MP）体积很大："
            "逐帧 AI 超分将显著变慢并可能显存不足，建议先用小样试跑一段")
    if not d.get("up"):
        w.append(f"目标 {d['target']} 小于源 {d['source']}：这是缩小而非放大，画质不会提升")
    return w
