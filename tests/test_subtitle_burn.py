"""字幕识别 / 断行 / ASS 渲染几何单元测试（不依赖 GPU，不调用 whisper）。

覆盖重点：断行不越界、字号可调、安全区、手画框定位、时间轴单调不重叠。
"""

from pathlib import Path

from vidpure.core import subtitle_burn as sb


def _cues(*triples):
    return [sb.Cue(start=a, end=b, text=t) for a, b, t in triples]


# ------------------------------------------------------------------ 断行
def test_split_two_lines_zh_prefers_punctuation():
    lc = sb.lang_cfg("zh")
    lines = sb.split_two_lines("老娘就说你活蹦乱跳的怎么可能得癌症！把我那两万块吐出来！", lc)
    assert len(lines) == 2
    assert lines[0].endswith("！")          # 在标点处断，不把词切开
    assert "".join(lines) == "老娘就说你活蹦乱跳的怎么可能得癌症！把我那两万块吐出来！"


def test_split_two_lines_zh_fallback_hard_cut_respects_width():
    lc = {**sb.lang_cfg("zh"), "max_line_chars": 8, "line_tolerance_chars": 0}
    lines = sb.split_two_lines("一二三四五六七八九十一二三四五六", lc)
    assert all(len(x) <= 8 for x in lines)


def test_split_two_lines_en_never_breaks_word():
    lc = sb.lang_cfg("en")
    text = "We need a batch processing solution for sixty clips"
    lines = sb.split_two_lines(text, lc)
    assert len(lines) == 2
    # 断点必须落在空格上：首行空格数为 0 或行尾随空格
    assert not lines[0].endswith(" ") or lines[0].strip() == lines[0]
    assert all(w in text.split() for w in lines[1].split())


def test_split_long_cue_splits_over_limit():
    lc = sb.lang_cfg("zh")
    long_text = "第一句话结束了。" * 12
    parts = sb.split_long_cue(long_text, lc)
    assert len(parts) > 1
    assert all(len(p) <= lc["max_cue_chars"] for p in parts)


# ------------------------------------------------------------------ 布局 / 字号
def _style_fields(ass_text: str) -> list:
    """取出 ASS 的 `Style: Default,` 行并切字段（供产物级断言）。"""
    for ln in ass_text.splitlines():
        if ln.startswith("Style: Default,"):
            return ln.split(",")
    raise AssertionError("ASS 里找不到 Default 样式行")


def test_bottom_layout_auto_font_and_width_rule():
    lay = sb.compute_layout((1920, 1080), sb.Placement(), sb.lang_cfg("zh"))
    assert lay["mode"] == "bottom" and lay["anchor"] == "ratio"
    assert lay["text_w_pct"] <= sb.SAFE_MAX_TEXT_WIDTH_PCT + 0.05   # 文本框 ≤70%
    assert lay["font_size"] > 0
    assert sb._safe_area_check(lay, 1920) is True


def test_default_anchor_is_6_4_across_resolutions():
    """默认：文字块**视觉中心**落在 0.6H ⇒ 距顶:距底 = 6:4，横竖屏一致且随分辨率等比缩放。"""
    assert sb.ANCHOR_TOP_DEFAULT == 0.6
    assert sb.ratio_label(0.6) == "6:4"
    for w, h in ((1920, 1080), (1080, 1920), (2560, 1440), (720, 1280), (3840, 2160)):
        lay = sb.compute_layout((w, h), sb.Placement(), sb.lang_cfg("zh"))
        want = round(h * 0.6)
        assert lay["center_y"] == want, (w, h)
        assert lay["pos"] == (w // 2, want)            # 水平居中 + 垂直 0.6H
        assert lay["bottom_to_center_px"] == h - want  # 中心距底 = 0.4H
        assert lay["margin_v"] == 0                    # ratio 模式不用样式 MarginV
        assert lay["safe_area"]["clamped_by_canvas"] is False
        assert sb._safe_area_check(lay, w) is True
    # 关键性质：随高度**等比**缩放，而不是固定像素
    a = sb.compute_layout((1920, 1080), sb.Placement(), sb.lang_cfg("zh"))["center_y"]
    b = sb.compute_layout((3840, 2160), sb.Placement(), sb.lang_cfg("zh"))["center_y"]
    assert abs(b - 2 * a) <= 1


def test_anchor_top_is_configurable():
    lay = sb.compute_layout((1920, 1080), sb.Placement(anchor_top=0.7), sb.lang_cfg("zh"))
    assert lay["center_y"] == round(1080 * 0.7)
    assert lay["safe_area"]["anchor_label"] == "7:3"


def test_ass_pos_reads_back_at_6_4(tmp_path: Path):
    """断言看产物：回读写出的 ASS，逐条 `\\pos` 的 y 必须等于 0.6H。"""
    out = tmp_path / "m.ass"
    sb.render_ass(_cues((0.0, 1.0, "第一句"), (1.5, 3.0, "第二句")), out,
                  (1920, 1080), sb.Placement())
    qa = sb.read_ass_anchor(out)
    assert qa["PlayResX"] == 1920 and qa["PlayResY"] == 1080
    assert qa["pos_ys"] == [648, 648]                    # 1080 × 0.6
    f = _style_fields(out.read_text(encoding="utf-8"))
    assert f[18] == "5"                                  # 用 \an5 以块中心为锚点
    assert int(f[21]) == 0                               # 样式 MarginV 不参与
    assert int(f[19]) == int(f[20])                      # MarginL == MarginR


def test_ratio_centers_both_one_and_two_line_cues(tmp_path: Path):
    """1 行与 2 行字幕的中心都落在同一条线上（\\an5 以块中心为锚点）。"""
    short = "短句"
    long_text = "这是一句很长的台词需要折成两行才能放下"
    cues = _cues((0.0, 1.0, short), (1.2, 2.6, long_text))
    out = tmp_path / "c.ass"
    m = sb.render_ass(cues, out, (1280, 720), sb.Placement())
    qa = sb.read_ass_anchor(out)
    assert qa["pos_ys"] == [432, 432]
    assert m["dialogue_lines"] == 2


def test_pixel_anchor_uses_style_margin_v(tmp_path: Path):
    """显式 pixel 模式保留旧行为：样式 Alignment=2 + MarginV=距底像素。"""
    out = tmp_path / "p.ass"
    sb.render_ass(_cues((0.0, 1.0, "固定距底")), out, (1920, 1080),
                  sb.Placement(anchor="pixel", margin_bottom_px=300))
    txt = out.read_text(encoding="utf-8")
    assert "\\pos(" not in txt
    f = _style_fields(txt)
    assert f[18] == "2" and int(f[21]) == 300
    assert sb.read_ass_anchor(out)["style_margin_v"] == 300
    lay = sb.compute_layout((1920, 1080),
                            sb.Placement(anchor="pixel", margin_bottom_px=300),
                            sb.lang_cfg("zh"))
    assert lay["anchor"] == "pixel" and lay["pos"] is None
    assert sb._safe_area_check(lay, 1920) is True


def test_explicit_margin_bottom_implies_pixel_mode():
    """只给固定像素、不显式写 anchor 时，也应理解成 pixel 模式。"""
    lay = sb.compute_layout((1920, 1080), sb.Placement(margin_bottom_px=260), sb.lang_cfg("zh"))
    assert lay["anchor"] == "pixel" and lay["margin_v"] == 260 and lay["pos"] is None
    assert lay["safe_area"]["clamped_by_canvas"] is False


def test_pixel_default_is_400_and_clamped_on_short_canvas():
    lay = sb.compute_layout((1920, 1080), sb.Placement(anchor="pixel"), sb.lang_cfg("zh"))
    assert lay["margin_v"] == sb.DEFAULT_MARGIN_BOTTOM_PX == 400
    short = sb.compute_layout((640, 280), sb.Placement(anchor="pixel"), sb.lang_cfg("zh"))
    assert short["margin_v"] < 400
    assert short["safe_area"]["clamped_by_canvas"] is True
    assert short["center_y"] + short["block_h_px"] / 2 <= 280     # 仍在画面内
    assert sb._safe_area_check(short, 640) is True                # 不误报为不合规


def test_ratio_mode_needs_no_clamp_on_short_canvas():
    """比例模式天然安全：0.6H 必在画面内，不需要任何画布兜底。"""
    lay = sb.compute_layout((640, 280), sb.Placement(), sb.lang_cfg("zh"))
    assert lay["center_y"] == round(280 * 0.6) == 168
    assert lay["safe_area"]["clamped_by_canvas"] is False
    assert sb._safe_area_check(lay, 640) is True


def test_box_mode_ignores_anchor_settings():
    p = sb.Placement(mode="box", box=(0.1, 0.3, 0.9, 0.5), anchor="ratio", anchor_top=0.9)
    lay = sb.compute_layout((1000, 1000), p, sb.lang_cfg("zh"))
    assert lay["anchor"] == "box" and lay["pos"] == (500, 400)


def test_bottom_layout_explicit_font_wins():
    lay = sb.compute_layout((1920, 1080), sb.Placement(font_size_px=100),
                            sb.lang_cfg("zh"))
    assert lay["font_size"] == 100 and lay["font_size_by"] == "explicit"


def test_box_layout_positions_at_box_center_and_honours_font():
    p = sb.Placement(mode="box", box=(0.1, 0.6, 0.9, 0.8), font_size_px=48)
    lay = sb.compute_layout((1000, 1000), p, sb.lang_cfg("zh"))
    assert lay["mode"] == "box"
    assert lay["font_size"] == 48
    cx, cy = lay["pos"]
    assert abs(cx - 500) <= 2 and abs(cy - 700) <= 2


def test_box_width_bound_wrapping_never_exceeds_box():
    """画框 + 大字号时，容差必须被收敛，断行不得超出框宽。"""
    p = sb.Placement(mode="box", box=(0.1, 0.4, 0.5, 0.6), font_size_px=60)
    lay = sb.compute_layout((640, 360), p, sb.lang_cfg("zh"))
    cw = sb.lang_cfg("zh")["char_width_em"]
    assert lay["max_line_chars"] * lay["font_size"] * cw <= lay["usable_w"] + 1
    assert lay["line_tolerance"] == 0 and lay["tolerance_collapsed"] is True


def test_auto_font_keeps_tolerance_when_roomy():
    """字号小时容差应保持原值（保住「不切词」的既有行为）。"""
    lay = sb.compute_layout((1920, 1080), sb.Placement(), sb.lang_cfg("zh"))
    assert lay["line_tolerance"] == sb.lang_cfg("zh")["line_tolerance_chars"]


# ------------------------------------------------------------------ ASS 渲染
def test_render_ass_writes_playres_style_and_dialogues(tmp_path: Path):
    cues = _cues((0.5, 2.0, "好啊！好你个陆鸣！"))
    out = tmp_path / "a.ass"
    m = sb.render_ass(cues, out, (1280, 720), sb.Placement(font_size_px=48))
    txt = out.read_text(encoding="utf-8")
    assert "PlayResX: 1280" in txt and "PlayResY: 720" in txt
    assert "Microsoft YaHei" in txt
    assert "WrapStyle: 2" in txt
    assert txt.count("Dialogue:") == 1
    assert m["dialogue_lines"] == 1 and m["cues_in"] == 1


def test_render_ass_box_mode_injects_an5_pos(tmp_path: Path):
    cues = _cues((0.0, 1.0, "测试字幕"))
    out = tmp_path / "b.ass"
    sb.render_ass(cues, out, (1080, 1920),
                  sb.Placement(mode="box", box=(0.2, 0.7, 0.8, 0.8), font_size_px=40))
    txt = out.read_text(encoding="utf-8")
    assert "\\an5\\pos(" in txt
    assert "Alignment, MarginL" in txt or ",5," in txt


def test_render_ass_escapes_braces(tmp_path: Path):
    out = tmp_path / "c.ass"
    sb.render_ass(_cues((0.0, 1.0, "花括号{危险}文本")), out, (640, 360), sb.Placement())
    txt = out.read_text(encoding="utf-8")
    assert "{危险}" not in txt and "(危险)" in txt


# ------------------------------------------------------------------ 时间轴
def test_normalize_cues_removes_overlap_and_fills_min_duration():
    cues = _cues((0.0, 3.0, "第一句"), (1.0, 2.0, "第二句"))
    out = sb.normalize_cues(cues, sb.lang_cfg("zh"))
    assert out[0].end <= out[1].start + 1e-6      # 不重叠
    assert all(c.end > c.start for c in out)


def test_normalize_cues_drops_empty_text():
    cues = _cues((0.0, 1.0, "  "), (1.0, 2.0, "有字"))
    out = sb.normalize_cues(cues, sb.lang_cfg("zh"))
    assert len(out) == 1 and out[0].text == "有字"


# ------------------------------------------------------------------ SRT 解析 / 导出
def test_parse_srt_and_export_roundtrip(tmp_path: Path):
    srt = tmp_path / "in.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500\n你好，世界\n\n"
                   "2\n00:00:03,000 --> 00:00:04,000\n第二句\n", encoding="utf-8")
    entries = sb.parse_srt(srt)
    assert len(entries) == 2 and entries[0]["text"] == "你好，世界"
    assert abs(entries[0]["start"] - 1.0) < 1e-6

    cues = _cues((0.0, 1.25, "导出测试"))
    out = sb.export_srt(cues, tmp_path / "o.srt")
    back = sb.parse_srt(out)
    assert back[0]["text"] == "导出测试"
    assert back[0]["start"] == 0.0 and abs(back[0]["end"] - 1.25) < 1e-3


def test_group_entries_merges_words_and_breaks_on_gap():
    lc = sb.lang_cfg("zh")
    entries = [
        {"start": 0.0, "end": 0.2, "text": "你"},
        {"start": 0.2, "end": 0.4, "text": "好"},
        {"start": 0.4, "end": 0.6, "text": "世"},
        {"start": 0.6, "end": 0.8, "text": "界"},
        {"start": 3.0, "end": 3.2, "text": "再"},   # 间隔 > 0.7s ⇒ 切句
        {"start": 3.2, "end": 3.4, "text": "见"},
    ]
    cues = sb.group_entries(entries, lc)
    assert len(cues) == 2
    assert cues[0].text == "你好世界" and cues[1].text == "再见"


def test_group_entries_drops_non_speech_markers():
    lc = sb.lang_cfg("zh")
    entries = [{"start": 0, "end": 1, "text": "[音乐]"},
               {"start": 1, "end": 2, "text": "(掌声)"},
               {"start": 2, "end": 3, "text": "正文"}]
    cues = sb.group_entries(entries, lc)
    assert [c.text for c in cues] == ["正文"]


# ------------------------------------------------------------------ 工具
def test_esc_filter_path_escapes_colon_and_quotes():
    out = sb.esc_filter_path(r"D:\a b\t.ass")
    assert out == "filename='D\\:/a b/t.ass'"


def test_capabilities_shape():
    caps = sb.capabilities()
    for k in ("transcribe", "ctc_align", "whisper_cli", "whisper_model", "aligner",
              "anchor", "anchor_top", "anchor_label", "margin_bottom_px_default",
              "safe_max_text_width_pct"):
        assert k in caps
    assert caps["anchor"] == "ratio" and caps["anchor_label"] == "6:4"


def test_tighten_returns_zero_without_audio(tmp_path: Path):
    """无 wav / 解析失败时必须安全返回 0，绝不抛异常（安全阀）。"""
    cues = _cues((0.0, 1.0, "abc"))
    assert sb.tighten_to_audio(cues, tmp_path / "missing.wav", sb.lang_cfg("zh")) == 0
