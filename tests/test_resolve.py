"""固定分辨率预设单元测试（不依赖 GPU / FFmpeg）。"""

import pytest

from vidpure.core import resolve


def test_preset_ids_and_order():
    ids = [p["id"] for p in resolve.list_presets()]
    assert ids[0] == "source"
    for k in ("720p", "1080p", "1440p", "2048p", "2160p", "4096p"):
        assert k in ids
    # 列表可安全修改，不影响模块常量
    resolve.list_presets()[0]["label"] = "x"
    assert resolve.list_presets()[0]["label"] != "x"


def test_landscape_short_edge():
    assert resolve.resolve_target(1920, 1080, "1080p") == (1920, 1080)
    assert resolve.resolve_target(1280, 720, "2160p") == (3840, 2160)


def test_portrait_short_edge_is_width():
    # 竖屏 768x1376：短边是宽度 ⇒ 1080p → 宽 1080，高按比例推算
    w, h = resolve.resolve_target(768, 1376, "1080p")
    assert w == 1080 and h == 1936 and h % 2 == 0


def test_source_and_none():
    assert resolve.resolve_target(640, 360, "source") is None
    assert resolve.resolve_target(640, 360, None) is None
    assert resolve.resolve_target(640, 360, "none") is None
    assert resolve.resolve_target(640, 360, 0) is None


def test_exact_dimensions_and_even_snap():
    assert resolve.resolve_target(640, 360, "1920x1080") == (1920, 1080)
    # 奇数尺寸向上取偶（yuv420p 要求偶数）
    assert resolve.resolve_target(640, 360, "1081x601") == (1082, 602)
    assert resolve.resolve_target(640, 360, "1920*1080") == (1920, 1080)
    assert resolve.resolve_target(640, 360, "1920:1080") == (1920, 1080)


def test_parse_spec_variants():
    assert resolve.parse_spec("1080p")["kind"] == "short"
    assert resolve.parse_spec("1080")["short_px"] == 1080
    assert resolve.parse_spec(720)["short_px"] == 720
    assert resolve.parse_spec("1440p")["id"] == "1440p"
    assert resolve.parse_spec("source")["kind"] == "source"


def test_parse_spec_invalid():
    with pytest.raises(ValueError):
        resolve.parse_spec("abc")
    with pytest.raises(ValueError):
        resolve.parse_spec("0x1080")


def test_describe_and_warnings():
    d = resolve.describe(640, 360, "1080p")
    assert d["target"] == "1920x1080" and d["up"] and not d["noop"]
    assert resolve.describe(640, 360, "source")["noop"] is True
    # 4096p 在横屏 640x360 上是巨量像素 ⇒ 必须给出提醒
    assert resolve.warnings_for(640, 360, "4096p")
    # 目标小于源 ⇒ 提示这是缩小
    assert resolve.warnings_for(4000, 2000, "720p")


def test_even_helper():
    assert resolve.even(3) == 4
    assert resolve.even(4) == 4
    assert resolve.even(0) == 2
