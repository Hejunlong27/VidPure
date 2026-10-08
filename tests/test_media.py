"""media / region 解析单元测试（不依赖 GPU）。

  pytest tests/ -q
"""

import subprocess
from pathlib import Path

import pytest

from vidpure.core.media import build_clean_filter, ffprobe, parse_region


@pytest.fixture(scope="module")
def sample(tmp_path_factory) -> Path:
    """生成 1 秒 640x360@30fps 测试视频（带正弦音轨）。"""
    p = tmp_path_factory.mktemp("media") / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(p)],
        capture_output=True, check=True)
    return p


def test_ffprobe(sample):
    info = ffprobe(sample)
    assert (info["w"], info["h"]) == (640, 360)
    assert info["has_audio"] and info["audio_codec"] == "aac"
    assert 0.9 < info["duration"] < 1.2


def test_parse_region_ratio():
    rects = parse_region("0.78,0.95", "ratio", 1920, 1080)
    x, y, w, h = rects[0]
    assert x == 1 and y == int(0.78 * 1080)
    assert w == 1918  # 全宽 - 边缘 clamp


def test_parse_region_pixel_multi():
    rects = parse_region("100,800,1600,90;100,910,1600,60", "pixel", 1920, 1080)
    assert len(rects) == 2
    assert rects[0] == (100, 800, 1600, 90)


def test_parse_region_clamp():
    # 越界区域被 clamp 而不是报错
    rects = parse_region("0.9,1.2", "ratio", 1920, 1080)
    x, y, w, h = rects[0]
    assert y + h <= 1079


def test_build_filter_delogo():
    rects = [(10, 800, 1800, 120)]
    key, filt = build_clean_filter("delogo", rects, 1920, 1080)
    assert key == "vf"
    assert filt.startswith("delogo=x=10:y=800")


def test_build_filter_blur():
    key, filt = build_clean_filter("blur", [(10, 800, 1800, 120)], 1920, 1080)
    assert key == "filter_complex"
    assert "gblur" in filt and filt.rstrip().endswith("[vout]")
