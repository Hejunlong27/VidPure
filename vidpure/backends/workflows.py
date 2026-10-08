"""API 格式工作流模板（提交给 ComfyUI /prompt 的 JSON）。

节点依赖：ComfyUI-VideoHelperSuite + ComfyUI-IOPaint。
节点类名以社区版本为准，若你安装的版本类名不同，请在此处对应修改。
"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def subtitle_workflow(video_path: str, region: str = "0.78,0.95",
                      model: str = "lama", out_name: str = "cleaned.mp4") -> dict:
    """去字幕工作流：VHS 载入视频 → IOPaint(LaMa) 修复 → VHS 合帧（保留音频）。"""
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_path, "custom_width": 0, "custom_height": 0,
            "frame_rate": 0, "frame_load_cap": 0, "skip_first_frames": 0,
            "select_every_nth": 1, "prefer_vae_device": "cpu"}},
        "2": {"class_type": "IOPaint", "inputs": {
            "image": ["1", 0], "mask": ["3", 0], "model": model,
            "device": "cuda", "invert_mask": False}},
        "3": {"class_type": "VidPureRegionMask", "inputs": {
            "image": ["1", 0], "region": region}},
        "4": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["2", 0], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": out_name.rsplit(".", 1)[0],
            "format": "video/h264-mp4", "pix_fmt": "yuv420p",
            "crf": 18, "save_metadata": True,
            "audio": ["1", 1]}},
    }


def upscale_workflow(video_path: str, model_name: str = "4x-UltraSharp.pth",
                     out_name: str = "upscaled.mp4") -> dict:
    """放大工作流：VHS 载入 → ComfyUI 原生超分模型 → VHS 合帧。"""
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {
            "video": video_path, "custom_width": 0, "custom_height": 0,
            "frame_rate": 0, "frame_load_cap": 0, "skip_first_frames": 0,
            "select_every_nth": 1, "prefer_vae_device": "cpu"}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {
            "upscale_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["3", 0], "frame_rate": 24, "loop_count": 0,
            "filename_prefix": out_name.rsplit(".", 1)[0],
            "format": "video/h264-mp4", "pix_fmt": "yuv420p",
            "crf": 18, "save_metadata": True, "audio": ["1", 1]}},
    }


def dump(workflow: dict) -> str:
    return json.dumps(workflow, ensure_ascii=False, indent=2)


def save(workflow: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(dump(workflow), encoding="utf-8")
    return p
