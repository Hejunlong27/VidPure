"""VidPure Subtitle Cleaner — ComfyUI 自定义节点（去字幕）。

节点清单：
  VidPureSubtitleCleaner : IMAGE(+可选 MASK) → LaMa 修复 → IMAGE
  VidPureRegionMask      : IMAGE + 区域参数 → 矩形 MASK（白=重绘区）

自包含设计：把本目录（vidpure_nodes）整体拷入 ComfyUI/custom_nodes 即可，
不依赖 vidpure 主包；唯一可选依赖为 iopaint（LaMa 推理）。

推荐工作流：
  VHS Load Video → VidPureRegionMask → VidPureSubtitleCleaner → VHS Video Combine
"""

import torch


def _to_pil(img_t):
    """ComfyUI IMAGE tensor [H,W,3](0-1) → PIL"""
    from PIL import Image
    import numpy as np
    arr = (img_t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _from_pil(pil, batch_device="cpu"):
    """PIL → ComfyUI IMAGE tensor [1,H,W,3](0-1)"""
    import numpy as np
    arr = np.array(pil.convert("RGB")).astype("float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(batch_device)


_MM_CACHE: dict = {}


def _get_model_manager(model: str, device: str):
    """IOPaint ModelManager 单例缓存。"""
    if model in _MM_CACHE:
        return _MM_CACHE[model]
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy
    except ImportError as e:
        raise RuntimeError(
            "VidPureSubtitleCleaner 需要 iopaint："
            "在 ComfyUI 的 python 环境中执行 pip install iopaint") from e
    mm = ModelManager(name=model, device=device)
    _MM_CACHE[model] = (mm, HDStrategy)
    return _MM_CACHE[model]


class VidPureSubtitleCleaner:
    """按 mask 或矩形区域对每帧执行 LaMa 修复，消除烧录字幕/水印。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["lama", "sttn"], {"default": "lama"}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
            },
            "optional": {
                "mask": ("MASK",),          # 优先使用外部 mask（如 OCR 节点产出）
                "region": ("STRING", {"default": "0.78,0.95",
                                      "tooltip": "无 mask 时生效：y0,y1 高度比例，多段用分号分隔"}),
                "crop_margin": ("INT", {"default": 64, "min": 0, "max": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "VidPure"

    def run(self, image, model, device, mask=None, region="0.78,0.95", crop_margin=64):
        from PIL import Image

        mm, HDStrategy = _get_model_manager(model, device)
        batch = image.shape[0]
        # 无外部 mask 时按 region 生成矩形 mask
        gen_mask = None
        if mask is None:
            gen_mask = VidPureRegionMask.build_mask(image, region)

        outs = []
        for i in range(batch):
            pil = _to_pil(image[i])
            m = (mask[i] if mask is not None else gen_mask[i])
            mask_pil = Image.fromarray(
                (m.clamp(0, 1).cpu().numpy() * 255).astype("uint8"), "L")
            result = mm(pil, mask_pil,
                        hd_strategy=HDStrategy.CROP,
                        hd_strategy_crop_margin=crop_margin,
                        hd_strategy_crop_trigger_size=800,
                        hd_strategy_resize_limit=1600)
            outs.append(_from_pil(Image.fromarray(result), image.device))
        return (torch.cat(outs, dim=0),)


class VidPureRegionMask:
    """按比例/像素区域生成矩形 mask（白=需修复区）。多行字幕用分号给多段。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region": ("STRING", {"default": "0.78,0.95",
                                      "tooltip": "比例: y0,y1 或 x0,y0,x1,y1；像素: x,y,w,h；多段分号分隔"}),
                "region_format": (["auto", "ratio", "pixel"], {"default": "auto"}),
                "grow": ("INT", {"default": 4, "min": 0, "max": 128,
                                 "tooltip": "区域向外扩张像素，避免字幕描边残留"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "VidPure"

    @staticmethod
    def build_mask(image, region: str, region_format: str = "auto", grow: int = 0):
        _, h, w = image[0].shape  # [B,H,W,3]
        rects = _parse_region(region, region_format, w, h, grow)
        b = image.shape[0]
        m = torch.zeros((b, h, w), dtype=torch.float32, device=image.device)
        for x, y, rw, rh in rects:
            m[:, y:y + rh, x:x + rw] = 1.0
        return m

    def run(self, image, region, region_format, grow):
        return (self.build_mask(image, region, region_format, grow),)


def _parse_region(spec: str, fmt: str, w: int, h: int, grow: int = 0) -> list:
    rects = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        nums = [float(p) for p in parts]
        use = fmt if fmt != "auto" else (
            "ratio" if (len(nums) == 2 or (len(nums) == 1 and nums[0] <= 1)) else "pixel")
        if use == "ratio":
            if len(parts) == 1:
                x0, y0, x1, y1 = 0.0, nums[0], 1.0, 1.0
            elif len(parts) == 2:
                x0, y0, x1, y1 = 0.0, nums[0], 1.0, nums[1]
            elif len(parts) == 4:
                x0, y0, x1, y1 = nums
            else:
                raise ValueError(chunk)
            rx, ry, rw, rh = x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h
        else:
            if len(parts) != 4:
                raise ValueError(chunk)
            rx, ry, rw, rh = nums
        x = max(1, int(rx) - grow)
        y = max(1, int(ry) - grow)
        rw = min(int(rw) + grow * 2, w - 1 - x)
        rh = min(int(rh) + grow * 2, h - 1 - y)
        if rw > 0 and rh > 0:
            rects.append((x, y, rw, rh))
    if not rects:
        raise ValueError(f"区域解析为空: {spec}")
    return rects


NODE_CLASS_MAPPINGS = {
    "VidPureSubtitleCleaner": VidPureSubtitleCleaner,
    "VidPureRegionMask": VidPureRegionMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VidPureSubtitleCleaner": "VidPure 字幕清除 (LaMa)",
    "VidPureRegionMask": "VidPure 区域 Mask",
}
WEB_DIRECTORY = "./web"  # 预留前端目录
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
