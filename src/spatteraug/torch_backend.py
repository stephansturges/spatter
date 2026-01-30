from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def torch_available() -> bool:
    return torch is not None


def to_torch_image(image: np.ndarray, device: str) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    return torch.from_numpy(image).to(device)


def to_numpy_image(image: "torch.Tensor") -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    return image.detach().cpu().numpy()


def blend_roi_torch(
    base_rgb: "torch.Tensor",
    overlay_rgba: "torch.Tensor",
    x0: int,
    y0: int,
) -> None:
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    if base_rgb.device != overlay_rgba.device:
        raise ValueError("Torch blend requires base and overlay on the same device.")
    h, w = overlay_rgba.shape[:2]
    rgb = overlay_rgba[..., :3].to(dtype=torch.uint16)
    alpha = overlay_rgba[..., 3:4].to(dtype=torch.uint16)
    inv = 255 - alpha
    roi = base_rgb[y0 : y0 + h, x0 : x0 + w].to(dtype=torch.uint16)
    out = (rgb * alpha + roi * inv + 127) // 255
    base_rgb[y0 : y0 + h, x0 : x0 + w] = out.to(dtype=torch.uint8)


def blend_rois_torch(
    base_rgb: "torch.Tensor",
    overlays: list["torch.Tensor"],
    xs: list[int],
    ys: list[int],
) -> None:
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    for overlay, x0, y0 in zip(overlays, xs, ys):
        blend_roi_torch(base_rgb, overlay, x0, y0)


def resize_rgba_torch(
    image: "torch.Tensor", new_h: int, new_w: int, device: Optional[str] = None
) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not installed.")
    if device is not None:
        image = image.to(device)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("Expected RGBA image with shape HxWx4.")
    img = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    resized = torch.nn.functional.interpolate(img, size=(new_h, new_w), mode="area")
    resized = resized.squeeze(0).permute(1, 2, 0)
    return torch.clamp(resized.round(), 0, 255).to(dtype=torch.uint8)
