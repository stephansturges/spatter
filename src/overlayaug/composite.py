from __future__ import annotations

from typing import Tuple

import numpy as np


def blend_roi(base_rgb: np.ndarray, overlay_rgba: np.ndarray, x0: int, y0: int) -> None:
    h, w = overlay_rgba.shape[:2]
    rgb = overlay_rgba[..., :3].astype(np.uint16)
    alpha = overlay_rgba[..., 3:4].astype(np.uint16)
    inv = 255 - alpha

    roi = base_rgb[y0 : y0 + h, x0 : x0 + w].astype(np.uint16)
    out = (rgb * alpha + roi * inv + 127) // 255
    base_rgb[y0 : y0 + h, x0 : x0 + w] = out.astype(np.uint8)


def update_coverage(
    coverage: np.ndarray, alpha_patch: np.ndarray, x0: int, y0: int, alpha_thr: int
) -> None:
    h, w = alpha_patch.shape
    y1 = y0 + h
    x1 = x0 + w
    coverage[y0:y1, x0:x1] |= alpha_patch > alpha_thr


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def clip_patch(
    overlay_rgba: np.ndarray, x0: int, y0: int, width: int, height: int
) -> Tuple[np.ndarray, int, int]:
    h, w = overlay_rgba.shape[:2]
    x1 = x0 + w
    y1 = y0 + h
    crop_x0 = max(0, -x0)
    crop_y0 = max(0, -y0)
    crop_x1 = w - max(0, x1 - width)
    crop_y1 = h - max(0, y1 - height)
    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return overlay_rgba[:0, :0], 0, 0
    clipped = overlay_rgba[crop_y0:crop_y1, crop_x0:crop_x1]
    return clipped, x0 + crop_x0, y0 + crop_y0
