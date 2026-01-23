from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Instances:
    boxes_xyxy: np.ndarray
    labels: np.ndarray
    polygons: List[Optional[np.ndarray]]
    source: np.ndarray

    def __post_init__(self) -> None:
        if self.boxes_xyxy.ndim != 2 or self.boxes_xyxy.shape[1] != 4:
            raise ValueError("boxes_xyxy must be [N,4]")
        if self.labels.shape[0] != self.boxes_xyxy.shape[0]:
            raise ValueError("labels must match boxes length")
        if len(self.polygons) != self.boxes_xyxy.shape[0]:
            raise ValueError("polygons must match boxes length")
        if self.source.shape[0] != self.boxes_xyxy.shape[0]:
            raise ValueError("source must match boxes length")


def empty_instances() -> Instances:
    return Instances(
        boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.int64),
        polygons=[],
        source=np.zeros((0,), dtype=np.int8),
    )


def bbox_xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes_xywh = boxes_xywh.astype(np.float32)
    x_c, y_c, w, h = boxes_xywh.T
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def bbox_xyxy_to_xywh(boxes_xyxy: np.ndarray) -> np.ndarray:
    boxes_xyxy = boxes_xyxy.astype(np.float32)
    x1, y1, x2, y2 = boxes_xyxy.T
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    return np.stack([x_c, y_c, w, h], axis=1)


def clip_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, height)
    return clipped


def filter_small_boxes(boxes: np.ndarray, min_area_px: float) -> np.ndarray:
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    return areas >= min_area_px


def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    return boxes * scale


def scale_polygons(polygons: List[Optional[np.ndarray]], scale: float) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for poly in polygons:
        if poly is None:
            out.append(None)
        else:
            out.append(poly.astype(np.float32) * scale)
    return out


def flip_boxes(
    boxes: np.ndarray, width: int, height: int, hflip: bool, vflip: bool
) -> np.ndarray:
    out = boxes.copy()
    if hflip:
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]
        out[:, 0] = x1
        out[:, 2] = x2
    if vflip:
        y1 = height - boxes[:, 3]
        y2 = height - boxes[:, 1]
        out[:, 1] = y1
        out[:, 3] = y2
    return out


def flip_polygons(
    polygons: List[Optional[np.ndarray]], width: int, height: int, hflip: bool, vflip: bool
) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for poly in polygons:
        if poly is None:
            out.append(None)
            continue
        pts = poly.copy().astype(np.float32)
        if hflip:
            pts[:, 0] = width - pts[:, 0]
        if vflip:
            pts[:, 1] = height - pts[:, 1]
        out.append(pts)
    return out


def translate_boxes(boxes: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] += dx
    out[:, [1, 3]] += dy
    return out


def translate_polygons(
    polygons: List[Optional[np.ndarray]], dx: float, dy: float
) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for poly in polygons:
        if poly is None:
            out.append(None)
        else:
            pts = poly.copy().astype(np.float32)
            pts[:, 0] += dx
            pts[:, 1] += dy
            out.append(pts)
    return out


def polygons_from_flat(points: Sequence[float]) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.size % 2 != 0:
        raise ValueError("Polygon points must be even length")
    return coords.reshape(-1, 2)


def normalize_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    norm = boxes.astype(np.float32).copy()
    norm[:, [0, 2]] /= float(width)
    norm[:, [1, 3]] /= float(height)
    return norm


def normalize_polygons(polygons: Iterable[Optional[np.ndarray]], width: int, height: int) -> List[Optional[np.ndarray]]:
    out: List[Optional[np.ndarray]] = []
    for poly in polygons:
        if poly is None:
            out.append(None)
            continue
        pts = poly.astype(np.float32).copy()
        pts[:, 0] /= float(width)
        pts[:, 1] /= float(height)
        out.append(pts)
    return out


def denormalize_boxes_xywh(
    boxes_xywh: np.ndarray, width: int, height: int
) -> np.ndarray:
    denorm = boxes_xywh.astype(np.float32).copy()
    denorm[:, [0, 2]] *= float(width)
    denorm[:, [1, 3]] *= float(height)
    return denorm


def denormalize_polygons(
    polygons: Iterable[np.ndarray], width: int, height: int
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).copy()
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        out.append(pts)
    return out


def boxes_from_polygons(polygons: List[Optional[np.ndarray]]) -> np.ndarray:
    boxes = []
    for poly in polygons:
        if poly is None or poly.size == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            x1 = float(np.min(poly[:, 0]))
            y1 = float(np.min(poly[:, 1]))
            x2 = float(np.max(poly[:, 0]))
            y2 = float(np.max(poly[:, 1]))
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32)
